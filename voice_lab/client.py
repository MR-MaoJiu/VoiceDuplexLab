import argparse
import asyncio
import base64
from collections import deque
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from urllib.parse import urlencode
import numpy as np
import websockets


class Playback:
    def __init__(self):
        self.blocks = deque()
        self.lock = threading.Lock()
        self.playing = False

    def clear(self):
        with self.lock:
            self.blocks.clear()
            self.playing = False

    def append(self, data):
        with self.lock:
            self.blocks.append(np.frombuffer(data, dtype="<i2").copy())

    def callback(self, outdata, frames, timing, status):
        result = np.zeros(frames, dtype=np.int16)
        offset = 0
        with self.lock:
            while self.blocks and offset < frames:
                block = self.blocks[0]
                count = min(len(block), frames - offset)
                result[offset : offset + count] = block[:count]
                offset += count
                if count == len(block):
                    self.blocks.popleft()
                else:
                    self.blocks[0] = block[count:]
            self.playing = bool(offset)
        outdata[:] = result.tobytes()


async def run(args):
    import sounddevice as sd

    token = os.environ.get(args.token_env, "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = args.url.rstrip("/")
    url += "?" + urlencode({"voice": args.voice}) if args.voice else ""
    recording = Path(args.record) if args.record else None
    if recording:
        if recording.exists():
            raise FileExistsError("录音输出目录已存在，请使用新的目录")
        recording.mkdir(parents=True)
    playback = Playback()
    capture = queue.Queue(maxsize=100)
    overflow = threading.Event()
    last_interrupt = 0.0
    started = time.monotonic()
    audio_history = []
    events = []
    loop = asyncio.get_running_loop()
    commands = asyncio.Queue()
    epoch = 0

    def microphone(data, frames, timing, status):
        nonlocal last_interrupt
        block = bytes(data)
        try:
            capture.put_nowait(block)
        except queue.Full:
            overflow.set()
        if args.auto_interrupt and playback.playing:
            samples = np.frombuffer(block, dtype=np.int16).astype(np.float32) / 32768
            energy = float(np.sqrt(np.mean(samples**2)))
            now = time.monotonic()
            if energy > args.interrupt_threshold and now - last_interrupt > 1.5:
                last_interrupt = now
                playback.clear()
                loop.call_soon_threadsafe(commands.put_nowait, {"type": "interrupt"})

    async with websockets.connect(
        url, additional_headers=headers, max_size=2_000_000, open_timeout=600
    ) as ws:
        first = json.loads(await ws.recv())
        if first.get("type") != "ready":
            raise RuntimeError(f"服务未准备好 {first}")
        print("已连接。闲聊直接回答；工具结果会自动回注。建议使用耳机。")
        print(
            "输入 /interrupt 打断，/cancel 取消后台任务，/task 描述 可测试后台，/quit 结束。"
        )

        def stdin_ready():
            line = sys.stdin.readline()
            if not line:
                loop.remove_reader(sys.stdin.fileno())
                return
            line = line.strip()
            mapping = {"/interrupt": "interrupt", "/cancel": "cancel", "/quit": "close"}
            if line in mapping:
                playback.clear()
                commands.put_nowait({"type": mapping[line]})
            elif line.startswith("/task "):
                commands.put_nowait({"type": "delegate", "task": line[6:]})

        has_stdin = sys.stdin.isatty()
        if has_stdin:
            loop.add_reader(sys.stdin.fileno(), stdin_ready)

        async def send_audio():
            pending = bytearray()
            size = first["first_samples"] * 2
            while True:
                await asyncio.sleep(0.01)
                if overflow.is_set():
                    raise RuntimeError("客户端录音积压，请检查网络或服务器性能")
                while not capture.empty():
                    data = capture.get_nowait()
                    if recording:
                        audio_history.append(data)
                    pending.extend(data)
                while len(pending) >= size:
                    await ws.send(bytes(pending[:size]))
                    del pending[:size]
                    size = first["samples"] * 2

        async def send_commands():
            while True:
                value = await commands.get()
                await ws.send(json.dumps(value, ensure_ascii=False))
                if value["type"] == "close":
                    return

        async def receive():
            nonlocal epoch
            async for raw in ws:
                event = json.loads(raw)
                if recording:
                    events.append(
                        {
                            "client_time": time.monotonic() - started,
                            **{
                                key: value
                                for key, value in event.items()
                                if key != "audio"
                            },
                        }
                    )
                kind = event["type"]
                if kind == "interrupted":
                    epoch = event["epoch"]
                    playback.clear()
                elif kind == "speech":
                    if event["epoch"] < epoch:
                        continue
                    print(event["text"], end="\n" if event["end"] else "", flush=True)
                    playback.append(base64.b64decode(event["audio"], validate=True))
                else:
                    print("\n" + json.dumps(event, ensure_ascii=False), flush=True)
                if kind == "error":
                    raise RuntimeError(event["message"])

        tasks = []
        try:
            with (
                sd.RawInputStream(
                    samplerate=16000,
                    blocksize=1600,
                    channels=1,
                    dtype="int16",
                    callback=microphone,
                ),
                sd.RawOutputStream(
                    samplerate=24000,
                    blocksize=2400,
                    channels=1,
                    dtype="int16",
                    callback=playback.callback,
                ),
            ):
                tasks = [
                    asyncio.create_task(send_audio()),
                    asyncio.create_task(send_commands()),
                    asyncio.create_task(receive()),
                ]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if has_stdin:
                loop.remove_reader(sys.stdin.fileno())
            if recording:
                import soundfile as sf

                audio = np.frombuffer(b"".join(audio_history), dtype="<i2")
                sf.write(recording / "user.wav", audio, 16000, subtype="PCM_16")
                (recording / "events.json").write_text(
                    json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    f"输入音频与会话事件已保存至 {recording}，训练前需要人工校对时间与目标文本。"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--voice", help="当前用户拥有的音色 ID")
    parser.add_argument("--token-env", default="VOICE_TOKEN")
    parser.add_argument("--record", help="可选，保存输入音频与事件到新的目录")
    parser.add_argument(
        "--auto-interrupt",
        action="store_true",
        help="启用音量阈值打断，需耳机并按环境调整阈值",
    )
    parser.add_argument("--interrupt-threshold", type=float, default=0.08)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n已结束")


if __name__ == "__main__":
    main()
