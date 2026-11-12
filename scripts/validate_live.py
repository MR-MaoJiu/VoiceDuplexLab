import argparse
import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import re
import secrets
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi
import uvicorn
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus
from voice_lab import backend as backend_module
from voice_lab.config import ROOT, load_config
from voice_lab.data import load_manifest
from voice_lab.download import atomic_json
from voice_lab.model import VoiceRuntime, read_audio
from voice_lab.pipeline import save_yaml
from voice_lab.server import create_app

parser = argparse.ArgumentParser(description='真实工具、等待期间插话、双用户音色与转写验收')
parser.add_argument('--config', default='configs/repair-candidate.yaml')
parser.add_argument('--output', default='runs/repair-multiuser-validation')
parser.add_argument('--fixtures', default='runs/runtime-fixtures/queries.json')
parser.add_argument('--tool-delay-seconds', type=float, default=12)
args = parser.parse_args()
if args.tool_delay_seconds < 0:
    parser.error('工具验收延迟不能小于0')
OUTPUT = ROOT / args.output
OUTPUT.mkdir(parents=True, exist_ok=True)
options = onnxruntime.SessionOptions()
options.intra_op_num_threads = 1
speaker = onnxruntime.InferenceSession(
    str(ROOT / "models/MiniCPM-o-4_5/assets/token2wav/campplus.onnx"),
    sess_options=options, providers=["CPUExecutionProvider"],
)


def speaker_vector(path):
    audio = read_audio(path)
    active = np.flatnonzero(np.abs(audio) > 0.001)
    if len(active):
        audio = audio[active[0]:active[-1]+1]
    if len(audio) < 8000:
        raise ValueError("可用于音色验证的实际语音不足半秒")
    features = kaldi.fbank(torch.from_numpy(audio).unsqueeze(0), num_mel_bins=80,
                           dither=0, sample_frequency=16000)
    features -= features.mean(dim=0, keepdim=True)
    value = speaker.run(None, {speaker.get_inputs()[0].name: features.unsqueeze(0).numpy()})[0].reshape(-1)
    return value / max(float(np.linalg.norm(value)), 1e-8)


async def main():
    config = load_config(args.config)
    checkpoint = Path(config["model"]["adapter"]) if config["model"].get("adapter") else None
    config["model"].update(adapter=str(checkpoint) if checkpoint else None, decode_mode="greedy")
    config["server"]["worker_devices"] = ["cuda:0", "cuda:0"]
    serving = copy.deepcopy(config)
    records = load_manifest(ROOT / "data/formal/full-corpus-v1/real_eval.jsonl")
    candidates = list(dict.fromkeys(row["voice"] for row in records if Path(row["voice"]).is_file() and sf.info(row["voice"]).duration >= 3))[:48]
    vectors = np.stack([speaker_vector(path) for path in candidates])
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, 2)
    left, right = np.unravel_index(np.argmin(similarity), similarity.shape)
    references = {"alice": candidates[left], "bob": candidates[right]}
    reference_vectors = np.stack([vectors[left], vectors[right]])
    tokens = {name: secrets.token_urlsafe(24) for name in references}
    for name, token in tokens.items():
        os.environ["ACCEPTANCE_"+name.upper()] = token
    config["server"].update(host="127.0.0.1", port=18765,
                              voice_store=str(OUTPUT / "voices"),
                              users={name: "ACCEPTANCE_"+name.upper() for name in tokens})
    starts, actual_calls = [], []
    from voice_lab.asr import SpeechTranscriber
    transcriber = SpeechTranscriber(config['asr'])
    original_start = VoiceRuntime.start

    def start(runtime, voice=None):
        starts.append({"model": id(runtime), "voice": str(voice)})
        return original_start(runtime, voice)

    VoiceRuntime.start = start
    original_tool = backend_module.execute_tool

    async def tool(name, arguments, settings):
        # 仅验收注入固定延迟，以稳定检查任务等待期间的插话。
        await asyncio.sleep(args.tool_delay_seconds)
        value = await original_tool(name, arguments, settings)
        actual_calls.append({"name": name, "arguments": arguments, "result": value})
        return value

    backend_module.execute_tool = tool
    server = uvicorn.Server(uvicorn.Config(create_app(config), host="127.0.0.1", port=18765,
                                           log_level="warning", loop="asyncio"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    report = {"checkpoint": str(checkpoint), "decode_mode": "greedy", "passed": False,
              "reference_pair_similarity": float(similarity[left, right]), "controlled_tool_delay_seconds": args.tool_delay_seconds, "cases": {}}
    fixtures = json.loads((ROOT / args.fixtures).read_text())

    def headers(user):
        return {"Authorization": "Bearer "+tokens[user]}

    async def talk(user, case, query_name, tool_case=False):
        events, waves, texts = [], [], []
        completed = asyncio.Event()
        status = {"task_started": None, "task_completed": None, "error": None,
                  "premature_result": False, "speech_while_pending": False}
        source = read_audio(fixtures[query_name]["path"])
        followup = read_audio(fixtures["followup"]["path"])
        offset, followup_sent = 0, False
        async with connect(f"ws://127.0.0.1:18765/ws?voice={voice_ids[user]}",
                           additional_headers=headers(user), open_timeout=60,
                           max_size=4*1024*1024) as websocket:
            ready = json.loads(await asyncio.wait_for(websocket.recv(), 90))
            if ready.get("type") != "ready":
                raise RuntimeError(str(ready))
            began = time.monotonic()

            async def receive():
                try:
                    async for raw in websocket:
                        event = json.loads(raw)
                        now = time.monotonic()-began
                        event["at"] = now
                        encoded = event.pop("audio", None)
                        if encoded:
                            value = np.frombuffer(base64.b64decode(encoded), dtype="<i2").astype(np.float32)/32768
                            waves.append(value)
                            event["audio_samples"] = len(value)
                        events.append(event)
                        if event["type"] == "task_started":
                            status["task_started"] = now
                        elif event["type"] == "task_completed":
                            status["task_completed"] = now
                        elif event["type"] in ("error", "control_error"):
                            status["error"] = event
                            completed.set()
                        elif event["type"] == "speech":
                            texts.append(event["text"])
                            joined = "".join(texts)
                            if status["task_started"] is not None and status["task_completed"] is None:
                                if event.get("audio_samples", 0):
                                    status["speech_while_pending"] = True
                                if "893" in joined or "八百九十三" in joined or re.search(r"结果(?:是|为|显示).{0,10}[0-9零一二三四五六七八九十百千万]", joined):
                                    status["premature_result"] = True
                            if tool_case:
                                if status["task_completed"] is not None and event.get("end") and any(number in joined for number in ("893", "八百九十三", "八九三")):
                                    completed.set()
                            elif event.get("end") and waves:
                                from voice_lab.floor import is_preface
                                from voice_lab.dialogue_data import normalize_audio_text
                                if len(normalize_audio_text(joined)) >= 8 and not is_preface(joined):
                                    completed.set()
                except ConnectionClosed:
                    pass

            receiver = asyncio.create_task(receive())
            try:
                for index in range(75 if tool_case else 28):
                    frame_started = time.monotonic()
                    if tool_case and not followup_sent and offset >= len(source) and status["task_started"] is not None and status["task_completed"] is None and frame_started-began-status["task_started"] >= 3:
                        await websocket.send(json.dumps({"type": "interrupt"}))
                        source, offset, followup_sent = followup, 0, True
                    size = ready["first_samples"] if index == 0 else ready["samples"]
                    chunk = source[offset:offset+size]
                    offset += size
                    chunk = np.pad(chunk, (0, size-len(chunk)))
                    await websocket.send((np.clip(chunk,-1,1)*32767).astype("<i2").tobytes())
                    if completed.is_set() and (tool_case or offset >= len(source)):
                        break
                    await asyncio.sleep(max(0,1-(time.monotonic()-frame_started)))
                await websocket.send(json.dumps({"type": "close"}))
                await asyncio.wait_for(receiver, 15)
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
        path = OUTPUT / (case+".wav")
        wave = np.concatenate(waves) if waves else np.zeros(0)
        if len(wave):
            sf.write(path, wave, 24000, subtype="PCM_16")
        value = {**status, "user": user, "text": "".join(texts), "events": events,
                 "audio": str(path), "seconds": len(wave)/24000,
                 "audio_valid": bool(len(wave)>12000 and np.sqrt(np.mean(wave**2))>0.001)}
        if tool_case:
            value["passed"] = value["audio_valid"] and status["task_started"] is not None and status["task_completed"] is not None and status["speech_while_pending"] and not status["premature_result"] and completed.is_set() and status["error"] is None
        else:
            value["passed"] = value["audio_valid"] and status["task_started"] is None and status["error"] is None
        value['user_transcripts'] = [event['text'] for event in events if event['type'] == 'user_transcript']
        if config.get('asr', {}).get('enabled'):
            value['passed'] = value['passed'] and bool(value['user_transcripts'])
        if len(wave):
            from difflib import SequenceMatcher
            from voice_lab.dialogue_data import normalize_audio_text, spoken_numbers
            actual = transcriber.recognize(wave, 24000)
            ratio = SequenceMatcher(None, normalize_audio_text(value['text']), normalize_audio_text(actual)).ratio()
            value.update(transcript=actual, audio_text_similarity=ratio)
            value['audio_numbers_match'] = spoken_numbers(value['text']) == spoken_numbers(actual)
            value['passed'] = value['passed'] and ratio >= 0.8 and value['audio_numbers_match']
        print(case, value["passed"], value["text"], flush=True)
        return value

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            for attempt in range(90):
                try:
                    if (await client.get("http://127.0.0.1:18765/health")).json().get("ready"):
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
            else:
                raise RuntimeError("真实语音服务启动超时")
            voice_ids = {}
            for user, reference in references.items():
                response = await client.post("http://127.0.0.1:18765/voices?name="+user,
                                             headers=headers(user), content=Path(reference).read_bytes())
                response.raise_for_status()
                voice_ids[user] = response.json()["id"]
            bob_voices = (await client.get("http://127.0.0.1:18765/voices",headers=headers("bob"))).json()["voices"]
            ownership = all(value["id"] != voice_ids["alice"] for value in bob_voices)
            denied = False
            try:
                async with connect(f"ws://127.0.0.1:18765/ws?voice={voice_ids['alice']}",additional_headers=headers("bob")):
                    pass
            except InvalidStatus as error:
                denied = error.response.status_code == 403
            report["owner_isolation"] = ownership and denied
            a, b = await asyncio.gather(talk("alice","alice-greet","greet"),talk("bob","bob-greet","greet"))
            report["cases"].update(alice_greet=a,bob_greet=b)
            report["cases"]["bob_tool"] = await talk("bob","bob-tool","calc",True)
            report["cases"]["bob_reuse"] = await talk("bob","bob-reuse","greet")
            generated = np.stack([speaker_vector(a["audio"]),speaker_vector(b["audio"])])
            scores = generated @ reference_vectors.T
            report["speaker_similarity_matrix"] = scores.tolist()
            report["different_voice_conditioning"] = bool(scores[0,0] > scores[0,1] and scores[1,1] > scores[1,0])
            reused = np.stack([speaker_vector(report["cases"][name]["audio"]) for name in ("bob_tool","bob_reuse")]) @ reference_vectors.T
            report["reused_speaker_similarity"] = reused.tolist()
            report["reused_voice_conditioning"] = bool(np.all(reused[:,1] > reused[:,0]))
            histories = {}
            for item in starts:
                histories.setdefault(item["model"],set()).add(item["voice"])
            report["model_reuse_with_voice_switch"] = any(len(values)>1 for values in histories.values())
            report["actual_tool_calls"] = actual_calls
            report["actual_calculator"] = any(item["name"]=="calculate" and item["result"].get("value")==893 for item in actual_calls)
            report["passed"] = report["owner_isolation"] and report["different_voice_conditioning"] and report["reused_voice_conditioning"] and report["model_reuse_with_voice_switch"] and report["actual_calculator"] and all(value["passed"] for value in report["cases"].values())
            report["references"] = references
            if report["passed"]:
                save_yaml(ROOT / "configs/repair-runtime-verified.yaml",serving)
    except Exception as error:
        report["error"] = type(error).__name__+": "+str(error)
    finally:
        atomic_json(OUTPUT / "report.json",report)
        transcriber.close()
        server.should_exit = True
        await asyncio.to_thread(thread.join,30)
    print(json.dumps({key:value for key,value in report.items() if key!="cases"},ensure_ascii=False),flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == '__main__':
    asyncio.run(main())
