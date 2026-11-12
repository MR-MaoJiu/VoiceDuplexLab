import argparse
import asyncio
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import asynccontextmanager
import functools
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
import uvicorn
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .backend import Backend, TaskCoordinator, result_context
from .config import load_config
from .model import InferenceInterrupted
from .voices import VoiceStore, principal
from .expression import UserAudioHistory, voice_commands

log = logging.getLogger("voice_lab")


def create_app(config, runtime_factory=None, transcriber_factory=None):
    available = asyncio.Queue()
    slots = []
    transcriber = None
    store = VoiceStore(config["server"]["voice_store"])

    async def call(slot, function, *args):
        return await asyncio.get_running_loop().run_in_executor(
            slot["executor"], functools.partial(function, *args)
        )

    @asynccontextmanager
    async def lifespan(app):
        nonlocal transcriber
        if config["server"]["host"] not in ("127.0.0.1", "localhost", "::1"):
            import os

            if not config["server"].get("users") and not os.environ.get(
                "VOICE_SERVER_TOKEN", config["server"].get("token")
            ):
                raise RuntimeError(
                    "公开监听需要配置用户令牌，或使用默认 localhost 加 SSH 隧道"
                )
        try:
            asr_options = config.get("asr", {})
            if asr_options.get("enabled", False):
                if transcriber_factory is None:
                    from .asr import SpeechTranscriber

                    transcriber = await asyncio.to_thread(SpeechTranscriber, asr_options)
                else:
                    transcriber = transcriber_factory(asr_options)
            for device in config["server"]["worker_devices"]:
                slot_config = copy.deepcopy(config)
                slot_config["model"]["device"] = device
                slot = {
                    "executor": ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="voice-gpu"
                    )
                }
                slots.append(slot)
                if runtime_factory is None:
                    from .model import VoiceRuntime
                    architecture = slot_config['model'].get('architecture', 'native_e2e')
                    if architecture not in ('native_e2e', 'legacy_v6'):
                        raise ValueError('未知语音模型架构')
                    if architecture == 'native_e2e':
                        from .native import NativeVoiceRuntime as VoiceRuntime

                    slot["runtime"] = await call(slot, VoiceRuntime, slot_config)
                else:
                    slot["runtime"] = runtime_factory(slot_config)
                available.put_nowait(slot)
            yield
        finally:
            for slot in slots:
                slot["executor"].shutdown(wait=True, cancel_futures=True)
            if transcriber is not None:
                transcriber.close()

    app = FastAPI(title="端到端语音实验服务", lifespan=lifespan)

    web_directory = Path(__file__).with_name("web")
    app.mount("/web", StaticFiles(directory=web_directory), name="web")

    @app.get("/")
    async def web_index():
        return FileResponse(web_directory / "index.html")

    @app.get("/health")
    async def health():
        return {
            "authentication_required": bool(
                config["server"].get("users")
                or os.environ.get("VOICE_SERVER_TOKEN", config["server"].get("token"))
            ),
            "ready": bool(slots),
            "slots": len(slots),
            "available": available.qsize(),
            "mode": "native-duplex",
            "transcription_enabled": transcriber is not None,
            "model": config["model"]["path"],
            **({"sessions": [{**slot["runtime"].diagnostics(), "events": slot.get("debug_events", [])} for slot in slots if hasattr(slot.get("runtime"), "diagnostics")]} if config["server"].get("debug_state", False) else {}),
        }

    def request_user(request):
        user = principal(config, request.headers.get("authorization", ""))
        if user is None:
            raise HTTPException(status_code=401, detail="认证失败")
        return user

    @app.get("/voices")
    async def list_voices(request: Request):
        return {"voices": store.list(request_user(request))}

    @app.post("/voices")
    async def upload_voice(request: Request):
        user = request_user(request)
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 20_000_000:
                raise HTTPException(status_code=413, detail="参考音频文件超过 20 MB")
        try:
            return await asyncio.to_thread(
                store.put,
                user,
                bytes(data),
                request.query_params.get("name", "我的音色"),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="无法读取参考音频，请上传有效的短音频文件"
            ) from exc

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        accepted = False
        authorization = ws.headers.get("authorization", "")
        if "voice-lab.v1" in ws.scope.get("subprotocols", []):
            # 浏览器通过首个消息认证，避免令牌进入 URL 与访问日志。
            await ws.accept(subprotocol="voice-lab.v1")
            accepted = True
            try:
                event = await asyncio.wait_for(ws.receive_text(), timeout=10)
                if len(event) > 4096:
                    raise ValueError("认证消息过大")
                authentication = json.loads(event)
                token = authentication.get("token", "")
                if authentication.get("type") != "auth" or not isinstance(token, str):
                    raise ValueError("认证消息错误")
                authorization = "Bearer " + token
            except WebSocketDisconnect:
                return
            except (TimeoutError, ValueError, AttributeError, KeyError):
                await ws.close(code=1008, reason="认证消息错误或超时")
                return
        user = principal(config, authorization)
        if user is None:
            await ws.close(code=1008, reason="认证失败")
            return
        voice = None
        if ws.query_params.get("voice"):
            try:
                voice = store.get(user, ws.query_params["voice"])
            except FileNotFoundError:
                await ws.close(code=1008, reason="音色不存在或不属于当前用户")
                return
        try:
            slot = available.get_nowait()
        except asyncio.QueueEmpty:
            await ws.close(code=1013, reason="所有语音槽位正忙，请稍后重试")
            return
        runtime = slot["runtime"]
        receiver = None
        coordinator = None
        transcription_task = None
        transcription_queue = None
        try:
            if not accepted:
                await ws.accept()
                accepted = True
            queue = asyncio.Queue(maxsize=config["server"]["max_pending_frames"])
            send_lock = asyncio.Lock()
            state = {"epoch": 0, "reset": False, "playback": False, "playback_text": None}
            pending_result = []
            pending_settings = deque(maxlen=8)
            user_audio = UserAudioHistory()
            slot["debug_events"] = []

            async def report(event):
                if config["server"].get("debug_state", False):
                    # 排查仅记录事件类型与时间，不保存用户文字或音频。
                    slot["debug_events"] = (slot["debug_events"] + [{"type": event["type"], "at": time.time()}])[-24:]
                async with send_lock:
                    await ws.send_json(event)

            if transcriber is not None:
                from .asr import TranscriptSession

                transcript = TranscriptSession(transcriber)
                transcription_queue = asyncio.Queue(maxsize=16)

                async def transcribe():
                    try:
                        while True:
                            chunk = await transcription_queue.get()
                            final = chunk is None
                            audio = np.empty(0, dtype=np.float32) if final else chunk
                            events = await asyncio.get_running_loop().run_in_executor(
                                transcriber.executor, transcript.feed, audio, final
                            )
                            for event in events:
                                samples = event.pop('_audio', None)
                                if samples is not None:
                                    user_audio.add(samples)
                                if getattr(runtime, 'native_e2e', False):
                                    event = {key: value for key, value in event.items()
                                             if key in ('type', 'id', 'text', 'start', 'end', 'final')}
                                await report(event)
                                if final or not hasattr(runtime, 'configure') or getattr(runtime, 'native_e2e', False):
                                    continue
                                settings = voice_commands(event['text'])
                                if settings:
                                    settings['source_end'] = event['end']
                                if 'emotion' in event:
                                    settings.update(emotion=event.get('dialogue_emotion', event['emotion']), event=event['vocal_event'], emotion_source=event.get('emotion_source', 'audio_model'))
                                if settings.get('voice') == 'self':
                                    try:
                                        settings['voice_audio'] = user_audio.reference()
                                    except ValueError as exc:
                                        settings.pop('voice')
                                        settings['voice_error'] = 'reference_unusable'
                                        await report({'type': 'settings_error', 'message': str(exc)})
                                if settings:
                                    pending_settings.append(settings)
                            if final:
                                return
                    except WebSocketDisconnect:
                        return
                    except Exception:
                        log.exception("用户发言转写失败")
                        try:
                            await report({"type": "transcript_error", "message": "用户文字转写暂时不可用"})
                        except (WebSocketDisconnect, RuntimeError):
                            return

                transcription_task = asyncio.create_task(transcribe())

            async def deliver(revision, value):
                pending_result.append((revision, value))
                await report({"type": "task_completed", "revision": revision})

            coordinator = TaskCoordinator(Backend(config["backend"]), report, deliver)
            sizes = await call(slot, runtime.start, voice)
            await report(
                {
                    "type": "ready",
                    "input_rate": 16000,
                    "output_rate": 24000,
                    "first_samples": sizes["first_samples"],
                    "samples": sizes["samples"],
                    "voice": ws.query_params.get("voice", "default"),
                    "voice_reference": sizes.get("voice_reference"),
                    "transcription_enabled": transcriber is not None,
                    "expression_controls": hasattr(runtime, 'configure'),
                    "architecture": sizes.get('architecture', 'legacy_v6'),
                    "capabilities_verified": sizes.get('capabilities_verified', False),
                }
            )

            async def receive():
                first = True
                try:
                    while True:
                        message = await ws.receive()
                        if message["type"] == "websocket.disconnect":
                            break
                        raw = message.get("bytes")
                        if raw is not None:
                            expected_size = (
                                sizes["first_samples"] if first else sizes["samples"]
                            ) * 2
                            if len(raw) != expected_size:
                                await report(
                                    {
                                        "type": "error",
                                        "message": f"音频帧大小错误，预期 {expected_size} 字节",
                                    }
                                )
                                break
                            first = False
                            if queue.full():
                                await report(
                                    {
                                        "type": "error",
                                        "message": "推理速度低于输入速度，音频队列已满，请降低负载或更换 GPU",
                                    }
                                )
                                break
                            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                            queue.put_nowait((audio, state['playback'], state['playback_text']))
                            if transcription_queue is not None and not transcription_task.done():
                                if transcription_queue.full():
                                    await report({"type": "transcript_error", "message": "转写处理积压，请稍后重新连接"})
                                    transcription_task.cancel()
                                else:
                                    transcription_queue.put_nowait(audio.copy())
                            continue
                        if len(message.get("text", "")) > 20000:
                            raise ValueError("控制消息过大")
                        event = json.loads(message.get("text", "{}"))
                        kind = event.get("type")
                        if kind == 'playback':
                            if not isinstance(event.get('active'), bool):
                                raise ValueError('播放状态必须是布尔值')
                            state['playback'] = event['active']
                            text = event.get('text')
                            if text is not None and not isinstance(text, str):
                                raise ValueError('播放文字必须是字符串')
                            state['playback_text'] = text[-500:] if text is not None else None
                        elif kind in ("interrupt", "cancel"):
                            state["epoch"] += 1
                            state["reset"] = True
                            runtime.interrupt()
                            if kind == "cancel":
                                await coordinator.cancel()
                                pending_result.clear()
                            await report(
                                {"type": "interrupted", "epoch": state["epoch"]}
                            )
                        elif kind == "delegate":
                            # 调试入口，用于独立验证后台链路。
                            task = event.get("task", "")
                            if not isinstance(task, str) or not 1 <= len(task) <= 4000:
                                raise ValueError("任务描述长度应为 1 到 4000")
                            await coordinator.submit(task)
                        elif kind == 'settings':
                            settings = event.get('settings')
                            if not hasattr(runtime, 'configure'):
                                await report({'type': 'settings_error', 'message': '当前运行时不支持表达设置'})
                                continue
                            if not isinstance(settings, dict) or set(settings) - {'style', 'rate', 'voice'}:
                                await report({'type': 'settings_error', 'message': '无效的表达设置'})
                                continue
                            settings = settings.copy()
                            if settings.get('style') in ('laugh', 'cry') and not getattr(runtime, 'native_e2e', False):
                                settings.update(style_once=True, vocal_action=settings['style'])
                            selected = settings.get('voice')
                            if 'voice' in settings:
                                if not isinstance(selected, str):
                                    await report({'type': 'settings_error', 'message': '音色标识必须是字符串'})
                                    continue
                                if selected == 'self' and not getattr(runtime, 'native_e2e', False):
                                    try:
                                        settings['voice_audio'] = user_audio.reference()
                                    except ValueError as exc:
                                        settings.pop('voice')
                                        settings['voice_error'] = 'reference_unusable'
                                        await report({'type': 'settings_error', 'message': str(exc)})
                                elif selected not in ('original', 'default', 'self'):
                                    try:
                                        settings['voice_path'] = store.get(user, selected)
                                        settings['voice_id'] = selected
                                    except FileNotFoundError:
                                        await report({'type': 'settings_error', 'message': '音色不存在或不属于当前用户'})
                                        continue
                            if len(pending_settings) == pending_settings.maxlen:
                                await report({'type': 'settings_error', 'message': '设置请求过多，请稍后重试'})
                                continue
                            pending_settings.append(settings)
                            message = ('设置已提交给主语音模型，实际表达与节奏请试听验证'
                                       if getattr(runtime, 'native_e2e', False)
                                       else '设置将在下一说话轮次生效')
                            await report({'type': 'settings_queued', 'message': message})
                        elif kind == "close":
                            state["epoch"] += 1
                            runtime.interrupt()
                            break
                finally:
                    # 断开时丢弃未处理音频，保证 GPU 会话及时释放。
                    while not queue.empty():
                        queue.get_nowait()
                    queue.put_nowait(None)

            receiver = asyncio.create_task(receive())
            while True:
                packet = await queue.get()
                if packet is None:
                    break
                audio, playback, playback_text = packet
                if hasattr(runtime, 'set_playback'):
                    await call(slot, runtime.set_playback, playback, playback_text)
                if state["reset"]:
                    await call(slot, runtime.resume_after_interrupt)
                    state["reset"] = False
                while pending_settings:
                    try:
                        await call(slot, runtime.configure, pending_settings.popleft())
                    except (ValueError, TypeError) as exc:
                        await report({'type': 'settings_error', 'message': str(exc)})
                context = None
                if pending_result:
                    revision, value = pending_result.pop(0)
                    if revision == coordinator.revision:
                        context = result_context(value, prioritize=True)
                if context is None:
                    context = coordinator.pending_context()
                epoch = state["epoch"]
                try:
                    result = await call(slot, runtime.step, audio, context)
                except InferenceInterrupted:
                    if epoch != state['epoch']:
                        continue
                    raise
                if epoch != state["epoch"]:
                    continue
                for event in result.get('events', []):
                    if event['type'] == 'barge_in':
                        state['epoch'] += 1
                        epoch = state['epoch']
                        await report({'type': 'interrupted', 'epoch': epoch, 'reason': 'user_barge_in'})
                    elif event['type'] == 'voice_changed' and playback:
                        state['epoch'] += 1
                        epoch = state['epoch']
                        await report({'type': 'interrupted', 'epoch': epoch, 'reason': 'voice_change'})
                        await report(event)
                    else:
                        await report(event)
                kind = result["kind"]
                if kind == "delegate":
                    await coordinator.submit(result["text"])
                elif kind == "cancel":
                    await coordinator.cancel()
                    pending_result.clear()
                    if hasattr(runtime, "confirm_cancelled"):
                        await call(slot, runtime.confirm_cancelled)
                    await report({"type": "task_cancelled"})
                elif kind == "invalid":
                    await report({"type": "control_error", "message": result["text"]})
                elif kind == "speech":
                    for block in result["blocks"]:
                        pcm = (
                            (np.clip(block["audio"], -1, 1) * 32767)
                            .astype("<i2")
                            .tobytes()
                        )
                        await report(
                            {
                                "type": "speech",
                                "epoch": epoch,
                                "text": block["text"],
                                "audio": base64.b64encode(pcm).decode(),
                                "rate": 24000,
                                "end": block["end"],
                            }
                        )
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("语音会话异常")
            if accepted:
                try:
                    await ws.send_json(
                        {"type": "error", "message": "会话处理失败，请查看服务端日志"}
                    )
                except Exception:
                    pass
        finally:
            if coordinator:
                await coordinator.cancel()
            if receiver:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
            if transcription_task is not None:
                if not transcription_task.done():
                    try:
                        await asyncio.wait_for(transcription_queue.put(None), timeout=1)
                        await asyncio.wait_for(transcription_task, timeout=4)
                    except (TimeoutError, WebSocketDisconnect):
                        transcription_task.cancel()
                await asyncio.gather(transcription_task, return_exceptions=True)
            if runtime:
                runtime.interrupt()
                # 等待同一槽位当前的模型操作结束，避免下一会话继承未完成的推理。
                await call(slot, lambda: None)
                if hasattr(runtime, 'end'):
                    await call(slot, runtime.end)
            if accepted:
                try:
                    await ws.close()
                except Exception:
                    pass
            available.put_nowait(slot)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    uvicorn.run(
        create_app(config),
        host=config["server"]["host"],
        port=config["server"]["port"],
        log_level="info",
        ws_max_size=65536,
    )


if __name__ == "__main__":
    main()
