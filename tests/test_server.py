import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from voice_lab.config import load_config
from voice_lab.server import create_app
from test_voices import wav


class FakeRuntime:
    """只用于协议回归测试，不参与真实模型服务。"""

    starts = []

    def __init__(self, config):
        self.voice = None

    def start(self, voice=None):
        self.voice = voice
        self.starts.append(voice)
        return {"first_samples": 16560, "samples": 16000}

    def step(self, audio, context=None):
        return {
            "kind": "speech",
            "blocks": [
                {"text": "你好", "audio": np.zeros(24, dtype=np.float32), "end": True}
            ],
        }

    def interrupt(self):
        pass

    def resume_after_interrupt(self):
        pass


def config(tmp_path):
    value = load_config()
    value["server"]["voice_store"] = str(tmp_path)
    return value


def test_health_and_pcm(tmp_path):
    with TestClient(create_app(config(tmp_path), FakeRuntime)) as client:
        assert client.get("/health").json()["available"] == 1
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(bytes(16560 * 2))
            assert ws.receive_json()["type"] == "speech"
            ws.send_json({"type": "close"})


def test_multiuser_reference_isolation(tmp_path, monkeypatch):
    value = config(tmp_path)
    value["server"]["users"] = {"alice": "ALICE_TEST_TOKEN", "bob": "BOB_TEST_TOKEN"}
    monkeypatch.setenv("ALICE_TEST_TOKEN", "alice")
    monkeypatch.setenv("BOB_TEST_TOKEN", "bob")
    alice = {"Authorization": "Bearer alice"}
    bob = {"Authorization": "Bearer bob"}
    with TestClient(create_app(value, FakeRuntime)) as client:
        response = client.post("/voices", content=wav(), headers=alice)
        assert response.status_code == 200
        identifier = response.json()["id"]
        assert client.get("/voices", headers=bob).json()["voices"] == []
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?voice={identifier}", headers=bob):
                pass
        with client.websocket_connect(f"/ws?voice={identifier}", headers=alice) as ws:
            assert ws.receive_json()["voice"] == identifier
            ws.send_json({"type": "close"})
        assert FakeRuntime.starts[-1].name == identifier + ".wav"


def test_slot_capacity(tmp_path):
    with TestClient(create_app(config(tmp_path), FakeRuntime)) as client:
        with client.websocket_connect("/ws") as first:
            first.receive_json()
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws"):
                    pass
            first.send_json({"type": "close"})


def test_bad_frame_rejected(tmp_path):
    with TestClient(create_app(config(tmp_path), FakeRuntime)) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_bytes(b"bad")
            assert ws.receive_json()["type"] == "error"


def test_unauthenticated_upload_rejected(tmp_path, monkeypatch):
    value = config(tmp_path)
    value["server"]["users"] = {"alice": "ALICE_TEST_TOKEN"}
    monkeypatch.setenv("ALICE_TEST_TOKEN", "alice")
    with TestClient(create_app(value, FakeRuntime)) as client:
        assert client.post("/voices", content=wav()).status_code == 401


def test_two_slots_can_use_different_user_voices(tmp_path, monkeypatch):
    value = config(tmp_path)
    value["server"]["worker_devices"] = ["cuda:0", "cuda:1"]
    value["server"]["users"] = {"alice": "ALICE_TEST_TOKEN", "bob": "BOB_TEST_TOKEN"}
    monkeypatch.setenv("ALICE_TEST_TOKEN", "alice")
    monkeypatch.setenv("BOB_TEST_TOKEN", "bob")
    alice = {"Authorization": "Bearer alice"}
    bob = {"Authorization": "Bearer bob"}
    with TestClient(create_app(value, FakeRuntime)) as client:
        a = client.post("/voices", content=wav(), headers=alice).json()["id"]
        b = client.post("/voices", content=wav(), headers=bob).json()["id"]
        with client.websocket_connect(f"/ws?voice={a}", headers=alice) as first:
            assert first.receive_json()["voice"] == a
            with client.websocket_connect(f"/ws?voice={b}", headers=bob) as second:
                assert second.receive_json()["voice"] == b
                assert client.get("/health").json()["available"] == 0
                second.send_bytes(bytes(16560 * 2))
                assert second.receive_json()["type"] == "speech"
                second.send_json({"type": "close"})
            first.send_bytes(bytes(16560 * 2))
            assert first.receive_json()["type"] == "speech"
            first.send_json({"type": "close"})


def test_browser_auth_and_static_assets(tmp_path, monkeypatch):
    value = config(tmp_path)
    value["server"]["users"] = {"alice": "ALICE_TEST_TOKEN"}
    monkeypatch.setenv("ALICE_TEST_TOKEN", "alice")
    with TestClient(create_app(value, FakeRuntime)) as client:
        assert "语音工作台" in client.get("/").text
        assert client.get("/web/app.js").status_code == 200
        assert client.get("/web/capture.js").status_code == 200
        with client.websocket_connect("/ws", subprotocols=["voice-lab.v1"]) as ws:
            ws.send_json({"type": "auth", "token": "alice"})
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(bytes(16560 * 2))
            assert ws.receive_json()["type"] == "speech"
            ws.send_json({"type": "close"})
        with client.websocket_connect("/ws", subprotocols=["voice-lab.v1"]) as ws:
            ws.send_bytes(b"invalid-auth-frame")
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
        for payload in ({"type": "auth", "token": "wrong"}, {"type": "close"}, []):
            with client.websocket_connect("/ws", subprotocols=["voice-lab.v1"]) as ws:
                ws.send_json(payload)
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_json()


def test_browser_voice_ownership_and_busy(tmp_path, monkeypatch):
    value = config(tmp_path)
    value["server"]["users"] = {"alice": "ALICE_TEST_TOKEN", "bob": "BOB_TEST_TOKEN"}
    monkeypatch.setenv("ALICE_TEST_TOKEN", "alice")
    monkeypatch.setenv("BOB_TEST_TOKEN", "bob")
    with TestClient(create_app(value, FakeRuntime)) as client:
        voice = client.post("/voices", content=wav(), headers={"Authorization": "Bearer alice"}).json()["id"]
        with client.websocket_connect(f"/ws?voice={voice}", subprotocols=["voice-lab.v1"]) as ws:
            ws.send_json({"type": "auth", "token": "bob"})
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
        with client.websocket_connect(f"/ws?voice={voice}", subprotocols=["voice-lab.v1"]) as first:
            first.send_json({"type": "auth", "token": "alice"})
            assert first.receive_json()["voice"] == voice
            with client.websocket_connect("/ws", subprotocols=["voice-lab.v1"]) as second:
                second.send_json({"type": "auth", "token": "bob"})
                with pytest.raises(WebSocketDisconnect):
                    second.receive_json()
            first.send_json({"type": "close"})


def test_user_transcript_is_delivered_and_flushed(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from voice_lab import asr

    class Engine:
        def __init__(self, options):
            self.executor = ThreadPoolExecutor(max_workers=1)
        def close(self):
            self.executor.shutdown(wait=True)

    class Transcript:
        def __init__(self, engine):
            self.count = 0
        def feed(self, audio, final):
            self.count += 1
            return [{'type': 'user_transcript', 'id': self.count,
                     'text': '最后一句' if final else '用户发言',
                     'start': 0, 'end': 1, 'final': True}]

    monkeypatch.setattr(asr, 'TranscriptSession', Transcript)
    value = config(tmp_path)
    value['asr'] = {'enabled': True}
    with TestClient(create_app(value, FakeRuntime, Engine)) as client:
        assert client.get('/health').json()['transcription_enabled'] is True
        with client.websocket_connect('/ws', subprotocols=['voice-lab.v1']) as ws:
            ws.send_json({'type': 'auth', 'token': ''})
            assert ws.receive_json()['transcription_enabled'] is True
            ws.send_bytes(bytes(16560 * 2))
            events = [ws.receive_json(), ws.receive_json()]
            assert {event['type'] for event in events} == {'speech', 'user_transcript'}
            ws.send_json({'type': 'close'})
            assert ws.receive_json()['text'] == '最后一句'


def test_voice_cancellation_wakes_confirmation_after_coordinator(tmp_path):
    class Cancelling(FakeRuntime):
        confirmed = False
        def step(self, audio, context=None):
            return {'kind':'cancel','text':'取消'}
        def confirm_cancelled(self):
            type(self).confirmed = True
    with TestClient(create_app(config(tmp_path), Cancelling)) as client:
        with client.websocket_connect('/ws') as ws:
            ws.receive_json()
            ws.send_bytes(bytes(16560 * 2))
            assert ws.receive_json()['type'] == 'task_cancelled'
            assert Cancelling.confirmed
            ws.send_json({'type':'close'})


def test_close_during_prefill_is_a_normal_disconnect(tmp_path):
    import threading
    from voice_lab.model import InferenceInterrupted
    entered, cancelled = threading.Event(), threading.Event()
    class Interruptible(FakeRuntime):
        def step(self, audio, context=None):
            entered.set()
            assert cancelled.wait(2)
            raise InferenceInterrupted('已停止')
        def interrupt(self):
            cancelled.set()
    with TestClient(create_app(config(tmp_path), Interruptible)) as client:
        with client.websocket_connect('/ws') as ws:
            ws.receive_json()
            ws.send_bytes(bytes(16560 * 2))
            assert entered.wait(2)
            ws.send_json({'type':'close'})
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
            assert closed.value.code == 1000


def test_live_settings_reject_other_user_voice_and_keep_session(tmp_path, monkeypatch):
    class Configurable(FakeRuntime):
        def configure(self, settings):
            from voice_lab.expression import validate_rate
            if 'rate' in settings:validate_rate(settings['rate'])
            self.settings = settings
        def step(self, audio, context=None):
            return {'kind':'listen','events':[{'type':'expression_changed',**self.settings}]}
    value = config(tmp_path)
    value['server']['users'] = {'alice':'ALICE_TEST_TOKEN','bob':'BOB_TEST_TOKEN'}
    monkeypatch.setenv('ALICE_TEST_TOKEN','alice')
    monkeypatch.setenv('BOB_TEST_TOKEN','bob')
    with TestClient(create_app(value,Configurable)) as client:
        voice=client.post('/voices',content=wav(),headers={'Authorization':'Bearer alice'}).json()['id']
        with client.websocket_connect('/ws',headers={'Authorization':'Bearer bob'}) as ws:
            ws.receive_json()
            ws.send_json({'type':'settings','settings':{'voice':voice}})
            assert ws.receive_json()['type']=='settings_error'
            ws.send_json({'type':'settings','settings':{'style':'happy','rate':1.2}})
            assert ws.receive_json()['type']=='settings_queued'
            ws.send_bytes(bytes(16560 * 2))
            assert ws.receive_json()['style']=='happy'
            ws.send_json({'type':'close'})


def test_spoken_clone_keeps_reference_private(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from voice_lab import asr
    class Engine:
        def __init__(self, options):self.executor=ThreadPoolExecutor(max_workers=1)
        def close(self):self.executor.shutdown(wait=True)
    class Transcript:
        def __init__(self, engine):self.count=0
        def feed(self, audio, final):
            self.count+=1
            if final or self.count>1:return []
            return [{'type':'user_transcript','id':1,'text':'请模仿我的声音','start':0,'end':4,'final':True,
                     'emotion':'happy','vocal_event':'speech','emotion_confidence':None,
                     '_audio':np.sin(np.arange(64000)*0.1).astype(np.float32)*0.1}]
    class Configurable(FakeRuntime):
        captured=False
        def configure(self, settings):
            type(self).captured=settings.get('voice')=='self' and len(settings['voice_audio'])==64000
    monkeypatch.setattr(asr,'TranscriptSession',Transcript)
    value=config(tmp_path);value['asr']={'enabled':True}
    with TestClient(create_app(value,Configurable,Engine)) as client:
        with client.websocket_connect('/ws') as ws:
            ws.receive_json();ws.send_bytes(bytes(16560*2))
            events=[ws.receive_json(),ws.receive_json()]
            user=next(e for e in events if e['type']=='user_transcript')
            assert '_audio' not in user and user['emotion']=='happy'
            ws.send_bytes(bytes(16000*2));ws.receive_json()
            assert Configurable.captured
            assert client.get('/voices').json()['voices']==[]
            ws.send_json({'type':'close'})
