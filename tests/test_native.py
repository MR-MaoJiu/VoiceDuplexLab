from types import SimpleNamespace

import numpy as np
import pytest

from voice_lab.control import ControlParser
from voice_lab.native import NativeVoiceRuntime


def test_winik_persona_reaches_native_runtime(monkeypatch):
    """角色提示必须进入实际运行配置，同时保留调用方的自定义提示。"""
    from voice_lab.model import VoiceRuntime
    from voice_lab.prompts import WINIK_PERSONA

    received = {}
    monkeypatch.setattr(VoiceRuntime, '__init__', lambda self, config: received.update(config))
    config = {'model': {'system_prompt': '用简洁中文回答。'}}
    NativeVoiceRuntime(config)
    prompt = received['model']['system_prompt']
    assert WINIK_PERSONA in prompt
    assert '我叫winik' in prompt
    assert '普通自我介绍' in prompt
    assert '是否真人、是否虚拟角色' in prompt
    assert '用简洁中文回答。' in prompt
    assert config['model']['system_prompt'] == '用简洁中文回答。'


@pytest.mark.parametrize('marker,expected', [('@@VOICE SELF@@', 'self'), ('@@VOICE ORIGINAL@@', 'original'), ('@@VOICE DEFAULT@@', 'default')])
def test_native_voice_marker_every_character(marker, expected):
    parser = ControlParser()
    values = [parser.push(char) for char in marker]
    assert all(value.kind == 'pending' for value in values[:-1])
    assert values[-1].kind == 'voice'
    assert values[-1].text == expected


def test_native_settings_reach_model_without_audio_processing():
    runtime = NativeVoiceRuntime.__new__(NativeVoiceRuntime)
    runtime.model_hints = []
    runtime.configure({'rate': 1.5, 'style': 'cry'})
    assert len(runtime.model_hints) == 2
    assert '1.5' in runtime.model_hints[1]
    assert not hasattr(runtime, 'expression')
    with pytest.raises(ValueError):
        runtime.configure({'emotion': 'sad'})


def test_native_rejects_legacy_adapter_before_loading():
    with pytest.raises(ValueError, match='旧版'):
        NativeVoiceRuntime({'model': {'acoustic_adapter': 'legacy'}})


def test_speech_loss_reaches_language_backbone():
    """专门防止再次把语义条件detach后只训练独立语音分支。"""
    import torch
    from torch import nn
    from voice_lab.train_native import joint_loss

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(4, 4)
            self.dtype = torch.float32

        def forward(self, inputs_embeds, **kwargs):
            hidden = self.project(inputs_embeds)
            return SimpleNamespace(loss=hidden.sum() * 0, hidden_states=[hidden])

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(4, 4)

        def forward(self, inputs_embeds, **kwargs):
            return SimpleNamespace(last_hidden_state=self.project(inputs_embeds.cumsum(dim=1)))

    language = Language()
    tts = SimpleNamespace(device='cpu', emb_text=nn.Embedding(8, 4), emb_code=[nn.Embedding(8, 4)],
                          projector_semantic=nn.Linear(4, 4), audio_bos_token_id=0,
                          model=Decoder(), head_code=[nn.Linear(4, 8)])
    model = SimpleNamespace(device='cpu', llm=language, tts=tts)
    record = {'inputs_embeds': torch.randn(5, 4), 'labels': torch.tensor([-100, -100, 2, 3, 4]),
              'spans': [{'start': 2, 'ids': torch.tensor([2, 3]), 'codes': torch.tensor([1, 2, 7])}]}
    total, _, speech, _, _ = joint_loss(model, record)
    assert torch.isfinite(total)
    speech.backward()
    assert language.project.weight.grad.abs().sum() > 0
    assert tts.projector_semantic.weight.grad.abs().sum() > 0
    assert tts.model.project.weight.grad.abs().sum() > 0


def test_native_render_preserves_hidden_and_speaker_reference():
    import torch
    runtime = NativeVoiceRuntime.__new__(NativeVoiceRuntime)
    condition = torch.zeros(1, 3, 4)
    states = [[2, torch.ones(1, 1, 4), True]]
    calls = []
    def convert(hidden):
        assert hidden is states
        return condition
    def generate(**kwargs):
        assert kwargs['inputs_embeds'] is condition
        return torch.tensor([[1, 2]]), None
    def waveform(tokens, reference, end, force_flush):
        calls.append(reference)
        return np.ones(16, dtype=np.float32)
    runtime.model = SimpleNamespace(tts=SimpleNamespace(generate_chunk=generate))
    runtime.voice_path = 'same-speaker.wav'
    runtime.duplex = SimpleNamespace(_convert_results_to_tts_input=convert, tts_text_start_pos=0,
                                    tts_temperature=0.8, tts_repetition_penalty=1,
                                    tts_eos_token=7, tts_past_key_values=None, tts_logits_processors=[],
                                    _generate_waveform_from_tokens=waveform,
                                    _reset_token2wav_for_new_turn=lambda: None)
    assert len(runtime._render(states, True, 0)) == 16
    assert calls == ['same-speaker.wav']


def test_unified_flow_preserves_speaker_and_semantic_condition():
    import torch
    from torch import nn
    from voice_lab.native_flow import flow_loss

    class Estimator(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.1))

        def forward(self, x, mask, mu, t, spks, cond):
            assert torch.equal(mu, torch.ones_like(mu))
            assert torch.equal(spks, torch.full_like(spks, 0.5))
            return x * self.scale + mu + cond

    decoder = nn.Module()
    decoder.estimator = Estimator()
    row = {'mu': torch.ones(1, 2, 4), 'target': torch.zeros(1, 2, 4),
           'condition': torch.zeros(1, 2, 4), 'speaker': torch.full((1, 2), 0.5),
           'mask': torch.ones(1, 1, 4), 'loss_mask': torch.tensor([[[0., 0., 1., 1.]]])}
    loss = flow_loss(decoder, row)
    loss.backward()
    assert torch.isfinite(loss)
    assert decoder.estimator.scale.grad.abs() > 0


def test_native_transcript_cannot_control_voice_or_emotion(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient
    from voice_lab import asr, server
    from voice_lab.config import load_config
    from test_server import FakeRuntime

    class Engine:
        def __init__(self, options):
            self.executor = ThreadPoolExecutor(max_workers=1)

        def close(self):
            self.executor.shutdown(wait=True)

    class Transcript:
        def __init__(self, engine):
            pass

        def feed(self, audio, final):
            if final:
                return []
            return [{'type': 'user_transcript', 'id': 1, 'text': '请模仿我的声音，哭着说快一点',
                     'start': 0, 'end': 4, 'emotion': 'sad', 'vocal_event': 'crying',
                     '_audio': np.ones(64000, dtype=np.float32) * 0.1}]

    class Runtime(FakeRuntime):
        native_e2e = True

        def configure(self, settings):
            raise AssertionError('字幕支路不得修改主模型状态')

    def forbidden(text):
        raise AssertionError('字幕支路不得解析语音指令')
    monkeypatch.setattr(asr, 'TranscriptSession', Transcript)
    monkeypatch.setattr(server, 'voice_commands', forbidden)
    config = load_config()
    config['asr'] = {'enabled': True}
    config['server']['voice_store'] = str(tmp_path)
    with TestClient(server.create_app(config, Runtime, Engine)) as client:
        with client.websocket_connect('/ws') as ws:
            ws.receive_json()
            ws.send_bytes(bytes(16560 * 2))
            events = [ws.receive_json(), ws.receive_json()]
            transcript = next(event for event in events if event['type'] == 'user_transcript')
            assert 'emotion' not in transcript
            assert '_audio' not in transcript
            ws.send_json({'type': 'close'})


@pytest.mark.parametrize('selected,expected', [(0, None), (1, 'self'), (2, 'original'), (3, 'default'), (4, 'cancel')])
def test_voice_action_branch_preserves_conversation_cache(selected, expected):
    import torch
    from transformers import DynamicCache
    from voice_lab.voice_action import decide_voice_action

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [int(text)] if text in ('0', '1', '2', '3', '4') else [5, 6, 7]

    cache = DynamicCache()
    original = torch.ones(1, 1, 2, 4)
    cache.update(original, original.clone(), 0)
    class Model:
        device = 'cpu'

        def __init__(self):
            self.calls = 0

        def __call__(self, inputs_embeds, position_ids, use_cache, **kwargs):
            assert not use_cache
            assert position_ids.tolist() == [list(range(inputs_embeds.shape[1]))]
            logits = torch.zeros(1, 1, 8)
            choice = int(selected > 0) if self.calls == 0 else selected
            self.calls += 1
            logits[0, -1, choice] = 10
            return SimpleNamespace(logits=logits)
    decoder = SimpleNamespace(cache=cache, tokenizer=Tokenizer(), m=Model(),
                              embed_tokens=lambda ids: torch.zeros(len(ids), 4))
    assert decide_voice_action(decoder, [torch.ones(2, 4)]) == expected
    assert cache.get_seq_length() == 2
    assert torch.equal(cache.key_cache[0], original)


def test_voice_action_runs_once_at_user_turn_boundary(monkeypatch):
    from voice_lab import voice_action
    runtime = NativeVoiceRuntime.__new__(NativeVoiceRuntime)
    runtime.floor = SimpleNamespace(active=False, generation=1)
    runtime.duplex = SimpleNamespace(current_turn_ended=True, decoder=object())
    runtime.parser = ControlParser()
    runtime.voice_decision_generation = 0
    runtime.current_voice = 'original'
    runtime.voice_input_units = []
    selected = []
    monkeypatch.setattr(voice_action, 'decide_voice_action', lambda decoder, units, **kwargs: selected.append(decoder))
    assert runtime._voice_action() == []
    assert runtime._voice_action() == []
    assert len(selected) == 1


def test_voice_action_capture_excludes_backend_text():
    import torch
    runtime = NativeVoiceRuntime.__new__(NativeVoiceRuntime)
    def original(*args, **kwargs):
        return None
    decoder = SimpleNamespace(feed=original)
    def prefill(audio_waveform, text_list):
        decoder.feed(torch.ones(1, 4))
        decoder.feed(torch.ones(2, 4) * 2)
        decoder.feed(torch.ones(1, 4) * 99)
        return {'success': True}
    runtime.duplex = SimpleNamespace(decoder=decoder, streaming_prefill=prefill)
    runtime.voice_input_units = []
    runtime._prefill(np.zeros(16000), '后台文字', True)
    assert decoder.feed is original
    assert runtime.voice_input_units[0].shape == (3, 4)
    assert runtime.voice_input_units[0].max() == 2


def test_voice_action_capture_restores_decoder_after_failure():
    runtime = NativeVoiceRuntime.__new__(NativeVoiceRuntime)
    def original(*args, **kwargs):
        return None
    decoder = SimpleNamespace(feed=original)
    def prefill(**kwargs):
        raise RuntimeError('模拟打断')
    runtime.duplex = SimpleNamespace(decoder=decoder, streaming_prefill=prefill)
    with pytest.raises(RuntimeError, match='模拟打断'):
        runtime._prefill(np.zeros(16000), None, True)
    assert decoder.feed is original


@pytest.mark.parametrize('current,requested', [('original', None), ('selected', None), ('self', 'original')])
def test_stopping_mimic_preserves_other_reference_voices(monkeypatch, current, requested):
    from voice_lab import voice_action
    runtime = NativeVoiceRuntime.__new__(NativeVoiceRuntime)
    runtime.floor = SimpleNamespace(active=False, generation=1)
    runtime.duplex = SimpleNamespace(current_turn_ended=True, decoder=object())
    runtime.parser = ControlParser()
    runtime.voice_decision_generation = 0
    runtime.voice_input_units = []
    runtime.current_voice = current
    configured = []
    runtime.configure = configured.append
    runtime._apply_settings = lambda: []
    monkeypatch.setattr(voice_action, 'decide_voice_action', lambda *args, **kwargs: 'cancel')
    runtime._voice_action()
    assert configured == ([{'voice': requested}] if requested else [])


def test_native_short_voice_request_can_use_existing_reference():
    from voice_lab.expression import UserAudioHistory
    runtime = NativeVoiceRuntime.__new__(NativeVoiceRuntime)
    runtime.input_audio = UserAudioHistory()
    runtime.input_audio.add(np.sin(np.arange(24000) * 0.1).astype(np.float32) * 0.1)
    runtime.model_hints = []
    runtime.pending_settings = {}
    runtime.floor = SimpleNamespace(generation=1)
    runtime.configure({'voice': 'self'})
    assert len(runtime.pending_settings['voice_audio']) == 24000


def test_native_compiler_masks_automatically_inserted_unit_end(monkeypatch):
    import sys
    import torch
    from voice_lab import train_native
    monkeypatch.setitem(sys.modules, 's3tokenizer', SimpleNamespace())
    monkeypatch.setattr(train_native, 'read_audio', lambda path: np.zeros(8, dtype=np.float32))

    class Tokenizer:
        def encode(self, text, **kwargs):
            return {'<|listen|>': [7], '</unit>': [9], '<|speak|>': [1],
                    '<|chunk_eos|></unit>': [3, 9]}.get(text, [4])

        def convert_tokens_to_ids(self, token):
            return 2

    decoder = SimpleNamespace(feed=lambda *args, **kwargs: None,
                              embed_tokens=lambda ids: torch.zeros(len(ids), 4), reset=lambda: None)
    def prepare(**kwargs):
        decoder.feed(torch.zeros(1, 4))
    def prefill(**kwargs):
        decoder.feed(torch.zeros(1, 4))
        return {'success': True}
    duplex = SimpleNamespace(decoder=decoder, tokenizer=Tokenizer(), prepare=prepare,
                             streaming_prefill=prefill, audio_buffer=[],
                             processor=SimpleNamespace(get_streaming_chunk_size=lambda: 4))
    model = SimpleNamespace(reset_session=lambda **kwargs: None, init_tts=lambda **kwargs: None,
                            as_duplex=lambda **kwargs: duplex)
    config = {'model': {'device': 'cpu'}, 'training': {'max_tokens': 100}}
    record = {'id': 'test', 'reference_audio': 'reference.wav', 'user_audio': 'user.wav',
              'outputs': [{'text': 'hello'}]}
    compiled = train_native.compile_record(model, config, record)
    assert 7 in compiled['labels']
    assert 9 not in compiled['labels']
