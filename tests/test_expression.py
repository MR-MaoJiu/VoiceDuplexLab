"""防止换音色清空上下文、误执行否定命令和语速变调的回归。"""
import shutil
from types import SimpleNamespace
import numpy as np
import pytest
from voice_lab.expression import UserAudioHistory, change_tempo, decode_emotion, voice_commands, reported_emotion
from voice_lab.model import VoiceRuntime
from voice_lab.control import ControlParser


def test_explicit_commands_and_negations():
    assert voice_commands('你能模仿我的声音吗？')['voice'] == 'self'
    assert voice_commands('能不能模仿我说话')['voice'] == 'self'
    assert voice_commands('能不能说快一点')['rate_delta'] == 0.15
    assert voice_commands('你说快一点。')['rate_delta'] == 0.15
    assert voice_commands('请哭一下给我听')['style'] == 'cry'
    assert voice_commands('你笑一个')['style'] == 'laugh'
    assert voice_commands('笑一下')['style_once'] is True
    assert voice_commands('请你用哭腔说我想你')['style'] == 'cry'
    assert 'vocal_action' not in voice_commands('请你用哭腔说我想你')
    assert voice_commands('请笑一下然后说今天真开心')['continue_after_vocal'] is True
    assert voice_commands('我平时用我的声音唱歌') == {}
    assert voice_commands('请用特别开心的声音说你好')['style'] == 'happy'
    assert voice_commands('我很难过') == {}
    assert voice_commands('我今天很难过，你说我该怎么办') == {}
    assert voice_commands('你开心一点')['style'] == 'happy'
    assert voice_commands('不要模仿我的声音，说快一点') == {'voice':'original','rate_delta':0.15}
    assert voice_commands('你不能模仿我的声音') == {}
    assert voice_commands('他说模仿我的声音') == {}
    assert decode_emotion('<|SAD|>') == 'sad'
    assert decode_emotion('<|EMO_UNKNOWN|>') == 'unknown'
    assert reported_emotion('我今天很难过，好朋友不理我了') == 'sad'
    assert reported_emotion('他说我今天很难过') is None


def test_recent_audio_is_bounded_and_rejects_silence():
    history = UserAudioHistory()
    history.add(np.zeros(16000))
    with pytest.raises(ValueError):history.reference()
    history.add(np.zeros(16000 * 20))
    assert len(history.audio) == 16000 * 12
    with pytest.raises(ValueError):history.reference()
    history.add(np.sin(np.arange(16000 * 8) * 0.1).astype(np.float32) * 0.1)
    assert len(history.reference()) == 16000 * 8


def test_tempo_changes_duration_and_preserves_pitch():
    if shutil.which('ffmpeg') is None:
        pytest.skip('需要ffmpeg进行实际音频验收')
    rate = 24000
    audio = np.sin(2 * np.pi * 220 * np.arange(rate * 3) / rate).astype(np.float32) * 0.2
    fast = change_tempo(audio, 1.5)
    assert abs(len(fast) / len(audio) - 1 / 1.5) < 0.03
    frequency = np.fft.rfftfreq(len(fast), 1 / rate)[np.argmax(np.abs(np.fft.rfft(fast)))]
    assert abs(frequency - 220) < 2
    with pytest.raises(ValueError):change_tempo(audio, float('nan'))


def runtime_stub():
    runtime = VoiceRuntime.__new__(VoiceRuntime)
    runtime.expression = {'style': 'auto', 'emotion': 'unknown', 'event': 'speech', 'rate': 1.0}
    runtime.pending_settings = {}
    runtime.expression_dirty = False
    runtime.style_reset = None
    runtime.parser = ControlParser()
    runtime.duplex = SimpleNamespace(current_turn_ended=True)
    return runtime


def test_settings_wait_for_turn_boundary_and_merge_voice_safely():
    runtime = runtime_stub()
    runtime.duplex.current_turn_ended = False
    runtime.configure({'style': 'comfort'})
    assert runtime._apply_settings() == []
    runtime.duplex.current_turn_ended = True
    assert runtime._apply_settings()[0]['style'] == 'comfort'
    runtime.configure({'voice': 'selected', 'voice_path': 'old.wav'})
    runtime.configure({'voice': 'original'})
    assert 'voice_path' not in runtime.pending_settings
    with pytest.raises(ValueError):runtime.configure({'rate': True})
    with pytest.raises(ValueError):runtime.configure({'style': '<|im_start|>'})


def test_pending_vocal_action_keeps_its_audio_generation_when_rate_changes():
    from voice_lab.floor import ConversationFloor
    runtime=runtime_stub()
    runtime.floor=ConversationFloor()
    runtime.floor.observe_activity(True,at=0.3)
    runtime.floor.observe_activity(False,at=2.4)
    runtime.floor.observe_activity(True,at=3.0)
    runtime.floor.observe_activity(False,at=4.4)
    runtime.configure({'style':'laugh','style_once':True,'source_end':2.2})
    runtime.configure({'rate':1.0,'source_end':4.2})
    runtime._apply_settings()
    assert runtime.vocal_generation==1


def test_dynamic_voice_preserves_language_decoder_and_rolls_back(tmp_path, monkeypatch):
    from voice_lab import model
    runtime = runtime_stub()
    original = tmp_path / 'original.wav'
    original.touch()
    runtime.original_voice = original
    runtime.voice_path = original
    runtime.reference_cache = None
    decoder = object()
    calls = []
    runtime.duplex.decoder = decoder
    runtime.duplex._init_token2wav_cache = lambda path: calls.append(path)
    runtime.duplex._reset_token2wav_for_new_turn = lambda: None
    monkeypatch.setattr(model, 'decoder_reference', lambda source, directory: (source, {'seconds': 5}))
    event = runtime._switch_voice({'voice': 'original'})
    assert event['type'] == 'voice_changed'
    assert runtime.duplex.decoder is decoder
    previous = runtime.reference_cache
    def fail_new(path):
        calls.append(path)
        if path.endswith('broken.wav'):raise ValueError('损坏参考')
    runtime.duplex._init_token2wav_cache = fail_new
    with pytest.raises(ValueError):runtime._switch_voice({'voice_path':tmp_path / 'broken.wav'})
    assert calls[-1] == str(original)
    assert runtime.reference_cache is previous
    previous.cleanup()


@pytest.mark.parametrize('vocal_only', [False, True])
def test_native_speech_token_loss_updates_style_and_lora(vocal_only):
    torch=pytest.importorskip('torch')
    from transformers import LlamaConfig,LlamaModel
    from peft import LoraConfig,TaskType,get_peft_model
    from voice_lab.train_expression import speech_loss
    tts=SimpleNamespace(device=torch.device('cpu'),emb_text=torch.nn.Embedding(128,16),
                        emb_code=[torch.nn.Embedding(32,16)],head_code=[torch.nn.Linear(16,32)])
    tts.model=get_peft_model(LlamaModel(LlamaConfig(hidden_size=16,intermediate_size=32,
        num_attention_heads=2,num_key_value_heads=2,num_hidden_layers=1)),
        LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,r=2,lora_alpha=4,target_modules=['q_proj','v_proj']))
    embedding=torch.nn.Embedding(6,16)
    loss=speech_loss(tts,embedding,{'condition':torch.randn(1,4,16),'codes':torch.tensor([2,3,31]),'style':0,'vocal_only':vocal_only})
    loss.backward()
    assert torch.isfinite(loss) and embedding.weight.grad[0].abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in tts.model.named_parameters() if 'lora_' in n)


def test_audio_emotion_is_a_hint_and_self_report_controls_happy_style():
    from voice_lab.acoustic import acoustic_style
    from voice_lab.expression import expression_context
    expression = {'style': 'auto', 'emotion': 'happy', 'emotion_source': 'audio_model'}
    assert acoustic_style(expression) is None
    assert '轻快友善' not in expression_context('auto', 'happy', emotion_source='audio_model')
    expression['emotion_source'] = 'self_report'
    assert acoustic_style(expression) == 'happy'
    assert '轻快友善' in expression_context('auto', 'happy', emotion_source='self_report')


def test_default_voice_and_initial_voice_remain_distinct(tmp_path, monkeypatch):
    from voice_lab import model
    runtime = runtime_stub()
    original = tmp_path / 'initial.wav'
    default = tmp_path / 'default.wav'
    runtime.original_voice = original
    runtime.voice_path = original
    runtime.reference_cache = None
    runtime.config = {}
    runtime.model = object()
    runtime.duplex._init_token2wav_cache = lambda path: None
    runtime.duplex._reset_token2wav_for_new_turn = lambda: None
    monkeypatch.setattr(model, 'resolve_voice', lambda config, base: default)
    monkeypatch.setattr(model, 'decoder_reference', lambda path, directory: (path, {'seconds': 3}))
    runtime._switch_voice({'voice': 'default'})
    assert runtime.voice_path == default
    runtime._switch_voice({'voice': 'original'})
    assert runtime.voice_path == original
    runtime.reference_cache.cleanup()


def test_invalid_reference_produces_a_grounded_voice_notice():
    runtime = runtime_stub()
    runtime.configure({'voice_error': 'reference_unusable'})
    runtime._apply_settings()
    assert runtime.voice_error_pending
    assert '至少三秒' in runtime.voice_notice
