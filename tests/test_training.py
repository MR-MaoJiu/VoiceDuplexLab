import json
import pytest


def test_native_labels_mask_audio_and_context():
    torch = pytest.importorskip("torch")
    from voice_lab.prepare import Capture

    class Decoder:
        def feed(self, embeddings, *args, **kwargs):
            return None

    capture = Capture(Decoder(), max_tokens=20)
    capture.feed(torch.randn(4, 8))
    capture.target = [5, 6, 7]
    capture.feed(torch.randn(3, 8))
    labels = torch.cat(capture.labels)
    assert labels.tolist() == [-100, -100, -100, -100, 5, 6, 7]
    assert torch.cat(capture.embeddings).shape == (7, 8)


def test_feature_only_capture_does_not_run_decoder():
    torch = pytest.importorskip("torch")
    from voice_lab.prepare import Capture

    class Decoder:
        def feed(self, *args, **kwargs):
            raise AssertionError("特征编译不应调用语言解码器")

    capture = Capture(Decoder(), max_tokens=20, forward=False)
    assert capture.feed(torch.ones(3, 8), return_logits=True) == (None, None)
    capture.target = [1, 2]
    capture.feed(torch.ones(2, 8))
    assert torch.cat(capture.labels).tolist() == [-100, -100, -100, 1, 2]


def test_final_audio_chunk_keeps_the_complete_utterance():
    torch = pytest.importorskip("torch")
    import numpy as np
    from types import SimpleNamespace
    from voice_lab.model import VoiceRuntime

    required_tokens = 65

    def generate(**kwargs):
        return torch.ones(1, min(required_tokens, kwargs["max_new_token"]), dtype=torch.long), None

    runtime = object.__new__(VoiceRuntime)
    runtime.voice_path = "reference.wav"
    runtime.model = SimpleNamespace(tts=SimpleNamespace(generate_chunk=generate))
    runtime.duplex = SimpleNamespace(
        _convert_results_to_tts_input=lambda hidden: torch.zeros(1, 2, 8),
        tts_current_turn_start_time=None, tts_text_start_pos=0,
        tts_temperature=0.8, tts_repetition_penalty=1.05, tts_eos_token=None,
        tts_past_key_values=None, tts_logits_processors=None,
        _generate_waveform_from_tokens=lambda tokens, *args, **kwargs: np.ones(tokens.shape[-1]*960,dtype=np.float32),
        _reset_token2wav_for_new_turn=lambda: None,
    )
    audio = runtime._render([], True, 1)
    assert len(audio) == required_tokens*960


def test_backend_completion_wakes_the_native_speech_gate():
    torch = pytest.importorskip("torch")
    import numpy as np
    from types import SimpleNamespace
    from voice_lab.model import VoiceRuntime

    runtime = object.__new__(VoiceRuntime)
    runtime.config = {"model": {"max_speak_tokens": 40}}
    runtime.pending_settings = {}
    runtime.expression_dirty = False
    runtime.voice_notice = None
    runtime.vocal_followup = None
    duplex = SimpleNamespace(total_ids=[],res_ids=[],prefill_schema_tokens=[],total_hidden=[],
                              pending_logits=torch.tensor([[1.,0.,2.]]),speak_token_id=1)
    duplex.streaming_prefill = lambda **kwargs: {"success":True}

    def generate(**kwargs):
        assert duplex.pending_logits.argmax().item() == duplex.speak_token_id
        return {"is_listen":True}

    duplex.streaming_generate = generate
    runtime.duplex = duplex
    runtime.step(np.zeros(16000),"【后台结果】任务完成，实际结果893")


def test_tiny_qwen_lora_backward_and_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model
    from voice_lab.config import load_config
    from voice_lab.train import atomic_checkpoint

    torch.manual_seed(12)
    base = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    base.config._name_or_path = str(tmp_path / "base")
    base.name_or_path = str(tmp_path / "base")
    base.save_pretrained(tmp_path / "base")
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = ids.clone()
    labels[:, :3] = -100
    inputs = model.get_input_embeddings()(ids).detach()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )
    result = model(inputs_embeds=inputs, labels=labels, use_cache=False)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    optimizer.step()
    config = load_config()
    atomic_checkpoint(model, optimizer, tmp_path, {"step": 1}, config, "step-0000001")
    checkpoint = tmp_path / "step-0000001"
    assert (checkpoint / "adapter_model.safetensors").is_file()
    saved = torch.load(checkpoint / "training_state.pt", weights_only=True)
    assert saved["step"] == 1
    metadata = json.loads((checkpoint / "voice_lab_metadata.json").read_text())
    assert metadata["voice_conditioning"] == "per-sample-reference-audio"
    restored_base = Qwen3ForCausalLM.from_pretrained(tmp_path / "base")
    restored_model = PeftModel.from_pretrained(restored_base, checkpoint)
    model.eval()
    restored_model.eval()
    with torch.no_grad():
        expected = model(inputs_embeds=inputs, use_cache=False).logits
        actual = restored_model(inputs_embeds=inputs, use_cache=False).logits
    assert torch.allclose(expected, actual, atol=1e-6)


def test_targets_use_predictable_native_terminator():
    from voice_lab.prepare import target_ids

    class Tokenizer:
        def encode(self, text, **kwargs):
            values = {
                "<|speak|>": 1,
                "你好": 2,
                "<|turn_eos|>": 3,
                "<|chunk_eos|>": 4,
                "</unit>": 5,
                "<|listen|>": 6,
            }
            return [values[text]]

    assert target_ids({"text": "你好", "end": True}, Tokenizer()) == [1, 2, 3, 4, 5]
    assert target_ids({"listen": True}, Tokenizer()) == [6, 5]


def test_lazy_training_saves_and_resumes_without_recompiling(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    import numpy as np
    import soundfile as sf
    from types import SimpleNamespace
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from voice_lab import train, prepare
    from voice_lab.config import load_config
    from voice_lab.corpus import write_jsonl
    from voice_lab.model import model_fingerprint

    base = tmp_path / "base"
    Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8,
    )).save_pretrained(base)
    config = load_config()
    config["model"]["path"] = str(base)
    config["training"].update(
        prepared_dir=str(tmp_path / "features"), output_dir=str(tmp_path / "run"),
        backup_dir=str(tmp_path / "backup"), keep_checkpoints=1,
        gradient_accumulation=1, epochs=1, save_every=1, rank=2, alpha=4,
    )
    monkeypatch.setattr(train, "load_config", lambda path: config)
    monkeypatch.setattr(train, "load_base", lambda *a, **k: SimpleNamespace(
        llm=Qwen3ForCausalLM.from_pretrained(base)
    ))
    audio = tmp_path / "voice.wav"
    sf.write(audio, np.zeros(1600), 16000)
    manifest = tmp_path / "train.jsonl"
    write_jsonl(manifest, [
        {"id": str(i), "audio": str(audio), "voice": str(audio),
         "units": [{"text": "你好", "end": True}]}
        for i in range(2)
    ])
    compiled = []

    def compile_record(model, options, record):
        assert model.llm.model.embed_tokens is model.llm.get_input_embeddings()
        compiled.append(record["id"])
        return {
            "inputs_embeds": torch.ones(4, 16), "labels": torch.tensor([-100, 1, 2, 3]),
            "compiler": prepare.compiler_version(options),
            "fingerprint": model_fingerprint(options), "id": record["id"], "voice_id": "voice",
        }

    monkeypatch.setattr(prepare, "prepare_record", compile_record)
    monkeypatch.setattr("sys.argv", ["train", "--manifest", str(manifest), "--max-steps", "1"])
    train.main()
    first = tmp_path / "run/step-0000001"
    assert first.is_dir()
    assert (tmp_path / "backup/step-0000001/training_state.pt").is_file()
    monkeypatch.setattr("sys.argv", ["train", "--manifest", str(manifest),
                                    "--resume", str(first), "--max-steps", "2"])
    train.main()
    state = torch.load(tmp_path / "run/step-0000002/training_state.pt", weights_only=True)
    assert state["step"] == 2
    assert state["next_index"] == 2
    assert sorted(compiled) == ["0", "1"]
    assert not first.exists()
    assert (tmp_path / "backup/step-0000002/training_state.pt").is_file()
    assert not (tmp_path / "backup/step-0000001").exists()


def test_forced_unit_end_has_no_language_prediction_loss():
    from voice_lab.prepare import target_ids, target_labels
    class Tokenizer:
        mapping = {'<|speak|>': [1], '你好': [2], '<|turn_eos|>': [3], '<|chunk_eos|>': [4], '</unit>': [5], '<|listen|>': [6]}
        def encode(self, value, add_special_tokens=False):
            return self.mapping[value]
    tokenizer = Tokenizer()
    assert target_ids({'text': '你好', 'end': True}, tokenizer) == [1, 2, 3, 4, 5]
    assert target_labels({'text': '你好', 'end': True}, tokenizer, True) == [1, 2, 3, 4, -100]
    assert target_labels({'listen': True}, tokenizer, True) == [6, -100]
    assert target_labels({'listen': True}, tokenizer, False) == [6, 5]


def test_old_feature_versions_require_explicit_legacy_configuration(tmp_path):
    import json
    from voice_lab.train import read_index
    from voice_lab.prepare import compiler_version
    value = {'fingerprint': 'base', 'samples': [{'file': 'sample.pt'}]}
    (tmp_path / 'index.json').write_text(json.dumps(value))
    assert read_index(tmp_path, 'base', 4)[0]
    with pytest.raises(ValueError, match='标签版本'):
        read_index(tmp_path, 'base', 5)
    assert compiler_version({'training': {}}) == 4
    assert compiler_version({'training': {'mask_forced_tokens': True}}) == 5


def test_reference_semantics_change_requires_fresh_features():
    from voice_lab.prepare import compiler_version
    assert compiler_version({'model': {'reference_audio_in_context': False}, 'training': {'mask_forced_tokens': True}}) == 6
    with pytest.raises(ValueError, match='必须启用'):
        compiler_version({'model': {'reference_audio_in_context': False}, 'training': {}})


def test_long_voice_reference_is_bounded_without_overwriting_original(tmp_path,monkeypatch):
    import numpy as np
    import soundfile as sf
    from voice_lab import model
    rate=16000
    active=(0.2*np.sin(np.arange(8*rate)*2*np.pi*180/rate)).astype(np.float32)
    source=np.concatenate((np.zeros(4*rate),active,np.zeros(7*rate))).astype(np.float32)
    original=tmp_path/'original.wav'
    sf.write(original,source,rate,subtype='PCM_16')
    before=original.read_bytes()
    monkeypatch.setattr(model,'read_audio',lambda p:sf.read(p,dtype='float32')[0])
    target,info=model.decoder_reference(original,tmp_path)
    result,out_rate=sf.read(target)
    assert out_rate==rate and len(result)==8*rate and info['clipped']
    assert 3.9<=info['start_seconds']<=4.1
    assert original.read_bytes()==before


def test_short_voice_reference_is_preserved(tmp_path,monkeypatch):
    import numpy as np
    from voice_lab import model
    monkeypatch.setattr(model,'read_audio',lambda p:np.ones(3*16000,dtype=np.float32))
    original=tmp_path/'voice.wav'
    result,info=model.decoder_reference(original,tmp_path)
    assert result==original and not info['clipped'] and info['seconds']==3
