import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from voice_lab.corpus import convert_pair, timeline_units, write_jsonl
from voice_lab.data import load_manifest
from voice_lab.download import choose_pairs, retry
from voice_lab.pipeline import training_checkpoint, filter_records
from voice_lab.synthesis import build_synthetic_record, SCENARIOS, synthesize_query


class Tokenizer:
    def encode(self, text, **kwargs):
        if text.startswith("<") and text.endswith(">"):
            return [1]
        return [ord(char) % 100 for char in text]


def test_download_selection_is_reproducible_and_paired():
    files = [
        "none_edu_20h_2ch/a.wav",
        "none_edu_20h_2ch/a.json",
        "none_edu_20h_2ch/b.wav",
        "none_edu_20h_2ch/b.json",
        "none_edu_20h_3ch/c.wav",
    ]
    assert choose_pairs(files, 1, 42) == choose_pairs(list(reversed(files)), 1, 42)
    with pytest.raises(ValueError):
        choose_pairs(files, 3, 42)


def test_download_retry_backs_off_on_rate_limit(monkeypatch):
    delays = []
    monkeypatch.setattr("voice_lab.download.time.sleep", delays.append)
    calls = []

    class Error(Exception):
        response = type("Response", (), {"status_code": 429})()

    def action():
        calls.append(1)
        if len(calls) == 1:
            raise Error()
        return "ok"

    assert retry(action) == "ok"
    assert delays == [20]


def test_real_corpus_conversion_preserves_channels(tmp_path):
    rate = 16000
    t = np.arange(rate * 20) / rate
    audio = (
        np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)], axis=1)
        * 0.1
    )
    path = tmp_path / "source.wav"
    sf.write(path, audio, rate)
    segments = []
    for channel in (0, 1):
        for start, end, text in [
            (0.5, 3.5, "这是一段参考声音"),
            (7, 8, "你好"),
            (11, 12, "今天怎么样"),
            (15, 16, "再见"),
        ]:
            segments.append(
                {
                    "channelIndex": channel,
                    "start": start,
                    "end": end,
                    "text": text,
                    "attributes": {"speaker": "A" if channel == 0 else "B"},
                }
            )
    annotations = tmp_path / "source.json"
    annotations.write_text(json.dumps({"instances": segments}))
    records = convert_pair(path, annotations, tmp_path / "converted", window_seconds=6)
    assert len({item["voice_id"] for item in records}) == 2
    record = next(item for item in records if item["provenance"]["channel"] == 0)
    user_audio, _ = sf.read(record["audio"])
    reference, _ = sf.read(record["voice"])
    assert np.argmax(np.abs(np.fft.rfft(user_audio[:16000]))) == 330
    assert np.argmax(np.abs(np.fft.rfft(reference[:16000]))) == 220
    manifest = tmp_path / "train.jsonl"
    write_jsonl(manifest, records)
    assert len(load_manifest(manifest)) == len(records)


def test_segment_timeline_has_closed_turns():
    units = timeline_units(
        [
            {"start": 0, "end": 2, "text": "你好朋友"},
            {"start": 4, "end": 5, "text": "再见"},
        ],
        0,
        5,
    )
    assert units[0]["end"] is False
    assert units[1]["end"] is True
    assert units[2] == {"listen": True}
    assert units[-1]["end"] is True


def test_synthetic_instructions_are_labelled_and_valid(tmp_path):
    voice = tmp_path / "voice.wav"
    sf.write(voice, np.ones(16000) * 0.1, 16000)
    records = []
    for index, scenario in enumerate(SCENARIOS):
        record = build_synthetic_record(
            scenario,
            np.ones(20000) * 0.1,
            {"voice_id": "alice", "voice": str(voice)},
            str(index),
            Tokenizer(),
            40,
            tmp_path,
        )
        assert record["provenance"]["kind"] == "synthetic_instruction"
        records.append(record)
    manifest = tmp_path / "train.jsonl"
    write_jsonl(manifest, records)
    assert len(load_manifest(manifest)) == len(SCENARIOS)
    accepted, rejected = filter_records(records, Tokenizer(), 40)
    assert len(accepted) == len(records)
    assert rejected == []


def test_teacher_forced_speech_wraps_the_supplied_text(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from voice_lab import synthesis

    reference = tmp_path / "voice.wav"
    sf.write(reference, np.ones(16000) * 0.1, 16000)
    monkeypatch.setattr(synthesis, "read_audio", lambda path: sf.read(path, dtype="float32")[0])

    def chat(**kwargs):
        assert kwargs["teacher_forcing"] is True
        assert kwargs["msgs"][-1]["content"] == ["<|tts_bos|>你好<|tts_eos|>"]
        sf.write(kwargs["output_audio_path"], np.ones(16000) * 0.1, 16000)

    model = SimpleNamespace(
        reset_session=lambda **kwargs: None,
        get_sys_prompt=lambda **kwargs: {"role": "system", "content": []},
        chat=chat,
        tts=SimpleNamespace(audio_tokenizer=SimpleNamespace(cache=None)),
    )
    assert len(synthesize_query(model, "你好", reference, tmp_path / "query.wav")) == 16000


def test_resume_checkpoint_selection(tmp_path):
    assert training_checkpoint(tmp_path, 20) == (None, False)
    checkpoint = tmp_path / "step-0000020"
    checkpoint.mkdir()
    (checkpoint / "training_state.pt").write_bytes(b"test")
    assert training_checkpoint(tmp_path, 20) == (checkpoint, True)
    assert training_checkpoint(tmp_path, 0) == (checkpoint, False)
    partial = tmp_path / "step-0000030.tmp"
    partial.mkdir()
    (partial / "training_state.pt").write_bytes(b"unfinished")
    assert training_checkpoint(tmp_path, 0) == (checkpoint, False)


def test_generated_instruction_inputs_survive_instance_recreation(tmp_path):
    import shutil
    from voice_lab.config import load_config
    from voice_lab.pipeline import backup_job_inputs, restore_job_inputs

    job = tmp_path / "jobs/job123"
    (job / "queries").mkdir(parents=True)
    (job / "queries/query.wav").write_bytes(b"original-voice-input")
    (job / "instructions.jsonl").write_text('{"id":"fixture"}\n')
    (job / ".pipeline.lock").write_text("运行锁不应恢复")
    config = load_config()
    config["training"]["backup_dir"] = str(tmp_path / "shared")
    backup_job_inputs(job, config)
    shutil.rmtree(job)
    job.mkdir()
    (job / ".pipeline.lock").write_text("当前进程锁")
    restore_job_inputs(job, config)
    assert (job / "queries/query.wav").read_bytes() == b"original-voice-input"
    assert (job / "instructions.jsonl").is_file()
    assert (job / ".pipeline.lock").read_text() == "当前进程锁"


def test_training_checkpoint_is_restored_for_the_same_job_only(tmp_path):
    from voice_lab.config import load_config
    from voice_lab.pipeline import restore_training_checkpoint

    output = tmp_path / "run"
    backup = tmp_path / "shared/step-0000005"
    backup.mkdir(parents=True)
    (backup / "training_state.pt").write_bytes(b"saved-optimizer-state")
    config = load_config()
    config["training"].update(backup_dir=str(backup.parent), output_dir=str(output))
    (backup / "voice_lab_metadata.json").write_text(json.dumps({"config": config}))
    restore_training_checkpoint(output, config)
    assert (output / backup.name / "training_state.pt").read_bytes() == b"saved-optimizer-state"
    different = tmp_path / "different-run"
    restore_training_checkpoint(different, config)
    assert not different.exists()


def test_prepare_resumes_partial_features(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from voice_lab import prepare
    from voice_lab.model import model_fingerprint

    voice = tmp_path / "voice.wav"
    sf.write(voice, np.ones(16000) * 0.1, 16000)
    records = [
        {
            "id": str(index),
            "voice": str(voice),
            "audio": str(voice),
            "units": [{"listen": True}],
        }
        for index in range(2)
    ]
    manifest = tmp_path / "train.jsonl"
    write_jsonl(manifest, records)
    output = tmp_path / "prepared"
    calls = []
    fail = [True]

    def compile_record(model, config, record):
        calls.append(record["id"])
        if record["id"] == "1" and fail[0]:
            fail[0] = False
            raise RuntimeError("模拟中断")
        return {
            "inputs_embeds": torch.zeros(2, 8),
            "labels": torch.tensor([-100, 1]),
            "fingerprint": model_fingerprint(config),
            "id": record["id"],
            "voice_id": "voice",
        }

    monkeypatch.setattr(prepare, "load_base", lambda *args, **kwargs: object())
    monkeypatch.setattr(prepare, "prepare_record", compile_record)
    arguments = ["prepare", str(manifest), "--output", str(output)]
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(RuntimeError):
        prepare.main()
    monkeypatch.setattr("sys.argv", arguments + ["--resume"])
    prepare.main()
    assert calls == ["0", "1", "1"]
    assert len(json.loads((output / "index.json").read_text())["samples"]) == 2


def test_source_signature_changes_with_audio(tmp_path):
    from voice_lab.prepare import source_signature
    from voice_lab.config import load_config

    audio = tmp_path / "audio.wav"
    sf.write(audio, np.ones(16000) * 0.1, 16000)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("example")
    records = [{"voice": str(audio), "audio": str(audio)}]
    before = source_signature(manifest, records, load_config())
    sf.write(audio, np.ones(16000) * 0.2, 16000)
    assert source_signature(manifest, records, load_config()) != before


@pytest.mark.parametrize("offline", [False, True])
def test_oneclick_orchestrates_download_prepare_train_and_resume(
    tmp_path, monkeypatch, offline
):
    import yaml
    from voice_lab import pipeline
    from transformers import AutoTokenizer
    from voice_lab.config import load_config

    config = load_config()
    if offline:
        raw = tmp_path / "full-raw"
        raw.mkdir()
        (raw / "download_manifest.json").write_text("{}")
        config["data"] = {"raw_dir": str(raw)}
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "load_config", lambda *args: config)
    models = [{"path": str(tmp_path / "base")}, {"path": str(tmp_path / "backend")}]
    monkeypatch.setattr(pipeline, "download_models", lambda value: models)

    def fetch(*args):
        assert not offline
        return tmp_path / "raw"

    monkeypatch.setattr(pipeline, "download_dataset", fetch)
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer()
    )
    voice = tmp_path / "reference.wav"
    sf.write(voice, np.ones(16000) * 0.1, 16000)

    def record(identifier, kind):
        return {
            "id": identifier,
            "voice_id": "speaker",
            "voice": str(voice),
            "audio": str(voice),
            "units": [{"text": "你好", "end": True}],
            "provenance": {"kind": kind},
        }

    def convert(raw, output, *args):
        training = [record("real", "real_conversation")]
        evaluation = [record("eval", "real_conversation")]
        write_jsonl(output / "real_train.jsonl", training)
        write_jsonl(output / "real_eval.jsonl", evaluation)
        (output / "corpus_report.json").write_text("{}")
        return training, evaluation

    monkeypatch.setattr(pipeline, "convert_corpus", convert)
    calls = []

    def run(module, *arguments):
        calls.append(module)
        if module == "synthesis":
            output = Path(arguments[arguments.index("--output") + 1])
            write_jsonl(
                output / "instructions.jsonl",
                [record("synthetic", "synthetic_instruction")],
            )
        if module == "train":
            configuration = Path(arguments[arguments.index("--config") + 1])
            loaded = yaml.safe_load(configuration.read_text())
            name = "step-0000001" if "--max-steps" in arguments else "final"
            checkpoint = Path(loaded["training"]["output_dir"]) / name
            assert loaded["training"]["save_every"] == 5
            checkpoint.mkdir(parents=True)
            (checkpoint / "training_state.pt").write_bytes(b"test")
            (checkpoint / "adapter_config.json").write_text("{}")

    monkeypatch.setattr(pipeline, "run_module", run)
    monkeypatch.setattr("sys.argv", ["pipeline", "--clips", "1", "--max-steps", "1"])
    pipeline.main()
    assert calls == ["synthesis", "prepare", "prepare", "train"]
    active = yaml.safe_load((tmp_path / "configs/automatic-latest.yaml").read_text())
    assert active["model"]["adapter"].endswith("step-0000001")
    pipeline.main()
    assert calls.count("synthesis") == 1
    assert calls.count("train") == 1
    monkeypatch.setattr("sys.argv", ["pipeline", "--clips", "1", "--max-steps", "0"])
    pipeline.main()
    assert calls.count("train") == 2
    active = yaml.safe_load((tmp_path / "configs/automatic-latest.yaml").read_text())
    assert active["model"]["adapter"].endswith("final")


def test_download_only_entry_bypasses_gpu_setup(tmp_path):
    import shutil
    import subprocess
    from voice_lab.config import ROOT

    directory = tmp_path / "scripts"
    directory.mkdir()
    for name in ("common.sh", "oneclick.sh"):
        shutil.copyfile(ROOT / "scripts" / name, directory / name)
    (directory / "prefetch.sh").write_text('printf "prefetch %s\\n" "$*"\n')
    result = subprocess.run(
        ["bash", str(directory / "oneclick.sh"), "--clips", "2", "--download-only"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "prefetch --clips 2 --download-only" in result.stdout


def test_preconverted_inputs_do_not_require_original_recordings(tmp_path,monkeypatch):
    from voice_lab import pipeline
    from voice_lab.config import load_config

    config=load_config()
    config['data']={'raw_dir':str(tmp_path/'missing-original-wavs')}
    converted=tmp_path/'converted'
    converted.mkdir()
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'load_config',lambda *args:config)
    monkeypatch.setattr(pipeline,'download_models',lambda *args:[])

    def download(*args):
        raise AssertionError('已有预处理输入时不能重新下载原始录音')

    monkeypatch.setattr(pipeline,'download_dataset',download)
    monkeypatch.setattr('sys.argv',['pipeline','--corpus-dir',str(converted),'--download-only'])
    pipeline.main()


def test_zero_window_limit_uses_all_recording_windows(tmp_path):
    rate = 16000
    t = np.arange(rate * 20) / rate
    audio = (
        np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)], axis=1)
        * 0.1
    )
    audio_path = tmp_path / "source.wav"
    sf.write(audio_path, audio, rate)
    segments = []
    for channel in (0, 1):
        for start, end in [(0.5, 3.5), (7, 8), (11, 12), (15, 16)]:
            segments.append(
                {
                    "channelIndex": channel,
                    "start": start,
                    "end": end,
                    "text": "你好朋友",
                    "attributes": {"speaker": str(channel)},
                }
            )
    annotations = tmp_path / "source.json"
    annotations.write_text(json.dumps({"instances": segments}))
    limited = convert_pair(audio_path, annotations, tmp_path / "limited", 6, 1)
    unlimited = convert_pair(audio_path, annotations, tmp_path / "unlimited", 6, 0)
    assert len(unlimited) > len(limited)
