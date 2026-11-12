import hashlib
import json
from types import SimpleNamespace
from voice_lab.training_materials import (
    select_full_pairs,
    reusable_file,
    inventory,
    download_smoothconv,
)
from voice_lab.resources import CATALOG


def test_full_selection_covers_domains_excludes_multichannel_and_duplicates():
    files = {}
    for name, checksum in [
        ("edu_10_2ch/a", "one"),
        ("none_edu_20h_2ch/b", "two"),
        ("edu_20h_2ch/copy", "one"),
        ("none_edu_20h_3ch/c", "three"),
    ]:
        files[name + ".wav"] = {"size": 10, "sha256": checksum}
        files[name + ".json"] = {"size": 5}
    files["none_edu_20h_2ch/unpaired.wav"] = {"size": 10, "sha256": "four"}
    result = select_full_pairs(files)
    assert {item["audio"] for item in result} == {
        "edu_10_2ch/a.wav",
        "none_edu_20h_2ch/b.wav",
    }


def test_existing_recording_is_reused_and_verified(tmp_path):
    source = tmp_path / "existing.wav"
    source.write_bytes(b"real-audio-fixture")
    metadata = {
        "size": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    destination = tmp_path / "full/recording.wav"
    assert reusable_file(source, destination, metadata)
    assert destination.read_bytes() == source.read_bytes()
    assert reusable_file(source, destination, metadata)


def test_inventory_supports_lfs_metadata_forms():
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="a.wav", size=10, lfs={"sha256": "one"}),
            SimpleNamespace(rfilename="b.json", size=4, lfs=None),
        ]
    )
    assert inventory(info)["a.wav"]["sha256"] == "one"
    assert inventory(info)["b.json"]["sha256"] is None


def test_full_download_manifest_is_separate_from_smoke(tmp_path, monkeypatch):
    import voice_lab.training_materials as module

    pairs = [{"audio": "edu_10_2ch/a.wav", "annotation": "edu_10_2ch/a.json"}]
    files = {
        "README.md": {"size": 2},
        "edu_10_2ch/a.wav": {"size": 2},
        "edu_10_2ch/a.json": {"size": 2},
    }
    smoke = tmp_path / CATALOG["SmoothConv"]["root"]
    smoke.mkdir(parents=True)
    (smoke / "download_manifest.json").write_text('{"pairs": ["existing"]}')

    def fake_fetch(repo, revision, directory, name, metadata, previous=None):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")

    monkeypatch.setattr(module, "download_file", fake_fetch)
    destination = download_smoothconv(tmp_path, files, pairs, 1)
    value = json.loads((destination / "download_manifest.json").read_text())
    assert value["profile"] == "full-2ch"
    assert value["pairs"] == pairs
    assert json.loads((smoke / "download_manifest.json").read_text())["pairs"] == [
        "existing"
    ]
