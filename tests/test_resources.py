import json
import pytest
from voice_lab.resources import CATALOG, pack_group, restore_group


def test_split_resource_roundtrip_and_reuse(tmp_path):
    original = tmp_path / "original"
    source = original / CATALOG["MiniCPM"]["root"]
    source.mkdir(parents=True)
    (source / "config.json").write_text('{"model":"test"}')
    weights = bytes(range(256)) * 100
    (source / "model.safetensors").write_bytes(weights)
    packages = tmp_path / "packages"
    manifest = pack_group(original, packages, "MiniCPM", part_bytes=4096)
    value = json.loads(manifest.read_text())
    assert len(value["parts"]) > 1
    restored = tmp_path / "restored"
    restore_group(packages, restored, "MiniCPM")
    assert (
        restored / CATALOG["MiniCPM"]["root"] / "model.safetensors"
    ).read_bytes() == weights
    restore_group(packages, restored, "MiniCPM")
    assert pack_group(original, packages, "MiniCPM", part_bytes=4096) == manifest


def test_corrupted_part_stops_restore(tmp_path):
    source = tmp_path / "original" / CATALOG["Qwen3"]["root"]
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    packages = tmp_path / "packages"
    manifest = pack_group(tmp_path / "original", packages, "Qwen3", part_bytes=4096)
    part = json.loads(manifest.read_text())["parts"][0]["name"]
    (packages / part).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        restore_group(packages, tmp_path / "restored", "Qwen3")


def test_local_resources_avoid_model_hub(monkeypatch, tmp_path):
    from voice_lab import download
    import huggingface_hub

    base, backend = tmp_path / "base", tmp_path / "backend"
    base.mkdir()
    backend.mkdir()
    monkeypatch.setattr(download, "ROOT", tmp_path)

    def blocked(*args, **kwargs):
        raise AssertionError("本地资源模式不应下载模型")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", blocked)
    config = {
        "model": {"path": str(base), "revision": "voice-revision"},
        "backend": {
            "model": "Qwen/Qwen3-8B",
            "weights_path": str(backend),
            "revision": "backend-revision",
        },
    }
    result = download.download_models(config)
    assert result[1]["path"] == str(backend)


def test_complete_local_dataset_avoids_hub(monkeypatch, tmp_path):
    from voice_lab import download
    import huggingface_hub

    (tmp_path / "a.wav").write_bytes(b"wave")
    (tmp_path / "a.json").write_text("{}")
    manifest = {
        "repository": download.DATASET_REPO,
        "revision": download.DATASET_REVISION,
        "seed": 42,
        "pairs": [{"audio": "a.wav", "annotation": "a.json"}],
    }
    (tmp_path / "download_manifest.json").write_text(json.dumps(manifest))

    def blocked(*args, **kwargs):
        raise AssertionError("已完整下载的语料不应访问 Hub")

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", blocked)
    assert download.download_dataset(1, 42, tmp_path) == tmp_path


def test_gzip_resource_roundtrip(tmp_path):
    source = tmp_path / "original" / CATALOG["Qwen3"]["root"]
    source.mkdir(parents=True)
    payload = bytes(range(256)) * 500
    (source / "model.safetensors").write_bytes(payload)
    packages = tmp_path / "packages"
    manifest = pack_group(
        tmp_path / "original", packages, "Qwen3", part_bytes=128, compress=True
    )
    value = json.loads(manifest.read_text())
    assert value["compression"] == "gzip"
    assert len(value["parts"]) > 1
    assert all(".tar.gz.part" in part["name"] for part in value["parts"])
    restored = tmp_path / "restored"
    restore_group(packages, restored, "Qwen3")
    assert (
        restored / CATALOG["Qwen3"]["root"] / "model.safetensors"
    ).read_bytes() == payload


def test_manual_launcher_rebuilds_interrupted_gzip_package(tmp_path):
    import importlib.util
    from voice_lab.config import ROOT

    spec = importlib.util.spec_from_file_location(
        "local_resources", ROOT / "scripts/local_resources.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    source = tmp_path / CATALOG["Qwen3"]["root"]
    source.mkdir(parents=True)
    payload = b"interrupted-download-test" * 1000
    (source / "model.safetensors").write_bytes(payload)
    packages = tmp_path / "upload/Qwen3"
    packages.mkdir(parents=True)
    (packages / "Qwen3.tar.gz.part000.tmp").write_bytes(b"old partial data")
    manifest = launcher.finish_package(tmp_path, "Qwen3", 1)
    assert json.loads(manifest.read_text())["compression"] == "gzip"
    assert not (packages / "Qwen3.tar.gz.part000.tmp").exists()
    restored = tmp_path / "restored"
    restore_group(packages, restored, "Qwen3")
    assert (
        restored / CATALOG["Qwen3"]["root"] / "model.safetensors"
    ).read_bytes() == payload
    assert launcher.finish_package(tmp_path, "Qwen3", 1) == manifest


def test_browser_zip_parts_restore_without_extracting_wrappers(tmp_path):
    import zipfile

    source = tmp_path / "original" / CATALOG["Qwen3"]["root"]
    source.mkdir(parents=True)
    payload = bytes(range(256)) * 100
    (source / "model.safetensors").write_bytes(payload)
    packages = tmp_path / "packages"
    manifest = pack_group(
        tmp_path / "original", packages, "Qwen3", part_bytes=256, compress=True
    )
    value = json.loads(manifest.read_text())
    for part in value["parts"]:
        path = packages / part["name"]
        with zipfile.ZipFile(
            str(path) + ".zip", "w", compression=zipfile.ZIP_STORED
        ) as archive:
            archive.write(path, path.name)
        path.unlink()
    browser_manifest = packages / "Qwen3.browser.manifest.json"
    browser_manifest.write_text(json.dumps(value))
    manifest.unlink()
    restored = tmp_path / "restored"
    restore_group(packages, restored, "Qwen3")
    assert (
        restored / CATALOG["Qwen3"]["root"] / "model.safetensors"
    ).read_bytes() == payload


def test_resource_default_reuses_existing_volume_without_private_path(tmp_path,monkeypatch):
    from pathlib import Path
    from voice_lab.resources import default_resource_directory
    monkeypatch.delenv('VOICE_RESOURCES',raising=False)
    monkeypatch.setattr(Path,'home',lambda:tmp_path/'user')
    resources=tmp_path/'mounts/disk/VoiceDuplexLab-resources'
    resources.mkdir(parents=True)
    assert default_resource_directory(tmp_path/'project/code',tmp_path/'mounts')==str(resources)
    monkeypatch.setenv('VOICE_RESOURCES',str(tmp_path/'chosen'))
    assert default_resource_directory(tmp_path/'project/code',tmp_path/'mounts')==str(tmp_path/'chosen')
