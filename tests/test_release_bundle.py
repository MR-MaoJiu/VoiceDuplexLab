"""归档恢复中的路径边界和已有文件保护回归。"""
import importlib.util
import io
from pathlib import Path
import tarfile
import pytest

module_path = Path(__file__).resolve().parents[1] / 'scripts/restore_training_bundle.py'
spec = importlib.util.spec_from_file_location('restore_bundle', module_path)
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


def archive_at(tmp_path, name, body=b'checkpoint'):
    path = tmp_path / 'test.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        member = tarfile.TarInfo(name)
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))
    return path


def test_restore_is_idempotent_and_preserves_newer_file(tmp_path):
    archive = archive_at(tmp_path, 'runs/state.bin')
    target = tmp_path / 'restored'
    restore.restore_archive(archive, target)
    restore.restore_archive(archive, target)
    state = target / 'runs/state.bin'
    state.write_bytes(b'newer-checkpoint')
    with pytest.raises(FileExistsError):
        restore.restore_archive(archive, target)
    assert state.read_bytes() == b'newer-checkpoint'


@pytest.mark.parametrize('name', ['../escape', '/tmp/escape'])
def test_restore_rejects_unsafe_archive(tmp_path, name):
    archive = archive_at(tmp_path, name)
    with pytest.raises(ValueError):
        restore.restore_archive(archive, tmp_path / 'target')


def test_bundle_runtime_uses_active_adapter(tmp_path, monkeypatch):
    import json
    import shutil
    import sys
    import yaml
    script = module_path.with_name('bundle_runtime.py')
    runtime_spec = importlib.util.spec_from_file_location('bundle_runtime_test', script)
    runtime = importlib.util.module_from_spec(runtime_spec)
    runtime_spec.loader.exec_module(runtime)
    project = tmp_path / 'source/VoiceDuplexLab'
    (project / 'configs').mkdir(parents=True)
    shutil.copy2(module_path.parents[1] / 'configs/default.yaml', project / 'configs/default.yaml')
    (tmp_path / 'models/MiniCPM-o-4_5').mkdir(parents=True)
    adapter = tmp_path / 'artifacts/adapter-step0002179'
    adapter.mkdir(parents=True)
    (tmp_path / 'PROVENANCE.json').write_text(json.dumps({'active_adapter': 'artifacts/adapter-step0002179'}))
    monkeypatch.setattr(runtime, 'ROOT', project)
    monkeypatch.setattr(sys, 'argv', ['bundle_runtime', str(tmp_path)])
    runtime.main()
    config = yaml.safe_load((project / 'configs/bundle-runtime.yaml').read_text())
    assert config['model']['adapter'] == str(adapter)
