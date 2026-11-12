"""恢复归档选定检查点的原始路径、输入与优化器状态，保持续训数据指纹。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ORIGINAL_ROOT = Path('/root/gpufree-data/VoiceDuplexLab')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def restore_archive(archive, destination):
    with tarfile.open(archive) as source:
        for item in source.getmembers():
            if not (item.isfile() or item.isdir()) or item.name.startswith('/') or '..' in Path(item.name).parts:
                raise ValueError('归档含有异常路径')
        # 已存在且内容不同的文件交给调用者先备份，避免覆盖后续训练进度。
        for item in source.getmembers():
            target = destination / item.name
            if target.is_file() and item.isfile():
                incoming = source.extractfile(item)
                value = hashlib.sha256()
                for chunk in iter(lambda: incoming.read(4 * 1024 * 1024), b''):
                    value.update(chunk)
                if digest(target) != value.hexdigest():
                    raise FileExistsError(f'已有不同内容，请先备份 {target}')
        destination.mkdir(parents=True, exist_ok=True)
        source.extractall(destination, filter='data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / 'MANIFEST.json').read_text())
    metadata_path = bundle / 'PROVENANCE.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    checkpoint_archive = metadata.get(
        'active_checkpoint_archive',
        'checkpoints/VoiceDuplexLab-续训检查点-step0001133.tar.gz',
    )
    archives = metadata.get('training_archives') or [
        'training/VoiceDuplexLab-完整中文训练输入.tar.gz',
        checkpoint_archive,
        'training/bf7405ff68fd.inputs.tar.gz',
    ]
    entries = {entry['path']: entry for entry in manifest['files']}
    for name in archives:
        path = bundle / name
        if path.stat().st_size != entries[name]['bytes'] or digest(path) != entries[name]['sha256']:
            raise ValueError(f'文件校验失败 {name}')
        print(f'校验通过 {name}', flush=True)
    if args.verify_only:
        return
    if sys.platform != 'linux':
        raise RuntimeError('续训恢复请在 Linux NVIDIA 服务器执行')
    # 原清单、优化器配置和基座指纹保存了绝对路径，保持其字节内容以精确续训。
    if not ORIGINAL_ROOT.exists():
        shutil.copytree(bundle / 'source/VoiceDuplexLab', ORIGINAL_ROOT)
    for name in metadata.get('model_directories', ['MiniCPM-o-4_5', 'Qwen3-8B']):
        target = ORIGINAL_ROOT / 'models' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != (bundle / 'models' / name).resolve() and target.exists():
            for entry in manifest['files']:
                prefix = f'models/{name}/'
                if entry['path'].startswith(prefix):
                    existing = target / entry['path'].removeprefix(prefix)
                    if not existing.is_file() or digest(existing) != entry['sha256']:
                        raise ValueError(f'已有模型与固定版本不同 {existing}')
        if not target.exists():
            target.symlink_to(bundle / 'models' / name, target_is_directory=True)
    if metadata.get('training_archives'):
        for name in archives:
            restore_archive(bundle / name, ORIGINAL_ROOT)
    else:
        restore_archive(bundle / archives[0], ORIGINAL_ROOT / 'data/formal')
        restore_archive(bundle / archives[2], ORIGINAL_ROOT / 'data/automatic')
        restore_archive(bundle / archives[1], ORIGINAL_ROOT)
    if metadata.get('runtime_config'):
        source = bundle / metadata['runtime_config']
        target = ORIGINAL_ROOT / 'configs/repair-ready.yaml'
        if target.exists() and digest(target) != digest(source):
            raise FileExistsError(f'已有不同的运行配置，请先备份 {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print('已恢复归档选定的检查点、指令录音和完整训练输入。', flush=True)
    print('进入 /root/gpufree-data/VoiceDuplexLab 后安装 GPU 环境，再运行归档根目录的 resume-training.sh。')


if __name__ == '__main__':
    main()
