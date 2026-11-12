"""本地资源分卷、校验和服务器离线恢复。"""

import argparse
from contextlib import nullcontext, contextmanager, ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import zipfile


def default_resource_directory(project_root=None, mount_root='/Volumes'):
    """优先显式配置和已有资源库，不把开发机器的卷名写入发布代码。"""
    configured = os.environ.get('VOICE_RESOURCES')
    if configured:
        return str(Path(configured).expanduser())
    project = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    sibling = project.parent / 'VoiceDuplexLab-resources'
    for path in (sibling, Path.home() / 'Desktop/VoiceDuplexLab-resources'):
        if path.is_dir():
            return str(path)
    mounts = Path(mount_root)
    existing = [path / 'VoiceDuplexLab-resources' for path in mounts.iterdir() if (path / 'VoiceDuplexLab-resources').is_dir()] if mounts.is_dir() else []
    return str(existing[0] if len(existing) == 1 else sibling)

CATALOG = {
    "MiniCPM": {
        "root": "models/MiniCPM-o-4_5",
        "repository": "openbmb/MiniCPM-o-4_5",
        "revision": "fb09628e98b1f55d8d5b863e0eccfcdd11bff499",
    },
    "Qwen3": {
        "root": "models/Qwen3-8B",
        "repository": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    },
    "SmoothConv": {
        "root": "data/raw/smoothconv",
        "repository": "qualialabsAI/SmoothConv",
        "revision": "cd74b4fca285a66d6ac8c16228d0953ff1e0cda2",
    },
}

CATALOG.update(
    {
        "SmoothConvFull": {
            "root": "data/raw/smoothconv-full-2ch",
            "repository": "qualialabsAI/SmoothConv",
            "revision": "cd74b4fca285a66d6ac8c16228d0953ff1e0cda2",
        },
        "ForcedAligner": {
            "root": "models/Qwen3-ForcedAligner-0.6B",
            "repository": "Qwen/Qwen3-ForcedAligner-0.6B",
            "revision": "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
        },
        "ToolACE": {
            "root": "data/raw/toolace",
            "repository": "Team-ACE/ToolACE",
            "revision": "6bda777c88d21e5a204703c1ee45597a8fa4f734",
        },
    }
)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


class SplitWriter:
    """流式写入定长分卷，完整分卷通过原子重命名发布。"""

    def __init__(self, directory, group, limit, extension=".tar"):
        self.directory, self.group, self.limit = Path(directory), group, limit
        self.directory.mkdir(parents=True, exist_ok=True)
        self.extension = extension
        self.parts = []
        self.stream = None
        self.size = 0

    def write(self, data):
        original_length = len(data)
        offset = 0
        while offset < len(data):
            if self.stream is None:
                self.name = f"{self.group}{self.extension}.part{len(self.parts):03d}"
                self.path = self.directory / (self.name + ".tmp")
                if (self.directory / self.name).exists():
                    raise FileExistsError(f"分卷已存在 {self.name}")
                self.stream = self.path.open("wb")
                self.hash = hashlib.sha256()
                self.size = 0
            count = min(self.limit - self.size, len(data) - offset)
            chunk = data[offset : offset + count]
            self.stream.write(chunk)
            self.hash.update(chunk)
            self.size += count
            offset += count
            if self.size == self.limit:
                self._publish()
        return original_length

    def _publish(self):
        self.stream.close()
        self.path.replace(self.directory / self.name)
        self.parts.append(
            {"name": self.name, "size": self.size, "sha256": self.hash.hexdigest()}
        )
        self.stream = None
        print(f"分卷完成 {self.name} {self.size} 字节", flush=True)

    def finish(self):
        if self.stream is not None:
            self._publish()


def pack_group(resources, output, group, part_bytes=1024**3, compress=False):
    source = Path(resources) / CATALOG[group]["root"]
    output = Path(output)
    manifest_path = output / f"{group}.manifest.json"
    if manifest_path.exists():
        value = json.loads(manifest_path.read_text())
        if all(
            (output / part["name"]).stat().st_size == part["size"]
            and digest_file(output / part["name"]) == part["sha256"]
            for part in value["parts"]
        ):
            return manifest_path
        raise ValueError("现有分卷校验失败，请使用新的输出目录重新打包")
    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(source).parts
        and not path.name.endswith((".lock", ".incomplete"))
    )
    if not files:
        raise ValueError("资源目录为空")
    writer = SplitWriter(
        output, group, part_bytes, extension=".tar.gz" if compress else ".tar"
    )
    inventory = []
    try:
        # Python 3.11 的 tar 流接口不接收 compresslevel，压缩层单独管理。
        compression = (
            gzip.GzipFile(
                filename="", fileobj=writer, mode="wb", compresslevel=1, mtime=0
            )
            if compress
            else nullcontext(writer)
        )
        with compression as stream:
            with tarfile.open(fileobj=stream, mode="w|") as archive:
                for path in files:
                    if path.is_symlink():
                        raise ValueError("资源包要求使用普通文件")
                    name = str(Path(CATALOG[group]["root"]) / path.relative_to(source))
                    inventory.append(
                        {
                            "path": name,
                            "size": path.stat().st_size,
                            "sha256": digest_file(path),
                        }
                    )
                    archive.add(path, arcname=name, recursive=False)
        writer.finish()
    finally:
        if writer.stream is not None:
            writer.stream.close()
    manifest = {
        "format": 1,
        "compression": "gzip" if compress else "none",
        "group": group,
        **CATALOG[group],
        "parts": writer.parts,
        "files": inventory,
    }
    write_json(manifest_path, manifest)
    return manifest_path


@contextmanager
def open_part(path):
    """读取原始分卷或浏览器兼容的单分卷 ZIP 封装。"""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) != 1 or entries[0].filename != path.name[:-4]:
                raise ValueError(f"ZIP 分卷封装内容不符合清单 {path.name}")
            with archive.open(entries[0]) as stream:
                yield stream
    else:
        with path.open("rb") as stream:
            yield stream


def verify_part(path, record):
    checksum = hashlib.sha256()
    size = 0
    with open_part(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
            size += len(chunk)
    if size != record["size"] or checksum.hexdigest() != record["sha256"]:
        raise ValueError(f"分卷校验失败 {record['name']}")


class JoinedReader:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.stream = None
        self.contexts = ExitStack()

    def read(self, size):
        chunks = []
        remaining = size
        while remaining:
            if self.stream is None:
                try:
                    self.stream = self.contexts.enter_context(
                        open_part(next(self.paths))
                    )
                except StopIteration:
                    break
            data = self.stream.read(remaining)
            if data:
                chunks.append(data)
                remaining -= len(data)
            else:
                self.stream.close()
                self.stream = None
        return b"".join(chunks)

    def close(self):
        self.contexts.close()


def restore_group(source, target, group):
    source, target = Path(source), Path(target)
    manifest_path = source / f"{group}.browser.manifest.json"
    if not manifest_path.is_file():
        manifest_path = source / f"{group}.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if any(manifest[key] != value for key, value in CATALOG[group].items()):
        raise ValueError("资源来源与固定版本不一致")
    destination = target / manifest["root"]
    receipt = target / ".resource_receipts" / f"{group}.json"
    manifest_hash = digest_file(manifest_path)
    if receipt.exists():
        if json.loads(receipt.read_text())["manifest_sha256"] != manifest_hash:
            raise ValueError("已恢复资源与当前资源包版本不一致")
        for item in manifest["files"]:
            path = target / item["path"]
            if not path.is_file() or digest_file(path) != item["sha256"]:
                raise ValueError(f"已有资源校验失败 {item['path']}")
        return
    if destination.exists():
        raise FileExistsError(f"资源目标目录已存在 {destination}")
    parts = []
    for part in manifest["parts"]:
        if Path(part["name"]).name != part["name"]:
            raise ValueError("分卷名称格式错误")
        path = source / part["name"]
        if not path.is_file():
            path = source / (part["name"] + ".zip")
        verify_part(path, part)
        parts.append(path)
    temporary = target / f".restoring-{group}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    expected = {item["path"]: item for item in manifest["files"]}
    seen = set()
    reader = JoinedReader(parts)
    try:
        with tarfile.open(fileobj=reader, mode="r|*") as archive:
            for member in archive:
                if (
                    member.name not in expected
                    or not member.isfile()
                    or member.name in seen
                ):
                    raise ValueError("归档包含未登记的成员")
                path = (temporary / member.name).resolve()
                if not path.is_relative_to(temporary.resolve()):
                    raise ValueError("归档路径超出目标目录")
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as body, path.open("wb") as stream:
                    shutil.copyfileobj(body, stream)
                if (
                    path.stat().st_size != expected[member.name]["size"]
                    or digest_file(path) != expected[member.name]["sha256"]
                ):
                    raise ValueError(f"解包文件校验失败 {member.name}")
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError("归档文件数量与清单不一致")
        destination.parent.mkdir(parents=True, exist_ok=True)
        (temporary / manifest["root"]).rename(destination)
        write_json(receipt, {"manifest_sha256": manifest_hash})
    finally:
        reader.close()
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"离线资源已恢复 {group}", flush=True)


def activate(target):
    from .config import ROOT, load_config
    import yaml

    target = Path(target).resolve()
    config = load_config()
    dataset_group = (
        "SmoothConvFull"
        if (target / ".resource_receipts/SmoothConvFull.json").exists()
        else "SmoothConv"
    )
    for group in ("MiniCPM", "Qwen3", dataset_group):
        if not (target / ".resource_receipts" / f"{group}.json").is_file():
            raise ValueError(f"资源尚未恢复 {group}")
    config["model"]["path"] = str(target / CATALOG["MiniCPM"]["root"])
    config["backend"]["weights_path"] = str(target / CATALOG["Qwen3"]["root"])
    if dataset_group == "SmoothConvFull":
        config["data"] = {"raw_dir": str(target / CATALOG[dataset_group]["root"])}
    path = ROOT / "configs/resources-local.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"离线资源配置已生成 {path}")


def main():
    from .config import ROOT

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--resources", required=True)
    pack.add_argument("--output", required=True)
    pack.add_argument("--group", choices=CATALOG, required=True)
    pack.add_argument("--part-mib", type=int, default=1024)
    pack.add_argument("--gzip", action="store_true")
    restore = commands.add_parser("restore")
    restore.add_argument("--source", required=True)
    restore.add_argument("--target", default=str(ROOT))
    args = parser.parse_args()
    if args.command == "pack":
        if args.part_mib < 1:
            parser.error("分卷大小必须为正整数")
        print(
            pack_group(
                args.resources,
                args.output,
                args.group,
                args.part_mib * 1024**2,
                compress=args.gzip,
            )
        )
    else:
        if Path(args.target).resolve() != ROOT.resolve():
            parser.error("请把资源恢复到当前项目目录，以复用默认语料路径")
        groups = [
            "MiniCPM",
            "Qwen3",
            "SmoothConvFull"
            if (Path(args.source) / "SmoothConvFull.manifest.json").exists()
            else "SmoothConv",
        ]
        for optional in ("ToolACE", "ForcedAligner"):
            if (Path(args.source) / f"{optional}.manifest.json").exists():
                groups.append(optional)
        for group in groups:
            restore_group(args.source, args.target, group)
        activate(args.target)


if __name__ == "__main__":
    main()
