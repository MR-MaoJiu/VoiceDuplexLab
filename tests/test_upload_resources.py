import hashlib
import importlib.util
import json
from argparse import Namespace
import requests
import pytest
from voice_lab.config import ROOT

spec = importlib.util.spec_from_file_location(
    "upload_resources", ROOT / "scripts/upload_resources.py"
)
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


@pytest.mark.parametrize("value", ["test-token\n", "Bearer test-token\n", "\ufefftest-token\n"])
def test_reads_default_token_file(tmp_path, monkeypatch, value):
    monkeypatch.delenv("GPUFREE_AUTHORIZATION", raising=False)
    path = tmp_path / ".credentials/gpufree-token.txt"
    path.parent.mkdir()
    path.write_text(value, encoding="utf-8")
    args = Namespace(auth_pipe=None, token_file=None, resources=str(tmp_path))
    assert upload.authorization(args) == "Bearer test-token"


def test_explicit_token_file_overrides_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUFREE_AUTHORIZATION", "old-token")
    path = tmp_path / "token.txt"
    path.write_text("new-token\n", encoding="utf-8")
    args = Namespace(auth_pipe=None, token_file=str(path), resources=str(tmp_path))
    assert upload.authorization(args) == "Bearer new-token"


def test_empty_token_file_is_rejected(tmp_path):
    path = tmp_path / "token.txt"
    path.write_text("\n", encoding="utf-8")
    args = Namespace(auth_pipe=None, token_file=str(path), resources=str(tmp_path))
    with pytest.raises(ValueError, match="登录令牌为空"):
        upload.authorization(args)


def response(status=200, data=None):
    value = requests.Response()
    value.status_code = status
    value._content = json.dumps({"code": 200, "data": data}).encode()
    return value


def test_native_chunk_protocol_resumes_completed_chunks(tmp_path):
    payload = b"a" * upload.CHUNK_SIZE + b"last chunk"
    path = tmp_path / "resource.bin"
    path.write_bytes(payload)
    sent = []

    class Session:
        headers = {}

        def request(self, method, url, **kwargs):
            if url.endswith("file_check"):
                assert kwargs["params"]["md5"] == hashlib.md5(payload).hexdigest()
                return response(data=["0"])
            if url.endswith("upload"):
                fields = dict(kwargs["files"])
                assert fields["chunk_offset"] == (None, "1")
                assert fields["chunks"] == (None, "2")
                assert fields["md5"] == (None, hashlib.md5(payload).hexdigest())
                assert fields["file_md5"] == (
                    None,
                    hashlib.md5(b"last chunk").hexdigest(),
                )
                assert fields["file"][1] == b"last chunk"
                sent.append(1)
                return response()
            return response(
                data=[{"name": "resource.bin", "size": len(payload), "is_dir": False}]
            )

    client = upload.StorageClient(upload.DEFAULT_BASE, "Bearer test", session=Session())
    assert client.upload(path, path.name)["verified_size"]
    assert sent == [1]


def test_auth_failure_is_not_retried(monkeypatch):
    class Session:
        headers = {}

        def request(self, *args, **kwargs):
            return response(401)

    delays = []
    monkeypatch.setattr(upload.time, "sleep", delays.append)
    with pytest.raises(upload.AuthenticationError):
        upload.StorageClient(
            upload.DEFAULT_BASE, "Bearer test", session=Session()
        ).files()
    assert delays == []


def test_rate_limit_waits_twenty_seconds(monkeypatch):
    statuses = [429, 200]

    class Session:
        headers = {}

        def request(self, *args, **kwargs):
            return response(statuses.pop(0), data=[])

    delays = []
    monkeypatch.setattr(upload.time, "sleep", delays.append)
    assert (
        upload.StorageClient(
            upload.DEFAULT_BASE, "Bearer test", session=Session()
        ).files()
        == {}
    )
    assert delays == [20]


def test_plan_prefers_browser_parts_to_avoid_duplicate_model_uploads(tmp_path):
    browser = tmp_path / "upload-web/Qwen3"
    original = tmp_path / "upload/Qwen3"
    browser.mkdir(parents=True)
    original.mkdir(parents=True)
    (browser / "Qwen3.browser.manifest.json").write_text(
        json.dumps({"group": "Qwen3", "parts": [{"name": "Qwen3.webpart000"}]})
    )
    (browser / "Qwen3.webpart000.zip").write_bytes(b"fixture")
    (original / "Qwen3.manifest.json").write_text(
        json.dumps({"group": "Qwen3", "parts": [{"name": "oldpart000"}]})
    )
    (original / "oldpart000").write_bytes(b"old")
    plan = upload.upload_plan(tmp_path)
    assert set(plan) == {"Qwen3.browser.manifest.json", "Qwen3.webpart000.zip"}


@pytest.mark.parametrize(
    "url",
    [
        "http://ah01.storage.gpufree.cn",
        "https://example.com",
        "https://storage.gpufree.cn.evil.test",
    ],
)
def test_authentication_is_restricted_to_storage_hosts(url):
    with pytest.raises(ValueError):
        upload.validate_base(url)
