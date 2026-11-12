import json
import pytest

from voice_lab.hf_backend import parse_response


def test_qwen_reasoning_and_tool_calls_are_separated():
    message = parse_response('<think>需要准确计算</think><tool_call>{"name":"calculate","arguments":{"expression":"37*23"}}</tool_call>')
    assert message["content"] is None
    call = message["tool_calls"][0]
    assert call["function"]["name"] == "calculate"
    assert json.loads(call["function"]["arguments"]) == {"expression": "37*23"}


def test_normal_chat_remains_plain_content():
    assert parse_response("你好，今天怎么样？") == {"role": "assistant", "content": "你好，今天怎么样？"}


@pytest.mark.parametrize("value", ["<think>还在思考", '<tool_call>{"name":"calculate"}', '<tool_call>{"name":"calculate","arguments":[]}</tool_call>'])
def test_incomplete_or_invalid_tool_calls_are_rejected(value):
    with pytest.raises(ValueError):
        parse_response(value)


def test_default_backend_reuses_the_preuploaded_model(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import transformers
    from voice_lab import hf_backend
    from voice_lab.config import load_config

    path=tmp_path/"models/Qwen3-8B"
    path.mkdir(parents=True)
    calls=[]
    monkeypatch.setattr(hf_backend,"local_path",lambda value:tmp_path/value)

    class Model:
        def to(self,device):
            return self
        def eval(self):
            return self

    def tokenizer(source,**kwargs):
        calls.append(source)
        return object()

    def model(source,**kwargs):
        calls.append(source)
        return Model()

    monkeypatch.setattr(transformers.AutoTokenizer,"from_pretrained",tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM,"from_pretrained",model)
    with TestClient(hf_backend.create_app(load_config(),device="cpu")) as client:
        assert client.get('/health').json()['ready']
    assert calls==[str(path),str(path)]
