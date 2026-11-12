import json
import httpx
from voice_lab.backend import Backend


async def test_tool_roundtrip_without_network(monkeypatch):
    requests = []
    async_client = httpx.AsyncClient

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"expression":"1+2"}',
                        },
                    }
                ],
            }
        else:
            assert json.loads(body["messages"][-1]["content"])["value"] == 3
            message = {"role": "assistant", "content": "计算结果为 3"}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )
    events = []

    async def report(event):
        events.append(event)

    backend = Backend(
        {
            "base_url": "http://backend.test/v1",
            "model": "test",
            "timeout": 3,
            "max_rounds": 3,
        }
    )
    result = await backend.run("计算 1+2", report)
    assert result == "计算结果为 3"
    assert events == [{"type": "tool", "name": "calculate"}]
    assert len(requests) == 2
