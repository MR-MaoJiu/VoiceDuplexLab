import pytest
from voice_lab.tools import calculate, execute_tool


def test_calculate():
    assert calculate("(2 + 3) * 4 / 2") == 10


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "2**10000000",
        "9**9**9",
        "1/0",
        "True+1",
        "float('inf')",
    ],
)
def test_reject_bad_calculate(expression):
    with pytest.raises((ValueError, ZeroDivisionError)):
        calculate(expression)


async def test_time():
    value = await execute_tool("current_time", {}, {})
    assert value["timezone"] == "Asia/Shanghai"


async def test_knowledge(tmp_path):
    (tmp_path / "intro.txt").write_text("语音助手支持端到端对话", encoding="utf-8")
    result = await execute_tool(
        "knowledge_search", {"query": "语音"}, {"knowledge_dir": str(tmp_path)}
    )
    assert len(result["matches"]) == 1


async def test_schema_rejects_unknown_argument():
    with pytest.raises(ValueError):
        await execute_tool("current_time", {"command": "whoami"}, {})
