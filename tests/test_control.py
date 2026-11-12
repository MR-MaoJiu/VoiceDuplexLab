import pytest
from voice_lab.control import ControlParser


def test_chat_never_delegates():
    value = ControlParser().push("今天怎么样？", end=True)
    assert value.kind == "speech"


def test_task_split_every_character():
    parser = ControlParser()
    text = "@@TASK 查询上海天气 @@END"
    outputs = [parser.push(char) for char in text]
    assert all(item.kind == "pending" for item in outputs[:-1])
    assert outputs[-1].kind == "delegate"
    assert outputs[-1].text == "查询上海天气"


def test_cancel():
    parser = ControlParser()
    assert parser.push("@@CAN").kind == "pending"
    assert parser.push("CEL@@").kind == "cancel"


@pytest.mark.parametrize(
    "text",
    [
        "普通话@@TASK 查询 @@END",
        "@@TASK  @@END",
        "@@TASK 查询 @@END 继续说话",
        "@@INVALID",
    ],
)
def test_bad_control_is_muted(text):
    assert ControlParser().push(text, end=True).kind == "invalid"


def test_incomplete_at_end_is_muted():
    assert ControlParser().push("@@TASK 查询", end=True).kind == "invalid"


def test_overlong_control():
    assert ControlParser().push("@@TASK " + "字" * 4100).kind == "invalid"
