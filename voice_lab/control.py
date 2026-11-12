from dataclasses import dataclass


@dataclass
class Control:
    kind: str
    text: str = ""


class ControlParser:
    """控制单元跨分块缓存，控制内容不会送入语音解码器。"""

    def __init__(self):
        self.buffer = ""
        self.control = False

    def reset(self):
        self.buffer = ""
        self.control = False

    def push(self, text, end=False):
        if not text and not self.buffer:
            return Control("speech")
        self.buffer += text
        candidate = self.buffer.lstrip()
        if len(candidate) > 4096:
            self.reset()
            return Control("invalid", "控制单元超过长度限制")
        markers = ("@@TASK ", "@@CANCEL@@", "@@VOICE SELF@@", "@@VOICE ORIGINAL@@", "@@VOICE DEFAULT@@")
        actions = {'@@VOICE SELF@@': 'self', '@@VOICE ORIGINAL@@': 'original', '@@VOICE DEFAULT@@': 'default'}
        if candidate in actions:
            self.reset()
            return Control("voice", actions[candidate])
        if candidate and any(marker.startswith(candidate) for marker in markers):
            if candidate == "@@CANCEL@@":
                self.reset()
                return Control("cancel")
            if end:
                self.reset()
                return Control("invalid", "控制单元不完整")
            return Control("pending")
        if candidate.startswith("@@TASK "):
            self.control = True
            if "@@END" in candidate:
                task, remainder = candidate[7:].split("@@END", 1)
                self.reset()
                if not task.strip() or remainder.strip():
                    return Control("invalid", "任务为空或混合了口语文本")
                return Control("delegate", task.strip())
            if end:
                self.reset()
                return Control("invalid", "委派标记缺少结束符")
            return Control("pending")
        if self.control or "@@" in candidate:
            self.reset()
            return Control("invalid", "控制标记必须独占输出单元")
        speech = self.buffer
        self.reset()
        return Control("speech", speech)
