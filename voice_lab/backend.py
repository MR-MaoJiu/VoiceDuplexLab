import asyncio
import json
import os
import httpx
from .prompts import BACKEND_PROMPT
from .tools import TOOL_SCHEMAS, execute_tool


def result_context(value, prioritize=False):
    context = "【后台结果】" + json.dumps(value, ensure_ascii=False) + "\n请结合用户当前问题自然回答。"
    if prioritize:
        context += (
            "\n此前委派任务已经完成。请现在先用简短口语回应上面 task 对应的实际 result，"
            "说清核心结论。如果用户正在闲聊，先反馈任务结果，再继续自然回应用户。"
        )
    return context


class Backend:
    def __init__(self, config):
        self.config = config

    async def run(self, task, report):
        async with asyncio.timeout(self.config["timeout"]):
            return await self._run(task, report)

    async def _run(self, task, report):
        messages = [
            {"role": "system", "content": BACKEND_PROMPT},
            {"role": "user", "content": task},
        ]
        headers = {}
        key = os.environ.get(self.config.get("api_key_env", "BACKEND_API_KEY"))
        if key:
            headers["Authorization"] = f"Bearer {key}"
        async with httpx.AsyncClient(
            timeout=self.config["timeout"], headers=headers
        ) as client:
            for _ in range(self.config["max_rounds"]):
                response = await client.post(
                    self.config["base_url"].rstrip("/") + "/chat/completions",
                    json={
                        "model": self.config["model"],
                        "messages": messages,
                        "tools": TOOL_SCHEMAS,
                        "tool_choice": "auto",
                        "temperature": 0.2,
                        "max_tokens": self.config.get("max_tokens", 4096),
                        "chat_template_kwargs": {
                            "enable_thinking": self.config.get("enable_thinking", True)
                        },
                    },
                )
                response.raise_for_status()
                choice = response.json()["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise RuntimeError(
                        "后台输出达到 token 上限，请增加 max_tokens 或缩短任务"
                    )
                message = choice["message"]
                calls = message.get("tool_calls") or []
                if not calls:
                    result = message.get("content")
                    if not result or not result.strip():
                        raise ValueError("后台模型返回了空结果")
                    if "<think>" in result or "</think>" in result:
                        raise ValueError(
                            "后台推理内容未被解析，请检查 reasoning-parser 配置"
                        )
                    return result[:12000]
                if len(calls) > 8:
                    raise ValueError("单轮工具调用数量过多")
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content"),
                        "tool_calls": calls,
                    }
                )
                for call in calls:
                    function = call["function"]
                    await report({"type": "tool", "name": function["name"]})
                    try:
                        arguments = json.loads(function["arguments"])
                        result = await execute_tool(
                            function["name"], arguments, self.config
                        )
                    except Exception as exc:
                        result = {"status": "error", "reason": str(exc)[:500]}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(result, ensure_ascii=False)[:16000],
                        }
                    )
        raise RuntimeError("后台处理达到最大工具轮数")


class TaskCoordinator:
    """仅接受当前请求版本的结果，任务结束时清理引用。"""

    def __init__(self, backend, report, deliver):
        self.backend, self.report, self.deliver = backend, report, deliver
        self.revision = 0
        self.task = None
        self.lock = asyncio.Lock()
        self.notified_revision = -1

    def pending_context(self):
        if self.task is not None and not self.task.done() and self.notified_revision != self.revision:
            self.notified_revision = self.revision
            return (
                "【后台状态】本次委派任务仍在执行，工具结果尚未返回。"
                "可以继续闲聊、鼓励用户或说明处理进度。"
                "本次任务的数值和结论须等【后台结果】返回后再讲，不能提前把自行推测当作工具结果。"
            )
        return None

    async def _cancel(self):
        self.revision += 1
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None

    async def cancel(self):
        async with self.lock:
            await self._cancel()

    async def submit(self, description):
        async with self.lock:
            await self._cancel()
            revision = self.revision
            await self.report(
                {"type": "task_started", "revision": revision, "task": description}
            )
            self.task = asyncio.create_task(self._execute(description, revision))

    async def _execute(self, description, revision):
        async def report(event):
            if revision == self.revision:
                await self.report({**event, "revision": revision})

        try:
            result = await self.backend.run(description, report)
            status = "ok"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result, status = (
                f"后台任务失败，原因为 {type(exc).__name__}。请说明暂时无法完成，可邀请用户稍后重试。",
                "error",
            )
            await report({"type": "task_error", "error": type(exc).__name__})
        if revision == self.revision:
            await self.deliver(
                revision, {"status": status, "task": description, "result": result}
            )
