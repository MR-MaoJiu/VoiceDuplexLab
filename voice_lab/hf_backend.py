"""复用训练环境运行 Qwen 后台模型的 OpenAI 兼容接口。"""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import re
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from .config import load_config, local_path
from .download import BACKEND_REVISION


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    tools: list[dict] | None = None
    tool_choice: str = "auto"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    chat_template_kwargs: dict = Field(default_factory=dict)
    stream: bool = False


def parse_response(text):
    """将 Qwen 原生工具调用块转换为兼容接口消息。"""
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    elif "<think>" in text:
        raise ValueError("后台推理尚未完成")
    calls = []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        call = json.loads(match.group(1))
        arguments = call["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict) or not isinstance(call["name"], str):
            raise ValueError("后台工具调用格式错误")
        calls.append({
            "id": "call_" + uuid.uuid4().hex,
            "type": "function",
            "function": {"name": call["name"], "arguments": json.dumps(arguments, ensure_ascii=False)},
        })
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    if "<tool_call>" in content or "</tool_call>" in content:
        raise ValueError("后台工具调用不完整")
    message = {"role": "assistant", "content": content or None}
    if calls:
        message["tool_calls"] = calls
    return message


def create_app(config, device="cuda:0"):
    state = {}
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        options = config["backend"]
        name = options.get("weights_path", options["model"])
        path = local_path(name)
        cached = local_path("models/Qwen3-8B")
        if not options.get("weights_path") and name == "Qwen/Qwen3-8B" and options.get("revision") == BACKEND_REVISION and cached.is_dir():
            path = cached
        source = str(path) if path.is_dir() else name
        revision = {} if path.is_dir() else {"revision": options["revision"]}
        state["tokenizer"] = AutoTokenizer.from_pretrained(source, **revision)
        state["model"] = AutoModelForCausalLM.from_pretrained(
            source, torch_dtype=torch.bfloat16, attn_implementation="sdpa", **revision
        ).to(device).eval()
        yield
        state.clear()

    app = FastAPI(title="Qwen 工具后台", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ready": "model" in state, "model": config["backend"]["model"]}

    def generate(request):
        import torch

        tokenizer = state["tokenizer"]
        thinking = request.chat_template_kwargs.get("enable_thinking", config["backend"].get("enable_thinking", True))
        prompt = tokenizer.apply_chat_template(
            request.messages, tools=request.tools if request.tool_choice == "auto" else None,
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if inputs.input_ids.shape[1] + request.max_tokens > 16384:
            raise ValueError("后台上下文和输出合计超出16384个token")
        settings = {"do_sample": request.temperature > 0}
        if request.temperature > 0:
            settings["temperature"] = request.temperature
        with torch.inference_mode():
            result = state["model"].generate(
                **inputs, max_new_tokens=request.max_tokens,
                pad_token_id=tokenizer.eos_token_id, **settings
            )
        tokens = result[0, inputs.input_ids.shape[1]:]
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        for token in ("<|im_end|>", "<|endoftext|>"):
            text = text.replace(token, "")
        message = parse_response(text)
        reason = "length" if len(tokens) >= request.max_tokens else ("tool_calls" if message.get("tool_calls") else "stop")
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion", "created": int(time.time()), "model": request.model,
            "choices": [{"index": 0, "message": message, "finish_reason": reason}],
            "usage": {"prompt_tokens": inputs.input_ids.shape[1], "completion_tokens": len(tokens),
                      "total_tokens": inputs.input_ids.shape[1] + len(tokens)},
        }

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest):
        if request.stream or request.tool_choice not in ("auto", "none"):
            raise HTTPException(400, "当前后台支持非流式的auto或none工具选择")
        if request.model != config["backend"]["model"]:
            raise HTTPException(404, "后台模型名称不匹配")
        async with lock:
            try:
                return await asyncio.to_thread(generate, request)
            except ValueError as error:
                raise HTTPException(400, str(error)) from error

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(create_app(load_config(args.config), args.device), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
