import ast
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo
import httpx
from .config import local_path

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "查询指定时区当前时间",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数字四则运算、括号和乘方",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "按城市名查询今天起七日天气预报，包含来源与时间",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": "检索项目本地知识库",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


def calculate(expression):
    if len(expression) > 200:
        raise ValueError("算式过长")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("算式过于复杂")
    binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("指数超出范围")
            value = binary[type(node.op)](left, right)
        else:
            raise ValueError("只支持数字四则运算与乘方")
        if isinstance(value, complex) or not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError("数值超出范围")
        return value

    return visit(tree.body)


async def execute_tool(name, arguments, config):
    """固定工具注册表，所有返回内容都作为外部数据交给模型。"""
    schema = next(
        (
            item["function"]["parameters"]
            for item in TOOL_SCHEMAS
            if item["function"]["name"] == name
        ),
        None,
    )
    if schema is None:
        raise ValueError("未知工具")
    if not isinstance(arguments, dict) or set(arguments) - set(schema["properties"]):
        raise ValueError("工具参数不符合定义")
    for key in schema.get("required", []):
        if key not in arguments:
            raise ValueError(f"缺少参数 {key}")
    if any(
        not isinstance(value, str) or len(value) > 500 for value in arguments.values()
    ):
        raise ValueError("参数必须为长度不超过 500 的字符串")
    if name == "current_time":
        zone = arguments.get("timezone", "Asia/Shanghai")
        return {"timezone": zone, "time": datetime.now(ZoneInfo(zone)).isoformat()}
    if name == "calculate":
        return {"value": calculate(arguments["expression"])}
    if name == "knowledge_search":
        query = arguments["query"].strip().lower()
        if not query:
            raise ValueError("检索词为空")
        words = query.split()
        results = []
        directory = local_path(config["knowledge_dir"]).resolve()
        for path in sorted(directory.rglob("*.txt")):
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(directory)
                or path.stat().st_size > 1_000_000
            ):
                continue
            text = path.read_text(encoding="utf-8")
            lines = [
                line
                for line in text.splitlines()
                if any(word in line.lower() for word in words)
            ]
            if lines:
                results.append(
                    {
                        "source": str(path.relative_to(directory)),
                        "text": "\n".join(lines)[:2000],
                    }
                )
            if len(results) == 5:
                break
        return {"matches": results}
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": arguments["city"], "count": 1, "language": "zh"},
        )
        response.raise_for_status()
        places = response.json().get("results", [])
        if not places:
            return {"status": "not_found", "city": arguments["city"]}
        place = places[0]
        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
                "timezone": "auto",
                "forecast_days": 7,
            },
        )
        response.raise_for_status()
        return {
            "place": {key: place.get(key) for key in ("name", "country", "admin1")},
            "forecast": response.json(),
            "source": "https://open-meteo.com/",
            "retrieved_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        }
