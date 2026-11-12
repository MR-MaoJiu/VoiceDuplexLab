import hashlib
from difflib import SequenceMatcher
import json
import math
import re
from pathlib import Path
import numpy as np
import soundfile as sf
from .corpus import write_jsonl
from .download import atomic_json
from .model import load_base, read_audio
from .prepare import target_ids

SCENARIOS = [
    {
        "kind": "direct",
        "user": "今天有点累，想找个人说说话。",
        "answer": "辛苦了，今天发生了什么？我听着呢。",
    },
    {
        "kind": "direct",
        "user": "你觉得下雨天适合做什么？",
        "answer": "可以听听音乐，看看书，或者给自己做顿热乎的饭。",
    },
    {
        "kind": "direct",
        "user": "给我讲个简短的笑话吧。",
        "answer": "小猫问小狗，你为什么总追尾巴？小狗说，我在追求幸福呀。",
    },
    {"kind": "clarify", "user": "帮我查一下天气。", "answer": "你想查哪个城市的天气？"},
    {
        "kind": "tool",
        "user": "帮我算一下三十七乘以二十三。",
        "task": "计算37乘以23。",
        "result": "通过计算器得到37乘以23等于851。",
        "answer": "计算结果是八百五十一。",
    },
    {
        "kind": "tool",
        "user": "查一下上海明天的天气。",
        "task": "查询上海明天的天气。",
        "result": "训练用模拟工具结果，上海明天有小雨，气温18到23摄氏度。",
        "answer": "查询结果显示明天有小雨，气温十八到二十三度，出门记得带伞。",
    },
    {
        "kind": "error",
        "user": "帮我查一下北京现在的天气。",
        "task": "查询北京当前天气。",
        "result": "天气查询超时，未取得可靠的实时数据。",
        "answer": "这次天气查询超时了，暂时没拿到实时结果，要不要稍后再试？",
    },
    {
        "kind": "cancel",
        "user": "刚才让你查的天气不用查了，取消吧。",
        "answer": "好的，已经取消这次查询。",
    },
    {
        "kind": "tool",
        "user": "查一下这个语音项目支持哪些功能。",
        "task": "在项目知识库查询语音项目的功能。",
        "result": "项目支持端到端语音对话、每用户参考音色和异步工具协作。",
        "answer": "这个项目支持端到端语音对话，每位用户可以选自己的音色，也能异步调用工具。",
    },
    {
        "kind": "direct",
        "user": "帮我想想周末怎么放松一下。",
        "answer": "可以选一件轻松的小事，比如散步、做饭，或者和朋友见个面。",
    },
    {
        "kind": "clarify",
        "user": "帮我查一下现在几点了。",
        "answer": "你想查哪个城市或时区的时间？",
    },
    {
        "kind": "tool",
        "user": "一百二十除以五再加七是多少？",
        "task": "用计算器计算120除以5加7。",
        "result": "计算器结果为31。",
        "answer": "算出来是三十一。",
    },
]


def scenario_catalog():
    """提供可复现的中文工具、闲聊和等待期间继续说话的监督样本。"""
    scenarios = list(SCENARIOS)
    requests = ["帮我计算{a}乘以{b}。", "用计算器算一下{a}加上{b}。", "{a}减{b}等于多少？", "请算出({a}+{b})乘以2的结果。"]
    operations = [lambda a, b: a * b, lambda a, b: a + b,
                  lambda a, b: a - b, lambda a, b: (a + b) * 2]
    for index in range(48):
        a, b = 17 + index * 3, 5 + index % 19
        question = requests[index % 4].format(a=a, b=b)
        result = operations[index % 4](a, b)
        scenarios.append({
            "kind": "tool", "user": question, "task": "使用计算器完成：" + question,
            "result": f"计算器返回数值 {result}。", "answer": f"计算结果是{result}。",
            "waiting": "我正在核对计算结果，你也可以继续说。" if index % 3 == 0 else "",
        })
    for index, city in enumerate(("北京", "上海", "广州", "深圳", "杭州", "成都", "南京", "武汉")):
        scenarios.append({
            "kind": "tool", "user": f"查一下{city}明天的天气和温度。",
            "task": f"查询{city}明天的天气预报，返回温度与降雨信息。",
            "result": f"合成训练样本的模拟结果：{city}明天多云，气温{15+index}到{23+index}摄氏度。",
            "answer": f"预报是多云，气温{15+index}到{23+index}度。",
            "waiting": "天气结果还没回来，我们可以接着聊。",
        })
    return scenarios


def split_text(text, tokenizer, limit):
    chunks, pending = [], ""
    for char in text:
        candidate = pending + char
        if len(target_ids({"text": candidate, "end": True}, tokenizer)) > limit:
            if not pending:
                raise ValueError("单个字符超出当前 token 限制")
            chunks.append(pending)
            pending = char
        else:
            pending = candidate
    if pending:
        chunks.append(pending)
    return [
        {"text": chunk, "end": index == len(chunks) - 1}
        for index, chunk in enumerate(chunks)
    ]


def synthesize_query(model, text, reference, path):
    """用基座的已知文本语音生成接口合成工具训练的用户输入。"""
    path = Path(path)
    if path.is_file():
        audio = read_audio(path)
        if len(audio) >= 1600:
            return audio
    temporary = path.with_suffix(".partial.wav")
    model.reset_session(reset_token2wav_cache=True)
    model.tts.audio_tokenizer.cache = None
    system = model.get_sys_prompt(
        ref_audio=read_audio(reference), mode="audio_assistant", language="zh"
    )
    model.chat(
        msgs=[
            system,
            {"role": "user", "content": ["请朗读给定文本。"]},
            {"role": "assistant", "content": ["<|tts_bos|>" + text + "<|tts_eos|>"]},
        ],
        teacher_forcing=True,
        use_tts_template=True,
        generate_audio=True,
        output_audio_path=str(temporary),
        do_sample=False,
    )
    if not temporary.is_file():
        raise RuntimeError("基座语音合成未生成 WAV，请检查上游 Token2Wav 依赖与日志")
    audio = read_audio(temporary)
    if len(audio) < 1600 or np.max(np.abs(audio)) < 1e-5:
        raise RuntimeError("合成语音为空或静音，停止生成训练数据")
    temporary.replace(path)
    return audio


def build_synthetic_record(
    scenario, audio, voice, identifier, tokenizer, max_tokens, output,
    splitter=None, trailing_listen=0,
):
    splitter = splitter or split_text
    # 输入轨保留问题语音，工具等待和助手说话期间填静音，持续时间由真实合成结果决定。
    listen_units = math.ceil(len(audio) / 16000) + 1
    units = [{"listen": True} for _ in range(listen_units)]
    kind = scenario["kind"]
    if kind in ("tool", "error"):
        units.extend(splitter("我查一下。", tokenizer, max_tokens))
        control = "@@TASK " + scenario["task"] + " @@END"
        units.extend(split_text(control, tokenizer, max_tokens))
        units.append({"listen": True})
        if scenario.get("waiting"):
            units.extend(splitter(scenario["waiting"], tokenizer, max_tokens))
        units.append({"listen": True})
        result = {
            "status": "error" if kind == "error" else "ok",
            "task": scenario["task"],
            "result": scenario["result"],
            "provenance": "synthetic_tool_fixture",
        }
        answer = splitter(scenario["answer"], tokenizer, max_tokens)
        answer[0]["backend"] = result
        units.extend(answer)
    elif kind == "cancel":
        units.append({"text": "取消", "cancel": True, "end": True})
        units.extend(splitter(scenario["answer"], tokenizer, max_tokens))
    else:
        units.extend(splitter(scenario["answer"], tokenizer, max_tokens))
    units.extend({"listen": True} for _ in range(trailing_listen))
    required = int((len(units) + 0.1) * 16000)
    audio = np.pad(audio, (0, max(0, required - len(audio))))
    input_path = output / "audio" / (identifier + ".wav")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(input_path, audio, 16000, subtype="PCM_16")
    return {
        "id": identifier,
        "voice_id": voice["voice_id"],
        "voice": voice["voice"],
        "audio": str(input_path.resolve()),
        "units": units,
        "provenance": {
            "kind": "synthetic_instruction",
            "scenario": kind,
            "user_text": scenario["user"],
            "generator": "MiniCPM-o-4_5 teacher_forcing",
            "tool_results": "synthetic fixtures",
        },
    }


def create_instructions(config, real_records, output, count=24):
    output = Path(output)
    references = {
        item["voice_id"]: {"voice_id": item["voice_id"], "voice": item["voice"]}
        for item in real_records
    }
    voices = list(references.values())
    if not voices:
        raise ValueError("真实语料没有可用参考音色")
    model = load_base(config, with_adapter=False)
    model.init_tts()
    scenarios = scenario_catalog()
    records = []
    asr_validation = None
    directory = output / "queries"
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        scenario = scenarios[index % len(scenarios)]
        voice = voices[index % len(voices)]
        user_voice = voices[(index + 1) % len(voices)]
        digest = hashlib.sha256(
            (
                "teacher-forcing-v2|" + scenario["user"]
                + hashlib.sha256(Path(user_voice["voice"]).read_bytes()).hexdigest()
                + config["model"]["revision"]
            ).encode()
        ).hexdigest()[:20]
        query_path = directory / f"{digest}.wav"
        print(f"合成指令样本 {index + 1}/{count} {scenario['kind']}", flush=True)
        audio = synthesize_query(
            model, scenario["user"], user_voice["voice"], query_path
        )
        if index == 0:
            model.reset_session(reset_token2wav_cache=True)
            transcript = model.chat(
                msgs=[{"role": "user", "content": ["请仔细听这段音频片段，并将其内容逐字记录。", audio]}],
                do_sample=False, max_new_tokens=128, use_tts_template=True,
                generate_audio=False,
            )
            expected = re.sub(r"\W", "", scenario["user"])
            actual = re.sub(r"\W", "", str(transcript))
            similarity = SequenceMatcher(None, expected, actual).ratio()
            asr_validation = {"expected": scenario["user"], "transcript": str(transcript), "similarity": similarity}
            print("合成语音转写自检 " + json.dumps(asr_validation, ensure_ascii=False), flush=True)
            if similarity < 0.7:
                raise RuntimeError("首条合成录音与给定文本不符，停止生成训练数据")
        records.append(
            build_synthetic_record(
                scenario,
                audio,
                voice,
                f"instruction-{index:05d}-{digest}",
                model.processor.tokenizer,
                config["model"]["max_speak_tokens"],
                output,
            )
        )
    write_jsonl(output / "instructions.jsonl", records)
    atomic_json(
        output / "instruction_report.json",
        {
            "samples": len(records),
            "reference_voices": len(voices),
            "source": "公开参考音色条件下的基座语音合成，工具结果为显式模拟标注",
            "scenarios": len(scenarios),
            "synthesis_version": 2,
            "asr_validation": asr_validation,
        },
    )
    return records


def main():
    import argparse
    from .config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--config")
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    create_instructions(load_config(args.config), records, args.output, args.count)


if __name__ == "__main__":
    main()
