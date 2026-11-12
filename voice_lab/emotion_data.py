"""下载并编译SMIIP-NV原始中文录音，保留情绪与真实非语言发声监督。"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import re
import zipfile
import numpy as np
import soundfile as sf
from .download import atomic_json

REPOSITORY = 'xunyi/SMIIP-NV'
REVISION = '081cf86ac83fbc26253f1350dff6393d88c49a8e'
EMOTION_NAMES = {'高兴': 'happy', '开心': 'happy', '伤心': 'sad', '悲伤': 'sad', '生气': 'angry', '中性': 'neutral'}


def decoded_name(info):
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode('cp437').decode('gbk')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def build_candidates(archive):
    files = {decoded_name(info): info for info in archive.infolist()}
    speakers = sorted({Path(name).parts[1] for name in files if len(Path(name).parts) > 2 and Path(name).parts[1].startswith('MIC-')})
    if len(speakers) < 4:
        raise ValueError('原始录音需要至少四位说话人供独立划分')
    heldout = set(random.Random(20261112).sample(speakers, max(2, len(speakers) // 5)))
    candidates = defaultdict(list)
    for name, info in files.items():
        if not name.endswith('/annotation.txt'):
            continue
        path = Path(name)
        emotion = EMOTION_NAMES.get(path.parent.name)
        if emotion is None:
            continue
        speaker = path.parts[1]
        for line in archive.read(info).decode('utf-8-sig').splitlines():
            parts = line.replace('\\t', '\t').split('\t', 2)
            if len(parts) != 3:
                continue
            filename, text, times = parts
            text = text.strip()
            normalized = re.sub(r'<[^>]+>|\W', '', text)
            bucket = int(hashlib.sha256(normalized.encode()).hexdigest()[:8], 16) % 5
            # 同时保留未见说话人和未见文本，避免重复台词使验证结果虚高。
            if speaker in heldout and bucket == 0:
                split = 'eval'
            elif speaker not in heldout and bucket != 0:
                split = 'train'
            else:
                continue
            audio_name = str(path.parent / filename.strip())
            if audio_name not in files:
                continue
            tags = list(re.finditer(r'<(laughter|crying|cough)>', text))
            try:
                intervals = ast.literal_eval('[' + times.strip() + ']')
                if len(intervals) == 1 and isinstance(intervals[0], list) and intervals[0] and isinstance(intervals[0][0], list):
                    intervals = intervals[0]
            except (ValueError, SyntaxError):
                continue
            if len(tags) != len(intervals):
                continue
            origin = {'member': audio_name, 'speaker': speaker, 'split': split, 'source_text': text}
            if tags:
                prefix = text[:tags[0].start()].strip(' ，。！？')
                if len(prefix) >= 2 and float(intervals[0][0]) >= 0.6:
                    candidates[(emotion, split)].append({**origin, 'style': emotion, 'text': prefix,
                                                         'start': 0.0, 'end': float(intervals[0][0]), 'event': 'speech'})
            elif len(normalized) >= 2:
                candidates[(emotion, split)].append({**origin, 'style': emotion, 'text': text, 'start': 0.0, 'end': None, 'event': 'speech'})
            for number, (tag, interval) in enumerate(zip(tags, intervals)):
                kind = tag.group(1)
                if kind not in ('laughter', 'crying') or len(interval) != 2:
                    continue
                start, end = map(float, interval)
                if not 0.4 <= end - start <= 6:
                    continue
                style = 'laugh' if kind == 'laughter' else 'cry'
                candidates[(style, split)].append({**origin, 'style': style, 'text': '哈哈哈。' if style == 'laugh' else '呜呜。',
                                                   'start': start, 'end': end, 'event': kind, 'event_index': number})
    return files, candidates, sorted(heldout)


def prepare_archive(path, output, limit):
    from scipy.signal import resample_poly
    import math
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.jsonl').exists():
        raise FileExistsError('训练清单已存在，请复用该清单或选择新目录')
    rows = []
    rejected = Counter()
    with zipfile.ZipFile(path) as archive:
        files, candidates, heldout = build_candidates(archive)
        for (style, split), values in sorted(candidates.items()):
            random.Random(20261112).shuffle(values)
            accepted = 0
            quota = limit if split == 'train' else max(4, limit // 4)
            for item in values:
                if accepted >= quota:
                    break
                raw = archive.read(files[item['member']])
                audio, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
                audio = audio.mean(axis=1)
                start = max(0, int(item['start'] * rate))
                end = len(audio) if item['end'] is None else min(len(audio), int(item['end'] * rate))
                audio = audio[start:end]
                if not 0.4 <= len(audio) / rate <= 12 or not np.isfinite(audio).all():
                    rejected['duration_or_invalid'] += 1
                    continue
                if np.sqrt(np.mean(audio ** 2)) < 0.008 or np.mean(np.abs(audio) >= 0.995) > 0.03:
                    rejected['quiet_or_clipped'] += 1
                    continue
                divisor = math.gcd(rate, 16000)
                audio = resample_poly(audio, 16000 // divisor, rate // divisor)
                identity = hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
                relative = Path('audio') / (identity + '.wav')
                (output / 'audio').mkdir(exist_ok=True)
                sf.write(output / relative, audio, 16000, subtype='PCM_16')
                rows.append({**item, 'id': identity, 'audio': str(relative), 'seconds': len(audio) / 16000,
                             'audio_sha256': hashlib.sha256((output / relative).read_bytes()).hexdigest(),
                             'dataset': REPOSITORY, 'revision': REVISION, 'license': 'CC BY-NC-SA 4.0'})
                accepted += 1
    for style in ('happy', 'sad', 'laugh', 'cry'):
        for split in ('train', 'eval'):
            if sum(row['style'] == style and row['split'] == split for row in rows) < 4:
                raise ValueError(f'{style}/{split}有效样本不足，请检查原始标注')
    temporary = output / 'manifest.partial'
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    temporary.replace(output / 'manifest.jsonl')
    archive_hash = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            archive_hash.update(block)
    atomic_json(output / 'provenance.json', {
        'repository': REPOSITORY, 'revision': REVISION, 'license': 'CC BY-NC-SA 4.0',
        'archive_bytes': path.stat().st_size, 'archive_sha256': archive_hash.hexdigest(),
        'counts': dict(Counter(row['style'] + '/' + row['split'] for row in rows)),
        'seconds': sum(row['seconds'] for row in rows), 'heldout_speakers': heldout,
        'split_policy': '说话人与原始台词同时隔离，非语言片段保留原始事件时间范围',
        'rejected': dict(rejected), 'annotation_source': '原始人工标注，质量筛选不替代人工试听',
    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--resources', type=Path, default=Path('data/emotion-resources'))
    parser.add_argument('--output', type=Path, default=Path('data/emotion-acoustic-v1'))
    parser.add_argument('--per-style', type=int, default=24)
    args = parser.parse_args()
    if args.per_style < 4:
        parser.error('每类至少准备四条训练录音')
    archive = args.archive
    if archive is None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
        try:
            archive = Path(hf_hub_download(REPOSITORY, 'SMIIP_NV.zip', repo_type='dataset', revision=REVISION, local_dir=args.resources))
        except GatedRepoError:
            parser.error('请先在SMIIP-NV资料页申请访问并配置已获准账号的HF_TOKEN，或用--archive指定本地原始录音包')
    rows = prepare_archive(archive, args.output, args.per_style)
    print(json.dumps({'records': len(rows), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
