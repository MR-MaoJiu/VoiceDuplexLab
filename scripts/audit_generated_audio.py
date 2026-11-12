"""分段转写实际生成的连续对话，独立检查波形内容与关键结果。"""
import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from voice_lab.asr import SpeechTranscriber, TranscriptSession
from voice_lab.config import load_config
from voice_lab.dialogue_data import normalize_audio_text, spoken_numbers
from voice_lab.download import atomic_json
from voice_lab.model import read_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--required-number', action='append', default=[])
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    audio = read_audio(args.report.parent / 'continuous-three-topics.wav')
    engine = SpeechTranscriber(load_config(args.config)['asr'])
    session = TranscriptSession(engine)
    segments = []
    try:
        for offset in range(0, len(audio), 16000):
            segments.extend(session.feed(audio[offset:offset + 16000]))
        segments.extend(session.feed(np.empty(0, dtype=np.float32), final=True))
    finally:
        engine.close()
    expected = ''.join(row['text'] for row in report['continuous_topics'])
    actual = ''.join(row['text'] for row in segments)
    similarity = SequenceMatcher(None, normalize_audio_text(expected), normalize_audio_text(actual)).ratio()
    numbers = spoken_numbers(actual)
    number_check = all(normalize_audio_text(value) in numbers for value in args.required_number)
    value = {'passed': similarity >= 0.8 and number_check, 'expected': expected, 'transcript': actual,
             'audio_text_similarity': similarity, 'required_numbers': args.required_number,
             'spoken_numbers': numbers, 'required_numbers_present': number_check,
             'segments': [{'start': row['start'], 'end': row['end'], 'text': row['text']} for row in segments]}
    atomic_json(args.report.with_name('continuous-audio-audit.json'), value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    raise SystemExit(0 if value['passed'] else 2)


if __name__ == '__main__':
    main()
