"""独立CPU支路提供文字、声音情绪和明确的会话设置命令。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import urllib.request
import numpy as np
from .config import local_path
from .download import atomic_json, retry

MODEL_REPO = 'csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17'
MODEL_REVISION = '2365baeacb507f821a0c8120fcee3d484dba7a07'
CHECKSUMS = {
    'model.int8.onnx': 'c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51',
    'tokens.txt': 'f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc',
    'silero_vad.onnx': '9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6',
}


def download_models(directory):
    from huggingface_hub import snapshot_download
    directory = local_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    retry(lambda: snapshot_download(
        MODEL_REPO, revision=MODEL_REVISION, local_dir=directory,
        allow_patterns=['model.int8.onnx', 'tokens.txt', 'LICENSE', 'README.md'],
    ))
    target = directory / 'silero_vad.onnx'
    if not target.is_file():
        def fetch_vad():
            temporary = target.with_suffix('.partial')
            url = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx'
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            temporary.replace(target)
        retry(fetch_vad)
    files = []
    for name, expected in CHECKSUMS.items():
        path = directory / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'转写模型校验失败 {name}')
        files.append({'file': name, 'bytes': path.stat().st_size, 'sha256': actual})
    atomic_json(directory / 'resources.json', {
        'repository': MODEL_REPO, 'revision': MODEL_REVISION, 'files': files,
    })
    print('CPU转写模型与VAD校验完成', flush=True)


class SpeechTranscriber:
    def __init__(self, options):
        import sherpa_onnx
        self.options = options
        self.directory = local_path(options.get('model_dir', 'models/SenseVoice-int8'))
        for name in CHECKSUMS:
            if not (self.directory / name).is_file():
                raise FileNotFoundError(f'缺少 {name}，请先运行 quickstart.sh setup-asr')
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.directory / 'model.int8.onnx'),
            tokens=str(self.directory / 'tokens.txt'),
            num_threads=options.get('num_threads', 2),
            provider='cpu', language='zh', use_itn=True,
        )
        # 共享解码器的调用在同一线程串行执行，音频与VAD状态仍按会话隔离。
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='voice-transcription')

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def recognize(self, audio, rate=16000):
        return self.analyze(audio, rate)['text']

    def analyze(self, audio, rate=16000):
        from .expression import decode_emotion
        stream = self.recognizer.create_stream()
        stream.accept_waveform(rate, audio)
        self.recognizer.decode_stream(stream)
        result = stream.result
        event = str(result.event).lower()
        return {
            'text': result.text.strip(),
            'emotion': decode_emotion(result.emotion),
            'emotion_confidence': None,
            'vocal_event': 'laughter' if 'laughter' in event else 'crying' if 'cry' in event else 'speech',
        }


class TranscriptSession:
    def __init__(self, engine):
        import sherpa_onnx
        self.engine = engine
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(engine.directory / 'silero_vad.onnx')
        config.silero_vad.min_silence_duration = engine.options.get('silence_seconds', 0.6)
        config.silero_vad.min_speech_duration = 0.25
        config.silero_vad.max_speech_duration = 15
        config.sample_rate = 16000
        config.num_threads = 1
        self.window = config.silero_vad.window_size
        self.vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        self.pending = np.empty(0, dtype=np.float32)
        self.sequence = 0

    def feed(self, audio, final=False):
        self.pending = np.concatenate((self.pending, audio))
        consumed = len(self.pending) // self.window * self.window
        for offset in range(0, consumed, self.window):
            self.vad.accept_waveform(self.pending[offset:offset + self.window])
        self.pending = self.pending[consumed:]
        if final:
            if len(self.pending):
                self.vad.accept_waveform(np.pad(self.pending, (0, self.window - len(self.pending))))
                self.pending = np.empty(0, dtype=np.float32)
            self.vad.flush()
        events = []
        while not self.vad.empty():
            segment = self.vad.front
            start = segment.start / 16000
            samples = np.asarray(segment.samples, dtype=np.float32).copy()
            self.vad.pop()
            analysis = self.engine.analyze(samples)
            text = analysis['text']
            from .expression import reported_emotion
            stated = reported_emotion(text)
            analysis['dialogue_emotion'] = stated or analysis['emotion']
            analysis['emotion_source'] = 'self_report' if stated else 'audio_model'
            if text or analysis['vocal_event'] in ('laughter', 'crying'):
                self.sequence += 1
                events.append({
                    'type': 'user_transcript', 'id': self.sequence, 'text': text,
                    'start': start, 'end': start + len(samples) / 16000, 'final': True,
                    **analysis, '_audio': samples,
                })
        return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='models/SenseVoice-int8')
    args = parser.parse_args()
    download_models(args.output)


if __name__ == '__main__':
    main()
