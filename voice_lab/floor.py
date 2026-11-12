"""原生语音推理的发言权限，避免参考音频或静音触发自言自语。"""
import re
import numpy as np
from .config import local_path

FLOOR_VERSION = 7


def is_preface(text):
    value = re.sub(r'\W', '', text)
    if value in {'好的', '好', '当然可以', '没问题', '你好', '我查一下', '我来查一下', '稍等', '请稍等'}:
        return True
    lookup = re.fullmatch(r'(?:好的|好|没问题|当然可以|可以的)?我(?:来|先|去|正在)?(?:帮[你您]|给[你您]|为[你您])?(?:查一下|查下|查一查|查查|查询|核对|搜索|检索).{0,18}', value)
    if lookup and ':' not in text and '：' not in text:
        return True
    # 只识别整句动作承诺，含实际歌词或故事正文的回答不能因此被丢弃。
    action = re.fullmatch(r'(?:好的|好|没问题|当然可以|可以的)?我(?:现在|这就|马上)?(?:开始|来)(?:给[你您]|为[你您])?(?:唱(?:歌|一首歌|首歌|一段)?|讲(?:故事|一个故事|个故事)|读(?:一段|给[你您]听)?)(?:吧|了)?', value)
    if action:
        return True
    vocal_promise = re.fullmatch(r'(?:好的|好|没问题|当然可以)?我(?:现在|这就|马上|先)?(?:开始|来)?(?:给你|为你)?(?:开怀)?(?:大笑|笑|哭|抽泣)(?:三声|几声|一下|三秒|3秒|一小段)?', value)
    if vocal_promise:
        return True
    preface = re.fullmatch(r'(?:好的|好|没问题|当然可以|可以的)?我来(?:帮[你您]|给[你您])?(?:用.{0,6})?(?:写|讲|描述|解释|介绍|查|计算|整理|分析|安排|想).{0,18}', value)
    return bool(preface and ':' not in text and '：' not in text)


class ConversationFloor:
    def __init__(self):
        self.generation = 0
        self.answered = 0
        self.active = False
        self.notification = False
        self.text = ''
        self.output_generation = None
        self.idle_frames = 0
        self.wake_generation = -1
        self.wake_count = 0
        self.generation_starts = []

    def need_wake(self):
        if self.wake_generation != self.generation:
            self.wake_generation = self.generation
            self.wake_count = 0
            self.idle_frames = 0
        if not self.can_start or self.notification or self.output_generation is not None:
            self.idle_frames = 0
            return False
        self.idle_frames += 1
        # 一个用户发言最多唤醒两次，兼顾过渡语与独立工具控制单元。
        if self.idle_frames >= 2 and self.wake_count < 2:
            self.idle_frames = 0
            self.wake_count += 1
            return True
        return False

    def begin_output(self):
        self.idle_frames = 0
        if self.output_generation is None:
            self.output_generation = self.generation

    def observe_activity(self, active, at=None):
        if active and not self.active:
            self.generation += 1
            if at is not None:
                self.generation_starts = (self.generation_starts + [(at, self.generation)])[-64:]
        self.active = active

    def generation_at(self, audio_end=None):
        if audio_end is None:
            return self.generation
        return next((generation for start, generation in reversed(self.generation_starts) if start <= audio_end), 0)

    @property
    def pending(self):
        return self.generation > self.answered or self.notification

    @property
    def can_start(self):
        # 等待用户短暂停顿再开启新回答，仍持续接收音频并支持说话中打断。
        return self.notification or (self.generation > self.answered and not self.active)

    def notify(self):
        self.notification = True

    def consume_backchannel(self):
        self.answered = max(self.answered, self.generation)

    def complete(self, text, end, allow_laughter=False):
        self.begin_output()
        self.text += text
        if not end:
            return
        response_generation = self.output_generation
        self.output_generation = None
        original = self.text
        value = re.sub(r'\W', '', original)
        self.text = ''
        # 短语气词和承诺不消耗用户的问题，模型仍须给出实际内容。
        intentional_laughter = allow_laughter and re.fullmatch(r'[哈呵嘿]+', value)
        if not value or (not intentional_laughter and re.fullmatch(r'[嗯啊哦呃额唉噢哎诶呀唔哈]+', value)):
            return
        if is_preface(original):
            return
        if self.notification:
            self.notification = False
        else:
            # 回答只消费开始生成时对应的发言，不能吞掉生成期间的新输入。
            self.answered = max(self.answered, response_generation)

    def delegated(self):
        generation = self.output_generation if self.output_generation is not None else self.generation
        self.answered = max(self.answered, generation)
        self.output_generation = None
        self.notification = False
        self.text = ''

    def abandon_output(self):
        self.output_generation = None
        self.text = ''

    def interrupt(self):
        generation = self.output_generation if self.output_generation is not None else self.generation
        self.answered = max(self.answered, generation)
        self.output_generation = None
        self.notification = False
        self.text = ''


class SpeechActivity:
    def __init__(self, options):
        import sherpa_onnx
        path = local_path(options.get('model_dir', 'models/SenseVoice-int8')) / 'silero_vad.onnx'
        if not path.is_file():
            raise FileNotFoundError('缺少发言边界检测模型，请运行 ./quickstart.sh setup-asr')
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(path)
        config.silero_vad.min_speech_duration = 0.25
        config.silero_vad.min_silence_duration = 0.6
        config.silero_vad.max_speech_duration = 15
        config.sample_rate = 16000
        config.num_threads = 1
        self.window = config.silero_vad.window_size
        self.vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        self.pending = np.empty(0, dtype=np.float32)
        self.samples = 0

    def feed(self, audio, floor):
        segments = []
        self.pending = np.concatenate((self.pending, audio))
        consumed = len(self.pending) // self.window * self.window
        for offset in range(0, consumed, self.window):
            self.vad.accept_waveform(self.pending[offset:offset + self.window])
            self.samples += self.window
            floor.observe_activity(self.vad.is_speech_detected(), at=self.samples / 16000)
            while not self.vad.empty():
                segments.append(np.asarray(self.vad.front.samples, dtype=np.float32).copy())
                self.vad.pop()
        self.pending = self.pending[consumed:]
        return segments
