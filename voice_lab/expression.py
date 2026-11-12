"""会话内的表达条件、明确语音命令及保持音高的语速处理。"""
import re
import shutil
import subprocess
import numpy as np

EMOTIONS = {
    'neutral': '自然平静', 'happy': '开心轻快', 'sad': '低落悲伤',
    'angry': '生气', 'fearful': '害怕', 'surprised': '惊讶',
    'disgusted': '厌恶', 'unknown': '无法确定',
}
STYLES = {
    'auto': '', 'neutral': '自然平静地说话', 'happy': '用开心、明亮、有起伏的声音说话',
    'comfort': '用温柔、耐心、安抚的声音回应，先理解对方感受，再给简短具体回应',
    'sad': '用低落、悲伤的语调说话，保持内容清楚',
    'excited': '用兴奋、生动、有感染力的语调说话',
    'angry': '用生气、强烈的语调表演，保持内容清楚',
    'laugh': '发出一小段自然的真实笑声，然后自然回应，避免只念出哈哈两个字或舞台说明',
    'cry': '用带抽泣的哭腔表演一小段，再清楚说话，避免只念出呜呜或舞台说明',
}


def decode_emotion(value):
    label = str(value).replace('<|', '').replace('|>', '').lower()
    aliases = {'fear': 'fearful', 'surprise': 'surprised', 'disgust': 'disgusted'}
    label = aliases.get(label, label)
    return label if label in EMOTIONS else 'unknown'


def reported_emotion(text):
    """用户明确自述的感受作为补充，不把旁人的情绪归到用户身上。"""
    for pattern, emotion in (
        (r'我(?:今天|现在|最近|真的|感觉|觉得|有点|很|特别|非常|好|太|实在|心里|不太)*(?:不开心|不高兴|难过|伤心|低落|失落)', 'sad'),
        (r'我(?:今天|现在|最近|真的|感觉|觉得|有点|很|特别|非常|好|太|实在|心里)*(?:害怕|恐惧|恐慌)', 'fearful'),
        (r'我(?:今天|现在|最近|真的|感觉|觉得|有点|很|特别|非常|好|太|实在|心里)*(?:生气|愤怒)', 'angry'),
        (r'我(?:今天|现在|最近|真的|感觉|觉得|有点|很|特别|非常|好|太|实在|心里)*(?:开心|高兴|兴奋)', 'happy'),
    ):
        if re.search(r'(?:^|[，。！？,!?])\s*(?:其实|唉|哎)?' + pattern, text):
            return emotion
    return None


def voice_commands(text):
    """只把直接、明确的操作请求转换为会话控制，否定和引用不执行。"""
    value = re.sub(r'[\s，。！？、,.!?]', '', text)
    if re.search(r'他说|她说|有人说|如果|假如|为什么|怎么实现', value):
        return {}
    changes = {}
    withdrawal = re.search(r'(?:(?:不要|别|停止|不想再|取消)(?:再|继续)?(?:模仿|克隆|使用|用)|不用)(?:我(?:自己)?的(?:声音|音色)|我说话|我(?=了|啦|吧|$))', value)
    if withdrawal:
        changes['voice'] = 'original'
        value = value[:withdrawal.start()] + value[withdrawal.end():]
    if re.search(r'不要|(?<!特)别|(?<!能)不能|(?<!可)不可以|不想|不需要|不许|不准|请勿|禁止', value):
        return changes
    voice_request = re.match(r'^(?:那|现在|接下来|以后)?(?:请|麻烦)?(?:你|能不能|能否|可不可以|可以|帮我|给我|模仿|克隆|换成|换到|使用|用|恢复|我想让你|我想请你|我希望你)', value)
    if voice_request and re.search(r'(?:模仿|克隆|换成|换到|用|使用)(?:一下|成)?(?:我的|我自己的)(?:声音|音色)|模仿我说话|(?:能不能|可以|请你|你能)模仿我(?:吗|一下)?$', value):
        changes['voice'] = 'self'
    elif voice_request and re.search(r'(?:恢复|使用|用)(?:模型)?默认的?(?:声音|音色)', value):
        changes['voice'] = 'default'
    elif voice_request and re.search(r'恢复(?:原来|原始|之前|原)的?(?:声音|音色)', value):
        changes['voice'] = 'original'
    if re.search(r'(?:说|讲|语速|速度)(?:得|话|再|稍微)?快(?:一点|一些|点)|加快(?:语速|速度)', value):
        changes['rate_delta'] = 0.15
    elif re.search(r'(?:说|讲|语速|速度)(?:得|话|再|稍微)?慢(?:一点|一些|点)|放慢(?:语速|速度)', value):
        changes['rate_delta'] = -0.15
    elif re.search(r'恢复正常语速|正常速度说', value):
        changes['rate'] = 1.0
    # 情绪陈述用于感知，只有表达请求才切换助手的表演方式。
    expression_request = voice_request or re.match(r'^(?:请(?!问)|麻烦|来一|说|讲|开心|高兴|悲伤|难过|温柔|兴奋)', value)
    if expression_request or re.fullmatch(r'(?:笑|哭|大笑|抽泣)(?:一下|一段|一个|三声|三秒|3秒|给我听)', value):
        for pattern, style in (
            (r'哭(?:一下|一段|一个|给我|出来)|哭腔|哭着说|抽泣|哭声', 'cry'),
            (r'笑(?:一下|一段|一个|给我|出来|三声|几声)|笑着说|大笑|笑声', 'laugh'),
            (r'(?:开心|高兴|欢快).*(?:说|讲)|(?:说|讲).*开心|(?:开心|高兴|欢快)(?:一)?点', 'happy'),
            (r'(?:兴奋|激动).*(?:说|讲)|(?:兴奋|激动)(?:一)?点', 'excited'),
            (r'(?:温柔|安慰).*(?:说|讲)|安慰我|温柔(?:一)?点', 'comfort'),
            (r'(?:悲伤|难过|伤心).*(?:说|讲)|(?:悲伤|难过|伤心)(?:一)?点', 'sad'),
            (r'恢复正常(?:语气|情绪)|自然地说', 'auto'),
        ):
            if re.search(pattern, value):
                changes['style'] = style
                if style in ('laugh', 'cry'):
                    changes['style_once'] = True
                    speaking_style = re.search(r'哭腔|哭着说|笑着说|(?:声音|语气).*(?:说|讲)', value)
                    if not speaking_style:
                        changes['vocal_action'] = style
                        changes['continue_after_vocal'] = bool(re.search(r'然后|接着|并且|另外|还有|顺便|同时|(?:再|并).*(?:说|讲|回答|聊)', value))
                break
    return changes


def expression_context(style, emotion, event='speech', emotion_source='audio_model'):
    if style not in STYLES or emotion not in EMOTIONS:
        raise ValueError('表达条件无效')
    instruction = STYLES[style]
    if style == 'auto':
        instruction = '根据用户音频和上下文自然回应'
        if emotion in ('sad', 'fearful', 'angry') or event == 'crying':
            instruction = STYLES['comfort'] + '，避免机械模仿用户的负面情绪'
        elif (emotion == 'happy' and emotion_source == 'self_report') or event == 'laughter':
            instruction = '用轻快友善的语气回应，结合用户实际话题'
    return ('【会话表达设置】用户声音情绪初步判断为' + EMOTIONS[emotion] +
            '，仅作不确定线索，结合原始音频与上下文判断。' + instruction +
            '。直接给出实际内容，避免只承诺稍后开始。控制说明不要读出来。')


def validate_rate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0.75 <= value <= 1.5:
        raise ValueError('语速范围为0.75到1.5倍')
    return float(value)


def change_tempo(audio, rate, sample_rate=24000):
    """FFmpeg atempo 改变时长并保持音高，整轮处理保证音频块连续。"""
    rate = validate_rate(rate)
    audio = np.asarray(audio, dtype='<f4')
    if rate == 1.0 or not len(audio):
        return audio
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise RuntimeError('语速处理需要安装ffmpeg')
    result = subprocess.run([
        executable, '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-f', 'f32le', '-ar', str(sample_rate), '-ac', '1', '-i', 'pipe:0',
        '-af', f'atempo={rate}', '-f', 'f32le', '-ar', str(sample_rate), '-ac', '1', 'pipe:1',
    ], input=audio.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
    output = np.frombuffer(result.stdout, dtype='<f4').copy()
    if not len(output) or not np.isfinite(output).all():
        raise RuntimeError('语速处理未产生有效音频')
    return output


class UserAudioHistory:
    """仅保留本会话近期用户语音，捕获样本不进入对外JSON或音色库。"""

    def __init__(self):
        self.audio = np.empty(0, dtype=np.float32)

    def add(self, audio):
        self.audio = np.concatenate((self.audio, np.asarray(audio, dtype=np.float32)))[-16000 * 12:]

    def reference(self, minimum_seconds=3):
        if len(self.audio) < 16000 * minimum_seconds:
            raise ValueError(f'清晰语音不足{minimum_seconds:g}秒，请再说一句完整的话后重试')
        audio = self.audio[-16000 * 8:].copy()
        if not np.isfinite(audio).all() or np.sqrt(np.mean(audio ** 2)) < 0.008:
            raise ValueError('录音声音太小，请靠近麦克风再说一句话')
        if np.mean(np.abs(audio) >= 0.995) > 0.02:
            raise ValueError('录音削波过多，请降低麦克风音量后重试')
        return audio
