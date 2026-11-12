"""原音频驱动的全双工运行时，所有发声共用原生隐藏状态与声码器。"""
from copy import deepcopy

import numpy as np

from .expression import STYLES, UserAudioHistory, validate_rate
from .model import InferenceInterrupted, VoiceRuntime
from .prompts import NATIVE_VOICE_PROMPT

NATIVE_STYLES = {
    **STYLES,
    'angry': '用有力度的愤怒语气表达，保持文字清晰和当前音色',
    'surprised': '用惊讶的语调和重音表达，保持当前音色',
    'fearful': '用紧张害怕的语调表达，保持当前音色',
    'disgusted': '用厌恶的语调表达，保持当前音色',
    'laughing_speech': '带着自然笑意和笑腔说出实际内容，保持当前音色',
    'crying_speech': '用带抽泣的悲伤哭腔说出实际内容，保持当前音色',
}


class NativeVoiceRuntime(VoiceRuntime):
    native_e2e = True

    def __init__(self, config):
        config = deepcopy(config)
        for key in ('acoustic_adapter', 'vocoder_adapter'):
            if config['model'].get(key):
                raise ValueError('原生端到端模式需要联合训练权重，请移除旧版独立情绪适配器配置')
        config['model']['system_prompt'] = (
            config['model'].get('system_prompt', '') + '\n' + NATIVE_VOICE_PROMPT
        )
        # 音色参考进入主模型音频上下文，同时供同一原生声码器建立声线条件。
        config['model']['reference_audio_in_context'] = True
        config['model'].setdefault('conversation_floor', True)
        super().__init__(config)

    def start(self, voice_override=None):
        result = super().start(voice_override)
        self.input_audio = UserAudioHistory()
        self.latest_user_reference = None
        self.model_hints = []
        self.voice_decision_generation = 0
        self.current_voice = 'original'
        self.voice_input_generation = 0
        self.voice_input_units = []
        self.playback_active = False
        self.playback_text = None
        self.overlap_input = False
        self.yielded_generation = 0
        self.backchannel_generation = 0
        self.output_text = ''
        self.last_output_ended = True
        return {**result, 'architecture': 'native_e2e', 'capabilities_verified': False}

    def configure(self, settings):
        """网页显式设置转为主模型指令，参考声音更新属于本会话状态。"""
        settings = dict(settings)
        hints = []
        if 'style' in settings:
            style = settings.pop('style')
            if style not in NATIVE_STYLES:
                raise ValueError('不支持的表达方式')
            hints.append(NATIVE_STYLES[style] or '请根据对话自然表达情绪。')
        if 'rate' in settings:
            rate = validate_rate(settings.pop('rate'))
            hints.append(f'请在实际发声中使用正常语速的{rate:g}倍节奏，保持当前声线。')
        for key in ('style_once', 'vocal_action'):
            settings.pop(key, None)
        if settings.get('voice') == 'self':
            settings['voice_audio'] = self._clone_reference(settings.get('voice_audio'))
        if settings:
            allowed = {'voice', 'voice_audio', 'voice_path', 'voice_id', 'voice_error'}
            if set(settings) - allowed:
                raise ValueError('原生模式只接受模型指令与参考声音设置')
            super().configure(settings)
            if 'voice' in settings or 'voice_path' in settings:
                floor = getattr(self, 'floor', None)
                if floor is not None:
                    self.voice_decision_generation = floor.generation
        self.model_hints.extend(hints)

    def _clone_reference(self, current=None):
        if current is not None and len(current) >= 16000:
            reference = UserAudioHistory()
            reference.add(current)
            return reference.reference(minimum_seconds=1)
        latest = getattr(self, 'latest_user_reference', None)
        if latest is not None and len(latest) >= 16000:
            reference = UserAudioHistory()
            reference.add(latest)
            return reference.reference(minimum_seconds=1)
        return self.input_audio.reference(minimum_seconds=1)

    def _voice_action(self, reference_audio=None):
        floor = self.floor
        if (floor is None or floor.active or not self.duplex.current_turn_ended or
                floor.generation <= self.voice_decision_generation or self.parser.buffer):
            return []
        self.voice_decision_generation = floor.generation
        from .voice_action import decide_voice_action
        action = decide_voice_action(self.duplex.decoder, self.voice_input_units, current_voice=self.current_voice)
        if action == 'cancel':
            action = 'original' if self.current_voice == 'self' else None
        if action is None:
            return []
        events = [{'type': 'voice_action_selected', 'voice': action}]
        try:
            settings = {'voice': action}
            if action == 'self' and reference_audio is not None:
                settings['voice_audio'] = reference_audio
            self.configure(settings)
        except ValueError as exc:
            super().configure({'voice_error': 'reference_unusable'})
            events.append({'type': 'settings_error', 'message': str(exc)})
        events.extend(self._apply_settings())
        return events

    def _switch_voice(self, settings):
        event = super()._switch_voice(settings)
        self.current_voice = settings.get('voice') if settings.get('voice') in ('self', 'original', 'default') else 'selected'
        return event

    def set_playback(self, active, text=None):
        self.playback_active = active
        self.playback_text = text

    def _overlap_action(self):
        floor = self.floor
        if (floor is None or not self.overlap_input or not self.voice_input_units or
                floor.generation <= self.yielded_generation or
                floor.generation <= self.backchannel_generation):
            return []
        from .interaction_action import decide_overlap
        context = self.playback_text if self.playback_text is not None else self.output_text
        decision = decide_overlap(self.duplex.decoder, self.voice_input_units, context)
        if decision == 0 and not floor.active:
            floor.consume_backchannel()
            self.backchannel_generation = floor.generation
            self.voice_decision_generation = max(self.voice_decision_generation, floor.generation)
            return [{'type': 'backchannel', 'generation': floor.generation}]
        if decision != 1:
            return []
        d = self.duplex
        self.yielded_generation = floor.generation
        d.current_turn_ended = True
        d.tts_past_key_values = None
        d.tts_text_start_pos = 0
        d.tts_current_turn_start_time = None
        d._reset_token2wav_for_new_turn()
        self.parser.reset()
        self.pending_blocks = []
        floor.abandon_output()
        self.last_output_ended = True
        self.model_hints.append('【会话状态】用户已插话，上一段播报已停止。请倾听并根据最新用户语音回应。')
        return [{'type': 'barge_in', 'generation': floor.generation}]

    def _prefill(self, audio, context, capture):
        """复用本轮原始音频嵌入，字幕、助手发言与后台文字留在各自支路。"""
        import torch
        decoder = self.duplex.decoder
        original = decoder.feed
        captured = []
        def feed(embeddings, *args, **kwargs):
            captured.append(embeddings.detach().clone())
            return original(embeddings, *args, **kwargs)
        try:
            if capture:
                decoder.feed = feed
            result = self.duplex.streaming_prefill(audio_waveform=audio, text_list=[context] if context else None)
        finally:
            decoder.feed = original
        if capture and result['success']:
            if context:
                captured.pop()
            self.voice_input_units.append(torch.cat(captured))
            self.voice_input_units = self.voice_input_units[-32:]
        return result

    def _generate_chunk(self):
        d = self.duplex
        forbidden = d.decoder.forbidden_token_ids
        try:
            # 原生实现会把句中listen改成TTS BOS；直接屏蔽该候选可继续生成正文。
            if not d.current_turn_ended:
                d.decoder.forbidden_token_ids = [*forbidden, d.listen_token_id]
            return d.streaming_generate(
                max_new_speak_tokens_per_chunk=self.config['model']['max_speak_tokens'],
                decode_mode=self.config['model'].get('decode_mode', 'sampling'),
            )
        finally:
            d.decoder.forbidden_token_ids = forbidden

    def end(self):
        super().end()
        self.voice_input_units = []
        self.input_audio = UserAudioHistory()
        self.latest_user_reference = None
        self.model_hints = []
        self.playback_text = None
        self.output_text = ''

    def resume_after_interrupt(self):
        super().resume_after_interrupt()
        self.overlap_input = False
        if self.floor is not None:
            self.yielded_generation = max(self.yielded_generation, self.floor.generation)

    def step(self, audio, context=None):
        import torch

        with torch.inference_mode():
            updates = self._apply_settings()
            hints, self.model_hints = self.model_hints, []
            notice, self.voice_notice = self.voice_notice, None
            has_notice = bool(notice or hints or self.voice_error_pending)
            self.voice_error_pending = False
            context = '\n'.join(value for value in (context, notice, *hints) if value) or None
            d = self.duplex
            floor = self.floor
            capture = False
            segments = []
            if floor is not None:
                was_active = floor.active
                segments = self.activity.feed(audio, floor) or []
                if floor.generation != self.voice_input_generation:
                    self.voice_input_units = []
                    self.voice_input_generation = floor.generation
                    self.overlap_input = self.playback_active or not d.current_turn_ended
                    capture = True
                capture = capture or was_active or floor.active
                if capture:
                    self.overlap_input = self.overlap_input or self.playback_active or not d.current_turn_ended
                if has_notice or (context and context.startswith('【后台结果】')):
                    floor.notify()
            else:
                self.input_audio.add(audio)
            d.total_ids = d.total_ids[-8192:]
            d.res_ids = d.res_ids[-8192:]
            d.prefill_schema_tokens = d.prefill_schema_tokens[-32:]
            result = self._prefill(audio, context, capture)
            if not result['success']:
                if d.is_break_set() or d.is_session_stop_set():
                    raise InferenceInterrupted('会话已结束或打断')
                raise RuntimeError(f"音频预填充失败 {result.get('reason', '')}")
            if capture:
                updates.extend(self._overlap_action())
            if segments and (not self.overlap_input or self.yielded_generation == floor.generation):
                for segment in segments:
                    self.input_audio.add(segment)
                self.latest_user_reference = segments[-1]
            if any(event['type'] == 'barge_in' for event in updates):
                logits = torch.full_like(d.pending_logits, float('-inf'))
                logits[..., d.listen_token_id] = 0
                d.pending_logits = logits
                d.streaming_generate(max_new_speak_tokens_per_chunk=2)
                d.total_hidden.clear()
                return {'kind': 'listen', 'events': updates}
            updates.extend(self._voice_action(reference_audio=segments[-1] if segments else None))
            if self.voice_notice:
                # 将执行结果放入当前音频单元，起答依照实际缓存更新结果生成。
                tokens = d.tokenizer.encode(self.voice_notice, add_special_tokens=False)
                d.pending_logits, _ = d.decoder.feed(d.decoder.embed_tokens(tokens), return_logits=True)
                d._current_unit_prefill_tokens.extend(tokens)
                self.voice_notice = None
                self.voice_error_pending = False
                has_notice = True
                if floor is not None:
                    floor.notify()
            if floor is not None and d.current_turn_ended and not self.parser.buffer and d.pending_logits is not None:
                if has_notice or (context and context.startswith('【后台结果】')) or floor.need_wake():
                    logits = torch.full_like(d.pending_logits, float('-inf'))
                    logits[..., d.speak_token_id] = 0
                    d.pending_logits = logits
                elif not floor.can_start:
                    logits = torch.full_like(d.pending_logits, float('-inf'))
                    logits[..., d.listen_token_id] = 0
                    d.pending_logits = logits
            block = self._generate_chunk()
            if block['is_listen']:
                d.total_hidden.clear()
                return {'kind': 'listen', 'events': updates}
            if floor is not None:
                floor.begin_output()
            decision = self.parser.push(block['text'], end=block['end_of_turn'])
            hidden = list(d.total_hidden[-1])
            d.total_hidden.clear()
            self.pending_blocks.append((block, hidden))
            if decision.kind == 'pending':
                return {'kind': 'listen', 'events': updates}
            blocks, self.pending_blocks = self.pending_blocks, []
            if decision.kind != 'speech':
                if floor is not None:
                    if decision.kind in ('delegate', 'cancel'):
                        floor.delegated()
                    else:
                        floor.abandon_output()
                d.tts_past_key_values = None
                d.tts_text_start_pos = 0
                d.current_turn_ended = True
                d._reset_token2wav_for_new_turn()
                if decision.kind == 'voice':
                    try:
                        self.configure({'voice': decision.text})
                    except ValueError as exc:
                        super().configure({'voice_error': 'reference_unusable'})
                        updates.append({'type': 'settings_error', 'message': str(exc)})
                    return {'kind': 'listen', 'events': updates}
                return {'kind': decision.kind, 'text': decision.text, 'events': updates}
            rendered = []
            for item, states in blocks:
                self.output_text = item['text'] if self.last_output_ended else (self.output_text + item['text'])[-1000:]
                self.last_output_ended = item['end_of_turn']
                rendered.append({'text': item['text'],
                                 'audio': self._render(states, item['end_of_turn'], item['current_time']),
                                 'end': item['end_of_turn']})
                if floor is not None:
                    # 原生模型可以生成真实的非语言发声，文本层不裁剪其笑声。
                    floor.complete(item['text'], item['end_of_turn'], allow_laughter=True)
            return {'kind': 'speech', 'blocks': rendered, 'events': updates}

    def _render(self, hidden, end, current_time, vocal_action=False):
        """所有情绪沿用同一组权重、完整语义条件与同一声线缓存。"""
        d = self.duplex
        condition = d._convert_results_to_tts_input(hidden)
        first = d.tts_text_start_pos == 0
        tokens, cache = self.model.tts.generate_chunk(
            inputs_embeds=condition, temperature=d.tts_temperature,
            repetition_penalty=d.tts_repetition_penalty, eos_token=d.tts_eos_token,
            force_no_stop=False, max_new_token=512 if end else 26,
            min_new_tokens=0 if end or first else 26,
            past_key_values=d.tts_past_key_values,
            logits_processors=d.tts_logits_processors, text_start_pos=d.tts_text_start_pos,
        )
        if end:
            d.tts_text_start_pos = 0
            d.tts_past_key_values = None
            d.tts_current_turn_start_time = None
        else:
            d.tts_past_key_values = cache
            d.tts_text_start_pos += condition.shape[1] + tokens.shape[1]
        waveform = d._generate_waveform_from_tokens(tokens, str(self.voice_path), end, force_flush=first)
        if end:
            d._reset_token2wav_for_new_turn()
        if waveform is None:
            return np.zeros(0, dtype=np.float32)
        if hasattr(waveform, 'detach'):
            waveform = waveform.detach().float().cpu().numpy()
        return np.asarray(waveform, dtype=np.float32).reshape(-1)
