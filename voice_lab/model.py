import hashlib
from contextlib import nullcontext
from pathlib import Path
import tempfile
import numpy as np
from .config import local_path
from .control import ControlParser
from .prompts import VOICE_PROMPT
from .expression import STYLES, EMOTIONS, change_tempo, expression_context, validate_rate


class InferenceInterrupted(RuntimeError):
    """用户打断或结束会话导致的正常推理中止。"""


class VoiceStateError(RuntimeError):
    """新旧音色均无法装入时停止会话，避免继续使用损坏缓存。"""


def read_audio(path):
    import librosa

    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("参考音频为空或含有无效值")
    return audio.astype(np.float32)


def decoder_reference(path, directory, maximum_seconds=8):
    """保留原录音，为固定缓存的原生声码器选取短而清晰的参考窗口。"""
    import soundfile as sf
    path = Path(path)
    audio = read_audio(path)
    limit = int(maximum_seconds * 16000)
    info = {'original_seconds': len(audio) / 16000, 'seconds': len(audio) / 16000, 'clipped': False}
    if len(audio) <= limit:
        return path, info
    energy = np.concatenate(([0.0], np.cumsum(audio.astype(np.float64) ** 2)))
    starts = np.arange(0, len(audio) - limit + 1, 1600)
    start = int(starts[np.argmax(energy[starts + limit] - energy[starts])])
    target = Path(directory) / 'reference.wav'
    sf.write(target, audio[start:start + limit], 16000, subtype='PCM_16')
    info.update(seconds=maximum_seconds, clipped=True, start_seconds=start / 16000)
    return target, info


def load_base(config, with_adapter=True):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoProcessor

    item = config["model"]
    path = local_path(item["path"])
    if not path.is_dir():
        path = Path(snapshot_download(item["path"], revision=item["revision"]))
    if not torch.cuda.is_available() and item["device"].startswith("cuda"):
        raise RuntimeError(
            "未发现 CUDA GPU。此推理与训练配置需要 Linux NVIDIA 服务器。"
        )
    if item["device"].startswith("cuda"):
        torch.cuda.set_device(item["device"])
    dtype = getattr(torch, item["dtype"])
    model = (
        AutoModel.from_pretrained(
            str(path),
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            init_vision=False,
            init_audio=True,
            init_tts=True,
        )
        .to(item["device"])
        .eval()
    )
    model.processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
    if with_adapter and item.get("adapter"):
        import json
        from peft import PeftModel

        adapter = local_path(item["adapter"])
        metadata = json.loads((adapter / "voice_lab_metadata.json").read_text())
        if metadata["base_revision"] != item["revision"]:
            raise ValueError("适配器与基座版本不匹配")
        model.llm = PeftModel.from_pretrained(
            model.llm, str(adapter)
        ).merge_and_unload()
    model._voice_lab_base_path = str(path)
    model._voice_lab_acoustic_embedding = None
    if item.get('acoustic_adapter'):
        from .acoustic import load_acoustic_adapter
        model._voice_lab_acoustic_embedding = load_acoustic_adapter(model, config, local_path(item['acoustic_adapter']))
    if item.get('native_adapter'):
        from .native_weights import load_native_weights
        load_native_weights(model, config, local_path(item['native_adapter']))
    return model


def resolve_voice(config, model):
    value = config["model"].get("voice")
    path = (
        local_path(value)
        if value
        else Path(model._voice_lab_base_path) / "assets/HT_ref_audio.wav"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少参考音频，请在 model.voice 配置 WAV 路径。当前路径为 {path}"
        )
    return path


def prepare_duplex(model, config, voice_path, generate_audio=False, prompt=None):
    model.reset_session(reset_token2wav_cache=True)
    duplex = model.as_duplex(
        generate_audio=generate_audio,
        sliding_window_mode="basic",
        basic_window_high_tokens=config["model"]["context_high"],
        basic_window_low_tokens=config["model"]["context_low"],
    )
    if config['model'].get('native_adapter'):
        from .native_flow import load_flow_weights
        load_flow_weights(model, config)
    duplex.prepare(
        prefix_system_prompt=prompt or config["model"].get("system_prompt") or VOICE_PROMPT,
        ref_audio=read_audio(voice_path) if config["model"].get("reference_audio_in_context", True) else None,
        prompt_wav_path=str(voice_path),
    )
    return duplex


class VoiceRuntime:
    """每个 GPU 实例只运行一个会话，所有模型操作由同一执行线程串行调用。"""

    def __init__(self, config):
        self.config = config
        self.model = load_base(config)
        self.duplex = None
        self.parser = ControlParser()
        self.pending_blocks = []
        self.reference_cache = None

    def start(self, voice_override=None):
        import torch

        with torch.inference_mode():
            if self.reference_cache is not None:
                self.reference_cache.cleanup()
            self.reference_cache = tempfile.TemporaryDirectory(prefix='voice-lab-reference-')
            original_voice = voice_override or resolve_voice(self.config, self.model)
            self.original_voice = original_voice
            self.voice_path, self.reference_info = decoder_reference(original_voice, self.reference_cache.name)
            # 先使用原生语音初始化缓存，再关闭自动合成，控制单元经过解析后才选择合成。
            self.duplex = prepare_duplex(
                self.model, self.config, self.voice_path, generate_audio=True
            )
            self.duplex.generate_audio = False
            from .acoustic import load_vocoder_adapter
            self.vocoder_adapter, self.vocoder_embedding = load_vocoder_adapter(self.model, self.config)
        self.parser.reset()
        self.pending_blocks = []
        self.expression = {'style': 'auto', 'emotion': 'unknown', 'emotion_source': 'audio_model', 'event': 'speech', 'rate': 1.0}
        self.pending_settings = {}
        self.expression_dirty = False
        self.voice_notice = None
        self.voice_error_pending = False
        self.style_reset = None
        self.vocal_action = None
        self.vocal_followup = None
        self.tempo_blocks = []
        self.acoustic_blocks = []
        self.floor = None
        if self.config['model'].get('conversation_floor', False):
            from .floor import ConversationFloor, SpeechActivity
            self.floor = ConversationFloor()
            self.activity = SpeechActivity(self.config.get('asr', {}))
        return {
            "first_samples": self.duplex.processor.get_streaming_chunk_size(),
            "samples": 16000,
            "voice_reference": self.reference_info,
        }

    def configure(self, settings):
        """设置仅在同一推理线程排队，并在下一输出轮次边界生效。"""
        allowed = {'style', 'style_once', 'vocal_action', 'continue_after_vocal', 'source_end', 'emotion', 'emotion_source', 'event', 'rate', 'rate_delta', 'voice', 'voice_audio', 'voice_path', 'voice_id', 'voice_error'}
        if not isinstance(settings, dict) or set(settings) - allowed:
            raise ValueError('未知会话设置')
        if 'style' in settings and settings['style'] not in STYLES:
            raise ValueError('不支持的表达方式')
        if 'voice_error' in settings and settings['voice_error'] != 'reference_unusable':
            raise ValueError('不支持的音色错误类型')
        if 'style_once' in settings and not isinstance(settings['style_once'], bool):
            raise ValueError('临时表达标记必须是布尔值')
        if 'vocal_action' in settings and settings['vocal_action'] not in ('laugh', 'cry'):
            raise ValueError('不支持的非语言发声动作')
        if 'continue_after_vocal' in settings and not isinstance(settings['continue_after_vocal'], bool):
            raise ValueError('发声后续答标记必须是布尔值')
        if 'emotion' in settings and settings['emotion'] not in EMOTIONS:
            raise ValueError('不支持的情绪标签')
        if 'emotion_source' in settings and settings['emotion_source'] not in ('audio_model', 'self_report'):
            raise ValueError('不支持的情绪来源')
        if 'event' in settings and settings['event'] not in ('speech', 'laughter', 'crying'):
            raise ValueError('不支持的声音事件')
        if 'rate_delta' in settings:
            delta = settings['rate_delta']
            if delta not in (-0.15, 0.15):
                raise ValueError('语速调整幅度无效')
            settings = {**settings, 'rate': min(1.5, max(0.75, self.pending_settings.get('rate', self.expression['rate']) + delta))}
        if 'rate' in settings:
            validate_rate(settings['rate'])
            if settings['rate'] != 1.0:
                import shutil
                if shutil.which('ffmpeg') is None:
                    raise ValueError('请安装ffmpeg后再调整语速')
        if 'voice' in settings:
            for key in ('voice', 'voice_audio', 'voice_path', 'voice_id'):
                self.pending_settings.pop(key, None)
        if 'style' in settings:
            for key in ('style_once', 'vocal_action', 'continue_after_vocal', 'vocal_generation'):
                if key not in settings:
                    self.pending_settings.pop(key, None)
        if settings.get('style_once'):
            floor = getattr(self, 'floor', None)
            self.pending_settings['vocal_generation'] = floor.generation_at(settings.get('source_end')) if floor else None
        self.pending_settings.update(settings)

    def _switch_voice(self, settings):
        """重建声码器条件，保留语言解码器和完整对话状态。"""
        import soundfile as sf
        cache = tempfile.TemporaryDirectory(prefix='voice-lab-reference-')
        try:
            source = settings.get('voice_path') or self.original_voice
            if settings.get('voice') == 'default':
                source = resolve_voice(self.config, self.model)
            if settings.get('voice') == 'self':
                source = Path(cache.name) / 'captured.wav'
                sf.write(source, settings['voice_audio'], 16000, subtype='PCM_16')
            path, info = decoder_reference(source, cache.name)
            try:
                self._initialize_voice_cache(path)
            except Exception:
                # 上游初始化会修改全局声码器缓存，失败后必须重新装入旧条件。
                try:
                    self._initialize_voice_cache(self.voice_path)
                except Exception as exc:
                    raise VoiceStateError('音色恢复失败，请重新开始对话') from exc
                raise
        except Exception:
            cache.cleanup()
            raise
        previous = self.reference_cache
        self.reference_cache = cache
        self.voice_path, self.reference_info = path, info
        if previous is not None:
            previous.cleanup()
        return {'type': 'voice_changed', 'voice': settings.get('voice_id', settings.get('voice', 'selected')), 'voice_reference': info}

    def _initialize_voice_cache(self, path):
        adapter = getattr(self, 'vocoder_adapter', None)
        with adapter.disable_adapter() if adapter is not None else nullcontext():
            self.duplex._init_token2wav_cache(str(path))
            self.duplex._reset_token2wav_for_new_turn()

    def _apply_settings(self):
        if not self.pending_settings or not self.duplex.current_turn_ended or self.parser.buffer:
            return []
        settings, self.pending_settings = self.pending_settings, {}
        events = []
        if settings.get('voice_error'):
            self.voice_notice = '【音色切换结果】参考语音不足或不清晰，当前音色保留。请用户再说至少三秒清晰语音，随后重新提出模仿请求。'
            self.voice_error_pending = True
        if 'voice' in settings or 'voice_path' in settings:
            try:
                events.append(self._switch_voice(settings))
                self.voice_notice = '【音色切换结果】本次会话的声码器已成功切换参考音色，正常继续对话。'
                self.voice_error_pending = False
            except VoiceStateError:
                raise
            except Exception:
                import logging
                logging.getLogger('voice_lab').exception('动态换声失败')
                events.append({'type': 'settings_error', 'message': '动态换声失败，已尝试恢复原音色，请检查参考录音'})
                self.voice_notice = '【音色切换结果】本次换声失败，已恢复先前音色。请如实告知用户。'
                self.voice_error_pending = True
        if 'style' in settings:
            self.style_reset = (self.style_reset or self.expression['style']) if settings.get('style_once') else None
            self.vocal_action = settings.get('vocal_action')
            # requested等待发声，pending等待续答提示，active等待续答正文结束。
            self.vocal_followup = 'requested' if settings.get('continue_after_vocal') else None
            floor = getattr(self, 'floor', None)
            self.vocal_generation = settings.get('vocal_generation', floor.generation_at(settings.get('source_end')) if floor else None)
        for key in self.expression:
            if key in settings:
                self.expression[key] = settings[key]
                self.expression_dirty = True
        if self.expression_dirty:
            events.append({'type': 'expression_changed', **self.expression, 'tempo_mode': 'streaming' if self.expression['rate'] == 1.0 else 'buffered-turn'})
        return events

    def end(self):
        self.pending_settings = {}
        self.tempo_blocks = []
        self.acoustic_blocks = []
        self.vocal_action = None
        self.vocal_followup = None
        if self.style_reset is not None:
            self.expression['style'] = self.style_reset
            self.style_reset = None
            self.expression_dirty = True
        self.duplex = None
        self.model.reset_session(reset_token2wav_cache=True)
        if self.reference_cache is not None:
            self.reference_cache.cleanup()
            self.reference_cache = None

    def interrupt(self):
        # 可从接收线程设置，实际缓存修改仍由模型线程在边界处理。
        if self.duplex:
            self.duplex.set_break_event()

    def resume_after_interrupt(self):
        if getattr(self, 'floor', None):
            self.floor.interrupt()
        self.duplex.clear_break_event()
        self.duplex.current_turn_ended = True
        self.duplex.tts_past_key_values = None
        self.duplex.tts_text_start_pos = 0
        self.duplex.tts_current_turn_start_time = None
        self.duplex._reset_token2wav_for_new_turn()
        self.parser.reset()
        self.pending_blocks = []
        self.tempo_blocks = []
        self.acoustic_blocks = []
        self.vocal_action = None
        self.vocal_followup = None

        if self.style_reset is not None:
            self.expression['style'] = self.style_reset
            self.style_reset = None
            self.expression_dirty = True

    def diagnostics(self):
        floor = getattr(self, 'floor', None)
        return {
            'generation': floor.generation if floor else None,
            'answered': floor.answered if floor else None,
            'active': floor.active if floor else None,
            'pending': floor.pending if floor else None,
            'notification': floor.notification if floor else None,
            'output_generation': floor.output_generation if floor else None,
            'wake_count': floor.wake_count if floor else None,
            'idle_frames': floor.idle_frames if floor else None,
            'parser_characters': len(self.parser.buffer),
            'parser_control': self.parser.control,
            'buffered_blocks': len(self.pending_blocks),
            'turn_ended': self.duplex.current_turn_ended if self.duplex else None,
        }

    def confirm_cancelled(self):
        # 协调器实际完成取消后，允许模型说出确认语，再回到倾听。
        if getattr(self, 'floor', None):
            self.floor.notify()

    def _vocal_block(self):
        """明确的哭笑动作使用短原生发声条件，避免只承诺或无限生成语气词。"""
        d = self.duplex
        text = '哈哈哈。' if self.vocal_action == 'laugh' else '呜呜。'
        self.vocal_action = None
        tokens = d.tokenizer.encode(text, add_special_tokens=False)
        tokens.append(d.tokenizer.convert_tokens_to_ids('<|turn_eos|>'))
        d.decoder.feed(d.decoder.embed_token(d.speak_token_id))
        _, hidden = d.decoder.feed(d.decoder.embed_tokens(tokens), return_logits=True)
        d.total_hidden.append([[token, hidden[:, index:index + 1], index == len(tokens) - 1] for index, token in enumerate(tokens)])
        closing = d.tokenizer.encode('<|chunk_eos|></unit>', add_special_tokens=False)
        d.decoder.feed(d.decoder.embed_tokens(closing))
        d.total_ids.extend([d.speak_token_id, *tokens, *closing])
        d.res_ids.extend(tokens)
        d.current_turn_ended = True
        if self.floor is not None:
            self.floor.output_generation = self.vocal_generation
        d.decoder.register_unit_end(input_type=d.current_mode.lower(), generated_tokens=tokens,
                                    is_listen=False, generated_text=text)
        d.decoder.enforce_window()
        return {'is_listen': False, 'text': text, 'end_of_turn': True, 'current_time': float(d.audio_chunk_idx)}

    def step(self, audio, context=None):
        import torch

        with torch.inference_mode():
            updates = self._apply_settings()
            voice_error_response = getattr(self, 'voice_error_pending', False)
            self.voice_error_pending = False
            if self.vocal_followup == 'pending' and not (context and context.startswith('【后台结果】')):
                context = '【发声动作结果】已经完成刚才要求的哭笑动作，请直接继续用户要求的其余内容，不再重复表演或承诺。\n' + (context or '')
                self.vocal_followup = 'active'
            if self.voice_notice:
                context = (context + '\n' if context else '') + self.voice_notice
                self.voice_notice = None
            if self.expression_dirty:
                hint = expression_context(self.expression['style'], self.expression['emotion'], self.expression['event'], self.expression.get('emotion_source', 'audio_model'))
                context = (context + '\n' if context else '') + hint
                self.expression_dirty = False
            floor = getattr(self, 'floor', None)
            if floor is not None:
                self.activity.feed(audio, floor)
                if voice_error_response:
                    floor.notify()
                if context and context.startswith(('【后台结果】', '【发声动作结果】')):
                    floor.notify()
            self.duplex.total_ids = self.duplex.total_ids[-8192:]
            self.duplex.res_ids = self.duplex.res_ids[-8192:]
            self.duplex.prefill_schema_tokens = self.duplex.prefill_schema_tokens[-32:]
            result = self.duplex.streaming_prefill(
                audio_waveform=audio, text_list=[context] if context else None
            )
            if not result["success"]:
                if self.duplex.is_break_set() or self.duplex.is_session_stop_set():
                    raise InferenceInterrupted("当前推理已由用户结束或打断")
                raise RuntimeError(f"音频预填充失败 {result.get('reason', '')}")
            user_wake = floor.need_wake() if floor is not None and self.duplex.current_turn_ended and not self.parser.buffer else False
            if (voice_error_response or (context and context.startswith(('【后台结果】', '【发声动作结果】'))) or user_wake) and self.duplex.pending_logits is not None:
                # 用户结束发言后的有限起答唤醒，以及真实后台结果通知；正文仍由音频与实际结果条件生成。
                logits = torch.full_like(self.duplex.pending_logits, float("-inf"))
                logits[..., self.duplex.speak_token_id] = 0
                self.duplex.pending_logits = logits
            if floor is not None and not floor.can_start and self.duplex.current_turn_ended and self.duplex.pending_logits is not None:
                # 只约束原生发言选择，用户音频仍直接进入端到端模型。
                logits = torch.full_like(self.duplex.pending_logits, float('-inf'))
                logits[..., self.duplex.listen_token_id] = 0
                self.duplex.pending_logits = logits
            vocal_ready = (getattr(self, 'vocal_action', None) is not None and
                           getattr(self.model, '_voice_lab_acoustic_embedding', None) is not None and
                           self.duplex.current_turn_ended and (floor is None or not floor.active) and
                           not (context and context.startswith('【后台结果】')))
            if vocal_ready:
                block = self._vocal_block()
            else:
                block = self.duplex.streaming_generate(
                    max_new_speak_tokens_per_chunk=self.config["model"]["max_speak_tokens"],
                    decode_mode=self.config["model"].get("decode_mode", "sampling"),
                )
            if block["is_listen"]:
                self.duplex.total_hidden.clear()
                return {"kind": "listen", "events": updates}
            if floor is not None:
                floor.begin_output()
            decision = self.parser.push(block["text"], end=block["end_of_turn"])
            hidden = list(self.duplex.total_hidden[-1])
            self.duplex.total_hidden.clear()
            self.pending_blocks.append((block, hidden))
            if decision.kind == "pending":
                return {"kind": "listen", "events": updates}
            blocks, self.pending_blocks = self.pending_blocks, []
            if decision.kind != "speech":
                if floor is not None:
                    if decision.kind in ("delegate", "cancel"):
                        floor.delegated()
                    else:
                        floor.abandon_output()
                self.duplex.tts_past_key_values = None
                self.duplex.tts_text_start_pos = 0
                self.duplex.current_turn_ended = True
                self.duplex._reset_token2wav_for_new_turn()
                return {"kind": decision.kind, "text": decision.text, "events": updates}
            if floor is not None:
                for item, _ in blocks:
                    floor.complete(item['text'], item['end_of_turn'], allow_laughter=self.expression['style'] == 'laugh')
            rendered = []
            from .acoustic import acoustic_style
            acoustic = getattr(self.model, '_voice_lab_acoustic_embedding', None)
            style = acoustic_style(self.expression, vocal=vocal_ready)
            if acoustic is not None and style is not None:
                # 声学训练采用完整一轮的原生隐藏状态，推理使用相同的条件布局。
                self.acoustic_blocks.extend(blocks)
                if not blocks[-1][0]['end_of_turn']:
                    return {'kind': 'listen', 'events': updates}
                combined = self.acoustic_blocks
                self.acoustic_blocks = []
                blocks = [({**combined[-1][0], 'text': ''.join(item['text'] for item, _ in combined)},
                           [value for _, states in combined for value in states])]
            for item, hidden in blocks:
                rendered.append(
                    {
                        "text": item["text"],
                        "audio": self._render(
                            hidden, item["end_of_turn"], item["current_time"], vocal_action=vocal_ready
                        ),
                        "end": item["end_of_turn"],
                    }
                )
            if self.expression['rate'] != 1.0:
                self.tempo_blocks.extend(rendered)
                buffered_samples = sum(len(item['audio']) for item in self.tempo_blocks)
                if not rendered[-1]['end'] and buffered_samples < 24000 * 30:
                    rendered = []
                else:
                    combined = self.tempo_blocks
                    self.tempo_blocks = []
                    waveform = np.concatenate([item['audio'] for item in combined])
                    try:
                        waveform = change_tempo(waveform, self.expression['rate'])
                    except Exception:
                        updates.append({'type': 'settings_error', 'message': '语速处理失败，本轮使用原始音频'})
                    rendered = [{'text': ''.join(item['text'] for item in combined), 'audio': waveform, 'end': combined[-1]['end']}]
            continue_vocal = vocal_ready and self.vocal_followup == 'requested'
            if continue_vocal:
                self.vocal_followup = 'pending'
                if floor is not None:
                    floor.notify()
            followup_finished = not vocal_ready and self.vocal_followup == 'active' and rendered and rendered[-1]['end']
            if rendered and rendered[-1]['end'] and self.style_reset is not None and not continue_vocal and (vocal_ready or followup_finished or floor is None or not floor.pending):
                self.expression['style'] = self.style_reset
                self.style_reset = None
                self.vocal_action = None
                if self.vocal_followup == 'requested':
                    self.vocal_followup = None
                self.expression_dirty = True
                updates.append({'type': 'expression_changed', **self.expression, 'tempo_mode': 'streaming' if self.expression['rate'] == 1.0 else 'buffered-turn'})
            if followup_finished:
                self.vocal_followup = None
            return {"kind": "speech", "blocks": rendered, "events": updates}

    def _render(self, hidden, end, current_time, vocal_action=False):
        """沿用上游隐藏状态到语音 token 的路径，普通回答保持端到端声学条件。"""
        d = self.duplex
        condition = d._convert_results_to_tts_input(hidden)
        from .acoustic import ACOUSTIC_STYLES, acoustic_style, condition_with_style
        embedding = getattr(self.model, '_voice_lab_acoustic_embedding', None)
        style = acoustic_style(self.expression, vocal=vocal_action) if embedding is not None else None
        if style is not None:
            condition = condition_with_style(condition, embedding, ACOUSTIC_STYLES.index(style),
                                              vocal_only=vocal_action and getattr(self.model, '_voice_lab_acoustic_version', 1) >= 2)
        if d.tts_current_turn_start_time is None:
            d.tts_current_turn_start_time = current_time
        first = d.tts_text_start_pos == 0
        adapter_context = self.model.tts.model.disable_adapter() if embedding is not None and style is None else nullcontext()
        minimum_tokens = 0 if end or first else 26
        maximum_tokens = 512 if end else 26
        if vocal_action:
            durations = self.config['model'].get('vocal_min_seconds', {'laugh': 1.8, 'cry': 3.2})
            seconds = float(durations.get(self.expression['style'], 2.0))
            if getattr(self, 'vocoder_adapter', None) is not None:
                seconds += 0.32
            # 语速处理后仍保留足够长的动作，避免提前EOS变成短促叹气。
            minimum_tokens = int(np.ceil(seconds * 25 * self.expression['rate']))
            maximum_tokens = max(minimum_tokens + 1, 150)
        with adapter_context:
            tokens, cache = self.model.tts.generate_chunk(
                inputs_embeds=condition,
                temperature=d.tts_temperature,
                repetition_penalty=d.tts_repetition_penalty,
                eos_token=d.tts_eos_token,
                force_no_stop=False,
                # 末块继续解码到语音EOS，避免长句和数值被一秒码长上限截断。
                max_new_token=maximum_tokens,
                min_new_tokens=minimum_tokens,
                past_key_values=d.tts_past_key_values,
                logits_processors=d.tts_logits_processors,
                text_start_pos=d.tts_text_start_pos,
            )
        if end:
            d.tts_text_start_pos = 0
            d.tts_past_key_values = None
            d.tts_current_turn_start_time = None
        else:
            d.tts_past_key_values = cache
            d.tts_text_start_pos += condition.shape[1] + tokens.shape[1]
        flow_adapter = getattr(self, 'vocoder_adapter', None)
        if vocal_action and flow_adapter is not None:
            import io
            import soundfile as sf
            import torch
            engine = self.model.tts.audio_tokenizer
            decoder = engine.flow.decoder
            original_forward = decoder.forward
            reference_frames = engine.cache[3].shape[1]
            index = 0 if self.expression['style'] == 'laugh' else 1
            offset = self.vocoder_embedding(torch.tensor([index], device=self.model.tts.device)).unsqueeze(2)
            def expressive_forward(mu, mask, spks, cond, n_timesteps=10, temperature=1.0):
                adjusted = mu.clone()
                adjusted[:, :, reference_frames:] += offset.to(mu.dtype)
                return original_forward(mu=adjusted, mask=mask, spks=spks, cond=cond,
                                        n_timesteps=n_timesteps, temperature=temperature)
            try:
                decoder.forward = expressive_forward
                raw = engine(tokens.reshape(-1).tolist(), str(self.voice_path))
                waveform, rate = sf.read(io.BytesIO(raw), dtype='float32')
                if rate != 24000:
                    raise ValueError('声码器输出采样率无效')
            finally:
                decoder.forward = original_forward
        else:
            with flow_adapter.disable_adapter() if flow_adapter is not None else nullcontext():
                waveform = d._generate_waveform_from_tokens(tokens, str(self.voice_path), end, force_flush=first)
        if end:
            d._reset_token2wav_for_new_turn()
        if waveform is None:
            return np.zeros(0, dtype=np.float32)
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().float().cpu().numpy()
        return np.asarray(waveform, dtype=np.float32).reshape(-1)


def model_fingerprint(config):
    item = config["model"]
    return hashlib.sha256(f"{item['path']}@{item['revision']}".encode()).hexdigest()
