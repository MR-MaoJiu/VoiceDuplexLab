from voice_lab.dialogue_data import DIALOGUE_PROMPT, HELDOUT, clean_filler_turns, repair_scenarios, spoken_units


def test_filler_cleanup_preserves_semantic_turn_and_timing():
    units = [{'text': '嗯', 'end': True}, {'listen': True},
             {'text': '嗯，我们', 'end': False}, {'text': '接着聊。', 'end': True}]
    cleaned, count = clean_filler_turns(units)
    assert count == 1 and len(cleaned) == len(units)
    assert cleaned[0] == {'listen': True}
    assert cleaned[2:] == units[2:]
    assert units[0]['text'] == '嗯'


def test_complete_control_units_are_preserved():
    units = [{'text': '嗯', 'delegate': True, 'end': True}]
    assert clean_filler_turns(units) == (units, 0)


def test_repair_catalog_reserves_unseen_normal_questions():
    training = repair_scenarios()
    assert sum(row['kind'] == 'direct' for row in training) >= 60
    assert len(HELDOUT) >= 20
    assert not ({row['user'] for row in training} & {row['user'] for row in HELDOUT})
    assert all(row['keywords'] for row in HELDOUT)
    assert '回答完成后保持倾听' in DIALOGUE_PROMPT


def test_spoken_chunks_preserve_complete_answer():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text)))
    text = '可以先从简单的小目标开始，再慢慢增加难度。'
    # 测试分块的节奏、文本顺序和唯一结束标记。
    units = spoken_units(text, Tokenizer(), 200)
    assert ''.join(unit['text'] for unit in units) == text
    assert all(len(unit['text']) <= 6 for unit in units)
    assert sum(unit['end'] for unit in units) == 1 and units[-1]['end']


def test_transcript_numbers_accept_same_spoken_value():
    from voice_lab.dialogue_data import normalize_audio_text
    assert normalize_audio_text('八加六等于十四。') == normalize_audio_text('8加6等于14')
    assert normalize_audio_text('八九三') == '893'
    assert normalize_audio_text('一百零五') == '105'


def test_quality_check_rejects_continuous_filler_loop():
    import numpy as np
    from voice_lab.dialogue_quality import observe_turn
    class Runtime:
        def step(self, audio):
            return {'kind': 'speech', 'blocks': [{'text': '嗯', 'audio': np.zeros(2400), 'end': True}]}
    state = {'frames': 0, 'samples': 0, 'first_samples': 16000, 'play_end': 0, 'audio': []}
    result = observe_turn(Runtime(), np.zeros(16000), state, ['地球'], max_frames=15)
    assert not result['passed']
    assert len(result['filler_turns']) == 15
    assert result['repeated_turns'] == 14


def test_quality_check_waits_for_substantive_reply_and_silence():
    import numpy as np
    from voice_lab.dialogue_quality import observe_turn
    class Runtime:
        count = 0
        def step(self, audio):
            self.count += 1
            if self.count == 2:
                return {'kind': 'speech', 'blocks': [{'text': '地球的自转带来了白天和黑夜。', 'audio': np.zeros(2400), 'end': True}]}
            return {'kind': 'listen'}
    runtime = Runtime()
    state = {'frames': 0, 'samples': 0, 'first_samples': 16000, 'play_end': 0, 'audio': []}
    result = observe_turn(runtime, np.zeros(16000), state, ['地球'], max_frames=30)
    assert result['passed'] and runtime.count >= 12


def test_semantic_review_can_keep_valid_alternative_without_ignoring_audio_failure():
    from voice_lab.semantic_quality import structural
    row = {'response_completed': False, 'turns': ['你好呀，好久不见。'],
           'filler_turns': [], 'repeated_turns': 0, 'followup_turns': 0,
           'audio_text_similarity': 0.95}
    assert structural(row)
    assert not structural(row | {'audio_text_similarity': 0.3})
    assert not structural(row | {'repeated_turns': 2})


def test_balancing_repeats_only_control_and_memory_without_new_audio():
    from voice_lab.dialogue_data import balance_repair_records
    records = [{'id':'direct', 'audio':'a.wav', 'provenance':{'scenario':'direct'}},
               {'id':'memory', 'audio':'b.wav', 'provenance':{'scenario':'memory'}}]
    rows = balance_repair_records(records)
    assert len(rows) == 4 and len({row['id'] for row in rows}) == 4
    assert [row['audio'] for row in rows].count('b.wav') == 3
    assert 'sampling_repeat' not in records[1]['provenance']


def test_quality_silence_window_starts_after_answer_not_promise():
    import numpy as np
    from voice_lab.dialogue_quality import observe_turn
    class Runtime:
        count = 0
        def step(self, audio):
            self.count += 1
            text = {2:'好的，我来帮您用一句话描述秋天。', 10:'秋风把树叶染成金黄色。'}.get(self.count)
            if text:
                return {'kind':'speech','blocks':[{'text':text,'audio':np.zeros(2400),'end':True}]}
            return {'kind':'listen'}
    runtime=Runtime()
    state={'frames':0,'samples':0,'first_samples':16000,'play_end':0,'audio':[]}
    result=observe_turn(runtime,np.zeros(16000),state,['秋'],max_frames=30)
    assert result['passed'] and runtime.count >= 20


def test_number_audit_does_not_treat_selection_pronoun_as_quantity():
    from voice_lab.dialogue_data import spoken_numbers
    assert spoken_numbers('你喜欢蓝色的哪一种？') == []
    assert spoken_numbers('我查一下，一会儿一起说。') == []
    assert spoken_numbers('一共十四本书') == ['14']
    assert spoken_numbers('一本书，一加六等于七，一点五。') == ['1','1','6','7','1','5']
    assert spoken_numbers('八一八，一一一') == ['818','111']
    assert spoken_numbers('八加六等于十四') != spoken_numbers('二加六等于八')


def test_completed_repair_job_resumes_without_original_corpus(tmp_path,monkeypatch):
    import sys
    from voice_lab import repair_pipeline
    job=tmp_path/'restored-job'
    job.mkdir()
    (job/'dataset_report.json').write_text('{}')
    original='model:\n  reference_audio_in_context: false\ntraining:\n  learning_rate: 0.00002\n'
    (job/'config.yaml').write_text(original)
    monkeypatch.setattr(repair_pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['repair_pipeline','--config','missing.yaml','--output',str(job)])
    monkeypatch.setattr(repair_pipeline,'load_base',lambda *a,**k: (_ for _ in ()).throw(AssertionError('不应重新加载模型')))
    repair_pipeline.main()
    assert (job/'config.yaml').read_text()==original
