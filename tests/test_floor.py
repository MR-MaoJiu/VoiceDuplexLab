from voice_lab.floor import ConversationFloor, is_preface


def test_silence_and_filler_do_not_answer_user():
    floor = ConversationFloor()
    assert not floor.pending
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.complete('嗯。', True)
    assert floor.pending
    floor.complete('当然可以。', True)
    assert floor.pending
    floor.complete('每天练习十分钟，读完回听录音。', True)
    assert not floor.pending
    floor.observe_activity(True)
    assert floor.pending


def test_notification_does_not_consume_new_user_question():
    floor = ConversationFloor()
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.delegated()
    assert not floor.pending
    floor.notify()
    floor.observe_activity(True)
    floor.complete('计算结果为八百九十三。', True)
    assert floor.pending
    floor.complete('你新问的问题可以这样做。', True)
    assert not floor.pending


def test_interrupt_clears_old_reply():
    floor = ConversationFloor()
    floor.observe_activity(True)
    floor.complete('还没', False)
    floor.interrupt()
    assert not floor.pending and floor.text == ''


def test_new_question_before_tool_result_remains_pending():
    floor = ConversationFloor()
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.delegated()
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.notify()
    floor.complete('计算结果为八百九十三。', True)
    assert floor.pending
    floor.complete('关于你刚才说的安排，可以先休息。', True)
    assert not floor.pending


def test_waits_for_user_pause_and_background_result_can_wake():
    floor = ConversationFloor()
    floor.observe_activity(True)
    assert floor.pending and not floor.can_start
    floor.observe_activity(False)
    assert floor.can_start
    floor.complete('我是一个中文语音助手。', True)
    assert not floor.can_start
    floor.notify()
    assert floor.can_start


def test_promise_to_describe_is_not_an_actual_answer():
    floor = ConversationFloor()
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.complete('好的，我来帮您用一句话描述秋天。', True)
    assert floor.can_start
    floor.complete('秋风把树叶染成了金黄色。', True)
    assert not floor.can_start


def test_clarifying_question_does_not_consume_city_spoken_during_output():
    floor=ConversationFloor()
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.complete('你要查哪个',False)
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.complete('城市的天气？',True)
    assert floor.answered==1 and floor.generation==2 and floor.can_start
    floor.begin_output()
    floor.delegated()
    assert not floor.pending
    floor.observe_activity(True)
    floor.observe_activity(False)
    assert floor.can_start


def test_new_question_during_control_output_is_preserved():
    floor=ConversationFloor()
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.begin_output()
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.delegated()
    assert floor.pending and floor.answered==1


def test_failed_control_does_not_mark_question_answered():
    floor=ConversationFloor()
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.begin_output()
    floor.abandon_output()
    assert floor.can_start


def test_query_transition_does_not_consume_request():
    floor=ConversationFloor()
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.complete('好的，我帮你查一下上海天气。',True)
    assert floor.can_start
    floor.complete('我可以查资料，也能陪你聊天。',True)
    assert not floor.pending


def test_completed_user_turn_wakes_from_listen_with_finite_budget():
    floor=ConversationFloor()
    assert not floor.need_wake()
    floor.observe_activity(True)
    assert not floor.need_wake()
    floor.observe_activity(False)
    assert not floor.need_wake()
    assert floor.need_wake()
    floor.complete('我查一下。',True)
    assert not floor.need_wake()
    assert floor.need_wake()
    floor.complete('请稍等。',True)
    assert not any(floor.need_wake() for _ in range(20))
    floor.delegated()
    assert not any(floor.need_wake() for _ in range(20))
    floor.observe_activity(True)
    floor.observe_activity(False)
    assert not floor.need_wake()
    assert floor.need_wake()


def test_finished_answer_does_not_wake_again_in_silence():
    floor=ConversationFloor()
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.complete('上海天气结果稍后会回来。',True)
    assert not any(floor.need_wake() for _ in range(20))


def test_action_promise_leaves_question_pending():
    floor=ConversationFloor()
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.complete('好的，我现在开始唱。',True)
    assert floor.pending
    floor.complete('我现在开始讲一个故事。',True)
    assert floor.pending
    floor.complete('从前，有一只住在森林里的小兔子。',True)
    assert not floor.pending


def test_delivered_content_and_capability_explanation_are_not_prefaces():
    assert not is_preface('好的，我现在开始唱：春天来了，花儿开了。')
    assert not is_preface('我可以朗读歌词，但目前还不能保证唱出旋律。')
    assert not is_preface('我现在开始唱，春天来了花儿开了。')


def test_semantic_structure_rejects_promise_even_if_old_flag_passed():
    from voice_lab.semantic_quality import structural
    assert not structural({'response_completed':True,'turns':['好的，我现在开始唱。']})
    assert structural({'response_completed':True,'turns':['我来讲一个故事。','从前有一只小兔子。']})


def test_requested_laughter_finishes_but_spontaneous_filler_does_not():
    floor=ConversationFloor()
    floor.observe_activity(True)
    floor.observe_activity(False)
    floor.complete('好的，我先开怀大笑三声。',True)
    assert floor.pending
    floor.complete('哈哈哈。',True)
    assert floor.pending
    floor.complete('哈哈哈。',True,allow_laughter=True)
    assert not floor.pending


def test_late_transcript_action_is_bound_to_its_original_audio():
    floor=ConversationFloor()
    floor.observe_activity(True,at=0.3)
    floor.observe_activity(False,at=2.5)
    floor.observe_activity(True,at=3.0)
    floor.observe_activity(False,at=4.5)
    assert floor.generation_at(2.2)==1
    floor.output_generation=floor.generation_at(2.2)
    floor.complete('哈哈哈。',True,allow_laughter=True)
    assert floor.pending and floor.answered==1
