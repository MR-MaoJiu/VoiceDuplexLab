"""用主模型已有的原始音频上下文选择会话音色动作。"""

DIRECT_REQUEST_QUERY = """判断用户这段语音是否正在直接要求你执行音色操作，只输出0或1。
1 直接要求你模仿用户的声音、换回默认或初始声音、停止模仿用户声音。
0 普通聊天、朗读复述、情绪表演、语速要求、询问一般能力、讲述其他人的经历。
必须判断谁在要求谁做什么。用户讲别人要求用户去模仿声音，是叙述经历，选0。
例子：请模仿我说话，接下来用我的声音回答。答案1。
例子：能不能模仿我说话？答案1。
例子：现在模仿我的声音说话。答案1。
例子：学一下我说话的声音，接下来这样跟我聊。答案1。
例子：你会不会模仿声音？我只是问问你的能力。答案0。
例子：她让我模仿她的声音，这件事情挺有意思的。答案0。
例子：我在练习模仿别人的声音。答案0。
例子：不要再模仿我的声音了。答案1。
例子：切回默认声音。答案1。
例子：请直接说这句话，今天真是美好的一天。答案0。
例子：请说快一点，讲一下怎么安排阅读。答案0。
例子：用悲伤的语气说，我好想你。答案0。
只根据接下来的用户音频判断，输出一个数字。"""

VOICE_ACTION_QUERY = """
这是一次内部会话动作选择。只依据最近一轮用户真实语音的意思选择一个数字。
0 保持当前音色。普通聊天、讲故事、情绪或语速要求、询问能力、禁止换声、引用别人的要求均选0。
1 用户明确请求模仿用户自己的声音或使用用户现在的声音。
2 用户明确请求恢复本次会话开始时的声音、原来的声音。
3 用户明确请求恢复模型默认声音或默认音色。
4 用户要求停止、禁止模仿用户的声音。
优先判断用户明确指定的目标。说到“默认声音”“模型默认”“系统默认”“原厂声音”时选3。
同一句同时要求“恢复默认”和“停止模仿”时仍选3。
只有指定“本次会话开始”“最初”“原来的声音”而没有指定默认时才选2。
只判断用户请求，忽略助手自己说的“已经换好了”等承诺，也不要重复执行历史请求。
例子：你能模仿我的声音吗，我希望你接下来用我的声音聊天。答案1。
例子：模仿我说话。答案1。
例子：换回默认的声音吧。答案3。
例子：请你切回模型默认的声音，不要再用我的音色。答案3。
例子：用系统默认音色回答。答案3。
例子：恢复最开始的声音。答案2。
例子：你会不会模仿声音？答案0。
例子：别模仿我的声音。答案4。
例子：不要模仿我的声音，请继续讲故事。答案4。
例子：她让我模仿她的声音。答案0。
例子：请说快一点。答案0。
例子：给我讲个故事。答案0。
例子：请重复我刚才说的那句话。答案0。
例子：请直接说这句话，今天真是美好的一天。答案0。
例子：用开心的语气读这句话。答案0。
例子：你刚才说换好了，为什么听起来没变化？答案0。
请只输出用户音频请求对应的数字。"""


def decide_voice_action(decoder, audio_units, current_voice='original'):
    """复用本轮音频编码和主模型权重，动作分支的上下文独立。"""
    if not audio_units:
        return None
    if _select_action(decoder, audio_units, DIRECT_REQUEST_QUERY, 2) == 0:
        return None
    query = voice_action_query(current_voice)
    selected = _select_action(decoder, audio_units, query, 5)
    return (None, 'self', 'original', 'default', 'cancel')[selected]


def _select_action(decoder, audio_units, query, count):
    import torch
    tokenizer = decoder.tokenizer
    choices = [tokenizer.encode(str(index), add_special_tokens=False) for index in range(count)]
    if any(len(tokens) != 1 for tokens in choices):
        raise ValueError('当前分词器不能使用单token音色动作选择')
    inputs = action_embeddings(decoder, audio_units, query)
    positions = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0)
    with torch.inference_mode():
        output = decoder.m(inputs_embeds=inputs, position_ids=positions,
                           use_cache=False, logits_to_keep=1, return_dict=True, output_hidden_states=False)
        selected = int(output.logits[0, -1, [tokens[0] for tokens in choices]].argmax())
    return selected


def voice_action_query(current_voice):
    voices = {'original': '本次会话初始声线', 'self': '正在模仿用户声音',
              'default': '模型默认声线', 'selected': '手动选择的参考声线'}
    return VOICE_ACTION_QUERY.replace('这是一次内部会话动作选择。',
                                      '这是一次内部会话动作选择。当前声线状态为' + voices[current_voice] + '。')


def action_embeddings(decoder, audio_units, query):
    """训练与推理共用原始音频条件布局。"""
    import torch
    tokenizer = decoder.tokenizer
    prefix = '<|im_start|>system\n' + query + '<|im_end|><|im_start|>user\n'
    suffix = '\n请根据以上用户语音选择数字。<|im_end|><|im_start|>assistant\n<think>\n\n</think>\n'
    def embed(text):
        return decoder.embed_tokens(tokenizer.encode(text, add_special_tokens=False))
    parts = [embed(prefix)]
    closing = embed('<|listen|></unit>')
    for unit in audio_units:
        parts.extend((unit.to(device=parts[0].device, dtype=parts[0].dtype), closing))
    parts.append(embed(suffix))
    return torch.cat(parts).unsqueeze(0)
