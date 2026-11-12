import { ConversationRecorder } from './recorder.js';
// 从脚本地址计算入口，兼容HTTPS代理的路径前缀。
const apiBase = new URL('../', import.meta.url);
const apiURL = (path) => new URL(path.replace(/^\//, ''), apiBase);
const $ = (id) => document.getElementById(id);
let session = null;
let previewURL = null;
let uploadBusy = false;
let messageTimer;
const recordingURLs = [];
function notify(text) {
  $('message').textContent = text;
  $('message').hidden = false;
  clearTimeout(messageTimer);
  messageTimer = setTimeout(() => { $('message').hidden = true; }, 8000);
}
function headers() {
  const token = $('token').value.trim();
  return token ? { Authorization: `Bearer ${token}` } : {};
}
async function request(url, options = {}) {
  const response = await fetch(apiURL(url), { ...options, headers: { ...headers(), ...options.headers } });
  if (!response.ok) {
    let detail = `请求失败 (${response.status})`;
    try { detail = (await response.json()).detail || detail; } catch { /* 保留 HTTP 状态。 */ }
    throw new Error(detail);
  }
  return response.json();
}
async function health() {
  try {
    const value = await request('/health');
    $('health').textContent = value.ready ? `服务就绪 · ${value.available}/${value.slots} 空闲槽位` : '模型尚未就绪';
    return value;
  } catch { $('health').textContent = '语音服务不可达'; }
}
async function refreshVoices(selected = $('voice').value) {
  const { voices } = await request('/voices');
  $('voice').replaceChildren(new Option('模型默认音色', ''));
  for (const voice of voices) $('voice').add(new Option(`${voice.name} · ${voice.seconds} 秒`, voice.id));
  $('voice').value = voices.some((voice) => voice.id === selected) ? selected : '';
}
$('refresh').onclick = () => refreshVoices().catch((error) => notify(error.message));
$('token').oninput = () => { $('voice').replaceChildren(new Option('模型默认音色', '')); };
$('voice-file').onchange = () => {
  if (previewURL) URL.revokeObjectURL(previewURL);
  const file = $('voice-file').files[0];
  previewURL = file ? URL.createObjectURL(file) : null;
  $('preview').hidden = !file;
  if (file) $('preview').src = previewURL;
};
function encodeWav(audio, rate) {
  const body = new ArrayBuffer(44 + audio.length * 2);
  const view = new DataView(body);
  const text = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, 36 + audio.length * 2, true); text(8, 'WAVE');
  text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data');
  view.setUint32(40, audio.length * 2, true);
  audio.forEach((value, index) => view.setInt16(44 + index * 2, Math.round(Math.max(-1, Math.min(1, value)) * 32767), true));
  return body;
}
$('upload').onclick = async () => {
  if (uploadBusy) return;
  const file = $('voice-file').files[0];
  if (!file) return notify('请先选择参考录音');
  if (!$('consent').checked) return notify('请确认已获得使用声音的授权');
  if (file.size > 20_000_000) return notify('参考录音需要小于 20 MB');
  uploadBusy = true; $('token').disabled = true; $('upload').disabled = true; $('upload-state').textContent = '正在转换与上传';
  let decoder;
  try {
    decoder = new AudioContext();
    const decoded = await decoder.decodeAudioData(await file.arrayBuffer());
    if (!(decoded.duration > 0 && decoded.duration <= 60)) throw new Error('参考录音需要在 60 秒以内');
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
    const source = offline.createBufferSource(); source.buffer = decoded; source.connect(offline.destination); source.start();
    const rendered = await offline.startRendering();
    const name = $('voice-name').value.trim() || file.name.replace(/\.[^.]+$/, '');
    const voice = await request(`/voices?name=${encodeURIComponent(name)}`, {
      method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: encodeWav(rendered.getChannelData(0), 16000),
    });
    await refreshVoices(voice.id);
    $('upload-state').textContent = `已保存 ${voice.name}`;
    if (session?.ready) sendSettings({ voice: voice.id });
  } catch (error) { $('upload-state').textContent = '上传失败'; notify(`上传失败 · ${error.message}`); }
  finally { if (decoder) await decoder.close(); uploadBusy = false; $('token').disabled = !!session; $('upload').disabled = false; }
};
function eventLog(text) {
  if ($('events').querySelector('.empty')) $('events').replaceChildren();
  const item = document.createElement('p');
  item.textContent = `${new Date().toLocaleTimeString()}  ${text}`;
  $('events').append(item);
  while ($('events').children.length > 100) $('events').firstChild.remove();
  $('events').scrollTop = $('events').scrollHeight;
}
function sendSettings(settings) {
  if (!session?.ready || session.closing) return notify('请先开始对话');
  session.ws.send(JSON.stringify({ type: 'settings', settings }));
}
$('apply-voice').onclick = () => sendSettings({ voice: $('voice').value || 'default' });
$('clone-self').onclick = () => sendSettings({ voice: 'self' });
$('restore-voice').onclick = () => sendSettings({ voice: 'original' });
$('apply-expression').onclick = () => sendSettings({ style: $('expression').value, rate: Number($('speed').value) });
function dialogueMessage(role, text, time) {
  if ($('transcript').querySelector('.empty')) $('transcript').replaceChildren();
  const entry = document.createElement('article');
  entry.className = `dialogue-entry ${role}`;
  entry.dataset.time = String(time);
  const label = document.createElement('span'); label.className = 'speaker';
  label.textContent = role === 'user' ? '用户' : '模型';
  const paragraph = document.createElement('p'); paragraph.textContent = text;
  entry.append(label, paragraph);
  const following = [...$('transcript').children].find((item) => Number(item.dataset.time) > time);
  $('transcript').insertBefore(entry, following || null);
  $('transcript').scrollTop = $('transcript').scrollHeight;
  return paragraph;
}
function recordingChanged(event) {
  if (event.state === 'error') {
    $('recording-state').textContent = event.message;
    $('record').textContent = '开始录音';
    notify(event.message);
    return;
  }
  $('record').textContent = event.state === 'recording' ? '停止录音' : '开始录音';
  $('recording-state').textContent = event.state === 'recording' ? '录音中' : '录音已保存到本页';
  if (!event.blob?.size) return;
  if ($('recordings').querySelector('.empty')) $('recordings').replaceChildren();
  const url = URL.createObjectURL(event.blob); recordingURLs.push(url);
  const row = document.createElement('div'); row.className = 'recording';
  const audio = document.createElement('audio'); audio.controls = true; audio.src = url;
  const download = document.createElement('a'); download.href = url;
  const extension = event.mime.includes('mp4') ? 'm4a' : 'webm';
  download.download = `VoiceDuplexLab-${new Date().toISOString().replace(/[:.]/g, '-')}.${extension}`;
  download.textContent = `下载录音 · ${Math.round(event.seconds)} 秒`;
  row.append(audio, download); $('recordings').prepend(row);
}
function stopPlayback(current) {
  for (const source of current.sources) { try { source.stop(); } catch { /* 音源可能已结束。 */ } }
  current.sources.clear(); current.nextPlay = 0; current.paragraph = null;
  current.playbackSegments = []; current.heardText = '';
}
function playbackText(current) {
  const now = current.context.currentTime;
  let text = current.heardText;
  for (const segment of current.playbackSegments) {
    if (segment.start > now) break;
    const fraction = Math.min(1, (now - segment.start) / (segment.end - segment.start));
    text += segment.text.slice(0, Math.floor(segment.text.length * fraction));
  }
  return text.slice(-500);
}
function play(current, event) {
  if (event.epoch < current.epoch) return;
  const bytes = Uint8Array.from(atob(event.audio), (char) => char.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  if (bytes.length) {
    const buffer = current.context.createBuffer(1, bytes.length / 2, event.rate);
    const data = buffer.getChannelData(0);
    for (let index = 0; index < data.length; index++) data[index] = view.getInt16(index * 2, true) / 32768;
    const start = Math.max(current.context.currentTime + 0.03, current.nextPlay);
    if (start - current.context.currentTime > 20) throw new Error('播放积压过多，请重新连接');
    const source = current.context.createBufferSource(); source.buffer = buffer; source.connect(current.context.destination);
    current.recorder?.connectSpeaker(source);
    current.sources.add(source);
    current.playbackSegments.push({ source, start, end: start + buffer.duration, text: event.text || '' });
    source.onended = () => {
      if (current.sources.delete(source)) current.heardText = (current.heardText + (event.text || '')).slice(-500);
      current.playbackSegments = current.playbackSegments.filter((segment) => segment.source !== source);
      source.disconnect();
    };
    source.start(start); current.nextPlay = start + buffer.duration;
  }
  if (event.text) {
    if (!current.paragraph) {
      current.paragraph = dialogueMessage('assistant', '', current.clockOrigin + current.context.currentTime);
    }
    current.paragraph.textContent += event.text;
    $('transcript').scrollTop = $('transcript').scrollHeight;
  }
  if (event.end) current.paragraph = null;
}
function updateControls(active) {
  $('connect').textContent = active ? '结束对话' : '开始对话';
  $('interrupt').disabled = !active; $('cancel').disabled = !active;
  $('record').disabled = !active;
  $('token').disabled = active || uploadBusy; $('refresh').disabled = active;
  $('session-state').textContent = active ? '连接中' : '未连接';
  $('orb').classList.toggle('active', active);
}
async function disconnect(current = session) {
  if (!current || current.closed || current.closing) return;
  current.closing = true;
  clearTimeout(current.timeout);
  $('connect').disabled = true; $('connect').textContent = '正在结束对话';
  current.stream?.getTracks().forEach((track) => track.stop());
  current.capture?.disconnect(); current.source?.disconnect(); current.sink?.disconnect();
  stopPlayback(current);
  if (current.ws?.readyState === WebSocket.OPEN) {
    current.ws.send(JSON.stringify({ type: 'close' }));
    await Promise.race([current.serverClosed, new Promise((resolve) => setTimeout(resolve, 4000))]);
  }
  current.ws?.close();
  try { await current.recorder?.close(); } catch (error) { notify(error.message); }
  current.closed = true;
  if (session === current) session = null;
  updateControls(false); $('connect').disabled = false;
  $('hint').textContent = '选择音色，开始说话'; health();
  if (current.context.state !== 'closed') await current.context.close();
}
function sendSamples(current, samples) {
  if (current.closed || current.closing || !current.ready) return;
  for (let index = 0; index < samples.length; index++) {
    const value = $('mute').checked ? 0 : samples[index];
    current.frame.setInt16(current.offset * 2, Math.round(Math.max(-1, Math.min(1, value)) * 32767), true);
    current.offset++;
    if (current.offset === current.target) {
      if (current.ws.bufferedAmount > 128000) { notify('网络发送积压，请重新连接'); disconnect(current); return; }
      current.ws.send(JSON.stringify({ type: 'playback', active: current.sources.size > 0 && current.nextPlay > current.context.currentTime, text: playbackText(current) }));
      current.ws.send(current.frame.buffer);
      current.target = current.samples;
      current.frame = new DataView(new ArrayBuffer(current.target * 2)); current.offset = 0;
    }
  }
}
$('connect').onclick = async () => {
  if (session) return disconnect();
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) return notify('麦克风需要 HTTPS 或 localhost，请使用 SSH 隧道或配置 HTTPS');
  const current = { context: new AudioContext(), sources: new Set(), nextPlay: 0, epoch: 0, closed: false, closing: false, userMessages: new Map(), playbackSegments: [], heardText: '' };
  current.serverClosed = new Promise((resolve) => { current.resolveClosed = resolve; });
  session = current; updateControls(true); $('record').disabled = true; $('hint').textContent = '正在获取麦克风与连接模型';
  current.timeout = setTimeout(() => { notify('连接超时，请检查服务和空闲槽位'); disconnect(current); }, 30000);
  try {
    await current.context.resume();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    if (current.closed) { stream.getTracks().forEach((track) => track.stop()); return; }
    current.stream = stream;
    await current.context.audioWorklet.addModule(apiURL('web/capture.js'));
    if (current.closed) return;
    current.capture = new AudioWorkletNode(current.context, 'voice-capture');
    current.capture.port.onmessage = ({ data }) => sendSamples(current, data);
    current.source = current.context.createMediaStreamSource(stream);
    current.sink = current.context.createGain(); current.sink.gain.value = 0;
    current.recorder = new ConversationRecorder(current.context, current.source, recordingChanged);
    current.recorder.setMuted($('mute').checked);
    current.source.connect(current.capture); current.capture.connect(current.sink); current.sink.connect(current.context.destination);
    const url = apiURL('ws'); url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    if ($('voice').value) url.searchParams.set('voice', $('voice').value);
    const ws = new WebSocket(url, 'voice-lab.v1'); current.ws = ws;
    ws.onopen = () => ws.send(JSON.stringify({ type: 'auth', token: $('token').value.trim() }));
    ws.onmessage = ({ data }) => {
      if (current.closed) return;
      try {
        const event = JSON.parse(data);
        if (event.type === 'ready') {
          clearTimeout(current.timeout); current.clockOrigin = performance.now() / 1000 - current.context.currentTime;
          current.streamStart = performance.now() / 1000;
          $('record').disabled = false;
          $('asr-state').textContent = event.transcription_enabled ? '用户发言转写已启用' : '当前服务尚未启用用户转写';
          if (event.architecture === 'native_e2e') {
            $('emotion-state').textContent = '主语音模型直接根据原始声音与上下文理解情绪';
            $('expression-state').textContent = '情绪与语速要求将交给主模型，实际控制能力待试听验证';
          }
          if ($('record-auto').checked) { try { current.recorder.start(); } catch (error) { notify(error.message); } }
          current.samples = event.samples; current.target = event.first_samples;
          current.frame = new DataView(new ArrayBuffer(current.target * 2)); current.offset = 0; current.ready = true;
          $('session-state').textContent = '正在聆听'; $('hint').textContent = '可以直接说话，也可以随时插话'; health();
        } else if (event.type === 'speech' && !current.closing) play(current, event);
        else if (event.type === 'voice_changed') {
          $('voice-state').textContent = event.voice === 'self' ? '正在使用本次对话中你的声音' : event.voice === 'original' ? '已恢复本次连接的初始音色' : event.voice === 'default' ? '已切换到模型默认音色' : '音色已切换';
          eventLog($('voice-state').textContent);
        } else if (event.type === 'expression_changed') {
          $('expression').value = event.style;
          $('speed').value = event.rate.toFixed(2);
          $('expression-state').textContent = `设置已生效 · ${event.rate.toFixed(2)}倍语速${event.tempo_mode === 'buffered-turn' ? ' · 整轮缓冲播放' : ''}`;
        } else if (event.type === 'settings_queued') eventLog(event.message);
        else if (event.type === 'backchannel') eventLog('收到附和，继续当前回答');
        else if (event.type === 'settings_error') { notify(event.message); eventLog(event.message); }
        else if (event.type === 'user_transcript') {
          let paragraph = current.userMessages.get(event.id);
          if (!paragraph) {
            paragraph = dialogueMessage('user', '', current.streamStart + event.start);
            current.userMessages.set(event.id, paragraph);
          }
          paragraph.textContent = event.text || (event.vocal_event === 'laughter' ? '（笑声）' : event.vocal_event === 'crying' ? '（哭声）' : '');
          if (event.emotion) {
            const labels = { happy: '开心', sad: '低落', angry: '生气', fearful: '害怕', surprised: '惊讶', disgusted: '厌恶', neutral: '平静', unknown: '不确定' };
            $('emotion-state').textContent = `声音情绪初步判断 · ${labels[event.emotion] || '不确定'}`;
            if (event.emotion_source === 'self_report') $('emotion-state').textContent += ` · 用户自述${labels[event.dialogue_emotion] || '不确定'}`;
          }
        } else if (event.type === 'transcript_error') $('asr-state').textContent = event.message;
        else if (event.type === 'interrupted') { current.epoch = Math.max(current.epoch, event.epoch); stopPlayback(current); eventLog('已打断当前播放'); }
        else if (event.type === 'task_started') eventLog(`开始后台任务 · ${event.task}`);
        else if (event.type === 'tool') eventLog(`调用工具 · ${event.name}`);
        else if (event.type === 'task_completed') eventLog('后台结果已返回，等待语音模型整合');
        else if (event.type === 'task_cancelled') eventLog('后台任务已取消');
        else if (event.type === 'task_error') eventLog(`后台任务失败 · ${event.error}`);
        else if (event.type === 'error') { notify(event.message); disconnect(current); }
        else if (event.type === 'control_error') eventLog(event.message);
      } catch (error) { notify(error.message); disconnect(current); }
    };
    ws.onerror = () => { notify('WebSocket 连接失败，请检查服务和代理配置'); disconnect(current); };
    ws.onclose = (event) => {
      current.resolveClosed();
      if (!current.closed && !current.closing) { if (event.code !== 1000) notify(event.reason || `连接已断开 (${event.code})`); disconnect(current); }
    };
  } catch (error) { notify(`无法开始对话 · ${error.message}`); await disconnect(current); }
};
$('interrupt').onclick = () => {
  if (!session?.ready) return;
  session.epoch++; stopPlayback(session); session.ws.send(JSON.stringify({ type: 'interrupt' }));
};
$('cancel').onclick = () => {
  if (!session?.ready) return;
  session.epoch++; stopPlayback(session); session.ws.send(JSON.stringify({ type: 'cancel' }));
};
$('record').onclick = async () => {
  if (!session?.ready || session.closing) return;
  try { if (session.recorder.active) await session.recorder.stop(); else session.recorder.start(); }
  catch (error) { notify(error.message); }
};
$('mute').onchange = () => session?.recorder?.setMuted($('mute').checked);
$('clear').onclick = () => { if (session) session.paragraph = null; $('transcript').replaceChildren(); };
window.addEventListener('pagehide', () => { if (session) disconnect(); if (previewURL) URL.revokeObjectURL(previewURL); recordingURLs.forEach((url) => URL.revokeObjectURL(url)); });
health().then((value) => { if (value && !value.authentication_required) refreshVoices().catch((error) => notify(error.message)); });
