/* 录音仅保留在浏览器，用户声与模型实际播放声分左右声道。 */
export class ConversationRecorder {
  constructor(context, microphone, onChange) {
    this.context = context;
    this.onChange = onChange;
    this.microphoneGain = context.createGain();
    this.microphoneGain.channelCount = 1;
    this.microphoneGain.channelCountMode = 'explicit';
    this.merger = context.createChannelMerger(2);
    this.destination = context.createMediaStreamDestination();
    microphone.connect(this.microphoneGain);
    this.microphoneGain.connect(this.merger, 0, 0);
    this.merger.connect(this.destination);
    this.chunks = [];
    this.recorder = null;
    this.stopping = null;
    this.error = null;
  }

  connectSpeaker(source) {
    source.connect(this.merger, 0, 1);
  }

  setMuted(muted) {
    this.microphoneGain.gain.value = muted ? 0 : 1;
  }

  get active() {
    return this.recorder?.state === 'recording';
  }

  start() {
    if (this.active) return;
    if (!window.MediaRecorder) throw new Error('当前浏览器不支持录音');
    const type = ['audio/webm;codecs=opus', 'audio/mp4'].find((value) => MediaRecorder.isTypeSupported(value));
    this.recorder = new MediaRecorder(this.destination.stream, type ? { mimeType: type } : {});
    this.chunks = [];
    this.stopping = null;
    this.error = null;
    this.started = this.context.currentTime;
    this.recorder.ondataavailable = (event) => {
      if (event.data.size) this.chunks.push(event.data);
    };
    this.recorder.onerror = (event) => {
      this.error = event.error || new Error('录音失败');
      this.onChange({ state: 'error', message: this.error.message });
    };
    this.recorder.start(1000);
    this.onChange({ state: 'recording' });
  }

  stop() {
    if (this.stopping) return this.stopping;
    if (this.error) return Promise.reject(this.error);
    if (!this.active) return Promise.resolve();
    this.stopping = new Promise((resolve, reject) => {
      this.recorder.addEventListener('stop', () => {
        if (this.error) { reject(this.error); return; }
        const mime = this.recorder.mimeType || this.chunks[0]?.type || 'audio/webm';
        const blob = new Blob(this.chunks, { type: mime });
        this.onChange({ state: 'ready', blob, mime, seconds: this.context.currentTime - this.started });
        this.chunks = [];
        resolve();
      }, { once: true });
      this.recorder.addEventListener('error', (event) => reject(event.error || new Error('录音失败')), { once: true });
      this.recorder.stop();
    });
    return this.stopping;
  }

  async close() {
    try {
      await this.stop();
    } finally {
      this.microphoneGain.disconnect();
      this.merger.disconnect();
      this.destination.stream.getTracks().forEach((track) => track.stop());
    }
  }
}
