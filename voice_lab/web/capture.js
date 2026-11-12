/* 浏览器声卡采样率转换为连续的 16 kHz PCM，跨回调保留插值位置。 */
class VoiceCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buffer = []; this.position = 0; }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    this.buffer.push(...input);
    const ratio = sampleRate / 16000;
    const output = [];
    while (this.position + 1 < this.buffer.length) {
      const index = Math.floor(this.position);
      const fraction = this.position - index;
      output.push(this.buffer[index] * (1 - fraction) + this.buffer[index + 1] * fraction);
      this.position += ratio;
    }
    const consumed = Math.min(Math.floor(this.position), this.buffer.length);
    this.buffer.splice(0, consumed);
    this.position -= consumed;
    if (output.length) {
      const samples = Float32Array.from(output);
      this.port.postMessage(samples, [samples.buffer]);
    }
    return true;
  }
}
registerProcessor('voice-capture', VoiceCapture);
