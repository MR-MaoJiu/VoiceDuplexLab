// 验证跨音频块的连续采样位置，防止44.1 kHz麦克风长期漂移。
const path = require('node:path');
const vm = require('node:vm');
const fs = require('node:fs');
const results = [];
const source = fs.readFileSync(path.resolve(__dirname, '../voice_lab/web/capture.js'), 'utf8');

for (const rate of [44100, 48000, 16000]) {
  let count = 0;
  const sandbox = {
    sampleRate: rate,
    Float32Array,
    AudioWorkletProcessor: class {
      constructor() {
        this.port = {
          postMessage(samples) {
            count += samples.length;
          },
        };
      }
    },
    registerProcessor(name, Processor) {
      sandbox.Capture = Processor;
    },
  };
  vm.runInNewContext(source, sandbox);
  const processor = new sandbox.Capture();
  const frames = Math.floor(rate * 60 / 128);
  for (let index = 0; index < frames; index++) {
    processor.process([[new Float32Array(128).fill(0.1)]]);
  }
  const expected = frames * 128 / rate * 16000;
  if (Math.abs(count - expected) > 2 || processor.buffer.length > 5) {
    throw new Error(JSON.stringify({rate, count, expected, buffer: processor.buffer.length}));
  }
  results.push({
    inputRate: rate,
    outputRate: 16000,
    actualSamples: count,
    expectedSamples: expected,
    retainedSamples: processor.buffer.length,
  });
}
console.log(JSON.stringify({passed: true, cases: results}));
