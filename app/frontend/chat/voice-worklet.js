// AudioWorklet: microphone Float32 at the device rate -> 16 kHz mono PCM16 chunks.
// Box-filter decimation (average of the input samples per output sample) is enough for speech
// and works for any device rate (44.1 kHz, 48 kHz...). Chunks of 640 samples = 40 ms.
const TARGET_RATE = 16000;
const CHUNK = 640;

class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.phase = 0;
    this.acc = 0;
    this.count = 0;
    this.out = new Int16Array(CHUNK);
    this.len = 0;
    this.energy = 0;
  }

  emit(sample) {
    const s = Math.max(-1, Math.min(1, sample));
    this.out[this.len++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    this.energy += s * s;
    if (this.len === CHUNK) {
      const level = Math.sqrt(this.energy / CHUNK);
      const buf = this.out.buffer.slice(0);
      this.port.postMessage({ pcm: buf, level }, [buf]);
      this.len = 0;
      this.energy = 0;
    }
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      this.acc += channel[i];
      this.count++;
      this.phase += 1;
      if (this.phase >= this.ratio) {
        this.phase -= this.ratio;
        this.emit(this.acc / this.count);
        this.acc = 0;
        this.count = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCapture);
