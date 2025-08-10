class PCMQueue {
  constructor() {
    this._chunks = [];
    this._length = 0; // samples in queue
  }
  push(f32) {
    if (!f32 || f32.length === 0) return;
    this._chunks.push(f32);
    this._length += f32.length;
  }
  get length() { return this._length; }
  // Pop up to n samples into out Float32Array
  popInto(out) {
    let needed = out.length;
    let offset = 0;
    while (needed > 0 && this._chunks.length > 0) {
      const head = this._chunks[0];
      if (head.length <= needed) {
        out.set(head, offset);
        offset += head.length;
        needed -= head.length;
        this._chunks.shift();
        this._length -= head.length;
      } else {
        out.set(head.subarray(0, needed), offset);
        const remaining = head.subarray(needed);
        this._chunks[0] = remaining;
        this._length -= needed;
        offset += needed;
        needed = 0;
      }
    }
    return offset; // written samples
  }
}

class PCMPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sampleRateIn = 24000;
    this.queue = new PCMQueue();
    this.targetBufferSamples = Math.floor(0.1 * this.sampleRateIn); // 100ms default
    this.started = false;
    this._firstTick = false;
    this._logUnderflowCount = 0;
    this._stopped = false;
    // tone test
    this.toneActiveSamples = 0;
    this.tonePhase = 0;
    this.toneFreq = 440;
    this.toneGain = 0.1;

    this.port.onmessage = (ev) => {
      const msg = ev.data;
      if (!msg || !msg.type) return;
      if (msg.type === 'push' && msg.ab && Number.isFinite(msg.length)) {
        const f32 = new Float32Array(msg.ab, msg.offset || 0, msg.length);
        this.queue.push(f32);
      } else if (msg.type === 'setBufferMs') {
        const ms = Math.max(0, msg.ms|0);
        this.targetBufferSamples = Math.floor(ms * this.sampleRateIn / 1000);
      } else if (msg.type === 'reset') {
        this.queue = new PCMQueue();
        this.started = false;
        this._firstTick = false;
        this._stopped = false;
        this._logUnderflowCount = 0;
      } else if (msg.type === 'tone') {
        const ms = Math.max(0, msg.ms|0) || 500; this.toneActiveSamples = Math.floor(ms * this.sampleRateIn / 1000);
        this.toneFreq = msg.freq || 440; this.toneGain = msg.gain ?? 0.1;
      } else if (msg.type === 'stop') {
        this._stopped = true;
      }
    };

    // Notify main thread that processor is alive
    this.port.postMessage({ type: 'ready', sr: this.sampleRateIn });
  }

  process(inputs, outputs) {
    const output = outputs[0];
    const channel = output[0]; // mono
    const frames = channel.length; // usually 128

    if (!this._firstTick) { this._firstTick = true; this.port.postMessage({ type: 'tick' }); }

    if (this.toneActiveSamples > 0) {
      const dt = 1 / this.sampleRateIn; const w = 2 * Math.PI * this.toneFreq;
      for (let i = 0; i < frames; i++) {
        channel[i] = Math.sin(this.tonePhase) * this.toneGain;
        this.tonePhase += w * dt;
      }
      this.toneActiveSamples -= frames;
      return true;
    }

    if (!this.started) {
      if (this.queue.length >= this.targetBufferSamples) {
        this.started = true;
        this.port.postMessage({ type: 'started' });
      } else {
        // fill with silence until buffer reaches target
        channel.fill(0);
        return true;
      }
    }

    // Pop available samples
    const written = this.queue.popInto(channel);
    if (written < frames) {
      // underflow, pad with zeros
      for (let i = written; i < frames; i++) channel[i] = 0;
      if (!this._stopped && this._logUnderflowCount < 20) {
        this.port.postMessage({ type: 'underflow', have: this.queue.length });
        this._logUnderflowCount++;
      }
      // keep going; Worklet remains alive
    }

    return true;
  }
}

registerProcessor('pcm-player', PCMPlayerProcessor);
