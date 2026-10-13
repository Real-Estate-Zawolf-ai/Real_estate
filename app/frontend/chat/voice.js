// Voice conversation client: microphone -> 16 kHz PCM -> /website/voice -> 24 kHz PCM playback.
// No DOM here; the chatbot renders states, transcripts and agent results from the callbacks.
// The browser never sees a provider key: it gets a single-use token from POST /voice/token.
import { CONFIG } from './config.js';
import { requestVoiceToken, ApiError } from './api.js';

export const VOICE_STATES = ['idle', 'connecting', 'listening', 'thinking', 'speaking', 'error'];

export function voiceSupported() {
  return typeof window !== 'undefined'
    && window.isSecureContext !== false
    && !!navigator.mediaDevices?.getUserMedia
    && typeof window.AudioContext === 'function'
    && typeof window.AudioWorkletNode === 'function'
    && typeof window.WebSocket === 'function';
}

function wsUrl(path, token) {
  const base = CONFIG.apiBase ? new URL(CONFIG.apiBase, location.href) : new URL(location.href);
  base.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
  base.pathname = path;
  base.search = `?token=${encodeURIComponent(token)}`;
  base.hash = '';
  return base.toString();
}

export class VoiceSession {
  constructor({ getSessionId, setSessionId, getLang, onState, onTranscript, onAgentResult, onTurnComplete, onError, onLevel }) {
    Object.assign(this, { getSessionId, setSessionId, getLang, onState, onTranscript, onAgentResult, onTurnComplete, onError, onLevel });
    this.state = 'idle';
    this.muted = false;
    this.ws = null;
    this.ctx = null;
    this.stream = null;
    this.node = null;
    this.sources = new Set();
    this.playhead = 0;
    this.outputRate = 24000;
    this.dropAudio = false; // after a manual interrupt, discard the rest of that reply
    this.reconnects = 0;
    this.ending = false;
  }

  setState(state) {
    if (this.state === state) return;
    this.state = state;
    this.onState?.(state);
  }

  get active() { return this.state !== 'idle' && this.state !== 'error'; }

  async start() {
    if (this.active) return;
    if (!voiceSupported()) { this.fail('unsupported'); return; }
    this.ending = false;
    this.reconnects = 0;
    this.setState('connecting');
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (err) {
      this.fail(err?.name === 'NotAllowedError' || err?.name === 'SecurityError' ? 'permission' : err?.name === 'NotFoundError' ? 'no_microphone' : 'unavailable');
      return;
    }
    try {
      this.ctx = new AudioContext({ latencyHint: 'interactive' });
      await this.ctx.audioWorklet.addModule(new URL('./voice-worklet.js', import.meta.url));
      const source = this.ctx.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.ctx, 'pcm-capture');
      this.node.port.onmessage = (e) => this.sendAudio(e.data);
      source.connect(this.node);
      // The worklet must be pulled by the graph; a muted gain keeps the mic out of the speakers.
      const sink = this.ctx.createGain();
      sink.gain.value = 0;
      this.node.connect(sink).connect(this.ctx.destination);
      if (this.ctx.state === 'suspended') await this.ctx.resume();
      await this.connect();
    } catch (err) {
      this.fail(err instanceof ApiError && err.status === 503 ? 'not_configured' : 'unavailable');
    }
  }

  async connect() {
    const { token, ws_path: path, session_id: sessionId } = await requestVoiceToken({
      sessionId: this.getSessionId?.(),
      language: this.getLang?.() || 'en',
    });
    this.setSessionId?.(sessionId);
    await new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl(path, token));
      ws.binaryType = 'arraybuffer';
      this.ws = ws;
      let opened = false;
      ws.onopen = () => { opened = true; resolve(); };
      ws.onerror = () => { if (!opened) reject(new Error('ws')); };
      // Events from a socket we already replaced (reconnect) or closed are ignored.
      ws.onmessage = (e) => { if (this.ws === ws) this.handleMessage(e.data); };
      ws.onclose = (e) => {
        if (this.ws !== ws) return;
        if (!opened) { this.ws = null; reject(new Error('ws')); return; } // caller reports it
        this.handleClose(e);
      };
    });
  }

  sendAudio({ pcm, level }) {
    this.onLevel?.(this.muted ? 0 : level);
    if (this.muted || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    if (this.state === 'connecting') return; // wait for "ready"
    this.ws.send(pcm);
  }

  handleMessage(data) {
    if (data instanceof ArrayBuffer) { this.play(data); return; }
    let event;
    try { event = JSON.parse(data); } catch { return; }
    switch (event.type) {
      case 'ready':
        this.outputRate = event.output_sample_rate || 24000;
        this.setState('listening');
        break;
      case 'transcript':
        this.onTranscript?.(event.role, event.text);
        break;
      case 'tool_start':
        this.setState('thinking');
        break;
      case 'agent_result':
        this.onAgentResult?.(event);
        break;
      case 'interrupted':
        this.stopPlayback();
        this.setState('listening');
        break;
      case 'reconnecting':
        // The server is re-establishing the voice provider; the call stays open.
        this.setState('connecting');
        break;
      case 'reconnected':
        this.setState('listening');
        break;
      case 'turn_complete':
        this.dropAudio = false;
        this.onTurnComplete?.();
        if (!this.sources.size && this.state !== 'idle') this.setState('listening');
        break;
      case 'session_end':
        if (event.reason === 'reconnect' && this.reconnects < 1) { this.reconnect(); return; }
        this.stop(event.reason === 'idle' ? 'idle' : event.reason === 'max_duration' ? 'max_duration' : null);
        break;
      case 'error':
        // agent_unavailable is spoken by the assistant and the call continues.
      {
        const reasons = { busy: 'busy', agent_unavailable: null, provider_disconnected: 'lost' };
        this.fail(event.code in reasons ? reasons[event.code] : 'unavailable');
      }
        break;
      default:
        break;
    }
  }

  handleClose(e) {
    if (this.ending) return;
    this.ws = null;
    if (e.code === 1008) this.fail('unavailable');
    else if (e.code === 1013) this.fail('busy');
    else if (e.code !== 1000) this.fail('lost');
    else this.stop();
  }

  async reconnect() {
    this.reconnects += 1;
    const old = this.ws;
    this.ws = null;
    try { old?.close(1000); } catch { /* already closed */ }
    this.setState('connecting');
    try { await this.connect(); } catch { this.fail('lost'); }
  }

  play(buffer) {
    if (this.dropAudio || !this.ctx) return;
    const pcm = new Int16Array(buffer);
    if (!pcm.length) return;
    const audio = this.ctx.createBuffer(1, pcm.length, this.outputRate);
    const channel = audio.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 0x8000;
    const src = this.ctx.createBufferSource();
    src.buffer = audio;
    src.connect(this.ctx.destination);
    const startAt = Math.max(this.ctx.currentTime + 0.02, this.playhead);
    src.start(startAt);
    this.playhead = startAt + audio.duration;
    this.sources.add(src);
    src.onended = () => {
      this.sources.delete(src);
      if (!this.sources.size && this.state === 'speaking') this.setState('listening');
    };
    this.setState('speaking');
  }

  stopPlayback() {
    for (const src of this.sources) { try { src.stop(); } catch { /* not started */ } }
    this.sources.clear();
    this.playhead = 0;
  }

  // Cut the assistant off. Speaking also interrupts it (server-side barge-in).
  interrupt() {
    if (this.state !== 'speaking') return;
    this.dropAudio = true;
    this.stopPlayback();
    this.setState('listening');
  }

  setMuted(muted) {
    this.muted = muted;
    this.stream?.getAudioTracks().forEach((track) => { track.enabled = !muted; });
    if (muted && this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: 'audio_end' }));
  }

  fail(reason) {
    if (!reason) return; // non-fatal (e.g. the agent failed but the call continues)
    this.cleanup();
    this.setState('error');
    this.onError?.(reason);
  }

  stop(reason = null) {
    this.cleanup();
    this.setState('idle');
    if (reason) this.onError?.(reason);
  }

  cleanup() {
    this.ending = true;
    const ws = this.ws;
    this.ws = null;
    try { if (ws && ws.readyState <= WebSocket.OPEN) ws.close(1000); } catch { /* ignore */ }
    this.stopPlayback();
    try { this.node?.port.close(); this.node?.disconnect(); } catch { /* ignore */ }
    this.node = null;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    if (this.ctx && this.ctx.state !== 'closed') this.ctx.close().catch(() => {});
    this.ctx = null;
    this.muted = false;
    this.dropAudio = false;
    this.onLevel?.(0);
  }
}
