// API layer: no DOM, no UI state. The backend is the source of truth for properties, answers
// and the conversation itself (requirements, selection, bookings live server-side per session).
import { CONFIG } from './config.js';

export class ApiError extends Error {
  constructor(kind, status) {
    super(kind);
    this.kind = kind; // 'timeout' | 'network' | 'invalid' | 'rate_limited' | 'server'
    this.status = status;
  }
}

const ids = (v) => (Array.isArray(v) ? v.map(String) : []);

async function postJson(path, body, { timeoutMs = CONFIG.timeoutMs } = {}) {
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, timeoutMs);
  let res;
  try {
    res = await fetch(CONFIG.apiBase + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
  } catch {
    throw new ApiError(timedOut ? 'timeout' : 'network');
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) throw new ApiError(res.status === 422 ? 'invalid' : res.status === 429 ? 'rate_limited' : 'server', res.status);
  try { return await res.json(); } catch { throw new ApiError('server', res.status); }
}

export function buildPayload({ query, sessionId, language, clearRequirements }) {
  const payload = { query: String(query).trim().slice(0, 2000) };
  if (sessionId) payload.session_id = sessionId;
  if (language === 'ar' || language === 'en') payload.language = language;
  if (clearRequirements?.length) payload.clear_requirements = clearRequirements;
  return payload;
}

// Normalizes /chat responses and voice `agent_result` events (same shape).
export function parseTurn(data) {
  return {
    sessionId: typeof data?.session_id === 'string' ? data.session_id : null,
    intent: String(data?.intent || ''),
    answer: typeof data?.response === 'string' ? data.response.trim() : '',
    cited: ids(data?.cited_property_ids),
    grounding: ids(data?.grounding_property_ids),
    properties: Array.isArray(data?.candidates) ? data.candidates : Array.isArray(data?.properties) ? data.properties : [],
    requirements: data?.requirements && typeof data.requirements === 'object' ? data.requirements : {},
    appointment: data?.appointment || null,
    handoff: data?.handoff || null,
    degraded: data?.degraded === true || !!data?.error,
  };
}

export async function sendChat(payload, opts) {
  return parseTurn(await postJson(CONFIG.chatPath, payload, opts));
}

export async function resetConversation(sessionId) {
  if (!sessionId) return;
  try { await postJson(CONFIG.resetPath, { session_id: sessionId }, { timeoutMs: 8000 }); } catch { /* best effort */ }
}

export async function requestVoiceToken({ sessionId, language }) {
  // A sleeping free-tier API needs up to ~50 s to wake, so wait as long as a chat message does.
  return postJson(CONFIG.voiceTokenPath, { session_id: sessionId || null, language }, { timeoutMs: CONFIG.timeoutMs });
}

export async function checkHealth(timeoutMs = 4000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(CONFIG.apiBase + CONFIG.healthPath, { signal: ctrl.signal, cache: 'no-store' });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}
