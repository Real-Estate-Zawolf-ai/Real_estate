// API layer: no DOM, no UI state. The backend is the source of truth for properties and answers.
import { CONFIG } from './config.js';

export class ApiError extends Error {
  constructor(kind, status) {
    super(kind);
    this.kind = kind; // 'timeout' | 'network' | 'invalid' | 'server'
    this.status = status;
  }
}

export function buildPayload({ query, requirements, history }) {
  const payload = {
    query: String(query).trim().slice(0, 2000),
    requirements: requirements || {},
    top_k: CONFIG.topK,
    retrieval_top_k: CONFIG.retrievalTopK,
  };
  // Ready for multi-turn backends: flip SEND_HISTORY in runtime-config.js.
  if (CONFIG.sendHistory && history?.length) payload.history = history.slice(-CONFIG.maxHistoryTurns * 2);
  return payload;
}

const ids = (v) => (Array.isArray(v) ? v.map(String) : []);

export async function searchProperties(payload, { timeoutMs = CONFIG.timeoutMs } = {}) {
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, timeoutMs);
  let res;
  try {
    res = await fetch(CONFIG.apiBase + CONFIG.ragPath, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(payload),
      signal: ctrl.signal,
    });
  } catch {
    throw new ApiError(timedOut ? 'timeout' : 'network');
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) throw new ApiError(res.status === 422 ? 'invalid' : 'server', res.status);
  let data;
  try { data = await res.json(); } catch { throw new ApiError('server', res.status); }
  return {
    answer: typeof data?.answer === 'string' ? data.answer.trim() : '',
    cited: ids(data?.cited_property_ids),
    grounding: ids(data?.grounding_property_ids),
    properties: Array.isArray(data?.properties) ? data.properties : [],
  };
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
