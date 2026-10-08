// Single place where chatbot configuration is read. No secrets here — ever.
const rc = (typeof window !== 'undefined' && window.APP_CONFIG) || {};

export const CONFIG = Object.freeze({
  apiBase: String(rc.API_BASE_URL ?? '').replace(/\/+$/, ''),
  // Text and voice both go through the agent workflow (same session, same memory).
  chatPath: '/chat',
  resetPath: '/chat/reset',
  voiceTokenPath: '/voice/token',
  healthPath: '/health',
  timeoutMs: Number(rc.REQUEST_TIMEOUT_MS) > 0 ? Number(rc.REQUEST_TIMEOUT_MS) : 60000,
  defaultLang: rc.CHAT_DEFAULT_LANG === 'en' ? 'en' : 'ar',
  voiceEnabled: rc.VOICE_ENABLED !== false,
});
