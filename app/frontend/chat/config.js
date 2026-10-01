// Single place where chatbot configuration is read. No secrets here — ever.
const rc = (typeof window !== 'undefined' && window.APP_CONFIG) || {};

export const CONFIG = Object.freeze({
  apiBase: String(rc.API_BASE_URL ?? '').replace(/\/+$/, ''),
  ragPath: '/rag/search',
  healthPath: '/health',
  timeoutMs: Number(rc.REQUEST_TIMEOUT_MS) > 0 ? Number(rc.REQUEST_TIMEOUT_MS) : 60000,
  defaultLang: rc.CHAT_DEFAULT_LANG === 'en' ? 'en' : 'ar',
  topK: 5,
  retrievalTopK: 20,
  sendHistory: rc.SEND_HISTORY === true,
  maxHistoryTurns: 6,
  // frontend filter name -> backend requirement key. Only mapped keys are sent.
  requirementKeys: Object.freeze({
    property_type: 'property_type',
    district: 'district',
    bedrooms: 'bedrooms',
    ...(rc.REQUIREMENT_KEYS || {}),
  }),
});
