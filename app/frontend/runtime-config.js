// Runtime configuration — loaded as a classic script BEFORE app.js.
// Edit this file per environment (or overwrite it at deploy time). It must never
// contain secrets: everything here is public and readable in the browser.
window.APP_CONFIG = {
  // '' (empty) = same origin. Correct when FastAPI serves this frontend (the default setup).
  // Set e.g. 'https://api.example.com' when the API lives on another origin.
  API_BASE_URL: '',

  // Default language of the chat interface: 'ar' or 'en'. Users can switch in the chat header.
  CHAT_DEFAULT_LANG: 'ar',

  // How long to wait for /rag/search before showing the timeout message (ms).
  REQUEST_TIMEOUT_MS: 60000,

  // Send conversation history to the backend as `history` (the current RAGRequest ignores it).
  // Flip to true once the backend accepts it.
  SEND_HISTORY: false,

  // Extra requirement keys to send to /rag/search. By default only the three keys from the
  // API example are sent (property_type, district, bedrooms). Budget and buy/rent stay in the
  // query text until you map them to the keys build_metadata_filter() actually supports, e.g.:
  // REQUIREMENT_KEYS: { max_budget: 'max_price', listing_type: 'listing_type' },
  REQUIREMENT_KEYS: {},
};
