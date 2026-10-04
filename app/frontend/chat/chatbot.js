// Atelier property assistant — UI + conversation state.
// Builds DOM with textContent only (never innerHTML with data), so backend text cannot inject markup.
import { CONFIG } from './config.js';
import { t, quickActions } from './i18n.js';
import { searchProperties, checkHealth, buildPayload, ApiError } from './api.js';
import {
  parseFilters, mergeFilters, removeFilter, toRequirements, humanize, describeFilters,
  refineChips, typeChips, detectLang, normalize, GREETING,
} from './requirements.js';
import { normalizeProperty, formatPrice, formatNumber } from './normalize.js';

const STORE_KEY = 'atelier-chat-v1';
const MAX_STORED = 40;
const MAX_COMPARE = 3;

const ICON = {
  chat: '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v7a2.5 2.5 0 0 1-2.5 2.5H11l-4.2 3.4c-.5.4-1.3.1-1.3-.6V16A2.5 2.5 0 0 1 4 13.5z"/><path d="M12 7.6l.9 1.9 1.9.9-1.9.9-.9 1.9-.9-1.9-1.9-.9 1.9-.9z" fill="currentColor" stroke="none"/></svg>',
  close: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  min: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M6 12h12"/></svg>',
  expand: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 14l6-6 6 6"/></svg>',
  send: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5M6 11l6-6 6 6"/></svg>',
};

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'html') el.innerHTML = v; // static icon strings only
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid);
  return el;
}

const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const isMobile = () => window.matchMedia('(max-width: 640px)').matches;

function loadSaved() {
  try {
    const s = JSON.parse(sessionStorage.getItem(STORE_KEY) || 'null');
    if (s && Array.isArray(s.messages)) return s;
  } catch { /* storage unavailable or corrupt: start fresh */ }
  return {};
}

export function initChatbot({ resolvePropertyHref = () => null, appRoot = null } = {}) {
  if (document.getElementById('chatbot-root')) return null;

  const saved = loadSaved();
  const state = {
    mode: 'closed', // 'closed' | 'open' | 'min'
    lang: saved.lang === 'en' || saved.lang === 'ar' ? saved.lang : CONFIG.defaultLang,
    messages: saved.messages || [],
    filters: saved.filters || {},
    loading: false,
    health: 'checking',
    healthChecked: false,
    unread: false,
    compare: [],
  };
  const catalog = new Map();
  for (const m of state.messages) if (m.kind === 'results') m.properties.forEach((p) => catalog.set(p.id, p));

  // ---------- DOM skeleton ----------
  const launcherIcon = h('span', { class: 'cb-launcher-icon' });
  const launcherText = h('span', { class: 'cb-launcher-text' });
  const launcher = h('button', { type: 'button', class: 'cb-launcher', 'aria-controls': 'cb-panel', 'aria-expanded': 'false' }, launcherIcon, launcherText);

  const statusDot = h('span', { class: 'cb-dot', 'aria-hidden': 'true' });
  const statusText = h('span', { class: 'cb-status' });
  const titleEl = h('h2', { class: 'cb-title', id: 'cb-title' });
  const langBtn = h('button', { type: 'button', class: 'cb-iconbtn cb-lang' });
  const minBtn = h('button', { type: 'button', class: 'cb-iconbtn' });
  const closeBtn = h('button', { type: 'button', class: 'cb-iconbtn', html: ICON.close });
  const headMain = h('div', { class: 'cb-head-main' }, h('span', { class: 'cb-avatar', 'aria-hidden': 'true', html: ICON.chat }), h('div', { class: 'cb-head-text' }, titleEl, h('p', { class: 'cb-status-row' }, statusDot, statusText)));
  const header = h('header', { class: 'cb-head' }, headMain, h('div', { class: 'cb-head-actions' }, langBtn, minBtn, closeBtn));

  const log = h('div', { class: 'cb-log', role: 'log', 'aria-live': 'polite', 'aria-relevant': 'additions', tabindex: '0' });
  const filtersBar = h('div', { class: 'cb-filters', role: 'group', hidden: true });
  const compareInfo = h('span', { class: 'cb-compare-info' });
  const compareGo = h('button', { type: 'button', class: 'cb-btn cb-btn--primary cb-btn--sm' });
  const compareClear = h('button', { type: 'button', class: 'cb-btn cb-btn--sm' });
  const compareBar = h('div', { class: 'cb-comparebar', hidden: true }, compareInfo, h('span', { class: 'cb-compare-actions' }, compareClear, compareGo));
  const ta = h('textarea', { class: 'cb-input', rows: '1', maxlength: '500', dir: 'auto', autocomplete: 'off', enterkeyhint: 'send' });
  const sendBtn = h('button', { type: 'submit', class: 'cb-send', html: ICON.send, disabled: true });
  const form = h('form', { class: 'cb-form' }, ta, sendBtn);
  const panel = h('section', { class: 'cb-panel', id: 'cb-panel', role: 'dialog', 'aria-labelledby': 'cb-title', hidden: true }, header, log, filtersBar, compareBar, form);
  const root = h('div', { class: 'cb', id: 'chatbot-root', 'data-mode': 'closed' }, launcher, panel);
  document.body.append(root);

  // ---------- chrome (language-dependent static text) ----------
  function renderStatus() {
    const k = state.health === 'up' ? 'statusUp' : state.health === 'down' ? 'statusDown' : 'statusChecking';
    statusText.textContent = t(state.lang, k);
    statusDot.dataset.state = state.health;
  }
  function renderLauncher() {
    const open = state.mode === 'open';
    launcherIcon.innerHTML = open && !isMobile() ? ICON.close : ICON.chat;
    launcherText.textContent = t(state.lang, 'launcherLabel');
    launcher.setAttribute('aria-label', open ? t(state.lang, 'close') : t(state.lang, 'launcher'));
    launcher.setAttribute('aria-expanded', String(open));
    launcher.toggleAttribute('data-unread', state.unread && state.mode !== 'open');
  }
  function renderChrome() {
    const l = state.lang;
    panel.lang = l;
    panel.dir = l === 'ar' ? 'rtl' : 'ltr';
    titleEl.textContent = t(l, 'name');
    langBtn.textContent = t(l, 'langBtn');
    langBtn.setAttribute('aria-label', t(l, 'langToggle'));
    langBtn.setAttribute('lang', l === 'ar' ? 'en' : 'ar');
    const min = state.mode === 'min';
    minBtn.innerHTML = min ? ICON.expand : ICON.min;
    minBtn.setAttribute('aria-label', t(l, min ? 'expand' : 'minimize'));
    closeBtn.setAttribute('aria-label', t(l, 'close'));
    ta.placeholder = t(l, 'placeholder');
    ta.setAttribute('aria-label', t(l, 'inputLabel'));
    sendBtn.setAttribute('aria-label', t(l, 'send'));
    compareGo.textContent = t(l, 'compareGo');
    compareClear.textContent = t(l, 'compareClear');
    filtersBar.setAttribute('aria-label', t(l, 'filtersLabel'));
    renderStatus();
    renderLauncher();
    renderFilters();
    syncCompareUI();
  }

  // ---------- mode / focus / page lock ----------
  function lockPage(on) {
    document.documentElement.classList.toggle('cb-lock', on);
    if (appRoot) on ? appRoot.setAttribute('inert', '') : appRoot.removeAttribute('inert');
  }
  function setMode(mode, { focus = true } = {}) {
    const prev = state.mode;
    state.mode = mode;
    root.dataset.mode = mode;
    panel.hidden = mode === 'closed';
    if (mode === 'open') state.unread = false;
    lockPage(mode === 'open' && isMobile());
    renderChrome();
    if (mode === 'open') {
      ensureHealth();
      scrollLog(true);
      if (focus) ta.focus({ preventScroll: true });
    } else if (focus && prev !== mode) {
      (mode === 'closed' ? launcher : minBtn).focus({ preventScroll: true });
    }
  }
  function syncViewport() {
    const vv = window.visualViewport;
    if (!vv) return;
    root.style.setProperty('--cb-vh', `${vv.height}px`);
    root.style.setProperty('--cb-top', `${vv.offsetTop}px`);
  }
  window.visualViewport?.addEventListener('resize', syncViewport);
  window.visualViewport?.addEventListener('scroll', syncViewport);
  window.matchMedia('(max-width: 640px)').addEventListener?.('change', () => { lockPage(state.mode === 'open' && isMobile()); renderLauncher(); });
  syncViewport();

  async function ensureHealth() {
    if (state.healthChecked) return;
    state.healthChecked = true;
    renderStatus();
    state.health = (await checkHealth()) ? 'up' : 'down';
    renderStatus();
  }

  // ---------- persistence ----------
  function persist() {
    try {
      sessionStorage.setItem(STORE_KEY, JSON.stringify({ lang: state.lang, filters: state.filters, messages: state.messages.slice(-MAX_STORED) }));
    } catch { /* ignore quota / private mode */ }
  }
  const toHistory = () =>
    state.messages
      .filter((m) => m.kind === 'text' || m.kind === 'results' || m.role === 'user')
      .map((m) => ({ role: m.role, content: m.role === 'user' ? m.text : m.answer || m.text || '' }))
      .filter((m) => m.content);

  // ---------- scrolling ----------
  function scrollLog(toEnd, el) {
    requestAnimationFrame(() => {
      const top = toEnd || !el ? log.scrollHeight : Math.max(0, el.offsetTop - 8);
      log.scrollTo({ top, behavior: reducedMotion() ? 'auto' : 'smooth' });
    });
  }

  // ---------- answer text (paragraphs, bullets, **bold**, clickable property ids) ----------
  function formatAnswer(text, ids) {
    const frag = document.createDocumentFragment();
    const idAlt = [...new Set(ids.filter((i) => i && i.length >= 3))].sort((a, b) => b.length - a.length).map(escRe).join('|');
    const re = new RegExp(`\\*\\*(.+?)\\*\\*${idAlt ? `|(^|[^\\w-])(${idAlt})(?![\\w-])` : ''}`, 'g');
    const inline = (line) => {
      const nodes = [];
      let last = 0;
      for (const m of line.matchAll(re)) {
        if (m[1] !== undefined) {
          if (m.index > last) nodes.push(line.slice(last, m.index));
          nodes.push(h('strong', { text: m[1] }));
        } else {
          const start = m.index + m[2].length;
          if (start > last) nodes.push(line.slice(last, start));
          nodes.push(h('button', { type: 'button', class: 'cb-idref', 'data-goto': m[3], text: m[3] }));
        }
        last = m.index + m[0].length;
      }
      if (last < line.length) nodes.push(line.slice(last));
      return nodes;
    };
    let list = null;
    for (const raw of text.replace(/\r/g, '').split('\n')) {
      const line = raw.trim().replace(/^#{1,6}\s+/, '');
      if (!line) { list = null; continue; }
      const li = line.match(/^(?:[-*•]|\d+[.)])\s+(.*)$/);
      if (li) {
        if (!list) { list = h('ul', { class: 'cb-list' }); frag.append(list); }
        list.append(h('li', {}, inline(li[1])));
      } else {
        list = null;
        frag.append(h('p', {}, inline(line)));
      }
    }
    return frag;
  }

  // ---------- property card ----------
  function propertyCard(p, lang, showCompare) {
    const internal = resolvePropertyHref(p.id, p);
    const href = internal || p.url;
    const heading = p.title || [p.type, p.location].filter(Boolean).join(' — ') || `${t(lang, 'id')} ${p.id}`;
    const facts = [
      p.bedrooms && t(lang, p.bedrooms === 1 ? 'beds' : 'bedsN', { n: formatNumber(p.bedrooms, lang) }),
      p.bathrooms && t(lang, p.bathrooms === 1 ? 'baths' : 'bathsN', { n: formatNumber(p.bathrooms, lang) }),
      p.area != null && t(lang, 'sqm', { n: formatNumber(p.area, lang) }),
    ].filter(Boolean);
    const q = `property=${encodeURIComponent(p.id)}`;
    const more = p.description || p.source || p.id;
    const card = h('article', { class: `cb-prop${p.image ? ' has-img' : ''}`, 'data-pid': p.id, tabindex: '-1' },
      p.image && h('img', {
        class: 'cb-prop-img', src: p.image, alt: '', loading: 'lazy', decoding: 'async', referrerpolicy: 'no-referrer',
        onerror: (e) => { e.target.closest('.cb-prop')?.classList.remove('has-img'); e.target.remove(); },
      }),
      h('div', { class: 'cb-prop-main' },
        (p.type || p.inAnswer) && h('div', { class: 'cb-prop-tags' },
          p.type && p.type !== heading && h('span', { class: 'cb-tag', text: p.type }),
          p.inAnswer && h('span', { class: 'cb-tag cb-tag--ref', text: t(lang, 'inAnswer') })),
        h('h3', { class: 'cb-prop-title', dir: 'auto', text: heading }),
        p.location && p.title && h('p', { class: 'cb-prop-loc', dir: 'auto', text: p.location }),
        p.price != null && h('p', { class: 'cb-prop-price', text: formatPrice(p.price, p.currency, lang) }),
        facts.length > 0 && h('ul', { class: 'cb-facts' }, facts.map((f) => h('li', { text: f }))),
        more && h('details', { class: 'cb-more' },
          h('summary', { text: t(lang, 'details') }),
          p.description && h('p', { class: 'cb-more-text', dir: 'auto', text: p.description }),
          h('p', { class: 'cb-more-meta' }, `${t(lang, 'id')}: `, h('bdi', { text: p.id }), p.source && ` · ${t(lang, 'source')}: `, p.source && h('bdi', { text: p.source }))),
        h('div', { class: 'cb-prop-actions' },
          href && h('a', { class: 'cb-btn cb-btn--primary', href, ...(internal ? { 'data-nav': '' } : { target: '_blank', rel: 'noopener noreferrer' }), text: t(lang, 'view') }),
          h('a', { class: 'cb-btn', href: `#/contact?${q}&intent=contact`, 'data-nav': '', text: t(lang, 'contact') }),
          h('a', { class: 'cb-btn', href: `#/contact?${q}&intent=viewing`, 'data-nav': '', text: t(lang, 'viewing') }),
          showCompare && h('button', { type: 'button', class: 'cb-btn cb-btn--toggle', 'data-compare-id': p.id, 'aria-pressed': String(state.compare.includes(p.id)), text: t(lang, 'compare') }))));
    return card;
  }

  // ---------- comparison table (only rows with data; missing cells show an em dash) ----------
  function compareTable(items, lang) {
    const rows = [
      ['rowType', (p) => p.type],
      ['rowLocation', (p) => p.location],
      ['rowPrice', (p) => (p.price != null ? formatPrice(p.price, p.currency, lang) : undefined)],
      ['rowBedrooms', (p) => p.bedrooms && formatNumber(p.bedrooms, lang)],
      ['rowBathrooms', (p) => p.bathrooms && formatNumber(p.bathrooms, lang)],
      ['rowArea', (p) => (p.area != null ? t(lang, 'sqm', { n: formatNumber(p.area, lang) }) : undefined)],
    ].filter(([, get]) => items.some((p) => get(p)));
    return h('div', { class: 'cb-table-wrap' },
      h('table', { class: 'cb-table' },
        h('thead', {}, h('tr', {}, h('td', {}), items.map((p) => h('th', { scope: 'col' }, p.title || p.type || p.id, h('small', {}, h('bdi', { text: p.id })))))),
        h('tbody', {}, rows.map(([label, get]) => h('tr', {}, h('th', { scope: 'row', text: t(lang, label) }), items.map((p) => h('td', { text: get(p) || '—' })))))));
  }

  // ---------- messages ----------
  const bubble = (text, extra = '') => h('div', { class: `cb-bubble ${extra}`.trim(), dir: 'auto', text });
  function chipsEl(m, lang) {
    if (!m.chips?.length) return null;
    return h('div', { class: 'cb-chips' },
      m.hint && h('p', { class: 'cb-hint', text: m.hint }),
      h('div', { class: 'cb-chip-row' }, m.chips.map((c) => h('button', {
        type: 'button', class: 'cb-chip', text: c.label, dir: 'auto',
        onclick: () => (c.remove ? handleSend(c.text, { filtersOverride: removeFilter(state.filters, c.remove) })
          : c.action === 'guide' ? handleSend(c.label, { action: 'guide' }) : handleSend(c.text)),
      }))));
  }

  function buildMessage(m) {
    const lang = m.lang || state.lang;
    const wrap = h('div', { class: `cb-msg cb-msg--${m.role}`, 'data-id': m.id });
    if (m.role === 'user') { wrap.append(bubble(m.text)); return wrap; }
    const stack = h('div', { class: 'cb-stack' });
    wrap.append(stack);
    if (m.kind === 'welcome') {
      stack.append(bubble(t(state.lang, 'welcome')));
      if (m.showChips) stack.append(chipsEl({ chips: quickActions(state.lang) }, state.lang));
    } else if (m.kind === 'text') {
      stack.append(bubble(m.text), chipsEl(m, lang));
    } else if (m.kind === 'error') {
      const key = { timeout: 'errTimeout', invalid: 'errInvalid', network: 'errNetwork' }[m.errorKind] || 'errServer';
      stack.append(h('div', { class: 'cb-bubble cb-bubble--error', role: 'alert', dir: 'auto' }, h('p', { text: t(lang, key) }),
        m.req && h('button', { type: 'button', class: 'cb-btn cb-btn--sm', text: t(lang, 'retry'), onclick: () => retry(m) })));
    } else if (m.kind === 'compare') {
      stack.append(h('div', { class: 'cb-bubble cb-bubble--wide' }, h('p', { class: 'cb-compare-title', text: t(lang, 'compareTitle') }), compareTable(m.items, lang)));
    } else if (m.kind === 'results') {
      const ids = m.properties.map((p) => p.id);
      if (m.answer) stack.append(h('div', { class: 'cb-bubble cb-answer', dir: 'auto' }, formatAnswer(m.answer, ids)));
      else stack.append(bubble(t(lang, m.properties.length ? 'foundFallback' : 'empty')));
      const cited = m.properties.filter((p) => p.inAnswer);
      const others = m.properties.filter((p) => !p.inAnswer);
      const show = m.properties.length > 1;
      const group = (title, list) => {
        if (!list.length) return;
        if (title) stack.append(h('p', { class: 'cb-group', text: title }));
        list.forEach((p) => stack.append(propertyCard(p, lang, show)));
      };
      const both = cited.length && others.length;
      group(both ? t(lang, 'groupCited') : '', cited);
      group(both ? t(lang, 'groupOthers') : '', others);
      stack.append(chipsEl(m, lang));
    }
    return wrap;
  }

  function renderLog() {
    log.replaceChildren(...state.messages.map(buildMessage));
    scrollLog(true);
  }

  function pushMessage(m) {
    m.id = `m${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
    for (const prev of state.messages) { delete prev.chips; delete prev.hint; if (prev.kind === 'welcome') prev.showChips = false; }
    log.querySelectorAll('.cb-chips').forEach((n) => n.remove());
    state.messages.push(m);
    if (m.kind === 'results') m.properties.forEach((p) => catalog.set(p.id, p));
    const el = buildMessage(m);
    log.append(el);
    persist();
    return el;
  }

  // ---------- filters bar ----------
  function renderFilters() {
    const items = describeFilters(state.filters, state.lang);
    const started = state.messages.some((m) => m.role === 'user');
    filtersBar.hidden = !items.length && !started;
    filtersBar.replaceChildren(
      started && h('button', { type: 'button', class: 'cb-link', text: t(state.lang, 'newSearch'), onclick: newSearch }),
      ...items.map((f) => h('button', {
        type: 'button', class: 'cb-chip cb-chip--filter', 'aria-label': t(state.lang, 'removeFilter', { x: f.label }),
        onclick: () => !state.loading && handleSend(t(state.lang, 'removed', { x: f.label }), { filtersOverride: removeFilter(state.filters, f.key) }),
      }, h('span', { text: f.label }), h('span', { class: 'cb-x', 'aria-hidden': 'true', text: '×' }))),
    );
  }

  // ---------- compare ----------
  function syncCompareUI() {
    log.querySelectorAll('[data-compare-id]').forEach((b) => b.setAttribute('aria-pressed', String(state.compare.includes(b.dataset.compareId))));
    compareBar.hidden = state.compare.length === 0;
    compareInfo.textContent = t(state.lang, 'selected', { n: formatNumber(state.compare.length, state.lang) });
    compareGo.disabled = state.compare.length < 2;
  }
  function toggleCompare(id) {
    const i = state.compare.indexOf(id);
    if (i >= 0) state.compare.splice(i, 1);
    else if (state.compare.length >= MAX_COMPARE) { compareInfo.textContent = t(state.lang, 'compareMax'); return; }
    else state.compare.push(id);
    syncCompareUI();
  }
  function runCompare() {
    const items = state.compare.map((id) => catalog.get(id)).filter(Boolean).map((p) => ({ ...p }));
    if (items.length < 2) return;
    state.compare = [];
    const el = pushMessage({ role: 'assistant', kind: 'compare', lang: state.lang, items });
    syncCompareUI();
    scrollLog(false, el);
  }

  // ---------- conversation flow ----------
  function pushWelcome() { state.messages.push({ id: 'welcome', role: 'assistant', kind: 'welcome', showChips: true }); }
  function newSearch() {
    if (state.loading) return;
    state.filters = {};
    state.messages = [];
    state.compare = [];
    catalog.clear();
    pushWelcome();
    renderLog();
    renderFilters();
    syncCompareUI();
    persist();
  }

  const showLoader = (lang) => {
    const el = h('div', { class: 'cb-msg cb-msg--assistant cb-msg--loading' },
      h('div', { class: 'cb-bubble', role: 'status' }, h('span', { class: 'cb-dots', 'aria-hidden': 'true' }, h('i'), h('i'), h('i')), h('span', { text: t(lang, 'searching') })));
    log.append(el);
    scrollLog(true);
    return el;
  };

  function buildChips(lang, filters, empty) {
    if (empty) {
      const active = describeFilters(filters, lang);
      if (active.length) return { hint: t(lang, 'tryRelax'), chips: active.map((f) => ({ label: t(lang, 'without', { x: f.label }), text: t(lang, 'removed', { x: f.label }), remove: f.key })) };
    }
    const chips = refineChips(lang, filters);
    return chips.length ? { hint: t(lang, 'refineHint'), chips } : {};
  }

  async function runSearch(req) {
    state.loading = true;
    sendBtn.disabled = true;
    log.setAttribute('aria-busy', 'true');
    const loader = showLoader(req.lang);
    let el;
    try {
      // The current turn is sent in `query`; history only contains earlier turns.
      const payload = buildPayload({ query: req.query, requirements: req.requirements, history: toHistory().slice(0, -1) });
      const data = await searchProperties(payload);
      loader.remove();
      state.health = 'up';
      renderStatus();
      if (data.response_type === 'conversational') {
        el = pushMessage({ role: 'assistant', kind: 'text', lang: req.lang, text: data.answer || t(req.lang, 'greeting') });
        scrollLog(false, el);
        return;
      }
      // cited properties first, then other grounding properties, then the rest in backend (reranked) order
      const rank = (p) => (data.cited.includes(p.id) ? 0 : data.grounding.includes(p.id) ? 1 : 2);
      const props = data.properties.map(normalizeProperty).filter((p) => p.id)
        .map((p, i) => ({ p, i })).sort((a, b) => rank(a.p) - rank(b.p) || a.i - b.i).map(({ p }) => ({ ...p, inAnswer: data.cited.includes(p.id) }));
      el = pushMessage({ role: 'assistant', kind: 'results', lang: req.lang, answer: data.answer, properties: props, cited: data.cited, grounding: data.grounding, ...buildChips(req.lang, state.filters, props.length === 0) });
    } catch (err) {
      loader.remove();
      const kind = err instanceof ApiError ? err.kind : 'server';
      if (kind === 'network' || kind === 'server') { state.health = 'down'; renderStatus(); }
      el = pushMessage({ role: 'assistant', kind: 'error', lang: req.lang, errorKind: kind, req });
    } finally {
      state.loading = false;
      log.removeAttribute('aria-busy');
      sendBtn.disabled = !ta.value.trim();
      if (state.mode !== 'open') { state.unread = true; renderLauncher(); }
    }
    scrollLog(false, el);
  }

  function retry(m) {
    if (state.loading) return;
    state.messages = state.messages.filter((x) => x !== m);
    log.querySelector(`[data-id="${m.id}"]`)?.remove();
    runSearch(m.req);
  }

  async function handleSend(rawText, opts = {}) {
    const text = rawText.trim();
    if (!text || state.loading) return;
    if (state.mode !== 'open') setMode('open', { focus: false });
    const lang = detectLang(text, state.lang);
    pushMessage({ role: 'user', kind: 'text', text });
    scrollLog(true);
    renderFilters();

    if (opts.action === 'guide') {
      pushMessage({ role: 'assistant', kind: 'text', lang, text: t(lang, 'guide'), chips: typeChips(lang) });
      scrollLog(true);
      return;
    }
    const parsed = opts.filtersOverride ? {} : parseFilters(text);
    if (!opts.filtersOverride && !Object.keys(parsed).length && text.length <= 30 && GREETING.test(normalize(text))) {
      pushMessage({ role: 'assistant', kind: 'text', lang, text: t(lang, 'greeting'), chips: quickActions(lang) });
      scrollLog(true);
      return;
    }
    const hasCurrentFilters = Object.keys(parsed).length > 0;
    const merged = opts.filtersOverride ?? (hasCurrentFilters ? mergeFilters(state.filters, parsed) : state.filters);
    if (opts.filtersOverride || hasCurrentFilters) state.filters = merged;
    renderFilters();
    const query = opts.filtersOverride
      ? humanize(merged, lang) || t(lang, 'browseAll')
      : text;
    const isPropertyFilterTurn = Boolean(opts.filtersOverride) || hasCurrentFilters;
    await runSearch({ query, requirements: isPropertyFilterTurn ? toRequirements(merged) : {}, lang });
  }

  // ---------- events ----------
  launcher.addEventListener('click', () => setMode(state.mode === 'open' ? 'closed' : 'open'));
  closeBtn.addEventListener('click', () => setMode('closed'));
  minBtn.addEventListener('click', () => setMode(state.mode === 'min' ? 'open' : 'min'));
  headMain.addEventListener('click', () => state.mode === 'min' && setMode('open'));
  langBtn.addEventListener('click', () => {
    state.lang = state.lang === 'ar' ? 'en' : 'ar';
    renderChrome();
    renderLog();
    persist();
  });
  panel.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.preventDefault(); setMode('closed'); }
  });
  compareGo.addEventListener('click', runCompare);
  compareClear.addEventListener('click', () => { state.compare = []; syncCompareUI(); });

  log.addEventListener('click', (e) => {
    const cmp = e.target.closest('[data-compare-id]');
    if (cmp) return toggleCompare(cmp.dataset.compareId);
    const ref = e.target.closest('[data-goto]');
    if (ref) {
      const card = ref.closest('.cb-msg')?.querySelector(`[data-pid="${CSS.escape(ref.dataset.goto)}"]`);
      if (card) {
        card.scrollIntoView({ block: 'nearest', behavior: reducedMotion() ? 'auto' : 'smooth' });
        card.focus({ preventScroll: true });
        card.classList.add('is-flash');
        setTimeout(() => card.classList.remove('is-flash'), 1500);
      }
      return;
    }
    if (e.target.closest('a[data-nav]') && isMobile()) setMode('min', { focus: false }); // reveal the page on small screens
  });

  const autosize = () => { ta.style.height = 'auto'; ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`; };
  const submit = () => {
    const v = ta.value;
    if (!v.trim() || state.loading) return;
    ta.value = '';
    autosize();
    sendBtn.disabled = true;
    handleSend(v);
  };
  ta.addEventListener('input', () => { autosize(); sendBtn.disabled = state.loading || !ta.value.trim(); });
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
  });
  form.addEventListener('submit', (e) => { e.preventDefault(); submit(); });

  // Any element with data-chat-open opens the assistant (optionally sending data-chat-prompt). Survives page re-renders.
  document.addEventListener('click', (e) => {
    const trigger = e.target.closest?.('[data-chat-open]');
    if (!trigger) return;
    e.preventDefault();
    setMode('open');
    const prompt = trigger.getAttribute('data-chat-prompt');
    if (prompt) handleSend(prompt);
  });

  // ---------- boot ----------
  if (!state.messages.length) pushWelcome();
  renderChrome();
  renderLog();

  return { open: () => setMode('open'), close: () => setMode('closed'), send: handleSend };
}
