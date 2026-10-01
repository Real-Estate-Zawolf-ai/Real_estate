// Turns free text (Arabic/English) into structured search filters, and back into readable text.
// This only extracts what the user literally said. The backend still decides what matches.
// Vocabulary mirrors the values used by the site and the API example (Villa, Apartment, Sheikh Zayed…).
// If your index uses other values, edit TYPES / DISTRICTS below.
import { CONFIG } from './config.js';

export function normalize(s) {
  return String(s ?? '')
    .replace(/[٠-٩]/g, (d) => d.charCodeAt(0) - 0x0660)
    .replace(/[۰-۹]/g, (d) => d.charCodeAt(0) - 0x06f0)
    .replace(/(\d),(?=\d{3}(?!\d))/g, '$1')
    .replace(/[\u064B-\u065F\u0640]/g, '')
    .replace(/[أإآ]/g, 'ا').replace(/ى/g, 'ي').replace(/ة/g, 'ه')
    .toLowerCase();
}

export function detectLang(text, fallback = 'en') {
  const ar = (text.match(/[\u0600-\u06FF]/g) || []).length;
  const la = (text.match(/[a-z]/gi) || []).length;
  if (!ar && !la) return fallback;
  return ar >= la ? 'ar' : 'en';
}

const WORDS = {
  one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  'واحد': 1, 'واحده': 1, 'اتنين': 2, 'اثنين': 2, 'تلات': 3, 'تلاته': 3, 'ثلاث': 3, 'ثلاثه': 3,
  'اربع': 4, 'اربعه': 4, 'خمس': 5, 'خمسه': 5, 'ست': 6, 'سته': 6, 'سبع': 7, 'سبعه': 7,
  'تمان': 8, 'تمانيه': 8, 'ثمان': 8, 'ثمانيه': 8, 'تسع': 9, 'تسعه': 9, 'عشر': 10, 'عشره': 10,
};
const WORD_RE = Object.keys(WORDS).sort((a, b) => b.length - a.length).join('|');
const toNum = (s) => (/^\d/.test(s) ? Number(s) : WORDS[s]);

const TYPES = [
  ['Villa', /villa|فيلا|فيلل|فلل|فيلات/],
  ['Apartment', /apartment|\bflat|شقه|شقق/],
  ['Chalet', /chalet|شاليه/],
  ['Commercial', /commercial|office|\bshop|تجاري|مكتب|محل|اداري/],
];
const DISTRICTS = [
  ['Sheikh Zayed', /sheikh\s*zayed|\bzayed|الشيخ\s*زايد|زايد/],
  ['New Cairo', /new\s*cairo|tagamo|tagammu|التجمع|القاهره\s*الجديده/],
  ['North Coast', /north\s*coast|الساحل/],
];

const BED_RE = new RegExp(
  `(?:^|[^a-z0-9])(?:(\\d{1,2})|(${WORD_RE}))\\s*(\\+)?\\s*(?:bed(?:room)?s?|br|غرف|اوض|نوم)(?![a-z])(\\s*(?:او\\s*اكتر|او\\s*اكثر|or\\s*more))?`,
);
const BUDGET_RE = new RegExp(
  `(\\d+(?:\\.\\d+)?|${WORD_RE})\\s*(?:(million|mil|mn|m|مليون|ملايين)|(k|الف|thousand))(?![a-z0-9²])`,
);
const BIG_NUMBER_RE = /(?:^|[^\d.])(\d{6,})(?!\d)/;
const AROUND_RE = /(around|about|approx|approximately|تقريبا|حدود|حوالي|نحو)/;
const OVER_RE = /(above|over|more than|at least|اكتر من|اكثر من|فوق|علي الاقل)\s*$/;
const RENT_RE = /\brent|rental|lease|ايجار|اجار|تاجير/;
const BUY_RE = /\bbuy|\bpurchase|for sale|\bsale\b|شراء|اشتري|للبيع|تمليك|بيع/;

function uniqueMatch(t, table) {
  const hits = table.filter(([, re]) => re.test(t)).map(([v]) => v);
  return hits.length === 1 ? hits[0] : undefined; // ambiguous (e.g. "villa or apartment") -> no filter
}

export function parseFilters(text) {
  const t = normalize(text);
  const out = {};
  const type = uniqueMatch(t, TYPES);
  if (type) out.property_type = type;
  const district = uniqueMatch(t, DISTRICTS);
  if (district) out.district = district;

  if (/غرفتين|اوضتين/.test(t)) out.bedrooms = 2;
  else {
    const b = t.match(BED_RE);
    const n = b && toNum(b[1] ?? b[2]);
    if (n > 0 && n < 20) {
      out.bedrooms = n;
      if (b[3] || b[4]) out.bedrooms_min = true;
    }
  }

  const m = t.match(BUDGET_RE);
  let amount;
  if (m) {
    const n = toNum(m[1]);
    if (n > 0) amount = n * (m[3] ? 1000 : 1e6);
  } else {
    const big = t.match(BIG_NUMBER_RE);
    if (big) amount = Number(big[1]);
  }
  if (amount) {
    const before = t.slice(Math.max(0, (m?.index ?? t.search(BIG_NUMBER_RE)) - 24), m?.index ?? undefined);
    out.budget = amount;
    out.budget_mode = OVER_RE.test(before) ? 'min' : AROUND_RE.test(before) ? 'around' : 'max';
  }

  const rent = RENT_RE.test(t), buy = BUY_RE.test(t);
  if (rent !== buy) out.listing_type = rent ? 'rent' : 'buy';
  return out;
}

export function mergeFilters(prev, next) {
  const out = { ...prev, ...next };
  if ('bedrooms' in next && !('bedrooms_min' in next)) delete out.bedrooms_min;
  return out;
}

export function removeFilter(f, key) {
  const out = { ...f };
  delete out[key];
  if (key === 'bedrooms') delete out.bedrooms_min;
  if (key === 'budget') delete out.budget_mode;
  return out;
}

// Only filters mapped to a backend key in CONFIG.requirementKeys are sent as `requirements`.
export function toRequirements(f) {
  const K = CONFIG.requirementKeys;
  const src = {
    property_type: f.property_type,
    district: f.district,
    bedrooms: f.bedrooms,
    max_budget: f.budget_mode === 'min' ? undefined : f.budget,
    listing_type: f.listing_type,
  };
  const out = {};
  for (const [k, v] of Object.entries(src)) if (v != null && K[k]) out[K[k]] = v;
  return out;
}

const TYPE_LABEL = {
  Villa: { en: 'villa', ar: 'فيلا' }, Apartment: { en: 'apartment', ar: 'شقة' },
  Chalet: { en: 'chalet', ar: 'شاليه' }, Commercial: { en: 'commercial unit', ar: 'وحدة تجارية' },
};
const DISTRICT_LABEL = {
  'Sheikh Zayed': { en: 'Sheikh Zayed', ar: 'الشيخ زايد' },
  'New Cairo': { en: 'New Cairo', ar: 'القاهرة الجديدة' },
  'North Coast': { en: 'North Coast', ar: 'الساحل الشمالي' },
};
const trim = (n) => String(Math.round(n * 100) / 100);
const short = (n, lang) => (n >= 1e6 ? `${trim(n / 1e6)}${lang === 'ar' ? ' مليون' : 'M'}` : `${trim(n / 1e3)}${lang === 'ar' ? ' ألف' : 'K'}`);
const BUDGET_PRE = {
  max: { en: 'Under', ar: 'أقل من' }, around: { en: 'Around', ar: 'حوالي' }, min: { en: 'Over', ar: 'أكتر من' },
};
const bedsLabel = (f, lang) =>
  lang === 'ar'
    ? (f.bedrooms_min ? `${f.bedrooms} غرف أو أكتر` : `${f.bedrooms} غرف`)
    : (f.bedrooms_min ? `${f.bedrooms}+ bedrooms` : `${f.bedrooms} bedroom${f.bedrooms > 1 ? 's' : ''}`);

// Short labels for the removable filter chips.
export function describeFilters(f, lang) {
  const out = [];
  if (f.property_type) out.push({ key: 'property_type', label: TYPE_LABEL[f.property_type]?.[lang] ?? f.property_type });
  if (f.district) out.push({ key: 'district', label: DISTRICT_LABEL[f.district]?.[lang] ?? f.district });
  if (f.bedrooms) out.push({ key: 'bedrooms', label: bedsLabel(f, lang) });
  if (f.budget) out.push({ key: 'budget', label: `${BUDGET_PRE[f.budget_mode || 'max'][lang]} ${short(f.budget, lang)}` });
  if (f.listing_type) out.push({ key: 'listing_type', label: f.listing_type === 'rent' ? (lang === 'ar' ? 'إيجار' : 'Rent') : (lang === 'ar' ? 'شراء' : 'Buy') });
  return out;
}

// A readable sentence used to carry earlier filters into the (stateless) backend query.
export function humanize(f, lang) {
  const ar = lang === 'ar';
  const type = f.property_type ? TYPE_LABEL[f.property_type]?.[lang] ?? f.property_type : ar ? 'عقار' : 'property';
  const listing = f.listing_type === 'rent' ? (ar ? 'للإيجار' : 'for rent') : f.listing_type === 'buy' ? (ar ? 'للبيع' : 'for sale') : '';
  const where = f.district ? `${ar ? 'في' : 'in'} ${DISTRICT_LABEL[f.district]?.[lang] ?? f.district}` : '';
  const beds = f.bedrooms ? (ar ? bedsLabel(f, lang).replace('غرف', 'غرف نوم') : bedsLabel(f, lang)) : '';
  let budget = '';
  if (f.budget) {
    const amt = f.budget >= 1e6 ? `${trim(f.budget / 1e6)} ${ar ? 'مليون' : 'million'}` : `${trim(f.budget / 1e3)} ${ar ? 'ألف' : 'thousand'}`;
    budget = `${BUDGET_PRE[f.budget_mode || 'max'][lang].toLowerCase()} ${amt} ${ar ? 'جنيه' : 'EGP'}`;
  }
  const head = [type, listing, where].filter(Boolean).join(' ');
  const rest = [beds, budget].filter(Boolean);
  const hasAny = Object.keys(f).some((k) => !k.endsWith('_mode') && !k.endsWith('_min'));
  return hasAny ? [head, ...rest].join(ar ? '، ' : ', ') : '';
}

const REFINE = {
  property_type: [{ en: 'Villas', ar: 'فلل' }, { en: 'Apartments', ar: 'شقق' }],
  district: [{ en: 'Sheikh Zayed', ar: 'الشيخ زايد' }, { en: 'New Cairo', ar: 'التجمع' }],
  bedrooms: [{ en: '2 bedrooms', ar: 'غرفتين' }, { en: '3 bedrooms', ar: '3 غرف' }, { en: '4+ bedrooms', ar: '4 غرف أو أكتر' }],
  budget: [{ en: 'Under 5M', ar: 'أقل من 5 مليون' }, { en: 'Under 10M', ar: 'أقل من 10 مليون' }],
  listing_type: [{ en: 'Buy', ar: 'شراء' }, { en: 'Rent', ar: 'إيجار' }],
};
const DIM_ORDER = ['property_type', 'district', 'bedrooms', 'budget', 'listing_type'];

// Contextual suggestion chips for whatever the user hasn't specified yet. Sending one sends its label as a message.
export function refineChips(lang, f, limit = 6) {
  const chips = [];
  for (const dim of DIM_ORDER) {
    if (f[dim] != null) continue;
    for (const c of REFINE[dim]) chips.push({ label: c[lang], text: c[lang] });
  }
  return chips.slice(0, limit);
}

export const typeChips = (lang) =>
  Object.values(TYPE_LABEL).map((l) => ({ label: l[lang], text: l[lang] }));

export const GREETING = /^(hi|hello|hey|good (morning|evening)|مرحبا|اهلا|السلام عليكم|سلام|صباح|مساء|ازيك|هاي)/;
