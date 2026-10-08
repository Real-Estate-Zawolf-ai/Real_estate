// Filter chips for the requirements the backend collected (Arabic/English).
// Extraction itself happens server-side (app/agents/understanding.py) so text and voice
// share one source of truth; this module only maps server requirements to chip labels.

export function detectLang(text, fallback = 'en') {
  const ar = (text.match(/[\u0600-\u06FF]/g) || []).length;
  const la = (text.match(/[a-z]/gi) || []).length;
  if (!ar && !la) return fallback;
  return ar >= la ? 'ar' : 'en';
}

// Server requirement keys (canonical index vocabulary) -> chip filter model.
export function fromServerRequirements(r = {}) {
  const f = {};
  if (r.property_type) f.property_type = r.property_type;
  if (r.district) f.district = r.district;
  if (r.bedrooms || r.min_bedrooms) { f.bedrooms = r.bedrooms || r.min_bedrooms; if (r.min_bedrooms) f.bedrooms_min = true; }
  if (r.max_price) { f.budget = r.max_price; f.budget_mode = 'max'; }
  else if (r.min_price) { f.budget = r.min_price; f.budget_mode = 'min'; }
  if (r.transaction_type === 'rent') f.listing_type = 'rent';
  else if (r.transaction_type === 'sale') f.listing_type = 'buy';
  return f;
}

// Chip key -> server requirement keys to clear when the chip is removed.
export const SERVER_KEYS = {
  property_type: ['property_type'],
  district: ['district'],
  bedrooms: ['bedrooms', 'min_bedrooms'],
  budget: ['max_price', 'min_price'],
  listing_type: ['transaction_type'],
};

const TYPE_LABEL = {
  Villa: { en: 'villa', ar: 'فيلا' }, Apartment: { en: 'apartment', ar: 'شقة' },
  Chalet: { en: 'chalet', ar: 'شاليه' }, Commercial: { en: 'commercial unit', ar: 'وحدة تجارية' },
  Studio: { en: 'studio', ar: 'استوديو' }, Duplex: { en: 'duplex', ar: 'دوبلكس' }, Penthouse: { en: 'penthouse', ar: 'بنتهاوس' },
  Townhouse: { en: 'townhouse', ar: 'تاون هاوس' }, 'Twin House': { en: 'twin house', ar: 'توين هاوس' },
  Office: { en: 'office', ar: 'مكتب' }, Shop: { en: 'shop', ar: 'محل' },
};
const DISTRICT_LABEL = {
  'Sheikh Zayed': { en: 'Sheikh Zayed', ar: 'الشيخ زايد' },
  'New Cairo': { en: 'New Cairo', ar: 'القاهرة الجديدة' },
  'North Coast': { en: 'North Coast', ar: 'الساحل الشمالي' },
  '6th of October': { en: '6th of October', ar: '6 أكتوبر' },
  'New Capital': { en: 'New Capital', ar: 'العاصمة الإدارية' },
  Maadi: { en: 'Maadi', ar: 'المعادي' }, 'Nasr City': { en: 'Nasr City', ar: 'مدينة نصر' },
  'Mostakbal City': { en: 'Mostakbal City', ar: 'مدينة المستقبل' }, 'Ain Sokhna': { en: 'Ain Sokhna', ar: 'العين السخنة' },
  Heliopolis: { en: 'Heliopolis', ar: 'مصر الجديدة' },
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

