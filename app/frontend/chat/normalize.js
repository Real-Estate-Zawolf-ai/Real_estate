// Turns a backend property ({property_id, text, metadata, source_file, score}) into a card view-model.
// Rule: only fields that actually exist are returned. Nothing is invented or defaulted.
// The metadata key names below are aliases for the same concept; add yours if the index uses others.

const first = (obj, keys) => {
  for (const k of keys) {
    const v = obj?.[k];
    if (v !== undefined && v !== null && String(v).trim() !== '') return v;
  }
  return undefined;
};
const num = (v) => {
  if (typeof v === 'number') return Number.isFinite(v) ? v : undefined;
  if (typeof v === 'string') {
    const m = v.replace(/,/g, '').match(/^\s*(\d+(?:\.\d+)?)\s*(?:m2|m²|sqm|sq\.?\s*m)?\s*$/i);
    if (m) return Number(m[1]);
  }
  return undefined;
};
const str = (v) => (v === undefined ? undefined : String(v).trim());
// Only http(s) and same-site absolute paths. Blocks javascript:, data:, protocol-relative URLs.
const safeUrl = (v) => {
  if (typeof v !== 'string') return undefined;
  const s = v.trim();
  return /^(https?:\/\/|\/(?!\/))/i.test(s) ? s : undefined;
};
const truncate = (s, n) => (s.length > n ? s.slice(0, n).trimEnd() + '…' : s);

export function normalizeProperty(p) {
  const meta = p?.metadata && typeof p.metadata === 'object' ? p.metadata : {};
  const district = str(first(meta, ['district', 'neighborhood']));
  const city = str(first(meta, ['city', 'governorate', 'region']));
  const place = [district, city].filter((x, i, a) => x && a.indexOf(x) === i).join(', ')
    || str(first(meta, ['location', 'address']));
  const images = Array.isArray(meta.images) ? meta.images[0] : undefined;
  const desc = str(first(meta, ['description', 'summary']));
  const text = typeof p?.text === 'string' ? p.text.trim() : '';
  const bedrooms = num(first(meta, ['bedrooms', 'beds', 'bedroom_count']));
  const bathrooms = num(first(meta, ['bathrooms', 'baths', 'bathroom_count']));
  return {
    id: String(p?.property_id ?? ''),
    title: str(first(meta, ['title', 'name', 'project', 'project_name', 'compound'])),
    type: str(first(meta, ['property_type', 'type', 'unit_type'])),
    location: place || undefined,
    bedrooms: bedrooms > 0 ? bedrooms : undefined,
    bathrooms: bathrooms > 0 ? bathrooms : undefined,
    area: num(first(meta, ['area_sqm', 'area', 'size_sqm', 'size', 'built_area'])),
    price: num(first(meta, ['price', 'price_egp', 'total_price', 'sale_price', 'rent'])),
    currency: str(first(meta, ['currency'])),
    image: safeUrl(first(meta, ['image', 'image_url', 'thumbnail', 'photo', 'cover_image']) ?? images),
    url: safeUrl(first(meta, ['url', 'link', 'listing_url', 'property_url'])),
    description: desc ? truncate(desc, 600) : text ? truncate(text, 600) : undefined,
    source: typeof p?.source_file === 'string' && p.source_file ? p.source_file.split(/[\\/]/).pop() : undefined,
    isDemo: p?.metadata?.is_demo === true,
    inAnswer: false,
  };
}

export function formatNumber(n, lang) {
  return new Intl.NumberFormat(lang === 'ar' ? 'ar-EG' : 'en-EG', { maximumFractionDigits: 0 }).format(n);
}

// The existing site shows prices in EGP; a `currency` field in metadata overrides that.
export function formatPrice(n, currency, lang) {
  const cur = (currency || 'EGP').toUpperCase();
  try {
    return new Intl.NumberFormat(lang === 'ar' ? 'ar-EG' : 'en-EG', { style: 'currency', currency: cur, maximumFractionDigits: 0 }).format(n);
  } catch {
    return `${formatNumber(n, lang)} ${cur}`;
  }
}
