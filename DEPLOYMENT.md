# Deploying Aqar.ai on free tiers

| Part | Service | Free-tier facts that shape the setup |
|---|---|---|
| Website | **Cloudflare Pages** (Vercel works too) | Static files only; no limits that matter here. |
| API + voice WebSocket | **Render** web service (Docker) | 512 MB RAM, **sleeps after 15 min idle** (first request then takes ~50 s), 750 h/month. |
| Database + conversation memory | **Neon** PostgreSQL | 0.5 GB storage; compute suspends when idle and wakes in a few seconds. |
| Property search | **Qdrant Cloud** free cluster | 1 GB; suspended if unused for a while. |
| AI | Groq (answers, ranking, voice-note transcription), Google Gemini (embeddings, voice) | Free tiers are rate-limited; the app degrades with a polite message, never invents listings. |

Nothing below needs a credit card except what a provider asks for on sign-up. **Do every step yourself in the provider dashboards; nothing in this repo deploys anything.**

> **Do this first: rotate the leaked Groq key.** A real Groq key (`gsk_0Zx…`) is committed in the repository history (commits `f527cec`, `b92269f`, `a2e30da`, `1d68c4f`, `6b57938`, reachable from `origin/main`). Revoke it at <https://console.groq.com/keys> and create a new one. Deleting the file is not enough.

---

## 0. What you need locally

Python 3.12, Docker (optional), Git, and this repository. Check the configuration at any time with `python scripts/check_env.py` (never prints secrets; `--connect` also tests every credential).

Generate the two secrets you will need:

```bash
python -c "import secrets; print(secrets.token_hex(32))"      # SESSION_SECRET
python -c "import secrets; print(secrets.token_urlsafe(32))"  # ADMIN_API_KEY
```

## 1. Neon (PostgreSQL)

1. Sign up at <https://neon.tech>, create a project. **Region: AWS Europe (Frankfurt) `eu-central-1`**, the closest to Egypt and to the Render region used below.
2. Dashboard → **Connect**. **Turn Connection pooling OFF** and copy the connection string (host has no `-pooler`). The app holds transaction-level advisory locks and runs `CREATE INDEX CONCURRENTLY`, which need a direct connection.
3. That string is your `DATABASE_URL`, e.g. `postgresql://user:pass@ep-xxx.eu-central-1.aws.neon.tech/neondb?sslmode=require`. Keep `sslmode=require`; if the password contains `@ / : # %`, percent-encode it.

The schema is created by `scripts/migrate.py`, run automatically on every start. Nothing to do by hand.

## 2. Qdrant Cloud

1. Sign up at <https://cloud.qdrant.io>, create a **free cluster** (AWS `eu-central-1`).
2. Create a database API key. Note the cluster URL (`https://<id>.eu-central-1-0.aws.cloud.qdrant.io:6333`). These are `QDRANT_URL` and `QDRANT_API_KEY`.
3. Put your listings in it (done from your computer, not Render, because parsing PDFs/spreadsheets needs heavy libraries the free instance cannot hold):

```bash
pip install -r requirements-ingest.txt
export QDRANT_URL=... QDRANT_API_KEY=... GEMINI_API_KEY=...
python scripts/ingest.py data/raw --recursive --rebuild
```

**Test data is hidden from visitors.** Listings ingested from a file whose *name* contains `test_`, `demo` or `sample` (for example `test_properties_small.csv`) are treated as demo data: the website, search and the assistant do not show them. To present them on purpose (a demo), set `SHOW_DEMO_LISTINGS=true` on Render; every such listing then carries a visible "Demo listing — not for sale" label, and the site shows a demo banner. Turn it off, and ingest real listings from a normally named file, before real customers use the site. Change the markers with `DEMO_SOURCE_MARKERS`.

The website loads up to 500 listings (the first 500 in the collection); beyond that, search through the assistant.

Embeddings are Gemini `gemini-embedding-001` at 768 dimensions. **Queries and documents must use the same model and dimension**; the API refuses to search a collection built differently and `/health/ready` says so. Changing the embedding model means re-ingesting into a fresh collection.

## 3. API keys

* **Groq**: <https://console.groq.com/keys> → `GROQ_API_KEY` (new key, see the warning above).
* **Gemini**: <https://aistudio.google.com/apikey> → `GEMINI_API_KEY` (used for embeddings and website voice).

## 4. Staff alerts (set at least one)

Booking requests, cancellations and hand-offs must reach a person. Pick whichever you will actually watch:

* **Telegram (easiest):** talk to `@BotFather` → `/newbot` → `TELEGRAM_BOT_TOKEN`. Send your bot any message, open `https://api.telegram.org/bot<TOKEN>/getUpdates`, copy `chat.id` → `TELEGRAM_CHAT_ID`.
* **Slack / Discord / Teams:** create an incoming webhook → `STAFF_WEBHOOK_URL`.
* **E-mail:** `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`, `STAFF_EMAIL_TO` (Brevo's free SMTP works).

Alerts are stored in the database first and retried every minute, so a provider outage delays them but does not lose them. `GET /admin/notifications` shows their state.

## 5. Render (the API)

1. Push the repository to GitHub (your decision and your action).
2. Render → **New → Blueprint** → pick the repo. It reads `render.yaml`: Docker runtime, free plan, Frankfurt, health check `/health`, `autoDeploy: false`.
3. When prompted, fill the values marked `sync: false`:

| Variable | Value |
|---|---|
| `DATABASE_URL` | Neon direct string from step 1 |
| `SESSION_SECRET`, `ADMIN_API_KEY` | the two generated secrets |
| `CORS_ALLOWED_ORIGINS` | your Pages origin, e.g. `https://aqar.pages.dev` (exact, `https`, no trailing slash). Leave a placeholder until step 6, then update it. |
| `GROQ_API_KEY`, `GEMINI_API_KEY` | step 3 |
| `QDRANT_URL`, `QDRANT_API_KEY` | step 2 |
| `TELEGRAM_*` / `STAFF_WEBHOOK_URL` / `SMTP_*` | step 4 |
| `WHATSAPP_*` | step 8, or leave all four empty to run without WhatsApp |

The blueprint already sets `APP_ENV=production`, `TRUSTED_PROXY_HOPS=1`, `LOG_FORMAT=json` and the embedding/reranker choices.

4. Deploy. On start the container validates the configuration (a bad one stops the deploy with a list of what is wrong, never the values), applies migrations, then serves. Watch the logs for `Database schema at …`, `Conversation-memory tables ready`, `Startup complete`.

**Cold starts:** free instances sleep after 15 minutes without traffic and the first visitor then waits ~50 s (the chat shows its "taking longer than expected" message and the client waits up to 60 s). This deployment deliberately adds no keep-alive pings; an always-on instance would be a paid plan, so that is your decision.

## 6. Cloudflare Pages (the website)

The site is plain HTML/JS; a build step writes the API address and security headers into `dist/`.

1. Cloudflare dashboard → Workers & Pages → **Create → Pages → Connect to Git** → the repository.
2. Build settings: **Build command** `python3 scripts/build_frontend.py` · **Build output directory** `dist`.
3. Environment variables (Production): `API_BASE_URL=https://<service>.onrender.com` (https, no path). Optional: `CHAT_DEFAULT_LANG` (`ar`/`en`), `VOICE_ENABLED`. Leave `SHOW_PREVIEW_INVENTORY` unset so customers never see sample listings.
4. Deploy, then copy the Pages URL into Render's `CORS_ALLOWED_ORIGINS` and redeploy the API. (Both the REST calls and the voice WebSocket check this value.)

**Vercel instead:** import the repo; `vercel.json` already sets the build command and output directory. Set `API_BASE_URL` as an environment variable. Note that Vercel's free "Hobby" plan is for non-commercial use, so for a real-estate business Cloudflare Pages is the safer choice.

## 7. Verify the deployment

```bash
export ADMIN_API_KEY=...            # optional: also checks the staff API
python scripts/smoke_prod.py --api https://<service>.onrender.com --origin https://<pages-domain> --expect-indexed
pip install playwright
python scripts/browser_smoke.py --site https://<pages-domain> --expect-listings 1
```

`smoke_prod.py` checks health and readiness, that `/docs` and `/metrics` are closed, CORS, security headers, a chat turn, forged-session rejection, staff-API protection, WhatsApp webhook protection, and **that the rate limiter sees your real IP** (if `resolved` equals `peer`, raise `TRUSTED_PROXY_HOPS`). `browser_smoke.py` drives real Chrome against the live site. `/health/ready` returning 503 with `vector_search: collection does not exist` means step 2.3 has not been done yet.

**Voice needs a human:** open the site, press the microphone, allow access and speak Arabic and English. Nobody can automate that for you. Then book a viewing by voice, dictate a phone number, and confirm the read-back.

## 8. WhatsApp (Meta Cloud API)

1. <https://developers.facebook.com> → create an app (type *Business*) → add **WhatsApp**. Use the test number to start; a real number needs business verification.
2. Collect: `WHATSAPP_PHONE_NUMBER_ID` (API Setup page), `WHATSAPP_APP_SECRET` (App settings → Basic), `WHATSAPP_ACCESS_TOKEN` (create a **system user** in Business Settings, assign the app, generate a *permanent* token with `whatsapp_business_messaging` and `whatsapp_business_management`; the 24-hour test token expires), and choose any string for `WHATSAPP_VERIFY_TOKEN`.
3. Set all four in Render and redeploy.
4. Meta → WhatsApp → Configuration → **Callback URL** `https://<service>.onrender.com/whatsapp/webhook`, **Verify token** your string → *Verify and save*. Subscribe to the **messages** field.
5. Test: `python scripts/check_env.py --connect` (validates the token), then send a message from a number added to the test recipients list. Re-send the same webhook from Meta's dashboard: the customer must get one reply, not two.

Limits to know: free-form replies work inside 24 hours of the customer's last message; staff-initiated booking confirmations to a customer outside that window are rejected by Meta (logged, and the booking is still confirmed). Real-time WhatsApp *calls* are not supported; voice notes are transcribed and answered as text.

## 9. Running it day to day

* **Staff API** (all need header `X-Admin-Key`): `GET /admin/bookings?status=pending_confirmation`, `POST /admin/bookings/{id}/confirm` · `/cancel` · `/complete`, `GET /admin/handoffs`, `POST /admin/handoffs/{id}/claim` · `/resolve`, `GET /admin/leads`, `GET /admin/notifications`. Customers asking "is my viewing confirmed?" get the live status.
* **Add listings:** re-run `scripts/ingest.py` or `POST /properties` with the admin key.
* **Weekly:** `python scripts/maintenance.py --apply` (with `DATABASE_URL` set) keeps Neon under 0.5 GB.
* **Logs:** Render dashboard; one JSON object per line with a `request_id`. Secrets are scrubbed.
* **Rollback:** Render → Deploys → *Rollback*. Migrations are additive; to undo one, `alembic downgrade -1` from a machine with `DATABASE_URL` set.

## 10. Known limits (be honest with yourself about these)

* **One instance.** Rate limits and open voice calls are per process. Scaling out is safe for data (all state is in PostgreSQL, turns are locked across workers) but limits become per-instance.
* **Cold starts.** The first visitor after 15 idle minutes waits ~50 s; Neon and Qdrant add a few seconds when idle.
* **Gemini Live is a preview model** (`gemini-3.1-flash-live-preview`); Google may change or retire it. Voice quality and phone-number dictation depend on it. Text chat is unaffected if it fails.
* **Voice sessions are capped** at 10 minutes, 20 concurrent, 3 per client.
* **Booking confirmation is human.** The assistant records a *request*; an advisor must confirm it. It never tells a customer a visit is confirmed.
* **Customers are per channel.** The same person on the website and on WhatsApp is two CRM records, deliberately: merging by an unverified phone number would let anyone read another person's data.
