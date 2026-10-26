# PROJECT_CONTEXT.md

## 1. Core Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI 0.136 + Uvicorn (ASGI) |
| Agent Orchestration | LangGraph 1.2 (StateGraph + subgraphs + Postgres checkpointer) |
| LLM — Generation | Groq (configurable model via `GENERATION_MODEL` / `GROQ_MODEL`) |
| LLM — Reranking | Groq listwise via single LLM call (`RERANKER_GROQ_MODEL`); fallback: `sentence-transformers` CrossEncoder (`BAAI/bge-reranker-v2-m3`) |
| Embeddings | Gemini `gemini-embedding-001` (default, hosted, multilingual); fallback: `sentence-transformers` `intfloat/multilingual-e5-small` (local) |
| Vector Store | Qdrant Cloud (`qdrant-client>=1.10,<1.16`) |
| Relational DB | Neon PostgreSQL via `psycopg3` + SQLAlchemy 2.0 ORM + Alembic migrations |
| Agent Checkpointing | `langgraph-checkpoint-postgres` (per-thread conversation persistence) |
| Lexical Search | BM25 (`rank-bm25`) in-memory over corpus chunks |
| OCR | Conditional — `requirements-ingest.txt` stack (not in prod image) |
| CV / Floor Plans | Conditional — same ingest-only stack |
| Voice | Gemini Live (`app/voice/gemini_live.py`) |
| Frontend | Vanilla JS + HTML (`app/frontend/`) served via FastAPI static |
| Observability | `prometheus-client`, custom logging/tracing modules |
| Deployment | Docker + Render (`render.yaml`) + Vercel (frontend, `vercel.json`) |

---

## 2. Architecture & Data Flow

### Ingestion Pipeline (offline / on-demand)
```
Raw Source (PDF / DOCX / XLSX / CSV / Image)
  → Document type detection
  → Specialized loader (pdf_loader, office_loader, spreadsheet_loader, image_loader)
  → OCR (scanned) + CV (images/floor plans) — ingest-only stack
  → Information extraction → Normalization (field_mapper, property_normalizer)
  → Chunker: RecursiveCharacterTextSplitter (1500 chars / 100 overlap)
  → EmbeddingModel.embed_texts() [Gemini RETRIEVAL_DOCUMENT]
  → Qdrant upsert (chunk text + metadata payload)
```

### Customer Query Flow (runtime)
```
User message (Website / WhatsApp / Voice)
  → API route (chat / webhooks / voice)
  → AgentService → LangGraph graph.invoke(ConversationState, thread_id)
      → router node:
          detect_language() + extract_requirements() + classify_intent()
          + extract_qualification() + resolve_property_reference()
          → intent → INTENT_TO_AGENT mapping → route_agent()
      → Subgraph dispatch:
          ┌─ property_agent   → run_property_workflow()
          │     → rewrite_query() + build_metadata_filter()
          │     → hybrid_search() [Qdrant dense + optional BM25]
          │     → rerank() [Groq listwise / CrossEncoder / none]
          │     → generate_response() [Groq, grounded only on candidates]
          │     → post-generation financial claim sanitizer
          ├─ qualification_agent  → lead scoring (deterministic, 0-100)
          ├─ scheduling_agent     → availability + booking (Postgres)
          ├─ sales_agent          → payment plan / investment queries
          ├─ conversational_agent → chitchat / greetings
          └─ handoff_agent        → creates Handoff record + StaffNotification
      → record_reply node: appends AIMessage to checkpointed state (cap: 60 msgs)
  → Response → channel adapter (website JSON / WhatsApp / voice TTS)
```

---

## 3. Directory Structure & Key Files

```
app/
├── main.py                  FastAPI app factory, route registration, lifespan
├── agents/
│   ├── graph.py             Root LangGraph StateGraph: 6 agent nodes + router + record_reply
│   ├── state.py             ConversationState TypedDict; router() node; intent→agent mapping
│   ├── understanding.py     classify_intent, extract_requirements, detect_language,
│   │                        extract_qualification, resolve_property_reference (all LLM calls)
│   ├── service.py           AgentService: thread management, graph.invoke(), DB persistence
│   ├── checkpointer.py      Postgres checkpointer wiring (langgraph-checkpoint-postgres)
│   ├── handoff.py           handoff_to_human node: writes Handoff + StaffNotification
│   ├── messages.py          Localised fallback strings (say())
│   └── subgraphs/
│       ├── property/        RAG search subgraph (calls workflows/rag.py)
│       ├── qualification/   Lead scoring subgraph
│       ├── scheduling/      Booking state machine (need_property→confirm 6-stage FSM)
│       ├── sales/           Payment plan / investment intent handler
│       └── conversational/  General conversation handler
│   └── tools/
│       ├── property.py      LangGraph tool: search_properties
│       ├── crm.py           LangGraph tool: create/update lead in Postgres
│       └── scheduling.py    LangGraph tool: check availability, create booking
├── workflows/
│   └── rag.py               answer_query(), answer_about_properties(), search_index(),
│                            index_chunks(), list_indexed_properties() — shared RAG entry points
├── rag/
│   ├── embeddings.py        EmbeddingModel facade: Gemini (default) or local sentence-transformers
│   ├── generation.py        Groq chat completion; system prompt; citation extraction;
│   │                        post-gen financial claim sanitizer; GenerationResult dataclass
│   ├── retrieval/
│   │   ├── hybrid_search.py  Qdrant dense + BM25 lexical; min-max normalization; weighted merge
│   │   ├── metadata_filter.py build_metadata_filter(): dict → MetadataFilter (equality + range constraints)
│   │   └── reranker.py      Groq listwise (default) or CrossEncoder; graceful fallback to hybrid order
│   ├── ingestion/
│   │   ├── pipeline.py      Orchestrates multi-file ingestion; returns chunks[]
│   │   ├── chunker.py       RecursiveCharacterTextSplitter (1500/100)
│   │   ├── pdf_loader.py    Native text PDF extraction
│   │   ├── office_loader.py DOCX parser
│   │   ├── spreadsheet_loader.py XLSX/CSV → rows
│   │   ├── image_loader.py  Routes to OCR + CV
│   │   └── metadata.py      Attaches source_file, chunk_id, property_id to chunks
│   ├── ocr/                 extractor.py, document_parser.py, post_processor.py (ingest-only)
│   ├── cv/                  vision_extractor.py, image_processor.py, floorplan_parser.py (ingest-only)
│   └── normalization/       field_mapper.py, property_normalizer.py
├── database/
│   ├── models.py            SQLAlchemy ORM: Property, Customer, Lead, Interaction,
│   │                        Booking, Handoff, StaffNotification, ReplayGuard, AdminAudit
│   ├── vector_db.py         Qdrant client factory, ensure_collection, upsert_chunks
│   ├── migrations.py        Runtime migration helpers
│   └── replay.py            Webhook de-duplication (ReplayGuard first-writer-wins)
├── api/routes/
│   ├── chat.py              POST /chat — main conversational endpoint
│   ├── documents.py         POST /documents/ingest — file upload + ingestion
│   ├── properties.py        GET /properties — catalogue + search
│   ├── leads.py             GET/PATCH /leads — lead management
│   ├── voice.py             WebSocket voice endpoint
│   ├── webhooks.py          POST /webhooks/whatsapp — WhatsApp Cloud API
│   └── admin.py             Operator API (bookings, handoffs, audit)
├── api/
│   ├── security.py          API key auth, HMAC webhook verification
│   ├── ratelimit.py         Per-IP rate limiting
│   └── sessions.py          Session/thread_id management
├── leads/
│   ├── scoring.py           Deterministic 0-100 scorer (intent 25%, budget 25%,
│   │                        requirements 15%, timeline 20%, engagement 15%)
│   ├── qualification.py     Extracts qualification signals from ConversationState
│   └── handoff.py           Threshold-based hot-lead handoff trigger
├── scheduling/
│   ├── availability.py      Slot generation; conflict detection against Booking table
│   └── booking.py           create_booking(), confirm_booking(), cancel_booking()
├── crm/                     customers.py, leads.py, interactions.py — Postgres CRUD wrappers
├── channels/                website.py, whatsapp.py — message format adapters
├── notifications/           customer.py (confirmation SMS/WhatsApp), staff.py (StaffNotification dispatch)
├── voice/
│   └── gemini_live.py       Gemini Live API WebSocket bridge
├── llm/
│   ├── client.py            Groq client singleton
│   ├── prompts.py           Prompt templates
│   └── structured_output.py Pydantic-backed structured extraction helpers
├── helpers/
│   ├── config.py            Pydantic Settings; all env vars in one place
│   └── phone.py             E.164 normalization (phonenumbers)
├── observability/           logging.py, tracing.py (LangSmith-compatible), metrics.py (Prometheus)
└── schemas/                 Pydantic request/response schemas: property, customer, lead, interaction

alembic/versions/
├── ce49f8c2929c_create_initial_database_tables.py   Baseline schema
└── a7c1e0d2b9f4_production_persistence.py           Adds ReplayGuard, AdminAudit, Booking exclusion constraint

scripts/
├── ingest.py        CLI: run ingestion pipeline against data/raw/
├── evaluate.py      RAG evaluation: Recall@K, MRR, groundedness against golden_dataset.json
├── migrate.py       Alembic upgrade wrapper
└── maintenance.py   Expired booking cleanup, ReplayGuard pruning
```

---

## 4. Agents & Tools Summary

| Agent | Role |
|---|---|
| `router` | Every-turn pre-step: language detection, requirement extraction, qualification signal extraction, intent classification, agent dispatch. Not a specialist — sets shared state for all agents. |
| `property_agent` | RAG search subgraph. Rewrites query with carried requirements, runs `answer_query()` (hybrid search → rerank → Groq generation). Also handles `property_information` / `availability` detail intents against already-selected listing. |
| `qualification_agent` | Extracts lead signals from conversation, runs deterministic scorer, writes/updates Lead record in Postgres, triggers handoff if score ≥ threshold. |
| `scheduling_agent` | 6-stage FSM (`need_property → need_date → choose_slot → need_contact → verify_contact → confirm`). Calls `availability.py` + `booking.py`. Mid-booking short replies ("Saturday", "yes") are held in-flow by the router. |
| `sales_agent` | Handles `payment_plan`, `investment`, `sales` intents — generates grounded answers about financing from RAG evidence without inventing figures. |
| `conversational_agent` | Fallback: greetings, chitchat, clarification prompts, out-of-scope deflection. |
| `handoff_agent` | Creates `Handoff` DB record + `StaffNotification`, collects customer phone if missing (triggers `awaiting_contact` re-entry loop). |

**LangGraph tools (called by nodes, not as tool-calling LLM):**

| Tool | Function |
|---|---|
| `tools/property.py` | `search_properties` — wraps `workflows/rag.search_index()` |
| `tools/crm.py` | `upsert_customer`, `upsert_lead` — Postgres writes |
| `tools/scheduling.py` | `get_availability`, `create_booking` |

---

## 5. RAG & DB Specs

### Chunking
- Splitter: `RecursiveCharacterTextSplitter`
- `chunk_size = 1500` chars, `chunk_overlap = 100` chars
- Separator priority: `\n\n → \n → ". " → " " → ""`
- Each chunk carries: `text`, `property_id`, `chunk_id`, `source_file`, full property metadata dict

### Embedding
- Default: Gemini `gemini-embedding-001`, `output_dimensionality` = `EMBEDDING_DIMENSION` (configurable)
- Task types: `RETRIEVAL_DOCUMENT` (ingest) / `RETRIEVAL_QUERY` (search)
- Vectors L2-normalized before storage
- Batch size: 100 texts/request (Gemini limit); retry on 429/5xx with exponential backoff
- Local fallback: `intfloat/multilingual-e5-small` with `query:` / `passage:` E5 prefixes

### Retrieval Pipeline
```
Query → build_metadata_filter() [hard equality + range constraints on FILTER_KEYS]
  → hybrid_search(top_k=20):
      Qdrant dense search (FILTER applied as Qdrant must-conditions)
      + optional BM25 over in-memory corpus_chunks
      → min-max normalize each score set separately
      → weighted merge: semantic_weight=0.7, lexical_weight=0.3
  → rerank(top_k=5):
      Groq listwise (≤20 candidates, one LLM call, JSON {"ranking": [...]})
      Fallback: CrossEncoder.predict() / pass-through
  → generate_response() [Groq, temp=0.2, max_tokens=800]
      → citation extraction (JSON tail of response)
      → citation validation (IDs must exist in candidates)
      → post-gen financial claim sanitizer (removes invented EGP/% values)
```

**FILTER_KEYS** (applied as pre-retrieval hard filters):
`property_type`, `district`, `transaction_type`, `bedrooms`, `min_bedrooms`, `min_price`, `max_price`

### PostgreSQL Schema (key tables)

| Table | Purpose | Key Columns |
|---|---|---|
| `properties` | Structured property metadata mirror | `property_id` (unique, FK to Qdrant), `property_type`, `price`, `bedrooms`, `district`, `transaction_type`, `is_active` |
| `customers` | Conversation identities | `external_id` (channel key, unique), `public_id` (UUID), `phone` (E.164), `phone_verified`, `phone_verification_method` |
| `leads` | Sales qualification records | `customer_id` (FK), `score` (0-100), `status`, `intent`, `budget_min/max`, `preferred_location`, `property_ref` (Qdrant ID) |
| `interactions` | Per-turn message log | `customer_id`, `lead_id`, `channel` (`website:text`, `whatsapp:voice_note`…), `content`, `response` |
| `bookings` | Property viewings | `property_id` (Qdrant str), `starts_at/ends_at` (UTC), `status` (`pending_confirmation→confirmed→completed`), `contact_phone`, exclusion constraint (no overlap per property) |
| `handoffs` | Human escalation queue | `status` (`open→claimed→resolved`), `priority`, `summary` (JSON), `claimed_by` |
| `staff_notifications` | Outbox for staff alerts | `kind`, `status` (`pending→sent/failed`), `attempts`, retry logic |
| `replay_guard` | Webhook dedup / single-use tokens | `(kind, key)` composite PK, `expires_at` |
| `admin_audit` | Operator action log | `action`, `target`, `actor`, `detail` (JSON) |

### Qdrant Collection
- One collection (default name from `QDRANT_COLLECTION`)
- Vector dimension = `EMBEDDING_DIMENSION` (must match embedding model)
- Payload per point: `text`, `property_id`, `chunk_id`, `metadata{}` (full property fields), `source_file`
- Filters compiled from `MetadataFilter` → Qdrant `{"must": [...]}` syntax
- Demo listings filtered client-side by `DEMO_SOURCE_MARKERS` (never shown to customers unless `SHOW_DEMO_LISTINGS=true`)