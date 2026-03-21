# AEGIS — AI Police Assistance System: Project Context

> **For AI Agents:** This file is the authoritative source of truth for the project's architecture, data models, API contracts, and integration status. Update this file whenever you make significant changes to the codebase. Specifically: add a dated entry under [Changelog](#changelog) and update the relevant sections.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Technology Stack](#3-technology-stack)
4. [Component Details](#4-component-details)
   - 4.1 [Backend (FastAPI)](#41-backend-fastapi--backendapp)
   - 4.2 [Frontend (Next.js)](#42-frontend-nextjs--frontend)
   - 4.3 [RAG System (Microservice)](#43-rag-system-microservice--rag_system)
   - 4.4 [Multi-Agent System](#44-multi-agent-system)
5. [Database Schemas](#5-database-schemas)
6. [API Reference](#6-api-reference)
7. [Authentication & Authorization](#7-authentication--authorization)
8. [Environment Variables](#8-environment-variables)
9. [Data Flow](#9-data-flow)
10. [Integration Status](#10-integration-status)
11. [Known Issues & Gaps](#11-known-issues--gaps)
12. [Changelog](#12-changelog)

---

## 1. Project Overview

**AEGIS** is a full-stack case-centric intelligence management platform for police departments. It allows officers and administrators to:

- Manage investigation cases with role-based, clearance-level access control
- Upload evidentiary documents (PDFs, images) tied to cases
- Query those documents with AI via a Retrieval-Augmented Generation (RAG) system *(integration pending)*
- Get AI-generated insights via a multi-agent system *(stub only)*
- Analyze CCTV footage *(stub only)*

**Three independently runnable services:**

| Service | Language/Framework | Port | Status |
|---|---|---|---|
| `backend/` | Python / FastAPI | `8000` | Fully operational |
| `frontend/` | TypeScript / Next.js 16.1.7 | `3000` | Fully operational |
| `Rag_system/` | Python / FastAPI | `8080` | Fully operational, proxied via backend |

**Supporting infrastructure:**

| Service | Purpose | Port |
|---|---|---|
| PostgreSQL (Neon.tech cloud) | Main relational DB | cloud |
| Qdrant | Vector database for RAG | `6333` (moved to remote server — configure `QDRANT_HOST` in `Rag_system/.env`) |
| Redis | Celery broker + Blackboard | `6379` |
| Modal (Qwen 3 30b) | Remote LLM via native Ollama API | Set `LLM_BASE_URL` in `Rag_system/.env` |

---

## 2. Directory Structure

```
E:\Major-project\
├── PROJECT_CONTEXT.md            ← THIS FILE — update on every significant change
├── .gitignore
│
├── backend/                      ← Python/FastAPI REST API (port 8000)
│   ├── .env                      ← Active DB credentials (not committed)
│   ├── alembic.ini
│   ├── alembic/
│   │   ├── env.py
│   │   └── versions/
│   │       ├── 94a164febd44_initial_backend_structure.py
│   │       ├── de7856598563_add_rag_fields_to_document.py
│   │       └── a3f1c9b84e21_add_document_metadata_fields.py
│   ├── seed.py                   ← Seeds default admin (admin / admin123)
│   ├── requirements.txt
│   └── app/
│       ├── main.py               ← FastAPI app entry, CORS, router mount
│       ├── api/
│       │   ├── api.py            ← Registers all routers under /api/v1
│       │   ├── deps.py           ← Auth dependencies (get_db, get_current_user, etc.)
│       │   └── endpoints/
│       │       ├── auth.py       ← /auth/login, /refresh, /change-password
│       │       ├── admin.py      ← /admin/officers CRUD + /admin/cases management
│       │       ├── officer.py    ← /officer/me, /officer/cases, /officer/list, /officer/{id}
│       │       ├── cases.py      ← Case lifecycle + assignment endpoints
│       │       ├── documents.py  ← Document upload/list/delete + RAG proxy + metadata fields
│       │       └── analysis.py   ← STUB: returns "not enabled yet"
│       ├── core/
│       │   ├── config.py         ← Settings: SECRET_KEY, token expiry, DB URL
│       │   └── security.py       ← bcrypt hashing, JWT create/decode
│       ├── db/
│       │   └── database.py       ← Async SQLAlchemy engine + session factory
│       ├── models/               ← SQLAlchemy ORM models
│       │   ├── user.py
│       │   ├── case.py
│       │   ├── assignment.py
│       │   ├── document.py
│       │   ├── activity.py
│       │   └── analysis.py
│       └── schemas/              ← Pydantic v2 schemas
│           ├── token.py
│           ├── user.py
│           ├── case.py
│           ├── assignment.py
│           └── document.py
│
├── frontend/                     ← Next.js 14 App Router (port 3000)
│   ├── package.json
│   ├── next.config.mjs           ← ignoreBuildErrors: true (TypeScript/ESLint suppressed)
│   ├── tailwind.config.ts
│   ├── components.json           ← shadcn/ui config
│   ├── tsconfig.json
│   └── src/
│       ├── app/
│       │   ├── globals.css
│       │   ├── layout.tsx        ← Root layout
│       │   ├── page.tsx          ← Login page (/)
│       │   └── (dashboard)/
│       │       ├── layout.tsx    ← Protected layout with sidebar navigation
│       │       ├── admin/dashboard/page.tsx       ← Officer management
│       │       ├── officer/dashboard/page.tsx     ← Case grid
│       │       └── cases/[id]/
│       │           ├── page.tsx                   ← Case detail & settings
│       │           ├── chat/page.tsx              ← STUB: RAG chat
│       │           ├── insights/page.tsx          ← STUB: Multi-agent insights
│       │           └── cctv/page.tsx              ← STUB: CCTV analysis
│       ├── components/
│       │   ├── ui/               ← shadcn/ui + Radix UI primitives
│       │   ├── auth/             ← Login form component
│       │   ├── navigation/       ← Sidebar
│       │   ├── admin/            ← Admin-specific components (OfficerModal, etc.)
│       │   └── officer/          ← Officer-specific components (CaseCard, etc.)
│       └── lib/
│           ├── auth.ts           ← Server actions: login, logout, getAccessToken
│           └── utils.ts          ← cn() utility, etc.
│
├── Rag_system/                   ← RAG microservice (port 8080)
│   ├── .env.example              ← Template for RAG env config
│   ├── .venv/                    ← Local Python venv
│   ├── api/
│   │   ├── app.py                ← FastAPI RAG entry point
│   │   └── routes/
│   │       ├── ingest.py         ← /ingest/file, /ingest/batch, /ingest/status, /ingest/documents
│   │       └── query.py          ← /query/
│   ├── config/
│   │   └── settings.py           ← Pydantic settings loaded from .env
│   ├── ingestion/
│   │   ├── pipeline.py           ← Orchestrates: load → clean → chunk → embed → store
│   │   ├── loaders/
│   │   │   ├── pdf_loader.py     ← Docling PDF → markdown + images
│   │   │   └── image_loader.py   ← Docling + EasyOCR → text
│   │   └── processors/
│   │       └── cleaner.py        ← Unicode norm, hyphenation fix, dedup
│   ├── core/
│   │   ├── documents/
│   │   │   ├── models.py         ← DocumentChunk, DocumentMetadata dataclasses
│   │   │   └── chunker.py        ← Semantic chunker (50–512 tokens, cosine similarity)
│   │   ├── embeddings/
│   │   │   └── local_embedder.py ← BAAI/bge-base-en-v1.5 (768-dim, lazy load)
│   │   ├── retrieval/
│   │   │   ├── hybrid_retriever.py ← Dense + BM25 parallel, RRF fusion
│   │   │   └── bm25_retriever.py   ← In-memory BM25Okapi with pickle persistence
│   │   ├── reranking/
│   │   │   └── bge_reranker.py   ← BAAI/bge-reranker-base cross-encoder
│   │   └── generation/
│   │       └── llm_client.py     ← LLM client (native Ollama API, returns content + reasoning)
│   ├── stores/
│   │   ├── qdrant_store.py       ← Qdrant vector store client
│   │   └── document_store.py     ← SQLite audit/status store
│   ├── query/
│   │   ├── context_builder.py    ← Assembles cited prompt for LLM (2048-token window)
│   │   └── query_rewriter.py     ← Multi-query expansion via LLM
│   ├── agents/
│   │   └── witness_agent.py      ← Celery task: RAG query → Blackboard
│   └── orchestration/
│       ├── celery_app.py         ← Celery worker (Redis broker)
│       ├── blackboard.py         ← Redis-backed shared memory for agents
│       └── test_task.py          ← ping → pong health check task
│
└── data/
    └── qdrant/                   ← Qdrant persistent data volume
        └── collections/
            └── case_text_chunks/ ← Live vector collection data
```

---

## 3. Technology Stack

| Layer | Technology | Version/Notes |
|---|---|---|
| **Frontend framework** | Next.js (App Router) | 16.1.7 (upgraded from 14 via `npm audit fix --force`) |
| **Frontend language** | TypeScript | — |
| **UI library** | Tailwind CSS + shadcn/ui + Radix UI | — |
| **Form validation** | React Hook Form + Zod | — |
| **Icons** | Lucide React | — |
| **Backend framework** | FastAPI | Python |
| **Backend ORM** | SQLAlchemy (async) + asyncpg | — |
| **DB migrations** | Alembic | — |
| **Data validation** | Pydantic v2 | — |
| **Auth** | python-jose (JWT HS256) + passlib (bcrypt) | — |
| **Main database** | PostgreSQL | Neon.tech (cloud-hosted) |
| **Document loading** | Docling | PDF + image OCR |
| **OCR fallback** | EasyOCR | Images |
| **Embeddings** | sentence-transformers | BAAI/bge-base-en-v1.5, 768-dim |
| **Sparse retrieval** | rank-bm25 (BM25Okapi) | Pickle-persisted index |
| **Vector database** | Qdrant | Local, port 6333 |
| **Reranker** | sentence-transformers | BAAI/bge-reranker-base |
| **Remote LLM** | Qwen 3 30b via Modal (native Ollama API) | Model: `qwen35-fast`, thinking enabled, returns content + reasoning separately |
| **Task queue** | Celery | Redis broker/backend |
| **Cache / Blackboard** | Redis | port 6379 |

---

## 4. Component Details

### 4.1 Backend (FastAPI) — `backend/app`

**Entry point:** `app/main.py` — mounts all routers under `/api/v1`, configures CORS (`allow_origins=["*"]` — needs tightening in production).

**Config** (`app/core/config.py`):
- `SECRET_KEY` — hardcoded dev placeholder; **must be moved to env var before production**
- `ACCESS_TOKEN_EXPIRE_MINUTES` = 10080 (7 days)
- `REFRESH_TOKEN_EXPIRE_MINUTES` = 43200 (30 days)
- `DATABASE_URL` — read from env, fallback to `postgresql+asyncpg://postgres:postgres@localhost:5432/aegis`

**Auth dependencies** (`app/api/deps.py`):
- `get_db` — yields an async SQLAlchemy session
- `get_current_user` — decodes JWT Bearer token, returns `User` ORM object
- `get_current_active_admin` — calls `get_current_user`, checks `role == "ADMIN"`
- `get_current_active_officer` — calls `get_current_user`, checks `role == "OFFICER"`

**Security** (`app/core/security.py`):
- `hash_password(plain)` → bcrypt hash
- `verify_password(plain, hashed)` → bool
- `create_access_token(data, expires_delta)` → signed JWT
- `create_refresh_token(data)` → JWT with `"refresh": True` claim

**How to run:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**How to seed the default admin:**
```bash
cd backend
python seed.py   # creates admin / admin123
```

**How to run migrations:**
```bash
cd backend
alembic upgrade head
```
Current head revision chain: `94a164febd44` → `de7856598563` (RAG fields on `document`) → `a3f1c9b84e21` (upload metadata: `display_name`, `evidence_category`, `description`). Use the project venv so `asyncpg` and Alembic resolve correctly.

---

### 4.2 Frontend (Next.js) — `frontend/`

**Auth flow** (`src/lib/auth.ts`):
1. `login(username, password)` → POST to `http://localhost:8000/api/v1/auth/login`
2. Receives `access_token` + `refresh_token` → stores both in HTTP-only cookies
3. Probes `GET /api/v1/officer/me` with the access token
   - 200 → officer → redirect to `/officer/dashboard`
   - Non-200 → admin → redirect to `/admin/dashboard`
4. `logout()` → deletes both cookies
5. `getAccessToken()` → reads `access_token` cookie from server context

**Backend URL:** hardcoded as `http://localhost:8000` throughout — **no `NEXT_PUBLIC_API_URL` env var exists yet**.

**Protected route pattern:** The `(dashboard)/layout.tsx` calls `getAccessToken()` — if no token, redirects to `/`.

**Key pages:**

| Route | File | Status |
|---|---|---|
| `/` | `app/page.tsx` | Login page, fully working |
| `/officer/dashboard` | `(dashboard)/officer/dashboard/page.tsx` | Fully working |
| `/admin/dashboard` | `(dashboard)/admin/dashboard/page.tsx` | Fully working |
| `/cases/[id]` | `(dashboard)/cases/[id]/page.tsx` | Fully working |
| `/cases/[id]/chat` | `(dashboard)/cases/[id]/chat/page.tsx` | **STUB** |
| `/cases/[id]/insights` | `(dashboard)/cases/[id]/insights/page.tsx` | **STUB** |
| `/cases/[id]/cctv` | `(dashboard)/cases/[id]/cctv/page.tsx` | **STUB** |

**How to run:**
```bash
cd frontend
npm install
npm run dev
```

---

### 4.3 RAG System (Microservice) — `Rag_system/`

**Entry point:** `api/app.py` — FastAPI app, mounts `/ingest` and `/query` routers, initializes all heavy models (embedder, retriever, reranker, LLM client) at startup.

**Full query pipeline (per request to `POST /query/`):**
```
User query + optional conversation history
  → QueryRewriter        — generates 2 alternative phrasings via LLM
  → HybridRetriever      — runs dense (Qdrant) + sparse (BM25) search for each query variant
                         — fuses all results with Reciprocal Rank Fusion (RRF) → top-50
  → BGEReranker          — cross-encoder reranking of top-50 → top-5
  → context_builder      — assembles cited prompt with [Source N] labels (2048-token window)
  → LLMClient            — generates answer via native Ollama /api/chat (Qwen 3 30b on Modal)
                         — thinking enabled: returns content + reasoning separately
                         — includes conversation history for multi-turn context
  → QueryResponse        — answer text + reasoning chain + list of source references
```

**Qdrant collections:**
- `case_text_chunks` — text extracted from PDFs/images
- `case_image_chunks` — OCR text from standalone image files
- Both use 768-dim cosine-distance vectors
- Payload fields: `chunk_id`, `document_id`, `chunk_type`, `text`, `page_number`, `chunk_index`, `token_count`, `parent_text`, `case_id`, `source_path`
- Filtering by `case_id` is supported in all queries

**Document store** — dual implementation, selected at startup:
- If `RAG_DATABASE_URL` is set in `.env` → uses `stores/pg_document_store.py` (PostgreSQL/NeonDB, table: `rag_documents`)
- If not set → falls back to `stores/document_store.py` (SQLite at `data/document_store.db`)
- Both implement the same interface: `create()`, `update_status()`, `get()`, `exists_by_path()`, `list_by_case()`, `list_all()`
- Fields: `document_id`, `filename`, `source_path`, `file_type`, `case_id`, `officer_id`, `status`, `chunk_count`, `page_count`, `error_message`, `extra_metadata`, `created_at`, `updated_at`

**BM25 index** (`data/bm25_index.pkl`):
- Pickle file, rebuilt in full on each ingestion
- **Not auto-triggered from the API route** — the `HybridRetriever` must call `build_index()` manually or on startup

**How to run:**
```bash
cd Rag_system
# Copy and fill .env.example → .env
pip install -r requirements.txt   # (if requirements.txt exists; else use .venv)
uvicorn api.app:app --reload --port 8080
```

**Prerequisites:**
- Qdrant: hosted on Ubuntu server, accessed via ngrok tunnel — set `QDRANT_URL` in `Rag_system/.env` (update each time ngrok restarts). Also set `QDRANT_API_KEY` if auth is enabled.
- LLM: Modal-hosted Qwen 3 30b — no local setup required. Set `LLM_BASE_URL` and `LLM_MODEL` in `Rag_system/.env`
- Redis running (for Celery): `docker run -p 6379:6379 redis`

---

### 4.4 Multi-Agent System

**Infrastructure:**
- `Rag_system/orchestration/celery_app.py` — Celery app, Redis as broker + result backend, timezone `Asia/Kolkata`
- `Rag_system/orchestration/blackboard.py` — Redis-backed shared blackboard; agents post observations keyed by `case_id`

**Blackboard API:**
```python
blackboard.post_observation(case_id, agent_name, content)  # write
blackboard.get_observations(case_id)                        # read all for a case
blackboard.post_anomaly(case_id, agent_name, content)       # write anomaly
blackboard.get_anomalies(case_id)                           # read anomalies
```

**Implemented agents:**
| Agent | File | Celery Task | Description |
|---|---|---|---|
| WitnessAgent | `agents/witness_agent.py` | `analyze_witness_statements` | Runs RAG query about witness statements/timeline → posts to blackboard |

**Pending agents (not yet implemented):**
- SuspectAgent, TimelineAgent, EvidenceAgent, etc. — architecture is ready but tasks not written

**How to run Celery worker:**
```bash
cd Rag_system
celery -A orchestration.celery_app worker --loglevel=info
```

---

## 5. Database Schemas

### PostgreSQL (Main DB via SQLAlchemy / Alembic)

#### `user` table
| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK | |
| `username` | String | Unique, Indexed, Not Null | |
| `hashed_password` | String | Not Null | bcrypt |
| `role` | String | Not Null | `"ADMIN"` or `"OFFICER"` |
| `rank` | String | Nullable | e.g. "Inspector", "DSP" |
| `clearance_level` | Integer | Indexed, Nullable | 1–11 (see hierarchy below) |
| `badge_number` | String | Nullable | |
| `station_name` | String | Nullable | |
| `is_active` | Boolean | Default True | |
| `created_at` | DateTime(tz) | Default now | |

#### `case` table
| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK | |
| `title` | String | Indexed, Not Null | |
| `description` | Text | Not Null | |
| `created_by` | UUID | FK → user.id, Indexed | Case owner |
| `required_clearance_level` | Integer | Indexed, Not Null | |
| `status` | String | Not Null | `OPEN`, `UNDER_INVESTIGATION`, `CLOSED` |
| `created_at` | DateTime(tz) | Default now | |
| `updated_at` | DateTime(tz) | Auto-update | |

#### `case_assignment` table
| Column | Type | Constraints |
|---|---|---|
| `id` | UUID | PK |
| `case_id` | UUID | FK → case.id, Indexed |
| `officer_id` | UUID | FK → user.id, Indexed |

#### `document` table
| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `case_id` | UUID | FK → case.id, Indexed |
| `uploaded_by` | UUID | FK → user.id, Indexed |
| `document_type` | String | MIME type |
| `file_path` | String | RAG path reference |
| `filename` | String | Original file name |
| `display_name` | String | User-provided label for the document |
| `evidence_category` | String | Police file type: `case_diary`, `fir_file`, `statement_file`, `scene_of_crime`, `forensic_evidence`, `property_seizure`, `arrest_remand`, `other` |
| `description` | String | Optional notes about the document |
| `rag_document_id` | String | Links to RAG service document_id |
| `ingest_status` | String | `pending`, `processing`, `completed`, `failed`, `rag_unavailable` |
| `created_at` | DateTime(tz) | |

#### `activity_log` table
| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `case_id` | UUID | FK → case.id, Indexed |
| `user_id` | UUID | FK → user.id, Indexed |
| `action` | String | `CASE_CREATED`, `OFFICER_ASSIGNED`, `OFFICER_REMOVED`, `DOCUMENT_UPLOADED`, `DOCUMENT_DELETED` |
| `timestamp` | DateTime(tz) | |

#### `case_analysis` table
| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `case_id` | UUID | FK → case.id, Indexed |
| `analysis_type` | String | |
| `result_text` | Text | |
| `created_at` | DateTime(tz) | |

---

### SQLite (RAG Document Store — `data/document_store.db`)

| Column | Type | Notes |
|---|---|---|
| `document_id` | TEXT | UUID string (PK) |
| `filename` | TEXT | |
| `source_path` | TEXT | Local file path used during ingestion |
| `display_name` | TEXT | User label (mirrors backend) |
| `evidence_category` | TEXT | Police file type key |
| `file_type` | TEXT | `pdf` or `image` |
| `case_id` | TEXT | Foreign reference to PostgreSQL `case.id` (string only, no FK constraint) |
| `officer_id` | TEXT | Foreign reference to PostgreSQL `user.id` |
| `status` | TEXT | `pending`, `processing`, `completed`, `failed` |
| `chunk_count` | INTEGER | |
| `page_count` | INTEGER | |
| `error_message` | TEXT | |
| `extra_metadata` | TEXT | JSON-serialized dict |
| `created_at` | DATETIME | |
| `updated_at` | DATETIME | |

---

### Clearance Level Hierarchy

| Level | Rank |
|---|---|
| 1 | Constable |
| 2 | Head Constable |
| 3 | ASI (Assistant Sub Inspector) |
| **4** | **SI (Sub Inspector) — minimum to create/own cases** |
| 5 | Inspector |
| 6 | DSP / ACP |
| 7 | SP |
| 8 | DIG |
| 9 | IG |
| 10 | ADGP |
| 11 | DGP |

---

## 6. API Reference

All backend endpoints are prefixed `/api/v1`. Base URL: `http://localhost:8000`

### Auth

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/auth/login` | None | OAuth2 form data (`username`, `password`) → `{access_token, refresh_token, token_type}` |
| POST | `/auth/refresh` | None | Body: `{refresh_token}` → new token pair |
| POST | `/auth/change-password` | Any | Body: `{current_password, new_password}` |

### Admin (requires `role == "ADMIN"`)

| Method | Path | Description |
|---|---|---|
| GET | `/admin/officers` | List all officers |
| POST | `/admin/officers` | Create officer — body: `{username, password, rank, clearance_level, badge_number, station_name}` |
| PATCH | `/admin/officers/{id}` | Update officer fields and/or password |
| PATCH | `/admin/officers/{id}/status` | Toggle `is_active` |
| DELETE | `/admin/officers/{id}` | Delete officer (blocked if officer owns cases; cleans up assignments/logs/docs first) |
| GET | `/admin/cases?officer_id=` | List all cases, optionally filtered by creator officer ID |
| DELETE | `/admin/cases/{case_id}` | Admin delete any case (cascades assignments, activity logs, documents) |
| POST | `/admin/cases/{case_id}/officers` | Assign officer to case — officer clearance must be ≥ case required level |
| DELETE | `/admin/cases/{case_id}/officers/{officer_id}` | Remove officer from case (cannot remove creator) |

### Officer (requires `role == "OFFICER"` except `/{id}` which accepts any authenticated user)

| Method | Path | Description |
|---|---|---|
| GET | `/officer/me` | Current officer's profile |
| GET | `/officer/cases` | All cases accessible to the officer (created + assigned) |
| GET | `/officer/list` | All active officers (for assignment dropdowns) — must be defined BEFORE `/{id}` in router |
| GET | `/officer/{id}` | Get any officer's basic profile by UUID — used by case detail page to show creator info |

### Cases (any authenticated user)

| Method | Path | Access | Description |
|---|---|---|---|
| POST | `/cases` | clearance ≥ 4 | Create case — body: `{title, description, required_clearance_level, status}` |
| GET | `/cases/search?q=` | accessible only | Search by title |
| GET | `/cases/{id}` | accessible only | Case detail + activity log + assigned officers |
| DELETE | `/cases/{id}` | creator only | Delete case and all sub-records |
| POST | `/cases/{id}/transfer` | creator only | Transfer ownership — body: `{new_owner_id}` |
| GET | `/cases/{id}/officers` | accessible | List assigned officers |
| POST | `/cases/{id}/officers` | creator or higher-clearance assigned | Assign officer — body: `{officer_id}` |
| DELETE | `/cases/{id}/officers/{officer_id}` | creator or higher-clearance assigned | Remove officer |

### Documents (any authenticated user with case access)

| Method | Path | Description |
|---|---|---|
| POST | `/cases/{case_id}/documents` | Upload document (multipart: `file`, optional `display_name`, `evidence_category`, `description`) — real ingest via RAG proxy |
| GET | `/cases/{case_id}/documents` | List documents for case |
| DELETE | `/cases/{case_id}/documents/{document_id}` | Delete document — requires officer clearance > case required level; cascades to RAG (Qdrant vectors + document store + BM25) |

### Analysis (STUB — always returns "not enabled yet")

| Method | Path | Description |
|---|---|---|
| POST | `/cases/{case_id}/analysis` | Trigger analysis |
| GET | `/cases/{case_id}/analysis` | Get results |

---

### RAG Service Endpoints (`http://localhost:8080`)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/ingest/file` | Upload + ingest single file (multipart: `file`, optional `case_id`, `officer_id`) |
| POST | `/ingest/batch` | Upload + ingest multiple files |
| GET | `/ingest/status/{document_id}` | Ingestion status (polls SQLite) |
| GET | `/ingest/documents` | List all ingested documents |
| DELETE | `/ingest/documents/{document_id}` | Delete document: removes Qdrant vectors + document store record + rebuilds BM25 |
| POST | `/query/` | Ask a question — body: `{query, case_id?, top_k?, rewrite?, messages?}` → `{answer, reasoning, sources, queries_used, chunks_retrieved, chunks_after_rerank}` |

---

## 7. Authentication & Authorization

### Token Mechanics
- **Algorithm:** HS256 JWT
- **Access token expiry:** 7 days (10080 minutes)
- **Refresh token expiry:** 30 days; includes `"refresh": true` claim to prevent use as access token
- **Frontend storage:** HTTP-only cookies (both tokens); not accessible to client-side JS → prevents XSS theft

### Role-Based Access
| Role | Created by | Can do |
|---|---|---|
| `ADMIN` | Seeded or created by other admin | Manage officers, view all |
| `OFFICER` | Admin only | Create/manage cases (if clearance ≥ 4), query RAG |

### Business Access Rules for Cases
1. **View a case:** must be `created_by` OR in `case_assignment` table
2. **Create a case:** must be OFFICER with `clearance_level >= 4`
3. **Delete / transfer a case:** must be `created_by` (the creator)
4. **Assign officers to a case:** must be `created_by` OR (assigned AND `clearance_level > case.created_by.clearance_level`)
5. **Cannot remove case creator** from assignment
6. **Transfer target** must also have `clearance_level >= 4`

---

## 8. Environment Variables

### Backend (`backend/.env`)
```env
DATABASE_URL=postgresql+asyncpg://<user>:<password>@<host>/<db>
```
Other config is hardcoded in `app/core/config.py` — should be moved to env:
- `SECRET_KEY` (currently hardcoded)

### RAG System (`Rag_system/.env` — copy from `.env.example`)
```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_TEXT_COLLECTION=case_text_chunks
QDRANT_IMAGE_COLLECTION=case_image_chunks

EMBEDDER_MODEL=BAAI/bge-base-en-v1.5
EMBEDDER_DEVICE=cpu
EMBEDDING_DIM=768

RERANKER_MODEL=BAAI/bge-reranker-base

RETRIEVAL_TOP_K=50
RERANKER_TOP_K=5

CHUNK_MAX_TOKENS=512
CHUNK_MIN_TOKENS=50
SEMANTIC_SIMILARITY_THRESHOLD=0.3

LLM_BASE_URL=<your-ollama-endpoint>
LLM_MODEL=<model-name>
QUERY_REWRITE_COUNT=2

DOCUMENT_STORE_PATH=data/document_store.db
BM25_INDEX_PATH=data/bm25_index.pkl

API_HOST=0.0.0.0
API_PORT=8080
MAX_UPLOAD_SIZE_MB=50
```

### Frontend
- **No env file exists.** Backend URL is hardcoded as `http://localhost:8000` throughout `src/lib/auth.ts` and other server components.
- A `NEXT_PUBLIC_API_URL` environment variable should be introduced.

---

## 9. Data Flow

```
Browser (port 3000)
  │  HTTP-only cookie: access_token, refresh_token
  ↓
Next.js Server Components / Server Actions
  │  Authorization: Bearer <access_token>
  ↓
Backend FastAPI (port 8000)
  │
  ├──→ PostgreSQL (Neon.tech)
  │     └── users, cases, assignments, documents (mock), activities, analysis (stub)
  │
  └──→ RAG Service (port 8080) — proxied via /cases/{id}/query and /cases/{id}/documents
               │
               ├──→ Qdrant (port 6333)        — dense vector search
               ├──→ PostgreSQL / SQLite        — document status tracking
               ├──→ BM25 (in-memory)           — sparse retrieval index
               └──→ Modal (Qwen 3 30b)        — remote LLM via native Ollama /api/chat (thinking enabled)

Celery Workers (async, separate process)
  ├── Redis (port 6379)  — task broker + result backend + Blackboard store
  └── Uses: HybridRetriever, LLMClient, QdrantStore, Blackboard
```

### Remaining Stubs
The backend **`analysis.py`** endpoint is a stub. The frontend pages `/insights` and `/cctv` are stubs. The chat page (`/cases/[id]/chat`) is fully operational with multi-turn conversation and reasoning transparency.

---

## 10. Integration Status

| Feature | Frontend | Backend | RAG | Status |
|---|---|---|---|---|
| User login/logout | ✅ | ✅ | N/A | **Complete** |
| Admin officer management | ✅ | ✅ | N/A | **Complete** |
| Case CRUD | ✅ | ✅ | N/A | **Complete** |
| Case officer assignment | ✅ | ✅ | N/A | **Complete** |
| Document upload | ✅ (metadata modal: name, category, notes) | ✅ → RAG | ✅ | **Complete** — ingest + `display_name` / `evidence_category` in DB + Qdrant sources |
| Document ingestion to RAG | ✅ | ✅ (proxy) | ✅ | **Complete** |
| RAG chat (`/cases/[id]/chat`) | ✅ (full UI + multi-turn) | ✅ (proxy + history) | ✅ | **Complete** |
| Multi-agent insights | ⚠️ (stub UI) | ❌ | ⚠️ (1 agent only) | **Not wired** |
| CCTV analysis | ⚠️ (stub UI) | ❌ | ❌ | **Not started** |
| Celery task worker | N/A | N/A | ✅ | **Standalone only** |

---

## 11. Known Issues & Gaps

| # | Location | Issue | Severity |
|---|---|---|---|
| 1 | `backend/app/core/config.py` | `SECRET_KEY` is a hardcoded placeholder string | High |
| 2 | `backend/app/main.py` | CORS `allow_origins=["*"]` — open to all origins | Medium |
| 3 | `frontend/src/lib/auth.ts` | Backend URL hardcoded as `http://localhost:8000` throughout all pages — no `NEXT_PUBLIC_API_URL` env var | Medium |
| 4 | `frontend/next.config.mjs` | `ignoreBuildErrors: true` + `ignoreDuringBuilds: true` — TypeScript/ESLint errors silently suppressed at build time | Medium |
| 5 | ~~`backend/app/api/endpoints/documents.py`~~ | ~~File saving is mock~~ **FIXED** — backend now forwards file bytes to RAG `/ingest/file` via httpx; stores `rag_document_id` + `ingest_status` in PostgreSQL | ~~High~~ |
| 6 | `backend/app/api/endpoints/analysis.py` | Always returns "not enabled yet" — `case_analysis` table is never populated | High |
| 7 | ~~`frontend/.../cases/[id]/page.tsx`~~ | ~~Fetches `/officer/{created_by}` — endpoint did not exist~~ **FIXED** — `GET /officer/{id}` added | ~~High~~ |
| 8 | ~~`Rag_system/core/retrieval/bm25_retriever.py`~~ | ~~BM25 index is not auto-rebuilt after ingestion~~ **FIXED** — BM25 is now in-memory, rebuilt from Qdrant on startup + updated after each ingest via shared singleton | ~~Medium~~ |
| 9 | `frontend/src/app/(dashboard)/layout.tsx` | Mobile sidebar toggle button has no `onClick` handler | Low |
| 10 | Backend cases endpoints | No pagination on case/officer lists | Low |
| 11 | RAG ↔ Backend | No service-to-service auth (the RAG service has no API key protection) | Medium |
| 12 | `frontend/` | Next.js 16.1.7: `cookies()`, `headers()`, `params`, `searchParams` are all async Promises — must be `await`-ed. Chat page already uses `params: Promise<{id: string}>`. `insights` and `cctv` stubs still need this when implemented. | Medium |
| 13 | `backend/requirements.txt` | `bcrypt` must stay pinned at `==4.0.1` — version 4.2+ removed `__about__` module that `passlib` requires, causing password verification to crash | High |
| 14 | `Rag_system/` | `RAG_DATABASE_URL` must be set in `.env` for PostgreSQL document tracking; if empty, falls back to local SQLite | Medium |
| 15 | Large file uploads (>10MB PDFs) | Backend proxies file synchronously to RAG — long ingestion times may hit client/nginx timeout. Consider raising timeout or moving to async background task for production. | Medium |

---

## 12. Changelog

> Add a new entry here whenever you make significant changes. Format: `## YYYY-MM-DD — <short description>`

### 2026-03-21 — Upload metadata modal + document deletion + evidence categories

**New feature — Upload metadata modal:**
- Frontend: New `UploadModal` dialog component with fields: file picker, document name (defaults to filename), evidence category dropdown (Case Diary, FIR File, Statement File, Scene of Crime, Forensic/Evidence, Property/Seizure, Arrest & Remand, Other), and optional notes.
- Replaces raw file input in `ChatInterface` — upload button now opens the modal.
- Backend: `POST /cases/{case_id}/documents` now accepts `display_name`, `evidence_category`, `description` as additional form fields. Forwarded to RAG service.
- RAG service: `POST /ingest/file` accepts `display_name` and `evidence_category` form fields. Stored in document metadata and propagated to every Qdrant chunk payload.
- Alembic migration `a3f1c9b84e21_add_document_metadata_fields.py`: adds `display_name`, `evidence_category`, `description` columns to the `document` table. **Applied** via `alembic upgrade head` on the principal PostgreSQL (Neon) database (2026-03-21).
- RAG document stores (SQLite + PostgreSQL): new columns added with ALTER fallback for existing databases.
- Context builder: source headers now show `[Source N] [FIR File] Document Name (Page X)` instead of raw file paths.
- Source references in query responses now include `display_name` and `evidence_category` fields.
- Frontend `SourceCard`: shows document name in bold with evidence category badge.
- Frontend document cards (ChatInterface sidebar + case detail page): show `display_name` with evidence category label.
- **Case detail dashboard** (`/cases/[id]`): Documents sidebar uses the same `UploadModal` as chat (`DocumentList` with `enableUpload`). The previous Plus button was non-functional; upload from the case page now sends `display_name`, `evidence_category`, and `description` like chat. Activity log line prefers `display_name` when present.

### 2026-03-21 — Document deletion across full stack

**New feature — Delete uploaded documents:**
- **Backend:** New `DELETE /cases/{case_id}/documents/{document_id}` endpoint in `documents.py`. Access control: officer must have case access AND clearance level strictly higher than the case's `required_clearance_level`. Admins can always delete. Cascades to RAG service, logs `DOCUMENT_DELETED` activity, removes PostgreSQL `Document` row.
- **RAG service:** New `DELETE /ingest/documents/{document_id}` endpoint in `ingest.py`. Calls `qdrant_store.delete_document()` to remove all vectors from both text and image collections, deletes the metadata record from the document store, and rebuilds the in-memory BM25 index.
- **RAG document stores:** Added `delete(document_id)` method to both `DocumentStore` (SQLite) and `PgDocumentStore` (PostgreSQL).
- **BM25 fix:** `build_index()` now properly resets internal state when called with an empty list (previously left stale index in place).
- **Frontend — Chat page:** `ChatInterface` now accepts `canDeleteDocuments` prop. Chat page fetches `/officer/me` to compute clearance eligibility. Delete button (trash icon) appears on hover over each document card when permitted. Confirmation dialog before deletion.
- **Frontend — Case detail page:** New `DocumentList` client component replaces static document rendering. Shows delete button on hover when user has sufficient clearance.

### 2026-03-20 — Replace Ollama with Modal-hosted Qwen 3 30b + multi-turn chat + reasoning UI

**LLM migration (Ollama → Modal):**
- Replaced `OllamaClient` with `LLMClient` in `Rag_system/core/generation/llm_client.py`. Uses native Ollama `/api/chat` endpoint via `httpx`.
- LLM endpoint and model are read from `Rag_system/.env` (`LLM_BASE_URL`, `LLM_MODEL`) — no credentials in code.
- `LLMClient.generate()` returns an `LLMResponse` dataclass with separate `content` and `reasoning` fields. Thinking mode is enabled (`think: true`) — the model's chain-of-thought is captured in `reasoning` and stripped from `content`.
- Updated `QueryRewriter` to use `LLMClient` — uses only `.content` for generating query variants.
- Updated all agent files (`witness_agent.py`, `supervisor_agent.py`, `timeline_agent.py`, `suspect_agent.py`, `cctv_agent.py`) to use `.content` from `LLMResponse`.
- Settings: replaced `ollama_base_url` / `ollama_model` with `llm_base_url` / `llm_model` in `config/settings.py`.

**Multi-turn conversation history:**
- RAG `POST /query/` now accepts an optional `messages` field (list of `{role, content}` dicts). Previous conversation is passed to the LLM alongside the context-augmented prompt.
- Backend proxy `POST /cases/{case_id}/query` now accepts and forwards `messages` to the RAG service.
- Frontend `ChatInterface.tsx` now sends all previous non-error messages as conversation history with each query. Clicking "Clear" resets the history, starting a fresh conversation.

**Reasoning transparency UI:**
- RAG `POST /query/` response now includes a `reasoning` field with the model's chain-of-thought.
- Backend proxy forwards `reasoning` to frontend.
- Frontend shows a collapsible "View reasoning" button (Brain icon) on assistant messages. Reasoning is hidden by default and displayed in a scrollable container when toggled.
- Max tokens increased to 16384 for answer generation and 2048 for query rewriting to accommodate reasoning + content.

### 2026-03-17 — Case detail page fixes + officer profile endpoint

**Bug fixes:**
- Next.js 16: `params` is now a Promise — fixed `cases/[id]/page.tsx` to use `const { id } = await params` and updated type signature to `params: Promise<{ id: string }>`
- `GET /api/v1/officer/${created_by}` was a 404 (endpoint never existed) — added `GET /officer/{officer_id}` to `officer.py`; placed **after** `/list` in router order to avoid route shadowing
- React key warning on admin dashboard officer rows — replaced bare `<>` fragments with `<React.Fragment key={officer.id}>`

**Note for future agents implementing stub pages (`chat`, `insights`, `cctv`):**
All three pages live under `/cases/[id]/` and will need to accept `params` as a Promise:
```typescript
export default async function ChatPage({ params }: { params: Promise<{ id: string }> }) {
    const { id } = await params;
```

### 2026-03-17 — Admin case management + bug fixes

**Bug fixes:**
- `bcrypt 5.x` incompatibility with `passlib` — pinned `bcrypt==4.0.1` in `backend/requirements.txt`
- Next.js 16 breaking change: `cookies()` is now async — updated all `cookies()` calls in `frontend/src/lib/auth.ts` to `await cookies()`
- `redirect()` in Next.js server actions throws an error with `.digest` not `.message` — fixed catch block check
- `DELETE /admin/officers/{id}` crashed with `IntegrityError` (SQLAlchemy tried to SET `created_by = NULL`) — fixed by blocking deletion when officer owns cases, and manually deleting FK references before user row deletion

**New features — Admin Case Management:**
- Backend: 4 new endpoints under `/admin/cases` (list, delete case, assign officer, remove officer)
- Admin dashboard: officer rows now have a "Cases" expand button — shows a sub-table of all cases created by that officer
- Per case: delete button (admin can delete any case) and "Assign Officer" button (opens a modal listing only officers with clearance ≥ case required level)
- Admins cannot navigate into case details — management only

### 2026-03-17 — RAG-Cases integration + global storage fix

**New features:**
- Backend `POST /cases/{case_id}/documents`: now actually forwards file bytes to RAG `/ingest/file` via `httpx`. Stores `rag_document_id` and `ingest_status` back in PostgreSQL. File types validated; 300s timeout for large PDFs.
- New backend endpoint `POST /cases/{case_id}/query`: validates case access, proxies to RAG `/query/` with `case_id` filter. Returns `{answer, sources, chunks_retrieved, chunks_after_rerank}`.
- Frontend `/cases/[id]/chat` page: fully implemented with two-panel layout (evidence file list + chat), document upload, ingest status badges, message history, source citations.
- New `backend/app/api/endpoints/rag_query.py` registered in `api.py`.
- `backend/app/core/config.py`: added `RAG_SERVICE_URL` (defaults to `http://localhost:8080`).
- `backend/requirements.txt`: added `httpx`.

**RAG system changes:**
- New `Rag_system/stores/pg_document_store.py`: drop-in PostgreSQL replacement for SQLite `DocumentStore` using `psycopg2`. Auto-creates `rag_documents` table. Selected when `RAG_DATABASE_URL` is set in `.env`.
- `Rag_system/ingestion/pipeline.py`: uses `PgDocumentStore` if `RAG_DATABASE_URL` is configured, falls back to SQLite.
- `Rag_system/core/retrieval/bm25_retriever.py`: removed pickle persistence entirely. Index is always in-memory; rebuilt from Qdrant on startup and updated via `update_index()` after each ingest.
- New `Rag_system/api/shared_state.py`: module-level `BM25Retriever` singleton shared between ingest and query routes.
- `Rag_system/api/app.py` lifespan: on startup, scrolls all Qdrant text chunks via `get_all_texts()` and builds BM25 in-memory.
- `Rag_system/api/routes/ingest.py`: after each successful ingest, calls `bm25.update_index(new_pairs)` on the shared singleton.
- `Rag_system/api/routes/query.py`: uses `shared_state.bm25` instead of a separate module-level instance.
- `Rag_system/requirements.txt`: added `psycopg2-binary`.
- `Rag_system/.env.example`: added `RAG_DATABASE_URL` field.

**Database migration:**
- New Alembic migration `de7856598563_add_rag_fields_to_document.py`: adds `filename`, `rag_document_id`, `ingest_status` columns to `document` table. Applied to Neon.tech.

**Document model + schema:**
- `backend/app/models/document.py`: added `filename`, `rag_document_id`, `ingest_status` fields.
- `backend/app/schemas/document.py`: exposed new fields in response schema.

### 2026-03-17 — Initial context document created

- Performed full codebase exploration
- Documented all three services (backend, frontend, RAG system), database schemas, API endpoints, auth flow, data flow, integration status, and known issues
- No code changes made in this session
