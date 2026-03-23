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
- Query those documents with AI via a Retrieval-Augmented Generation (RAG) system
- Get AI-generated **investigation insights** via a **multi-agent pipeline** implemented inside `Rag_system/` (LangGraph + Celery + Redis blackboard) — **frontend Insights page is still a placeholder**; there is **no backend proxy** to `/agents` yet
- Analyze CCTV footage *(stub only)*

**Three independently runnable services:**

| Service | Language/Framework | Port | Status |
|---|---|---|---|
| `backend/` | Python / FastAPI | `8000` | Fully operational |
| `frontend/` | TypeScript / Next.js 16.1.7 | `3000` | Fully operational |
| `Rag_system/` | Python / FastAPI | `8080` | Operational — `/ingest`, `/query`, `/agents` (+ Celery worker for agent jobs) |

**Supporting infrastructure:**

| Service | Purpose | Port |
|---|---|---|
| PostgreSQL (Neon.tech cloud) | Main relational DB | cloud |
| Qdrant | Vector database for RAG | `6333` (moved to remote server — configure `QDRANT_HOST` in `Rag_system/.env`) |
| Redis | Celery broker + Blackboard | `6379` |
| Ollama | Local LLM serving | `11434` |

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
│       │       ├── documents.py  ← Document upload/list/delete + RAG proxy + metadata
│       │       ├── rag_query.py  ← POST /cases/{id}/query → RAG
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
│       │           ├── chat/page.tsx              ← RAG chat (full UI)
│       │           ├── insights/page.tsx          ← InsightsPanel → backend `/insights/*`
│       │           └── cctv/page.tsx              ← STUB: CCTV analysis
│       ├── components/
│       │   ├── ui/               ← shadcn/ui + Radix UI primitives
│       │   ├── auth/             ← Login form component
│       │   ├── navigation/       ← Sidebar (Insights link → `/cases/[id]/insights`)
│       │   ├── admin/            ← Admin-specific components (OfficerModal, etc.)
│       │   ├── officer/          ← Officer-specific components (CaseCard, etc.)
│       │   └── cases/            ← ChatInterface, UploadModal, DocumentList
│       └── lib/
│           ├── auth.ts           ← Server actions: login, logout, getAccessToken
│           └── utils.ts          ← cn() utility, etc.
│
├── Rag_system/                   ← RAG + Insights microservice (port 8080)
│   ├── .env.example              ← Template for RAG env config
│   ├── .venv/                    ← Local Python venv
│   ├── api/
│   │   ├── app.py                ← FastAPI: `/ingest`, `/query`, `/agents`, lifespan BM25 warmup
│   │   ├── shared_state.py       ← Shared Qdrant + BM25 singletons
│   │   └── routes/
│   │       ├── ingest.py         ← /ingest/* (file ingest, delete, BM25 rebuild)
│   │       ├── query.py          ← POST /query/ (RAG Q&A)
│   │       └── agents.py         ← /agents/* (Insights pipeline HTTP API — see §6)
│   ├── config/
│   │   └── settings.py           ← + `upload_dir`, `memory_dir`, Ollama, Qdrant, …
│   ├── ingestion/
│   │   ├── pipeline.py           ← Docling path: load → clean → chunk → embed → Qdrant
│   │   ├── extractor.py          ← pypdf / docx / txt / image OCR — used by **agent** document uploads
│   │   ├── loaders/
│   │   │   ├── pdf_loader.py
│   │   │   └── image_loader.py
│   │   └── processors/
│   │       └── cleaner.py
│   ├── core/
│   │   ├── documents/
│   │   │   ├── models.py
│   │   │   ├── chunker.py
│   │   │   ├── classifier.py     ← Police doc-type classifier (keywords + optional LLM)
│   │   │   └── manager.py        ← Per-case per-agent markdown memory (`MEMORY_DIR`)
│   │   ├── embeddings/
│   │   │   └── local_embedder.py
│   │   ├── retrieval/
│   │   │   ├── hybrid_retriever.py
│   │   │   ├── bm25_retriever.py ← In-memory BM25 (no pickle persistence in current code)
│   │   │   └── agent_retriever.py← `ingest_document` / `query_rag` for agent pipeline
│   │   ├── reranking/
│   │   │   └── bge_reranker.py
│   │   └── generation/
│   │       └── llm_client.py     ← OllamaClient (RAG answer + streaming)
│   ├── stores/
│   │   ├── qdrant_store.py
│   │   ├── document_store.py
│   │   └── pg_document_store.py  ← optional Neon `rag_documents`
│   ├── query/
│   │   ├── context_builder.py
│   │   └── query_rewriter.py
│   ├── agents/
│   │   ├── base_agent.py         ← Shared specialist logic: LLM extract → RAG → memory → `post_finding`
│   │   ├── specialists.py        ← 7 agents: FIR, case_diary, statement, scene_of_crime, forensic, seizure, arrest_remand
│   │   └── supervisor.py         ← LangChain Ollama: reads `format_brief()` → JSON cross-case analysis
│   └── orchestration/
│       ├── celery_app.py
│       ├── tasks.py              ← `process_document`, `run_supervisor`, `classify_only`
│       ├── blackboard.py         ← Redis lists + pub/sub (messages, anomalies, findings, insights)
│       └── graph/
│           ├── state.py          ← `InvestigationState`, `BlackboardMessage`, FILE_TYPE literals
│           └── graph.py          ← LangGraph: router → specialist node → supervisor → END
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
| **Sparse retrieval** | rank-bm25 (BM25Okapi) | In-memory index rebuilt from Qdrant (see `api/shared_state.py`) |
| **Vector database** | Qdrant | Local, port 6333 |
| **Reranker** | sentence-transformers | BAAI/bge-reranker-base |
| **Local LLM** | Ollama (HTTP) | `OLLAMA_BASE_URL` / `OLLAMA_MODEL` — **RAG:** `OllamaClient` (`core/generation/llm_client.py`) + `query/query_rewriter.py`; **Agents:** `langchain_ollama.ChatOllama` in `agents/base_agent.py` and `agents/supervisor.py` |
| **Agent orchestration** | LangGraph | `langgraph` — compiled graph in `orchestration/graph/graph.py` |
| **Agent framework** | LangChain | `langchain`, `langchain-core`, `langchain-ollama` |
| **Task queue** | Celery | Redis broker/backend — `process_document`, `run_supervisor` |
| **Cache / Blackboard** | Redis | port 6379 — keys `case:{id}:messages|anomalies|findings|insights|status` + pub/sub `blackboard:{id}` |

---

## 4. Component Details

### 4.1 Backend (FastAPI) — `backend/app`

**Entry point:** `app/main.py` — mounts all routers under `/api/v1`, configures CORS (`allow_origins=["*"]` — needs tightening in production).

**Config** (`app/core/config.py`):
- `SECRET_KEY` — hardcoded dev placeholder; **must be moved to env var before production**
- `ACCESS_TOKEN_EXPIRE_MINUTES` = 10080 (7 days)
- `REFRESH_TOKEN_EXPIRE_MINUTES` = 43200 (30 days)
- `DATABASE_URL` — read from env, fallback to `postgresql+asyncpg://postgres:postgres@localhost:5432/aegis`
- `RAG_SERVICE_URL` — default `http://localhost:8080` (document upload + chat query proxy)

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

---

### 4.2 Frontend (Next.js) — `frontend/`

**Auth flow** (`src/lib/auth.ts`):
1. `login(username, password)` → POST to `{getApiV1Url()}/auth/login` (default `http://localhost:8000/api/v1`)
2. Receives `access_token` + `refresh_token` → stores both in HTTP-only cookies
3. Probes `GET /api/v1/officer/me` with the access token
   - 200 → officer → redirect to `/officer/dashboard`
   - Non-200 → admin → redirect to `/admin/dashboard`
4. `logout()` → deletes both cookies
5. `getAccessToken()` → reads `access_token` cookie from server context

**Backend URL:** `getApiV1Url()` in `src/lib/api.ts` — set **`NEXT_PUBLIC_API_URL`** in `.env.local` (see `frontend/.env.example`).

**Protected route pattern:** The `(dashboard)/layout.tsx` calls `getAccessToken()` — if no token, redirects to `/`.

**Key pages:**

| Route | File | Status |
|---|---|---|
| `/` | `app/page.tsx` | Login page, fully working |
| `/officer/dashboard` | `(dashboard)/officer/dashboard/page.tsx` | Fully working |
| `/admin/dashboard` | `(dashboard)/admin/dashboard/page.tsx` | Fully working |
| `/cases/[id]` | `(dashboard)/cases/[id]/page.tsx` | Fully working |
| `/cases/[id]/chat` | `(dashboard)/cases/[id]/chat/page.tsx` | **Complete** — RAG via backend proxy |
| `/cases/[id]/insights` | `insights/page.tsx` + `InsightsPanel.tsx` | Blackboard, upload, report, task poll via backend proxy |
| `/cases/[id]/cctv` | `(dashboard)/cases/[id]/cctv/page.tsx` | **STUB** |

**How to run:**
```bash
cd frontend
npm install
npm run dev
```

---

### 4.3 RAG System (Microservice) — `Rag_system/`

**Entry point:** `api/app.py` — FastAPI app, mounts **`/ingest`**, **`/query`**, and **`/agents`** routers. Lifespan warms Qdrant + rebuilds BM25 from existing chunks; pre-loads embedder.

**Full query pipeline (per request to `POST /query/`):**
```
User query
  → QueryRewriter        — alternative phrasings (Ollama HTTP / local model)
  → HybridRetriever      — dense (Qdrant) + sparse (BM25) per variant → RRF → top-50
  → BGEReranker          — cross-encoder → top-k
  → context_builder      — cited context + source metadata for UI
  → OllamaClient         — final answer (streaming supported)
  → QueryResponse        — answer + sources (+ optional reasoning fields per branch)
```

**Insights / multi-agent pipeline (separate from `/query/`):**
- **HTTP:** `api/routes/agents.py` under prefix `/agents` (see §6).
- **Async:** Celery tasks in `orchestration/tasks.py` — **`process_document`** (full chain) and **`run_supervisor`** (re-read blackboard, post cross-case insights).
- **Flow:** upload file → extract text (`ingestion/extractor.py`) → classify (`core/documents/classifier.py`) → optional RAG ingest (`core/retrieval/agent_retriever.ingest_document`) → **`investigation_graph.invoke()`** (`orchestration/graph/graph.py`): **router** picks specialist by `file_type` → one of **7 specialist agents** (`agents/specialists.py` + `base_agent.py`) → **SupervisorAgent** (`supervisor.py`) consumes `format_brief(case_id)` → structured JSON + narrative report.
- **Blackboard:** Specialists call `post_finding` / supervisor calls `post_insight`; data stored in Redis and exposed via **GET** blackboard + **SSE** stream endpoints.
- **Per-agent disk memory:** `core/documents/manager.py` appends to markdown files under `MEMORY_DIR` (default `data/memory_store`).
- **Upload storage:** Agent uploads land in `UPLOAD_DIR` (default `data/uploads`), distinct from main-app document flow via backend `POST /cases/.../documents`.

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

**BM25 index:** In-memory singleton (`api/shared_state.bm25`); built from Qdrant on API startup and rebuilt after ingest/delete routes (see `ingest.py`). Agent code paths that construct their own `BM25Retriever()` may not share the API process index.

**How to run:**
```bash
cd Rag_system
# Copy and fill .env.example → .env
pip install -r requirements.txt   # (if requirements.txt exists; else use .venv)
uvicorn api.app:app --reload --port 8080
```

**Prerequisites:**
- Qdrant: hosted on Ubuntu server, accessed via ngrok tunnel — set `QDRANT_URL` in `Rag_system/.env` (update each time ngrok restarts). Also set `QDRANT_API_KEY` if auth is enabled.
- Ollama running with model: `ollama run qwen3.5:2b`
- Redis (Celery + Insights blackboard): locally `docker run -p 6379:6379 redis`, or **remote via ngrok TCP** — see note below (set `REDIS_URL` in `Rag_system/.env`).

**Redis via ngrok TCP (server has Redis in Docker; devs on different PCs):** Use `ngrok tcp 6379` on the server; put `REDIS_URL=redis://default:PASSWORD@HOST:PORT/0` on each machine (Redis 6+ ACL; legacy auth may use `redis://:PASSWORD@...`). Host/port change when ngrok restarts. Do **not** use the Qdrant HTTPS ngrok URL for Redis.

**How to run Celery worker (required for `/agents` queued jobs):**
```bash
cd Rag_system
celery -A orchestration.celery_app worker --loglevel=info
```

---

### 4.4 Multi-Agent / Insights System (inside `Rag_system/`)

This subsystem powers **investigation-style insights** (specialist document analysis + supervisor synthesis). It is **not** the same code path as **`POST /query/`** RAG chat.

| Layer | Role |
|---|---|
| **FastAPI** | `api/routes/agents.py` — upload, classify preview, task polling, blackboard JSON, SSE stream, per-agent memory read, supervisor report trigger |
| **Celery** | `orchestration/tasks.py` — `process_document`, `run_supervisor`, `classify_only` |
| **LangGraph** | `orchestration/graph/graph.py` — compiled graph `investigation_graph`: START → router → `{file_type}_node` → supervisor → END |
| **State** | `orchestration/graph/state.py` — `InvestigationState`, `BlackboardMessage` (includes `insights`, `inconsistencies`, `rag_queries_made`, …) |
| **Specialists** | `agents/specialists.py` — 7 `BaseAgent` subclasses (FIR, case diary, statement, scene of crime, forensic, seizure, arrest/remand) |
| **Supervisor** | `agents/supervisor.py` — reads markdown brief from `blackboard.format_brief()`, returns `supervisor_report`, `cross_inconsistencies`, `final_status` |
| **Blackboard** | `orchestration/blackboard.py` — `post_message`, `post_anomaly`, `post_finding`, `post_insight`, `read_all`, `subscribe_to_case`, `format_brief`, case status keys |

**Case ID bridge:** Map backend **`Case.id` (UUID)** → RAG **`/agents` int** with **`uuid_to_insights_case_id`** in `backend/app/core/insights_case_id.py` (keep in sync with `Rag_system/core/insights_case_id.py`). Backend proxy uses this when calling RAG.

**Removed / replaced:** Earlier `witness_agent.py`-style single task is **not** the current architecture; specialists are **`BaseAgent`-driven** LangGraph nodes.

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
| `file_path` | String | RAG reference path |
| `filename` | String | Original upload name |
| `display_name` | String | User label (upload modal) |
| `evidence_category` | String | Police category key |
| `description` | String | Optional notes |
| `rag_document_id` | String | RAG document UUID |
| `ingest_status` | String | Ingest state |
| `created_at` | DateTime(tz) | |

#### `activity_log` table
| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `case_id` | UUID | FK → case.id, Indexed |
| `user_id` | UUID | FK → user.id, Indexed |
| `action` | String | `CASE_CREATED`, `OFFICER_ASSIGNED`, `OFFICER_REMOVED`, `DOCUMENT_UPLOADED`, `DOCUMENT_DELETED`, … |
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
| `display_name` | TEXT | Optional user label |
| `evidence_category` | TEXT | Optional police category |
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

All backend endpoints are prefixed `/api/v1`. Base URL: `NEXT_PUBLIC_API_URL` or `http://localhost:8000`

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
| POST | `/cases/{case_id}/documents` | Upload + RAG ingest (multipart: `file`, **required** `display_name` + `evidence_category`, optional `description`) |
| GET | `/cases/{case_id}/documents` | List documents for case |
| DELETE | `/cases/{case_id}/documents/{document_id}` | Delete row + best-effort `DELETE /ingest/documents/{rag_document_id}` (vectors + BM25 rebuild on RAG) |

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
| POST | `/ingest/file` | Upload + ingest single file (multipart: `file`, optional `case_id`, `officer_id`, `display_name`, `evidence_category`) |
| POST | `/ingest/batch` | Upload + ingest multiple files |
| GET | `/ingest/status/{document_id}` | Ingestion status |
| GET | `/ingest/documents` | List ingested documents |
| DELETE | `/ingest/documents/{document_id}` | Delete vectors + store row + rebuild BM25 |
| POST | `/query/` | RAG Q&A — body includes `query`, optional `case_id`, `top_k`, `rewrite`, `messages`, … |

### RAG — Insights / Agents API (same host, **no JWT**; intended for internal or future backend proxy)

Base path: **`/agents`**. `case_id` in these routes is typed as **`int`** in FastAPI (Redis key namespace) — see §11 for UUID mismatch with main app.

| Method | Path | Description |
|---|---|---|
| POST | `/agents/cases/{case_id}/upload` | Multipart: `file`, optional `file_type`, `auto_classify` → queues **`process_document`** Celery task; returns `task_id`, poll/stream URLs |
| POST | `/agents/cases/{case_id}/classify-preview` | Quick classify upload (no full pipeline) |
| GET | `/agents/tasks/{task_id}` | Celery task status + result |
| GET | `/agents/cases/{case_id}/blackboard` | JSON: `messages`, `anomalies`, `findings`, `insights`, `status` |
| GET | `/agents/cases/{case_id}/blackboard/brief` | Markdown brief fed to supervisor |
| GET | `/agents/cases/{case_id}/stream` | **SSE** — live blackboard events (`text/event-stream`) |
| POST | `/agents/cases/{case_id}/report` | Queue **`run_supervisor`** (refresh consolidated report) |
| GET | `/agents/cases/{case_id}/memory/{agent_type}` | Raw markdown memory for agent (`fir`, `case_diary`, …) |

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
RERANKER_TOP_K=7

CHUNK_MAX_TOKENS=512
CHUNK_MIN_TOKENS=50
SEMANTIC_SIMILARITY_THRESHOLD=0.3

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:2b
QUERY_REWRITE_COUNT=2
# Final answer: room for visible answer + chain-of-thought in the same completion
LLM_MAX_TOKENS=8192
# Query rewrite completion (reasoning + lines); raise if rewrites stay empty
QUERY_REWRITE_MAX_TOKENS=2048
# Prior user/assistant turns in the answer prompt only (not sent to rewriter)
CHAT_HISTORY_MAX_MESSAGES=16
# Soft budget (estimated tokens) for prior chat + document passages in the user prompt
RAG_COMBINED_CONTEXT_BUDGET_TOKENS=16000
RAG_CONTEXT_PASSAGES_MIN_TOKENS=3500
# Ollama `think` for all generations; rewrites still use split content only (reasoning dropped)
OLLAMA_ENABLE_THINKING=true

DOCUMENT_STORE_PATH=data/document_store.db
BM25_INDEX_PATH=data/bm25_index.pkl
RAG_DATABASE_URL=   # optional — Neon PostgreSQL for rag_documents

UPLOAD_DIR=data/uploads      # Insights agent uploads (agents.py)
MEMORY_DIR=data/memory_store # Per-agent markdown memory

API_HOST=0.0.0.0
API_PORT=8080
MAX_UPLOAD_SIZE_MB=50
```

### Frontend
- Copy **`frontend/.env.example`** → `.env.local` and set **`NEXT_PUBLIC_API_URL`** if the API is not on localhost.
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
  └──→ [NOT YET CONNECTED] RAG Service (port 8080)
               │
               ├──→ Qdrant (port 6333)        — dense vector search
               ├──→ SQLite data/document_store.db — document status tracking
               ├──→ data/bm25_index.pkl        — sparse BM25 index
               └──→ Ollama (port 11434)        — Qwen3.5:2b LLM

Celery Workers (async, separate process)
  ├── Redis (port 6379)  — task broker + result backend + Blackboard store
  └── Uses: HybridRetriever, OllamaClient, QdrantStore, Blackboard
```

### Critical Missing Integration
The backend **`analysis.py`** endpoint is a stub. The three frontend pages (`/chat`, `/insights`, `/cctv`) are all stubs. **No HTTP call from backend → RAG service currently exists.**

---

## 10. Integration Status

| Feature | Frontend | Backend | RAG | Status |
|---|---|---|---|---|
| User login/logout | ✅ | ✅ | N/A | **Complete** |
| Admin officer management | ✅ | ✅ | N/A | **Complete** |
| Case CRUD | ✅ | ✅ | N/A | **Complete** |
| Case officer assignment | ✅ | ✅ | N/A | **Complete** |
| Document upload | ✅ | ✅ → RAG | ✅ | **Complete** — real ingest via proxy |
| Document ingestion to RAG | ✅ | ✅ (proxy) | ✅ | **Complete** |
| RAG chat (`/cases/[id]/chat`) | ✅ (full UI) | ✅ (proxy) | ✅ | **Complete** |
| Multi-agent insights | ✅ Insights UI | ✅ `rag_insights.py` | ✅ `/agents` + Celery | **Run RAG 8080 + worker + Redis** |
| CCTV analysis | ⚠️ (stub UI) | ❌ | ❌ | **Not started** |
| Celery (Insights pipeline) | N/A | N/A | ✅ `process_document`, `run_supervisor` | **Run worker manually** — not invoked from main app |

---

## 11. Known Issues & Gaps

| # | Location | Issue | Severity |
|---|---|---|---|
| 1 | `backend/app/core/config.py` | `SECRET_KEY` is a hardcoded placeholder string | High |
| 2 | `backend/app/main.py` | CORS `allow_origins=["*"]` — open to all origins | Medium |
| 3 | `frontend/src/lib/api.ts` | Use **`NEXT_PUBLIC_API_URL`** for non-local API (`getApiV1Url()`) | Low |
| 4 | `frontend/next.config.mjs` | `ignoreBuildErrors: true` + `ignoreDuringBuilds: true` — TypeScript/ESLint errors silently suppressed at build time | Medium |
| 5 | ~~`backend/app/api/endpoints/documents.py`~~ | ~~File saving is mock~~ **FIXED** — backend now forwards file bytes to RAG `/ingest/file` via httpx; stores `rag_document_id` + `ingest_status` in PostgreSQL | ~~High~~ |
| 6 | `backend/app/api/endpoints/analysis.py` | Always returns "not enabled yet" — `case_analysis` table is never populated | High |
| 7 | ~~`frontend/.../cases/[id]/page.tsx`~~ | ~~Fetches `/officer/{created_by}` — endpoint did not exist~~ **FIXED** — `GET /officer/{id}` added | ~~High~~ |
| 8 | ~~`Rag_system/core/retrieval/bm25_retriever.py`~~ | ~~BM25 index is not auto-rebuilt after ingestion~~ **FIXED** — BM25 is now in-memory, rebuilt from Qdrant on startup + updated after each ingest via shared singleton | ~~Medium~~ |
| 9 | `frontend/src/app/(dashboard)/layout.tsx` | Mobile sidebar toggle button has no `onClick` handler | Low |
| 10 | Backend cases endpoints | No pagination on case/officer lists | Low |
| 11 | RAG ↔ Backend | No service-to-service auth (the RAG service has no API key protection) | Medium |
| 12 | `frontend/` | Next.js 16: `insights/page.tsx` uses `await params` (like chat). `cctv` stub still static. | Medium |
| 13 | `backend/requirements.txt` | `bcrypt` must stay pinned at `==4.0.1` — version 4.2+ removed `__about__` module that `passlib` requires, causing password verification to crash | High |
| 14 | `Rag_system/` | `RAG_DATABASE_URL` must be set in `.env` for PostgreSQL document tracking; if empty, falls back to local SQLite | Medium |
| 15 | Large file uploads (>10MB PDFs) | Backend proxies file synchronously to RAG — long ingestion times may hit client/nginx timeout. Consider raising timeout or moving to async background task for production. | Medium |

---

## 12. Changelog

> Add a new entry here whenever you make significant changes. Format: `## YYYY-MM-DD — <short description>`

### 2026-03-21 — PROJECT_CONTEXT: document Insights / multi-agent stack

**Documentation sync (code already in repo):**
- Described **LangGraph** investigation graph (`orchestration/graph/`), **7 specialist agents** + **SupervisorAgent**, **Celery** tasks (`process_document`, `run_supervisor`), **Redis blackboard** API (`post_finding`, `post_insight`, `read_all`, SSE `stream`), and **`api/routes/agents.py`** REST surface under **`/agents`**.
- Updated directory tree, tech stack (`langgraph`, `langchain-ollama`), RAG §6 API tables, data-flow diagram, integration matrix, and **known gaps** (UUID vs int `case_id`, no backend/Next.js wiring for Insights, hardcoded Redis host, dual upload paths).
- Removed obsolete references to **`witness_agent.py`** as the primary multi-agent implementation.

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

### 2026-03-21 — Document upload metadata (display name, evidence type)

**Backend**
- Alembic `f3a8c2b91d4e`: `document.display_name`, `document.evidence_category`, `document.description` (nullable).
- `POST /cases/{case_id}/documents`: multipart `file` plus optional form fields `display_name`, `evidence_category`, `description`; values persisted on `Document` and forwarded to RAG `/ingest/file`.
- `rag_query.SourceReference`: optional `display_name`, `evidence_category` (proxied from RAG query JSON).

**RAG**
- `DocumentMetadata`: optional `display_name`, `evidence_category`, `description`; packed into existing `extra_metadata` JSON via `pack_document_extra_metadata` / `unpack_document_extra_blob`.
- `POST /ingest/file`: same optional form fields; pipeline copies display/category into Qdrant chunk payloads for citations; description stored in document record only (not duplicated on every chunk).
- `GET /ingest/status/{id}` and `GET /ingest/documents`: include display/category (and description on status) where set.
- Query `SourceReference` + `context_builder`: citations use `display_name` in headers when present; sources expose `evidence_category`.

**Frontend**
- Chat evidence panel: optional display name, evidence-type select, short note before upload; document cards and source citations show labels when present.
- Case detail page: activity + documents sidebar show display name and evidence type.

**Ops:** run `alembic upgrade head` in `backend/` after pull.

**Alembic / Neon:** If `alembic` errors with `Can't locate revision 'a3f1c9b84e21'`, pull the placeholder migration `a3f1c9b84e21_align_placeholder.py` (linearizes `de7856598563` → `a3f1c9b84e21` → `f3a8c2b91d4e`). Then `alembic upgrade head` or `alembic stamp f3a8c2b91d4e` as needed. Direct SQL fix: `python backend/scripts/neon_db_inspect.py --stamp f3a8c2b91d4e`.

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
