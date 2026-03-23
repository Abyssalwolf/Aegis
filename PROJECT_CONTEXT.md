# AEGIS — AI Police Assistance System: Project Context

> **For AI agents:** This file is the canonical map of architecture, env vars, APIs, and integration points. **Update it** when you change behavior across services—especially LLM wiring, Celery/RAG ingest, or Insights—and add a dated entry under [Changelog](#12-changelog).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Technology Stack](#3-technology-stack)
4. [Component Details](#4-component-details)
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

**AEGIS** is a case-centric intelligence platform for police workflows:

- Cases with clearance-based access, assignments, activity logs
- Evidence upload with **display name**, **evidence category**, optional description
- **RAG chat** over ingested case documents (`/cases/[id]/chat`)
- **AI Investigation / Insights**: specialist agents + supervisor over a **Redis blackboard**, triggered via **`InsightsPanel`** → backend **`rag_insights.py`** → RAG **`/agents`** (requires **RAG API**, **Celery worker**, **Redis**, **Qdrant**, **LLM**)

**Three runnable services:**

| Service | Stack | Default port | Role |
|--------|--------|--------------|------|
| `backend/` | FastAPI | `8000` | Auth, cases, documents, proxies to RAG (`/query`, `/ingest`, `/agents` patterns) |
| `frontend/` | Next.js 16 (App Router) | `3000` | Dashboard, case UI, chat, insights |
| `Rag_system/` | FastAPI + Celery | `8080` | Ingest, hybrid retrieval, `/query`, `/agents`, LangGraph agents |

**Infrastructure:**

| Component | Purpose | Notes |
|-----------|---------|--------|
| PostgreSQL (e.g. Neon) | Main app DB | Users, cases, documents, `case_analysis` |
| Qdrant | Vectors | `QDRANT_URL` or `QDRANT_HOST`/`PORT`; collections `case_text_chunks`, `case_image_chunks` |
| Redis | Celery broker + result backend + blackboard | **`REDIS_URL`** — prefer **local or managed**; **ngrok TCP** is fragile (idle disconnects → redelivery) |
| LLM | All RAG + agent + supervisor + classifier completions | **`LLM_BASE_URL` + `LLM_MODEL` (+ optional `LLM_API_KEY`)** → OpenAI-compatible `POST …/v1/chat/completions`. If `LLM_BASE_URL` is empty, **`OllamaClient`** falls back to **`OLLAMA_BASE_URL` + `/api/generate`** |

---

## 2. Directory Structure

```
Major-project/
├── PROJECT_CONTEXT.md
├── backend/                          # FastAPI — port 8000
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── config.py             # DATABASE_URL, RAG_SERVICE_URL, SECRET_KEY, …
│   │   │   ├── insights_case_id.py   # uuid → deterministic int for RAG /agents + Redis
│   │   │   └── evidence_category_mapping.py  # Maps UI category → agent file_type
│   │   ├── api/
│   │   │   ├── api.py                # Mounts routers under /api/v1
│   │   │   ├── deps.py
│   │   │   └── endpoints/
│   │   │       ├── auth.py
│   │   │       ├── admin.py
│   │   │       ├── officer.py
│   │   │       ├── cases.py
│   │   │       ├── documents.py      # Upload: parallel RAG ingest + agent queue
│   │   │       ├── rag_query.py
│   │   │       ├── rag_insights.py   # JWT proxies → RAG /agents (blackboard, SSE, tasks, report)
│   │   │       └── analysis.py       # case_analysis snapshots (brief + blackboard)
│   │   ├── models/, schemas/, db/
│   ├── alembic/versions/
│   ├── seed.py
│   └── requirements.txt
│
├── frontend/                         # Next.js — port 3000
│   └── src/
│       ├── app/(dashboard)/
│       │   ├── admin/dashboard/
│       │   ├── officer/dashboard/
│       │   └── cases/[id]/
│       │       ├── page.tsx
│       │       ├── chat/page.tsx
│       │       ├── insights/page.tsx
│       │       └── cctv/page.tsx     # stub
│       ├── components/cases/
│       │   ├── InsightsPanel.tsx     # Blackboard, SSE, supervisor report UI (task result)
│       │   ├── ChatInterface.tsx
│       │   ├── CaseAnalysisSnapshots.tsx
│       │   └── …
│       └── lib/api.ts, auth.ts
│
└── Rag_system/                       # RAG + agents — port 8080
    ├── api/
    │   ├── app.py                    # Lifespan: Qdrant + BM25 warmup; GET /health
    │   ├── shared_state.py           # Qdrant, BM25, embedder, reranker singletons
    │   └── routes/
    │       ├── ingest.py
    │       ├── query.py
    │       └── agents.py             # Upload (sync disk write), tasks, blackboard, SSE, report
    ├── config/settings.py
    ├── core/
    │   ├── generation/
    │   │   ├── llm_client.py         # OllamaClient: OpenAI-compatible OR Ollama /api/generate
    │   │   └── agent_chat.py         # get_agent_llm_client(), agent_llm_complete()
    │   ├── documents/                # chunker, classifier, manager (per-agent markdown memory)
    │   ├── embeddings/, reranking/, retrieval/
    └── ingestion/
        ├── pipeline.py               # ingest_file(…, logical_source_path=…) for dedup
        ├── extractor.py              # Used by Celery agent path (pypdf, docx, txt, images)
        └── loaders/pdf_loader.py     # Docling PDF → markdown + embedded images
    ├── agents/
    │   ├── base_agent.py             # agent_llm_complete (no ChatOllama)
    │   ├── specialists.py
    │   └── supervisor.py
    ├── orchestration/
    │   ├── celery_app.py             # Windows: worker_pool=solo
    │   ├── tasks.py                  # process_document, run_supervisor, classify_only
    │   ├── blackboard.py
    │   └── graph/graph.py            # LangGraph: router → specialist → supervisor
    ├── query/, stores/, core/retrieval/agent_retriever.py
    └── core/insights_case_id.py      # Mirror of backend mapping (if present for tooling)
```

---

## 3. Technology Stack

| Layer | Technology | Notes |
|-------|------------|--------|
| Frontend | Next.js **16.1.x**, React 18, TS | `getApiV1Url()` ← `NEXT_PUBLIC_API_URL` |
| UI | Tailwind, shadcn/ui, Radix, Lucide | |
| Backend | FastAPI, Pydantic v2, SQLAlchemy 2 async + asyncpg | |
| Auth | JWT (python-jose), bcrypt via passlib | Pin **`bcrypt==4.0.1`** for passlib |
| RAG ingest (PDF path) | **Docling** | `ingestion/loaders/pdf_loader.py` |
| Agent text extract | **pypdf**, python-docx, txt, images | `ingestion/extractor.py` |
| Embeddings / rerank | sentence-transformers | bge-base-en-v1.5, bge-reranker-base |
| Vector DB | qdrant-client | |
| Sparse | rank-bm25 | In-process; rebuilt from Qdrant + updates on ingest |
| **LLM (unified)** | **httpx** in `OllamaClient` | **`LLM_*`** → `/v1/chat/completions`; else Ollama `/api/generate` |
| Agents | **LangGraph** + **langchain-core** | **No `langchain-ollama`** in requirements |
| Queue | Celery 5, redis-py | Broker + backend = `REDIS_URL` |

---

## 4. Component Details

### 4.1 Backend (`backend/app`)

- **Entry:** `main.py` — CORS open in dev (`allow_origins=["*"]`).
- **Config:** `core/config.py` — `DATABASE_URL`, `RAG_SERVICE_URL` (default `http://localhost:8080`), `SECRET_KEY` (set in prod).
- **Document upload** (`documents.py`): **`display_name`** and **`evidence_category`** required; maps category → **`file_type`** for agents; runs **parallel** httpx calls to RAG **`/ingest/file`** and **`/agents/cases/{int}/upload`** (int from `uuid_to_insights_case_id`).
- **Insights proxy** (`rag_insights.py`): All under `/api/v1/cases/{uuid}/insights/...` with JWT + case access; forwards to `RAG_SERVICE_URL/agents/cases/{int}/...`; rewrites `poll_url` / relative URLs where applicable.
- **Run:** `uvicorn app.main:app --reload --port 8000` from `backend/`; `alembic upgrade head`; `python seed.py` for admin.

### 4.2 Frontend (`frontend/`)

- **Auth:** HTTP-only cookies; `login` → `/auth/login`; role routing via `/officer/me`.
- **Insights** (`InsightsPanel.tsx`): Loads blackboard + brief; **Live stream** via `fetch` + stream reader (Bearer token); **Supervisor report** POST → Celery `task_id` → polls **`/insights/tasks/{id}`**; on **SUCCESS** reads **`result.supervisor_report`** and shows scrollable panel; optional **sessionStorage** per case + **Clear report**.
- **Run:** `npm run dev` — `NEXT_PUBLIC_API_URL` in `.env`.

### 4.3 RAG microservice (`Rag_system/`)

- **Routers:** `/ingest/*`, `/query/`, `/agents/*`.
- **Health:** `GET /health` → `{ status, model, llm_backend: "openai_compatible" | "ollama" }`.
- **Query pipeline:** `QueryRewriter` → `HybridRetriever` → `BGEReranker` → `build_prompt` → **`OllamaClient.generate`** (same LLM rules as agents).
- **Ingestion:** `IngestionPipeline.ingest_file(..., logical_source_path=?)` — dedup key `exists_by_path` uses **effective source path** (stable upload path for agent temp-txt ingest) so **Celery retries** do not re-upsert Qdrant for the same logical file when status is `completed`.
- **`agent_retriever.ingest_document`:** Writes temp `.txt`, calls `ingest_file` with **`logical_source_path`** from metadata **`file_path`** (or fallback key).
- **BM25:** `api/shared_state.py` singleton; warmed in lifespan; ingest route updates index.
- **Run API:** `cd Rag_system` → `uvicorn api.app:app --reload --port 8080` (ensure `.env` cwd is `Rag_system`).
- **Run worker:** `celery -A orchestration.celery_app worker --loglevel=info` from **`Rag_system`** so **`.env`** loads.

### 4.4 Multi-agent system (inside `Rag_system`)

| Piece | Responsibility |
|-------|----------------|
| `api/routes/agents.py` | Upload (multipart `file` + `file_type` Form), classify-preview, task status, blackboard, brief, SSE, **`POST …/report`** → `run_supervisor.delay` |
| `orchestration/tasks.py` | **`process_document`**: extract → classify or stated type → **`ingest_document`** → **`investigation_graph.invoke`**; on failure **`retry`** (watch dedup). **`run_supervisor`**: `SupervisorAgent` → **`post_insight`** per cross issue → returns dict including **`supervisor_report`** |
| `graph/graph.py` | Router by `file_type` → one of 7 nodes → **supervisor** → END |
| `agents/base_agent.py` | **`agent_llm_complete`** for extract/analyse/RAG follow-up |
| `agents/supervisor.py` | **`format_brief`** → **`agent_llm_complete`** → markdown **`supervisor_report`** |
| `orchestration/blackboard.py` | Redis lists + pub/sub for live UI |

**UUID ↔ int:** `backend/app/core/insights_case_id.py` — deterministic; must stay aligned with any copy under `Rag_system` if used.

---

## 5. Database Schemas

### PostgreSQL (main app)

- **`user`** — role `ADMIN` | `OFFICER`, clearance, rank, station, etc.
- **`case`** — title, description, `required_clearance_level`, status, `created_by`
- **`case_assignment`** — case ↔ officer
- **`document`** — `file_path`, `filename`, `display_name`, `evidence_category`, `description`, `rag_document_id`, `ingest_status`, …
- **`activity_log`** — case actions
- **`case_analysis`** — snapshots (`analysis_type` e.g. `insights_snapshot`, `result_text`)

**Clearance:** minimum **4** (SI) to create cases — see product docs / `PROJECT_SUMMARY` for full rank ladder.

### RAG document tracking (`rag_documents` or SQLite `documents`)

- Selected by **`RAG_DATABASE_URL`**: PostgreSQL → `PgDocumentStore`; else SQLite file (`DOCUMENT_STORE_PATH`).
- **`exists_by_path(source_path)`** for skip checks uses **`status = completed`** and **`source_path`** (may be **logical** path for agent text ingest).

---

## 6. API Reference

Prefix: **`/api/v1`** on backend (`http://localhost:8000`).

### Auth
`POST /auth/login`, `/auth/refresh`, `/auth/change-password`

### Admin / Officer / Cases
See earlier context: paginated list endpoints return `{ items, total, skip, limit }`; admin case delete/assign; officer case list with access rules.

### Documents
- **`POST /cases/{case_id}/documents`** — multipart: `file`, **`display_name`**, **`evidence_category`**, optional `description`; parallel RAG + agent pipeline.
- **`GET/DELETE`** — list/delete with RAG cleanup where applicable.

### Analysis
- **`POST /cases/{id}/analysis`** — snapshot brief + blackboard subset into `case_analysis`
- **`GET /cases/{id}/analysis`** — list snapshots

### Insights proxy (`/cases/{uuid}/insights/...`)
| Method | Suffix | Purpose |
|--------|--------|---------|
| GET | `/blackboard`, `/blackboard/brief` | JSON / `{ brief }` |
| GET | `/stream` | SSE (proxied) |
| POST | `/report` | Queue supervisor Celery task |
| GET | `/tasks/{task_id}` | Status + **`result`** (includes **`supervisor_report`** on success) |
| POST | `/upload`, `/classify-preview` | Alternate upload/classify paths |
| GET | `/memory/{agent_type}` | Per-agent markdown memory |

### RAG service direct (`RAG_SERVICE_URL`, default `:8080`)

- **`/ingest/*`**, **`POST /query/`**, **`/agents/*`** as documented in `api/routes/*`
- **`GET /health`** — LLM backend probe

---

## 7. Authentication & Authorization

- JWT access (7d) + refresh (30d); refresh claim prevents misuse as access token.
- Case access: creator or assigned officer; assignment rules per `cases` / `assignments` endpoints.
- RAG `/agents` has **no** API key in-repo — treat as internal; only expose via backend with JWT.

---

## 8. Environment Variables

### Backend (`backend/.env`)
```env
DATABASE_URL=postgresql+asyncpg://...
RAG_SERVICE_URL=http://localhost:8080
SECRET_KEY=change-me-in-production
```

### Frontend (`frontend/.env`)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### RAG (`Rag_system/.env` — load from **`Rag_system` cwd**)

**Critical for LLM (preferred):**
```env
LLM_BASE_URL=https://your-host.example        # no trailing /v1 — OllamaClient appends /v1/chat/completions
LLM_MODEL=your-model-name
LLM_API_KEY=...                              # if required
```

**Fallback (local Ollama) when `LLM_BASE_URL` is empty:**
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_ENABLE_THINKING=true
```

**Infra:**
```env
QDRANT_URL=                    # full URL, OR use QDRANT_HOST + QDRANT_PORT
QDRANT_API_KEY=

REDIS_URL=redis://localhost:6379/0

RAG_DATABASE_URL=              # optional Neon/pg for rag_documents

UPLOAD_DIR=data/uploads
MEMORY_DIR=data/memory_store
```

**Tuning (examples):** `RETRIEVAL_TOP_K`, `RERANKER_TOP_K`, `LLM_MAX_TOKENS`, `QUERY_REWRITE_MAX_TOKENS`, `CHAT_HISTORY_MAX_MESSAGES`, `RAG_COMBINED_CONTEXT_BUDGET_TOKENS`, … — see `config/settings.py`.

---

## 9. Data Flow

```
Browser → Next.js → Backend (JWT) → PostgreSQL
                  → httpx → RAG :8080 (/ingest, /query, /agents patterns)

RAG API process:
  Qdrant + BM25 + embedder + reranker + OllamaClient (LLM_* or Ollama)

Celery worker (same code + .env from Rag_system cwd):
  Redis broker/backend + blackboard
  process_document → extractor → ingest (Qdrant) → LangGraph agents → supervisor
  run_supervisor → supervisor_report in task result → UI polls task → InsightsPanel
```

---

## 10. Integration Status

| Area | Status |
|------|--------|
| Login, roles, cases, assignments | Complete |
| Document upload → RAG + agents | Complete (parallel) |
| RAG chat | Complete |
| Insights UI + blackboard + SSE | Complete |
| Supervisor report in UI | Complete (from Celery **`result.supervisor_report`**) |
| Analysis snapshots | Complete |
| CCTV | Stub |
| Celery | **Manual** worker process; **Redis** must be reachable and stable |

---

## 11. Known Issues & Gaps

| Topic | Notes |
|-------|--------|
| `SECRET_KEY` / CORS | Dev defaults; tighten for production |
| RAG service auth | No API key between backend and RAG in repo |
| Large uploads | Long synchronous proxy may hit timeouts — consider async job or higher limits |
| **`bcrypt`** | Keep **4.0.1** for passlib compatibility |
| **Celery + `.env`** | Worker must start from **`Rag_system`** (or export env) or **`LLM_*` / `REDIS_URL`** may be wrong |
| **Redis over ngrok TCP** | Causes **connection closed** + redelivery; prefer **local Docker Redis** or **managed Redis** |
| **`next.config` ignoreBuildErrors** | Masks TS/ESLint failures at build |

---

## 12. Changelog

### 2026-03-23 — PROJECT_CONTEXT full refresh

- **LLM:** Documented **unified `OllamaClient`**: **`LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`** (OpenAI-compatible) vs legacy **Ollama** when base URL unset; agents/supervisor/classifier use **`agent_llm_complete`** (`core/generation/agent_chat.py`); removed **`langchain-ollama`** from deps narrative.
- **RAG:** **`logical_source_path`** on **`ingest_file`** + **`agent_retriever`** to avoid duplicate Qdrant upserts on Celery **retry**; **`GET /health`** returns **`llm_backend`**.
- **Agents route:** Upload uses **sync `Path.write_bytes`** after **`await file.read()`** (no aiofiles shutdown race on uvicorn stop).
- **Frontend:** **Supervisor report** rendered in **InsightsPanel** from polled task **`result.supervisor_report`**; **sessionStorage** + clear.
- **Ops:** Redis stability (ngrok TCP vs local/managed); worker **cwd** for `.env`.
- **Repo layout:** PDF **Docling** vs **extractor** for Celery path; directory tree and API tables synced.

### 2026-03-22 — Insights proxy, UUID→int, Redis `REDIS_URL`, live SSE, `NEXT_PUBLIC_API_URL`

- Backend **`rag_insights.py`**; **`insights_case_id`**; Celery **`REDIS_URL`**; InsightsPanel live stream + task polling.

### 2026-03-21 — Analysis snapshots, pagination, mobile nav, document metadata, RAG citation fields

- **`case_analysis`**, paginated admin/officer lists, **`display_name` / `evidence_category`** on documents and RAG ingest.

### 2026-03-17 — RAG integration, bcrypt pin, officer `GET /officer/{id}`, admin case tools

- Documented in git history; initial context file created.

---

*End of PROJECT_CONTEXT.md*
