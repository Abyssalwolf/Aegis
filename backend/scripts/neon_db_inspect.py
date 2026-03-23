#!/usr/bin/env python3
"""
Inspect Neon / Postgres state without the web console.

Loads env in order: backend/.env then Rag_system/.env (first wins for duplicate keys).
Uses asyncpg + DATABASE_URL from the backend (Alembic + `document` table live there).

Usage (from repo root):
    cd backend
    python scripts/neon_db_inspect.py

Optional — align alembic_version to a revision that exists in this repo (does not run migrations):
    python scripts/neon_db_inspect.py --stamp de7856598563
    python scripts/neon_db_inspect.py --stamp f3a8c2b91d4e

Then run: alembic upgrade head
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

# Backend package root (parent of scripts/)
BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
BACKEND_ENV = BACKEND_DIR / ".env"
RAG_ENV = REPO_ROOT / "Rag_system" / ".env"

# Revisions present in this repository (alembic/versions)
KNOWN_REVISIONS = frozenset(
    {"94a164febd44", "de7856598563", "a3f1c9b84e21", "f3a8c2b91d4e"}
)


def parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        val = rest.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def merge_env_files() -> None:
    merged: dict[str, str] = {}
    for p in (BACKEND_ENV, RAG_ENV):
        for k, v in parse_env_file(p).items():
            if k not in merged:
                merged[k] = v
    for k, v in merged.items():
        os.environ.setdefault(k, v)


def normalize_asyncpg_dsn(url: str) -> str:
    """asyncpg accepts postgresql:// ; strip SQLAlchemy async driver prefix."""
    url = url.strip()
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + url.split("postgresql+asyncpg://", 1)[1]
    return url


def redact_dsn(url: str) -> str:
    return re.sub(r":([^:@/]+)@", r":***@", url, count=1)


def suggest_stamp(columns: set[str]) -> str:
    """Suggest alembic stamp target from `document` columns (no DB writes)."""
    has_meta = {"display_name", "evidence_category", "description"}.issubset(columns)
    has_rag = {"filename", "rag_document_id", "ingest_status"}.issubset(columns)
    if has_meta:
        return "f3a8c2b91d4e"
    if has_rag:
        return "de7856598563"
    return "94a164febd44"


async def inspect_main(db_url: str, stamp: str | None) -> int:
    import asyncpg

    dsn = normalize_asyncpg_dsn(db_url)
    print(f"Connecting to: {redact_dsn(dsn)}\n")

    conn = await asyncpg.connect(dsn)
    try:
        try:
            rows = await conn.fetch("SELECT version_num FROM alembic_version")
        except asyncpg.exceptions.UndefinedTableError:
            print("alembic_version table not found — run migrations from a fresh DB or create schema first.")
            return 1
        versions = [r["version_num"] for r in rows]
        print("alembic_version.version_num:")
        for v in versions:
            flag = "OK (in repo)" if v in KNOWN_REVISIONS else "UNKNOWN — not in local alembic/versions"
            print(f"  - {v}  ({flag})")

        cols = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'document'
            ORDER BY ordinal_position
            """
        )
        print("\npublic.document columns:")
        if not cols:
            print("  (no rows — table missing?)")
            col_names: set[str] = set()
        else:
            col_names = {c["column_name"] for c in cols}
            for c in cols:
                print(f"  - {c['column_name']}: {c['data_type']} nullable={c['is_nullable']}")

        suggested = suggest_stamp(col_names)
        print("\n--- Suggested fix for 'Can't locate revision ...' ---")
        if versions and versions[0] not in KNOWN_REVISIONS:
            print(
                f"Your DB points at a revision that is not in this repo.\n"
                f"Stamp to a revision that matches your actual schema, then upgrade:\n"
            )
            print(f"  Recommended stamp (from column detection): {suggested}")
            print(f"  cd backend")
            print(f"  alembic stamp {suggested}")
            print(f"  alembic upgrade head")
        else:
            print("alembic_version matches a local revision (or empty). If upgrades fail, compare columns above to migrations.")

        if stamp:
            if stamp not in KNOWN_REVISIONS:
                print(f"\nError: --stamp {stamp!r} is not a known local revision.", file=sys.stderr)
                return 1
            await conn.execute("UPDATE alembic_version SET version_num = $1", stamp)
            print(f"\nUpdated alembic_version to {stamp!r}. Next: alembic upgrade head")

    finally:
        await conn.close()
    return 0


async def inspect_rag_db(rag_url: str) -> None:
    if not rag_url.strip():
        return
    import asyncpg

    dsn = normalize_asyncpg_dsn(rag_url)
    print(f"\n--- RAG_DATABASE_URL (rag_documents) ---\nConnecting: {redact_dsn(dsn)}")
    try:
        conn = await asyncpg.connect(dsn)
    except Exception as e:
        print(f"  Could not connect: {e}")
        return
    try:
        n = await conn.fetchval(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'rag_documents'
            """
        )
        print(f"  rag_documents table exists: {bool(n)}")
        if n:
            cnt = await conn.fetchval("SELECT COUNT(*) FROM rag_documents")
            print(f"  row count: {cnt}")
    finally:
        await conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Neon DB / Alembic state using .env files")
    parser.add_argument(
        "--stamp",
        metavar="REVISION",
        help="Set alembic_version via SQL (use when `alembic stamp` fails). Known: 94a164febd44, de7856598563, a3f1c9b84e21, f3a8c2b91d4e",
    )
    args = parser.parse_args()

    merge_env_files()

    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        print(
            "DATABASE_URL is not set. Checked:\n"
            f"  {BACKEND_ENV}\n"
            f"  {RAG_ENV}\n"
            "Set DATABASE_URL in backend/.env (postgresql+asyncpg://... is fine).",
            file=sys.stderr,
        )
        return 1

    rag_url = os.environ.get("RAG_DATABASE_URL", "").strip()

    print("Env files loaded (first key wins):")
    print(f"  {BACKEND_ENV}  exists={BACKEND_ENV.is_file()}")
    print(f"  {RAG_ENV}  exists={RAG_ENV.is_file()}")

    async def run_all() -> int:
        rc = await inspect_main(db_url, args.stamp)
        if rc != 0:
            return rc
        await inspect_rag_db(rag_url)
        return 0

    return asyncio.run(run_all())


if __name__ == "__main__":
    raise SystemExit(main())
