"""add document display_name evidence_category description

Revision ID: f3a8c2b91d4e
Revises: a3f1c9b84e21
Create Date: 2026-03-21

"""
from typing import Sequence, Union

from alembic import op

revision: str = "f3a8c2b91d4e"
down_revision: Union[str, Sequence[str], None] = "a3f1c9b84e21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Idempotent: Neon DBs may already have these columns from an out-of-repo migration.
    op.execute("ALTER TABLE document ADD COLUMN IF NOT EXISTS display_name VARCHAR")
    op.execute("ALTER TABLE document ADD COLUMN IF NOT EXISTS evidence_category VARCHAR")
    op.execute("ALTER TABLE document ADD COLUMN IF NOT EXISTS description VARCHAR")


def downgrade() -> None:
    op.execute("ALTER TABLE document DROP COLUMN IF EXISTS description")
    op.execute("ALTER TABLE document DROP COLUMN IF EXISTS evidence_category")
    op.execute("ALTER TABLE document DROP COLUMN IF EXISTS display_name")
