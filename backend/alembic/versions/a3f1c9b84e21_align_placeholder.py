"""Placeholder: align DBs stamped with this revision id (local ghost revision).

Some environments were migrated with revision a3f1c9b84e21 which was never merged
into this repo. This empty revision sits between de7856598563 and f3a8c2b91d4e
so Alembic can load the graph. Schema changes for display/evidence fields are in
f3a8c2b91d4e (idempotent ADD COLUMN IF NOT EXISTS).

Revision ID: a3f1c9b84e21
Revises: de7856598563
Create Date: 2026-03-21

"""
from typing import Sequence, Union

from alembic import op

revision: str = "a3f1c9b84e21"
down_revision: Union[str, Sequence[str], None] = "de7856598563"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
