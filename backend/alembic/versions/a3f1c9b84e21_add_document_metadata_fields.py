"""add_document_metadata_fields

Revision ID: a3f1c9b84e21
Revises: de7856598563
Create Date: 2026-03-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f1c9b84e21'
down_revision: Union[str, Sequence[str], None] = 'de7856598563'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('document', sa.Column('display_name', sa.String(), nullable=True))
    op.add_column('document', sa.Column('evidence_category', sa.String(), nullable=True))
    op.add_column('document', sa.Column('description', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('document', 'description')
    op.drop_column('document', 'evidence_category')
    op.drop_column('document', 'display_name')
