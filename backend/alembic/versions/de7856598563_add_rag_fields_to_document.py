"""add_rag_fields_to_document

Revision ID: de7856598563
Revises: 94a164febd44
Create Date: 2026-03-17 09:44:50.796669

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'de7856598563'
down_revision: Union[str, Sequence[str], None] = '94a164febd44'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('document', sa.Column('filename', sa.String(), nullable=True))
    op.add_column('document', sa.Column('rag_document_id', sa.String(), nullable=True))
    op.add_column('document', sa.Column('ingest_status', sa.String(), server_default='pending', nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('document', 'ingest_status')
    op.drop_column('document', 'rag_document_id')
    op.drop_column('document', 'filename')
