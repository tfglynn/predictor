"""Switch to dates with timezone

Revision ID: d60e808059e0
Revises: 
Create Date: 2022-02-04 18:47:07.315682

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd60e808059e0'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column("questions", "open_date", type_=sa.TIMESTAMP(timezone=True), existing_type=sa.DateTime)
    op.alter_column("questions", "close_date", type_=sa.TIMESTAMP(timezone=True), existing_type=sa.DateTime)
    op.alter_column("questions", "expiry_date", type_=sa.TIMESTAMP(timezone=True), existing_type=sa.DateTime)
    op.alter_column("predictions", "datetime", type_=sa.TIMESTAMP(timezone=True), existing_type=sa.DateTime)


def downgrade():
    op.alter_column("questions", "open_date", existing_type=sa.DateTime, type_=sa.TIMESTAMP(timezone=True))
    op.alter_column("questions", "close_date", existing_type=sa.DateTime, type_=sa.TIMESTAMP(timezone=True))
    op.alter_column("questions", "expiry_date", existing_type=sa.DateTime, type_=sa.TIMESTAMP(timezone=True))
    op.alter_column("predictions", "datetime", existing_type=sa.DateTime, type_=sa.TIMESTAMP(timezone=True))
