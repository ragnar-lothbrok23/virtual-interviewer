"""resume hardening: background analysis, encryption metadata, deadlines, withdrawal

Revision ID: 2eb572da9794
Revises: ede687b4685c
Create Date: 2026-10-18 23:36:43.715877

"""
import json

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '2eb572da9794'
down_revision = 'ede687b4685c'
branch_labels = None
depends_on = None


def upgrade():
    # New required columns are added as nullable, back-filled for existing applications, then made NOT NULL.
    with op.batch_alter_table('application', schema=None) as batch_op:
        batch_op.add_column(sa.Column('resume_kind', sa.String(length=8), nullable=True))
        batch_op.add_column(sa.Column('resume_sha256', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('embedding', sa.LargeBinary(), nullable=True))
        batch_op.add_column(sa.Column('analysis_status', sa.String(length=10), nullable=True))
        batch_op.add_column(sa.Column('analysis_error', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('analysis_attempts', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('analysed_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('status_source', sa.String(length=8), nullable=True))
        batch_op.add_column(sa.Column('withdrawn_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.alter_column('parsed', existing_type=sa.JSON(), nullable=True)
        batch_op.alter_column('similarity', existing_type=sa.Float(), nullable=True)
        batch_op.alter_column('match_score', existing_type=sa.Float(), nullable=True)
        batch_op.create_index(batch_op.f('ix_application_analysis_status'), ['analysis_status'], unique=False)
        batch_op.create_index(batch_op.f('ix_application_resume_sha256'), ['resume_sha256'], unique=False)

    with op.batch_alter_table('job', schema=None) as batch_op:
        batch_op.add_column(sa.Column('closes_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True))

    bind = op.get_bind()
    audit = sa.table('audit_log', sa.column('action', sa.String), sa.column('details', sa.JSON))
    application = sa.table('application', sa.column('id', sa.Integer), sa.column('resume_path', sa.String),
                           sa.column('created_at', sa.DateTime(timezone=True)),
                           sa.column('resume_kind', sa.String), sa.column('analysis_status', sa.String),
                           sa.column('analysis_attempts', sa.Integer), sa.column('analysed_at', sa.DateTime(timezone=True)),
                           sa.column('status_source', sa.String))

    # Applications whose status an HR user changed (recorded in the audit log since Phase 3) keep that decision
    hr_decided = set()
    for (details,) in bind.execute(sa.select(audit.c.details).where(audit.c.action == 'application.status_changed')):
        if isinstance(details, str):
            details = json.loads(details)
        if details and details.get('application_id') is not None:
            hr_decided.add(int(details['application_id']))

    for app_id, resume_path, created_at in bind.execute(
            sa.select(application.c.id, application.c.resume_path, application.c.created_at)).fetchall():
        suffix = (resume_path or '').rsplit('.', 1)[-1].lower()
        bind.execute(application.update().where(application.c.id == app_id).values(
            resume_kind=suffix if suffix in ('pdf', 'docx') else None,
            analysis_status='done',  # these were analysed synchronously before this phase
            analysis_attempts=1,
            analysed_at=created_at,
            status_source='hr' if app_id in hr_decided else 'auto'))

    with op.batch_alter_table('application', schema=None) as batch_op:
        batch_op.alter_column('analysis_status', existing_type=sa.String(length=10), nullable=False)
        batch_op.alter_column('analysis_attempts', existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column('status_source', existing_type=sa.String(length=8), nullable=False)


def downgrade():
    # Applications that were never analysed have no score; the old schema requires one.
    application = sa.table('application', sa.column('parsed', sa.JSON), sa.column('similarity', sa.Float),
                           sa.column('match_score', sa.Float))
    bind = op.get_bind()
    bind.execute(application.update().where(application.c.parsed.is_(None)).values(parsed={}))
    bind.execute(application.update().where(application.c.similarity.is_(None)).values(similarity=0.0))
    bind.execute(application.update().where(application.c.match_score.is_(None)).values(match_score=0.0))

    with op.batch_alter_table('job', schema=None) as batch_op:
        batch_op.drop_column('updated_at')
        batch_op.drop_column('closes_at')

    with op.batch_alter_table('application', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_application_resume_sha256'))
        batch_op.drop_index(batch_op.f('ix_application_analysis_status'))
        batch_op.alter_column('match_score',
               existing_type=sa.Float(),
               nullable=False)
        batch_op.alter_column('similarity',
               existing_type=sa.Float(),
               nullable=False)
        batch_op.alter_column('parsed',
               existing_type=sa.JSON(),
               nullable=False)
        batch_op.drop_column('updated_at')
        batch_op.drop_column('withdrawn_at')
        batch_op.drop_column('status_source')
        batch_op.drop_column('analysed_at')
        batch_op.drop_column('analysis_attempts')
        batch_op.drop_column('analysis_error')
        batch_op.drop_column('analysis_status')
        batch_op.drop_column('embedding')
        batch_op.drop_column('resume_sha256')
        batch_op.drop_column('resume_kind')

    # ### end Alembic commands ###
