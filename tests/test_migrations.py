"""The migrations must build exactly the schema the models describe, and be reversible."""
import tempfile
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from flask_migrate import downgrade, upgrade

from app import create_app
from app.config import TestingConfig
from app.extensions import db


def test_models_match_migrations(app):
    """Fails if someone changes a model without generating a migration (flask db migrate)."""
    with app.app_context(), db.engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        diff = compare_metadata(context, db.metadata)
    assert diff == [], f"Models and migrations differ; run 'flask db migrate'. Differences: {diff}"


@pytest.fixture
def fresh_app(database_uri):
    if database_uri.startswith("sqlite"):
        uri = f"sqlite:///{Path(tempfile.mkdtemp()) / 'roundtrip.db'}"
    else:  # a separate PostgreSQL database so the shared test schema isn't touched
        engine = sa.create_engine(database_uri, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            conn.execute(sa.text("DROP DATABASE IF EXISTS roundtrip"))
            conn.execute(sa.text("CREATE DATABASE roundtrip"))
        engine.dispose()
        uri = database_uri.rsplit("/", 1)[0] + "/roundtrip"

    class Config(TestingConfig):
        SQLALCHEMY_DATABASE_URI = uri

    return create_app(Config)


def test_upgrade_downgrade_roundtrip(fresh_app):
    with fresh_app.app_context():
        upgrade()
        assert {"user", "job", "application", "alembic_version"} <= set(sa.inspect(db.engine).get_table_names())
        downgrade(revision="base")
        assert set(sa.inspect(db.engine).get_table_names()) <= {"alembic_version"}
        upgrade()
        assert "application" in sa.inspect(db.engine).get_table_names()
        db.engine.dispose()


def test_accounts_migration_backfills_existing_users(fresh_app):
    """Upgrading a database that already has users must give each a session token and mark them active/verified."""
    with fresh_app.app_context():
        upgrade(revision="26b860603d35")  # schema before the accounts migration
        db.session.execute(sa.text(
            "INSERT INTO \"user\" (name, email, password_hash, role, created_at) "
            "VALUES ('Old', 'old@test.com', 'x', 'candidate', CURRENT_TIMESTAMP), "
            "('Older', 'older@test.com', 'y', 'hr', CURRENT_TIMESTAMP)"))
        db.session.commit()
        upgrade()
        rows = db.session.execute(sa.text(
            "SELECT active, session_token, failed_login_count, email_verified_at FROM \"user\" ORDER BY id")).fetchall()
        assert len(rows) == 2
        assert all(r[0] in (True, 1) and len(r[1]) == 32 and r[2] == 0 and r[3] is not None for r in rows)
        assert rows[0][1] != rows[1][1]  # each account gets its own token
        db.session.remove()
        db.engine.dispose()
