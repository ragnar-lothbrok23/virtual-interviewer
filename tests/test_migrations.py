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
