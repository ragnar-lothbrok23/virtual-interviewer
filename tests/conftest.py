"""Shared test fixtures.

    venv\\Scripts\\pytest                 # SQLite (fast)
    venv\\Scripts\\pytest --postgres      # real PostgreSQL via embedded pgserver (production engine)
"""
import os
import shutil
import tempfile

os.environ["APP_ENV"] = "testing"  # must be set before the app (and the Huey queue) is imported

import numpy as np
import pytest
from flask_migrate import upgrade

from app import create_app
from app.config import TestingConfig
from app.extensions import db, limiter
from app.models import User
from app.services import gemini


def pytest_addoption(parser):
    parser.addoption("--postgres", action="store_true", help="Run the suite against an embedded PostgreSQL server")


@pytest.fixture(scope="session")
def database_uri(request):
    if not request.config.getoption("--postgres"):
        yield "sqlite://"
        return
    import pgserver

    data_dir = tempfile.mkdtemp(prefix="vi-pg-")
    server = pgserver.get_server(data_dir, cleanup_mode="delete")
    yield server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)
    server.cleanup()
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def app(database_uri):
    upload_dir = tempfile.mkdtemp(prefix="vi-uploads-")

    class Config(TestingConfig):
        SQLALCHEMY_DATABASE_URI = database_uri
        UPLOAD_FOLDER = __import__("pathlib").Path(upload_dir)

    app = create_app(Config)
    with app.app_context():
        upgrade()  # build the schema through the real migrations
    yield app
    shutil.rmtree(upload_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def _clean_state(app):
    """Every test starts with empty tables, an empty upload folder and fresh rate-limit counters."""
    yield
    with app.app_context():
        db.session.remove()
        for table in reversed(db.metadata.sorted_tables):
            db.session.execute(table.delete())
        db.session.commit()
    for f in app.config["UPLOAD_FOLDER"].iterdir():
        f.unlink()
    limiter.reset()
    app.config["WTF_CSRF_ENABLED"] = False
    app.extensions.get("mail_outbox", []).clear()


@pytest.fixture(autouse=True)
def _no_real_gemini(monkeypatch):
    """Tests must never call the real Gemini API (it costs quota). Use the fake_gemini fixture instead."""
    def blocked(*args, **kwargs):
        raise AssertionError("A test tried to call the real Gemini API; use the fake_gemini fixture")

    monkeypatch.setattr(gemini, "parse_resume", blocked)
    monkeypatch.setattr(gemini, "embed", blocked)


class FakeGemini:
    """Deterministic stand-in: texts mentioning backend/Flask embed in one direction, everything else in another."""

    def __init__(self):
        self.parse_calls = 0
        self.embed_calls = 0
        self.error = None

    def parse_resume(self, text, job):
        self.parse_calls += 1
        if self.error:
            raise self.error
        lowered = text.lower()
        matched = [s for s in job.skill_list if s.lower() in lowered]
        return {
            "name": "Arjun Mehta", "email": "arjun@example.com", "phone": "+91 90000 00000",
            "summary": "Backend developer." if "flask" in lowered else "Frontend developer.",
            "skills": ["Python", "Flask"] if "flask" in lowered else ["React"],
            "experience": [{"role": "Intern", "organisation": "Zentrix", "duration": "6 months", "highlights": ["Built APIs"]}],
            "education": [{"degree": "B.E. CSE", "institution": "Anna University", "year": "2026", "score": "8.4"}],
            "projects": ["Booking API"], "certifications": [], "total_experience_years": 0.5,
            "matched_skills": matched, "missing_skills": [s for s in job.skill_list if s not in matched],
        }

    def embed(self, text):
        self.embed_calls += 1
        if self.error:
            raise self.error
        vec = np.zeros(8, dtype=np.float32)
        vec[0 if ("backend" in text.lower() or "flask" in text.lower()) else 1] = 1.0
        return vec


@pytest.fixture
def fake_gemini(monkeypatch):
    fake = FakeGemini()
    monkeypatch.setattr(gemini, "parse_resume", fake.parse_resume)
    monkeypatch.setattr(gemini, "embed", fake.embed)
    return fake


PASSWORD = "Correct-Horse-42"


@pytest.fixture
def make_user(app):
    def _make(email="user@test.com", password=PASSWORD, role=User.ROLE_CANDIDATE, name="Test User",
              verified=True, consent=True, active=True):
        with app.app_context():
            user = User(name=name, email=email, role=role, active=active)
            user.set_password(password)
            if verified:
                user.mark_verified()
            if consent:
                user.give_consent(app.config["PRIVACY_POLICY_VERSION"])
            db.session.add(user)
            db.session.commit()
            return user.id
    return _make


@pytest.fixture
def outbox(app):
    """Emails 'sent' during the test (memory mail backend)."""
    box = app.extensions.setdefault("mail_outbox", [])
    box.clear()
    return box


@pytest.fixture
def login(app):
    def _login(email, password=PASSWORD):
        client = app.test_client()
        response = client.post("/login", data={"email": email, "password": password})
        assert response.status_code == 302, "login failed"
        return client
    return _login


@pytest.fixture
def hr_client(make_user, login):
    make_user("hr@test.com", role=User.ROLE_HR, name="HR Admin")
    return login("hr@test.com")


@pytest.fixture
def candidate_client(make_user, login):
    make_user("cand@test.com", name="Arjun")
    return login("cand@test.com")
