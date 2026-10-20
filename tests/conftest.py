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
    parser.addoption("--e2e", action="store_true", help="Also run the slow real-browser end-to-end tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--e2e"):
        return
    skip = pytest.mark.skip(reason="real-browser test; run with --e2e")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


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
    """Tests must never call the real Gemini API (it costs quota and is slow). Every helper is replaced, and the
    client itself refuses to be created unless a test installs a fake one in app.extensions["gemini_client"]."""
    def blocked(*args, **kwargs):
        raise AssertionError("A test tried to call the real Gemini API; use the fake_gemini fixture")

    for name in ("parse_resume", "embed", "transcribe", "generate_reference", "score_answer", "generate_follow_up",
                 "generate_resume_questions", "generate_jd_questions"):
        monkeypatch.setattr(gemini, name, blocked)
    real_client = gemini._client

    def guarded_client():
        from flask import current_app
        if "gemini_client" not in current_app.extensions:
            raise AssertionError("A test tried to create a real Gemini client")
        return real_client()

    monkeypatch.setattr(gemini, "_client", guarded_client)


class FakeInterviewAI:
    """Deterministic interview AI: answers mentioning 'flask' are strong, others weak."""

    def __init__(self):
        self.calls = {"reference": 0, "score": 0, "follow_up": 0, "transcribe": 0, "embed": 0, "resume_questions": 0,
                      "jd_questions": 0}
        self.jd_questions = None  # None = the default four; tests set a list (or an error) to vary it
        self.jd_error = None
        self.last_jd_request = None
        self.resume_questions = None  # None = the default three; tests set a list (or an error) to vary it
        self.resume_error = None
        self.last_resume_request = self.last_follow_up_category = None
        self.transcript = "Um, I built REST APIs with Flask and PostgreSQL."
        self.rubric_overrides = {}  # tests change single rubric fields (verdict, a malformed score, ...)
        self.follow_up_overrides = {}
        self.last_upcoming = None

    def transcribe(self, media, mime_type):
        self.calls["transcribe"] += 1
        return {"speech_detected": bool(self.transcript), "language": "en", "transcript": self.transcript}

    def generate_reference(self, question, category, job_title, skills):
        self.calls["reference"] += 1
        return f"A strong answer explains Flask routing, REST APIs and SQL databases for: {question}"

    def embed(self, text):
        self.calls["embed"] += 1
        vec = np.zeros(8, dtype=np.float32)
        vec[0 if "flask" in text.lower() else 1] = 1.0
        return vec

    def score_answer(self, question, category, reference, transcript, job_title, speech):
        self.calls["score"] += 1
        strong = "flask" in transcript.lower()
        rubric = {"relevance": 9 if strong else 3, "accuracy": 8 if strong else 2, "fluency": 7, "clarity": 8,
                  "summary": "Answer summary.", "strengths": ["Concrete example"] if strong else [],
                  "gaps": [] if strong else ["Missing technical detail"],
                  "evidence": [{"quote": transcript.split(",")[-1].strip(), "criterion": "accuracy",
                                "note": "key point"},
                               {"quote": "a sentence that was never said", "criterion": "relevance",
                                "note": "invented"}],
                  "verdict": "correct" if strong else "partially_correct", "confidence": "high"}
        rubric.update(self.rubric_overrides)
        return rubric

    def generate_follow_up(self, question, reference, transcript, job_title, upcoming=(), category="technical"):
        self.calls["follow_up"] += 1
        self.last_upcoming = list(upcoming)
        self.last_follow_up_category = category
        follow_up = {"question": "Can you explain how you would design a REST endpoint in Flask?",
                     "target_concept": "REST endpoint design",
                     "reference_answer": "Flask route, HTTP methods, JSON, status codes, validation."}
        follow_up.update(self.follow_up_overrides)
        return follow_up

    def generate_resume_questions(self, profile, job_title, skills, existing_questions, count):
        self.calls["resume_questions"] += 1
        self.last_resume_request = {"profile": profile, "existing": list(existing_questions), "count": count}
        if self.resume_error:
            raise self.resume_error
        if self.resume_questions is not None:
            return [dict(q) for q in self.resume_questions]
        return [
            {"source": "experience", "subject": "Zentrix",
             "question": "At Zentrix, which API did you build yourself, and how did you design its endpoints?",
             "reference_answer": "Names a concrete Flask API, their own role, endpoint design, data model and testing."},
            {"source": "project", "subject": "Booking API",
             "question": "In your Booking API project, how did you stop two users booking the same seat?",
             "reference_answer": "Flask with transactions or row locks, unique constraints, handling the conflict response."},
            {"source": "skill", "subject": "Flask",
             "question": "How do you structure a larger Flask application, and why that way?",
             "reference_answer": "Application factory, blueprints, config per environment, extensions, tests."},
        ][:count]

    def generate_jd_questions(self, job_title, description, skills, existing_questions, count):
        self.calls["jd_questions"] += 1
        self.last_jd_request = {"title": job_title, "description": description, "skills": list(skills),
                                "existing": list(existing_questions), "count": count}
        if self.jd_error:
            raise self.jd_error
        if self.jd_questions is not None:
            return [dict(q) for q in self.jd_questions]
        return [
            {"skill": "Flask", "difficulty": "intermediate",
             "question": "How does Flask's application context differ from the request context, and when is each active?",
             "reference_answer": "Flask keeps an application context for app-level data such as current_app and g, and a "
                                 "request context for request and session; both are pushed per request and the app "
                                 "context can also be pushed manually for scripts and background jobs."},
            {"skill": "SQL", "difficulty": "intermediate",
             "question": "When would you add an index to a table, and what does it cost you on writes?",
             "reference_answer": "Indexes speed up lookups, joins and ordering on selective columns used by frequent "
                                 "queries, but every insert, update and delete must also update the index, so they cost "
                                 "write time and storage; check query plans before adding them in Flask apps."},
            {"skill": "REST APIs", "difficulty": "basic",
             "question": "What makes an API RESTful, and how do you choose status codes for create and validation errors?",
             "reference_answer": "Resources identified by URLs, standard HTTP methods, stateless requests and "
                                 "representations such as JSON; a successful create returns 201 with the location, "
                                 "invalid input returns 400 or 422 with a clear error body, as Flask views often do."},
            {"skill": "Docker", "difficulty": "advanced",
             "question": "How would you keep a Docker image for a Python service small and its builds fast?",
             "reference_answer": "Use a slim base image, multi-stage builds, copy the requirements first so dependency "
                                 "layers are cached, avoid dev tools in the final image, use a dockerignore file and run "
                                 "as a non-root user, for example when containerising a Flask service."},
        ][:count]


@pytest.fixture
def interview_ai(monkeypatch):
    fake = FakeInterviewAI()
    for name in ("transcribe", "generate_reference", "embed", "score_answer", "generate_follow_up",
                 "generate_resume_questions", "generate_jd_questions"):
        monkeypatch.setattr(gemini, name, getattr(fake, name))
    return fake


class FakeGemini:
    """Deterministic stand-in: texts mentioning backend/Flask embed in one direction, everything else in another."""

    def __init__(self):
        self.parse_calls = 0
        self.embed_calls = 0
        self.error = None

    def parse_resume(self, text, job):
        from app.services.gemini import normalise_profile

        self.parse_calls += 1
        if self.error:
            raise self.error
        lowered = text.lower()
        flask_dev = "flask" in lowered
        evidence = []
        for skill in job.skill_list:
            if skill.lower() in lowered:
                level = "proven" if flask_dev and skill.lower() in ("python", "flask") else "listed"
                evidence.append({"skill": skill, "level": level,
                                 "evidence": ["Project: Booking API"] if level == "proven" else ["Skills section"]})
        profile = {
            "name": "Arjun Mehta", "email": "arjun@example.com", "phone": "+91 90000 00000",
            "summary": "Backend developer." if flask_dev else "Frontend developer.",
            "skills": ["Python", "Flask"] if flask_dev else ["React"],
            "experience": [{"role": "Intern", "organisation": "Zentrix", "duration": "6 months", "highlights": ["Built APIs"]}],
            "education": [{"degree": "B.E. CSE", "institution": "Anna University", "year": "2026", "score": "8.4"}],
            "projects": ([{"name": "Booking API", "description": "REST API for event bookings",
                           "tech_stack": ["Flask", "PostgreSQL"], "outcome": "500+ bookings"}] if flask_dev else
                         [{"name": "Recipe Finder", "description": "Search app for recipes",
                           "tech_stack": ["React"], "outcome": ""}]),
            "certifications": [], "total_experience_years": 0.5, "skill_evidence": evidence,
        }
        return normalise_profile(profile, job.skill_list)

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
