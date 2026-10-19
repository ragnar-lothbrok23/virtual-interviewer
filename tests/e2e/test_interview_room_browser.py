"""End-to-end: a real Chrome, with a simulated camera and microphone, takes a whole interview through the actual page
and JavaScript. Slow, so it only runs with:  venv\\Scripts\\pytest --e2e
"""
import socket
import threading
from datetime import timedelta

import pytest
from werkzeug.serving import make_server

from app.extensions import db
from app.models import Application, Interview, InterviewAnswer, InterviewQuestion, Job, User, utcnow
from app.services import storage
from tests.conftest import PASSWORD

pytestmark = pytest.mark.e2e


@pytest.fixture
def live_server(app, interview_ai):
    interview_ai.transcript = "I am not sure."  # weak answers, so the technical question gets a follow-up
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def interview_now(app, make_user):
    """A shortlisted candidate whose interview starts in 5 minutes (join window open), with 2 short questions."""
    uid = make_user("browser@test.com", name="Browser Candidate")
    with app.app_context():
        job = Job(title="E2E Developer", skills="Python, Flask", description="End-to-end test job " * 5)
        db.session.add(job)
        db.session.flush()
        for i, (text, category) in enumerate([("Introduce yourself.", "behavioral"),
                                              ("Describe a Flask project you built.", "technical")]):
            db.session.add(InterviewQuestion(job_id=job.id, position=i, text=text, category=category, answer_seconds=30,
                                             reference_answer="Flask routes, blueprints, SQLAlchemy models."))
        application = Application(candidate_id=uid, job_id=job.id, resume_path="none.pdf.enc", resume_original_name="cv.pdf",
                                  resume_text="text", parsed={}, similarity=0.8, match_score=80, status="shortlisted",
                                  analysis_status="done")
        db.session.add(application)
        db.session.flush()
        interview = Interview(application_id=application.id, starts_at=utcnow() + timedelta(minutes=5), duration_minutes=30)
        db.session.add(interview)
        db.session.commit()
        return interview.id


def test_full_interview_in_a_real_browser(app, live_server, interview_now):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True, args=[
                "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
        except Exception as e:  # pragma: no cover
            pytest.skip(f"Chrome not available: {e}")
        context = browser.new_context(permissions=["camera", "microphone"])
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("response", lambda r: errors.append(f"HTTP {r.status} {r.url}") if r.status >= 400 else None)

        page.goto(f"{live_server}/login")
        page.fill("#email", "browser@test.com")
        page.fill("#password", PASSWORD)
        page.click("#submit")
        page.goto(f"{live_server}/candidate/interviews/{interview_now}/join")

        page.check("#consent-box")
        page.click("#consent-next")
        page.wait_for_selector("#start-btn:not([disabled])", timeout=20000)  # browser, camera, mic, network all OK
        page.click("#start-btn")

        labels = []
        for position in range(3):  # 2 planned questions + 1 follow-up after the weak technical answer
            page.wait_for_selector("#phase-prep:not(.d-none)", timeout=30000)
            labels.append(page.inner_text("#q-category"))
            page.click("#answer-now")
            page.wait_for_selector("#submit-answer:not([disabled])", timeout=10000)
            page.wait_for_timeout(1500)  # record a little more video
            page.click("#submit-answer")

        page.wait_for_selector("#step-done:not(.d-none)", timeout=30000)
        assert "Interview complete" in page.inner_text("#step-done")
        browser.close()
    assert labels == ["Behavioural", "Technical", "Follow-up question"]

    assert not [e for e in errors if "favicon" not in e], errors
    with app.app_context():
        interview = db.session.get(Interview, interview_now)
        assert interview.status == "completed" and interview.end_reason == "completed"
        answers = InterviewAnswer.query.filter_by(interview_id=interview_now).order_by(InterviewAnswer.position).all()
        assert len(answers) == 3 and answers[2].is_follow_up and interview.follow_up_count == 1
        assert all(a.is_scored for a in answers)
        for answer in answers:
            media = storage.read_resume(answer.media_path)
            assert answer.media_type == "video/webm" and media[:4] == b"\x1a\x45\xdf\xa3" and len(media) > 10_000
            assert answer.duration_seconds and 3 <= answer.duration_seconds <= 30
            assert answer.transcript == "I am not sure."
