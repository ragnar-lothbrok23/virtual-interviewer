"""End-to-end: HR's page updates a score in place while HR is scrolled down; no reload, no scroll jump."""
from datetime import timedelta

import pytest

from app.extensions import db
from app.models import Application, Interview, InterviewAnswer, Job, User, utcnow
from tests.conftest import PASSWORD
from tests.e2e.test_interview_room_browser import live_server  # noqa: F401  (fixture)

pytestmark = pytest.mark.e2e


@pytest.fixture
def scoring_in_progress(app, make_user):
    """A completed interview with 6 answers; the last one is still being scored."""
    make_user("hr-e2e@test.com", role=User.ROLE_HR, name="HR")
    uid = make_user("cand-e2e@test.com", name="Live Candidate")
    with app.app_context():
        job = Job(title="Live Refresh Developer", skills="Python", description="Live refresh test job " * 5)
        db.session.add(job)
        db.session.flush()
        application = Application(candidate_id=uid, job_id=job.id, resume_path="none.enc", resume_original_name="cv.pdf",
                                  resume_text="text", parsed={}, similarity=0.8, match_score=80,
                                  status="shortlisted", analysis_status="done")
        db.session.add(application)
        db.session.flush()
        interview = Interview(application_id=application.id, starts_at=utcnow() - timedelta(hours=1),
                              duration_minutes=30, status=Interview.STATUS_COMPLETED, question_plan=[])
        db.session.add(interview)
        db.session.flush()
        for i in range(6):
            a = InterviewAnswer(interview_id=interview.id, position=i, question_text=f"Question number {i}?",
                                category="technical", answer_seconds=60, asked_at=utcnow(), submitted_at=utcnow(),
                                transcript="A long enough transcript for this answer. " * 8,
                                transcript_status=InterviewAnswer.TRANSCRIPT_DONE, speech_detected=True)
            if i < 5:
                a.score_status, a.scored_at, a.nlp_score = InterviewAnswer.SCORE_DONE, utcnow(), 70.0
                a.relevance = a.accuracy = a.fluency = a.clarity = 70.0
                a.similarity, a.similarity_score, a.filler_count = 0.8, 80.0, 0
            else:
                a.score_status, a.score_requested_at = InterviewAnswer.SCORE_PENDING, utcnow()
            db.session.add(a)
        db.session.commit()
        return application.id, a.id


def test_score_appears_without_reload_or_scroll_jump(app, live_server, scoring_in_progress):
    sync_api = pytest.importorskip("playwright.sync_api")
    application_id, answer_id = scoring_in_progress
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception as e:  # pragma: no cover
            pytest.skip(f"Chrome not available: {e}")
        page = browser.new_page(viewport={"width": 1280, "height": 700})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{live_server}/login")
        page.fill("#email", "hr-e2e@test.com")
        page.fill("#password", PASSWORD)
        page.click("#submit")
        page.goto(f"{live_server}/hr/applications/{application_id}#answer-{answer_id}")
        region = f'[data-live="answer-{answer_id}"]'
        assert "Scoring" in page.inner_text(region)
        # The redirect anchor brings HR to the answer they clicked (Bootstrap scrolls smoothly: wait until it arrives)
        page.wait_for_function("""id => {
            const top = document.getElementById('answer-' + id).getBoundingClientRect().top;
            const atEnd = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2;
            return top >= 0 && (top < 5 || atEnd);
        }""", arg=answer_id, timeout=10000)
        page.evaluate("window.__sameDocument = true")  # disappears if the page reloads
        page.evaluate("window.scrollTo({top: window.scrollY - 150, behavior: 'instant'})")  # HR scrolled a bit
        scroll_before = page.evaluate("window.scrollY")
        assert scroll_before > 300

        with app.app_context():  # the worker finishes scoring
            a = db.session.get(InterviewAnswer, answer_id)
            a.score_status, a.scored_at, a.nlp_score = InterviewAnswer.SCORE_DONE, utcnow(), 42.0
            a.relevance = a.accuracy = a.fluency = a.clarity = 40.0
            a.similarity, a.similarity_score, a.filler_count = 0.6, 20.0, 1
            db.session.commit()

        page.wait_for_function("sel => document.querySelector(sel).innerText.includes('NLP 42')", arg=region,
                               timeout=20000)
        assert page.evaluate("window.__sameDocument === true")  # updated in place, not reloaded
        assert page.evaluate("window.scrollY") == scroll_before  # the page didn't move
        browser.close()
    assert not errors, errors
