"""Pages that wait for background work refresh their live regions in place instead of reloading (no scroll jumps,
videos keep playing)."""
import re
from datetime import timedelta

from app.extensions import db
from app.models import Application, Interview, InterviewAnswer, Job, utcnow
from tests.test_interview_room import api, room, started  # noqa: F401  (fixtures)
from tests.test_scheduling import booked, get, job_id  # noqa: F401  (fixtures)
from tests.test_scoring import answer_question, answer_row

LIVE_SCRIPT = "js/live_refresh.js"


def busy_flag(html):
    return re.search(r'data-live-busy="(\d)"', html).group(1)


def hr_page(app, room, login):
    application_id = get(app, Interview, room.id).application_id
    return login("hr@test.com").get(f"/hr/applications/{application_id}").get_data(as_text=True)


def test_application_page_uses_live_regions_not_reloads(app, room, interview_ai, login):
    started(room)
    answer_question(room, 0)
    html = hr_page(app, room, login)
    assert LIVE_SCRIPT in html and "location.reload" not in html
    a = answer_row(app, 0)
    for key in ("match", "interview", "resume-questions", "results-heading", "interview-summary", "integrity",
                f"answer-{a.id}"):
        assert f'data-live="{key}"' in html
    # the video sits outside its answer's live region, so refreshing scores never restarts it
    card = html.index(f'id="answer-{a.id}"')
    assert card < html.index("<video", card) < html.index(f'data-live="answer-{a.id}"')


def test_busy_while_the_interview_runs_and_idle_afterwards(app, room, interview_ai, login):
    started(room)
    answer_question(room, 0)
    assert busy_flag(hr_page(app, room, login)) == "1"  # interview in progress
    with app.app_context():
        interview = db.session.get(Interview, room.id)
        interview.status = Interview.STATUS_COMPLETED
        db.session.commit()
    assert busy_flag(hr_page(app, room, login)) == "0"


def test_busy_while_an_answer_is_scored_or_analysed(app, room, interview_ai, login):
    started(room)
    answer_question(room, 0)
    with app.app_context():
        db.session.get(Interview, room.id).status = Interview.STATUS_COMPLETED
        a = InterviewAnswer.query.filter_by(position=0).one()
        a.score_status, a.score_requested_at = InterviewAnswer.SCORE_PENDING, utcnow()
        db.session.commit()
    assert busy_flag(hr_page(app, room, login)) == "1"
    with app.app_context():
        a = InterviewAnswer.query.filter_by(position=0).one()
        a.score_status = InterviewAnswer.SCORE_DONE
        a.emotion_status, a.emotion_requested_at = InterviewAnswer.EMOTION_PENDING, utcnow()
        db.session.commit()
    assert busy_flag(hr_page(app, room, login)) == "1"
    with app.app_context():
        a = InterviewAnswer.query.filter_by(position=0).one()
        a.emotion_requested_at = utcnow() - timedelta(minutes=30)  # lost to a worker restart: stop polling
        db.session.commit()
    assert busy_flag(hr_page(app, room, login)) == "0"


def test_busy_while_transcribing_but_not_forever(app, room, interview_ai, login):
    started(room)
    answer_question(room, 0)
    with app.app_context():
        db.session.get(Interview, room.id).status = Interview.STATUS_COMPLETED
        a = InterviewAnswer.query.filter_by(position=0).one()
        a.transcript_status, a.score_status = None, None
        a.submitted_at = utcnow()
        db.session.commit()
    assert busy_flag(hr_page(app, room, login)) == "1"
    with app.app_context():
        InterviewAnswer.query.filter_by(position=0).one().submitted_at = utcnow() - timedelta(minutes=20)
        db.session.commit()
    assert busy_flag(hr_page(app, room, login)) == "0"


def test_rescore_and_reanalyse_return_to_the_same_answer(app, room, interview_ai, login):
    started(room)
    answer_question(room, 0)
    a = answer_row(app, 0)
    hr = login("hr@test.com")
    for url in (f"/hr/answers/{a.id}/rescore", f"/hr/answers/{a.id}/emotion"):
        response = hr.post(url)
        assert response.status_code == 302 and response.headers["Location"].endswith(f"#answer-{a.id}")


def test_job_page_refreshes_the_applicant_table_while_analysing(app, hr_client, job_id, make_user):
    candidate_id = make_user("c@test.com")
    with app.app_context():
        db.session.add(Application(candidate_id=candidate_id, job_id=job_id, resume_path="x.enc",
                                   resume_original_name="cv.pdf", resume_text="text",
                                   analysis_status=Application.ANALYSIS_RUNNING))
        db.session.commit()
    html = hr_client.get(f"/hr/jobs/{job_id}").get_data(as_text=True)
    assert 'data-live="applicants"' in html and busy_flag(html) == "1" and LIVE_SCRIPT in html
    assert "location.reload" not in html
    with app.app_context():
        Application.query.one().analysis_status = Application.ANALYSIS_FAILED
        db.session.commit()
    assert busy_flag(hr_client.get(f"/hr/jobs/{job_id}").get_data(as_text=True)) == "0"


def test_questions_page_refreshes_while_jd_questions_are_written(app, hr_client, job_id):
    with app.app_context():
        job = db.session.get(Job, job_id)
        job.jd_questions_status, job.jd_questions_at = Job.JD_PENDING, utcnow()
        db.session.commit()
    html = hr_client.get(f"/hr/jobs/{job_id}/questions").get_data(as_text=True)
    assert busy_flag(html) == "1" and 'data-live="question-list"' in html and 'data-live="jd-card"' in html
    assert "location.reload" not in html and LIVE_SCRIPT in html


def test_live_refresh_script_is_served(app, hr_client):
    response = hr_client.get(f"/static/{LIVE_SCRIPT}")
    assert response.status_code == 200 and b"data-live" in response.data and b"replaceState" in response.data
