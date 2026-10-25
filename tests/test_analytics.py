"""Phase 10: HR analytics. The figures are computed from real interviews run through the app (fake AI and fake
emotion model), plus pure-function tests for the edge cases."""
import csv
import io
from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.extensions import db
from app.models import Application, AuditLog, Interview, InterviewAnswer, InterviewEvent, User
from app.services import analytics
from tests.test_interview_room import api, room, started  # noqa: F401  (fixtures)
from tests.test_scheduling import booked, get, job_id  # noqa: F401  (fixtures)
from tests.test_scorecard import STRONG, WEAK, card, run_interview

BANDS = [(75, "strong", "Strong candidate"), (60, "consider", "Worth considering"), (0, "weak", "Not recommended")]


@pytest.fixture(autouse=True)
def ai(interview_ai):
    return interview_ai


def page(client, **params):
    response = client.get("/hr/analytics", query_string=params)
    assert response.status_code == 200
    return " ".join(response.get_data(as_text=True).split())


# ---------- in the app ----------

def test_analytics_after_a_real_interview(app, room, ai, fake_emotion, login):
    run_interview(room, ai, {1: WEAK})
    sc = card(app, room)
    iv = get(app, Interview, room.id)
    with app.app_context():
        db.session.add(InterviewEvent(interview_id=room.id, kind="tab_hidden", details={}))
        db.session.commit()
    text = page(login("hr@test.com"))
    assert "Hiring pipeline" in text and "Interview completed" in text
    assert "Candidates by final score" in text and f"{sc.final_score:.1f}" in text
    assert f"/hr/interviews/{room.id}/scorecard" in text and f"/hr/applications/{iv.application_id}" in text
    assert "Left the interview tab" in text and "1 of 1 interview had at least one flag" in text
    assert "No processing problems" in text
    assert "Questions" in text and "Final interview scores" in text and "Show as table" in text
    with app.app_context():
        d = analytics.build(bands=BANDS)
    assert [s["count"] for s in d["funnel"]["stages"]][:5] == [1, 1, 1, 1, 1]
    assert d["scores"]["count"] == 1 and d["scores"]["mean"] == round(sc.final_score, 1)
    assert sum(b["count"] for b in d["scores"]["histogram"]) == 1
    assert d["ranking"][0]["flags"] == 1 and d["integrity"]["flagged_interviews"] == 1
    weak = [q for q in d["questions"] if q["original_average"] is not None]
    assert len(weak) == len(d["questions"]) and all(q["asked"] == 1 for q in d["questions"])
    assert d["timing"]["interview_minutes"] is not None


def test_one_job_view_and_filters(app, room, ai, fake_emotion, login, job_id):
    run_interview(room, ai)
    client = login("hr@test.com")
    text = page(client, job=job_id)
    assert "By job" not in text and "Candidates by final score" in text
    assert client.get("/hr/analytics", query_string={"job": 99999}).status_code == 404
    assert "Hiring pipeline" in page(client, period="nonsense")  # unknown period falls back to all time
    assert "By job" in page(client)


def test_period_counts_applications_by_their_date(app, room, ai, fake_emotion, login):
    run_interview(room, ai)
    room.frozen.tick(timedelta(days=100))
    client = login("hr@test.com")
    assert "No applications in this period" in page(client, period="90")
    assert "Hiring pipeline" in page(client, period="365")


def test_processing_problems_are_listed_with_links(app, room, ai, fake_emotion, login):
    run_interview(room, ai)
    with app.app_context():
        answer = InterviewAnswer.query.filter_by(interview_id=room.id, position=0).one()
        answer.score_status = InterviewAnswer.SCORE_FAILED
        db.session.commit()
        app_id = get(app, Interview, room.id).application_id
    text = page(login("hr@test.com"))
    assert "Answers whose scoring failed: 1" in text and f"/hr/applications/{app_id}" in text


def test_only_hr_can_see_analytics(app, room, ai, fake_emotion, login, candidate_client):
    assert candidate_client.get("/hr/analytics").status_code in (302, 403)
    assert candidate_client.get("/hr/analytics/export.csv").status_code in (302, 403)


def test_empty_analytics(app, hr_client):
    assert "No applications in this period" in page(hr_client)


def test_csv_export(app, room, ai, fake_emotion, login):
    run_interview(room, ai)
    sc = card(app, room)
    with app.app_context():
        candidate = User.query.filter_by(email="cand@test.com").one()
        candidate.name = "=HYPERLINK(\"http://evil\")"  # a name that would run as a formula in a spreadsheet
        db.session.commit()
    response = login("hr@test.com").get("/hr/analytics/export.csv")
    assert response.status_code == 200 and response.mimetype == "text/csv"
    assert "attachment" in response.headers["Content-Disposition"]
    rows = list(csv.reader(io.StringIO(response.data.decode("utf-8-sig"))))
    assert rows[0] == analytics.CSV_COLUMNS and len(rows) == 2
    row = dict(zip(rows[0], rows[1]))
    assert row["Candidate"].startswith("'=") and row["Email"] == "cand@test.com"
    assert row["Final score"] == f"{sc.final_score:.1f}" and row["Interview status"] == "Completed"
    with app.app_context():
        log = AuditLog.query.filter_by(action="analytics.exported").one()
        assert log.details["rows"] == 1 and log.details["period"] == "all"


def test_navigation_links_to_analytics(app, hr_client):
    assert "/hr/analytics" in hr_client.get("/hr/").get_data(as_text=True)


# ---------- pure calculations ----------

def scored_app(final, match=50.0, items=(), recommendation="weak", job=None, status="complete", app_status="shortlisted"):
    job = job or SimpleNamespace(id=1, title="Engineer")
    sc = SimpleNamespace(final_score=final, nlp_score=final, emotion_score=70.0, context_score=80.0,
                         recommendation=recommendation, status=status, details={"items": list(items)},
                         consistency_status=None)
    iv = SimpleNamespace(id=id(sc), status=Interview.STATUS_COMPLETED, scorecard=sc, started_at=None, completed_at=None)
    return SimpleNamespace(id=id(iv), job=job, job_id=job.id, match_score=match, interview=iv, status=app_status,
                           analysis_status=Application.ANALYSIS_DONE)


def item(score, question="Q1", category="technical", source=None, verdict="correct", follow_up=None, state="scored"):
    return {"question": question, "category": category, "source": source, "score": score, "state": state,
            "original": {"state": "scored" if score is not None else "unanswered", "nlp": score, "verdict": verdict},
            "follow_up": follow_up}


def test_histogram_edges():
    d = analytics.scores([scored_app(0.0), scored_app(9.9), scored_app(10.0), scored_app(99.9), scored_app(100.0)],
                         BANDS)
    counts = [b["count"] for b in d["histogram"]]
    assert counts[0] == 2 and counts[1] == 1 and counts[9] == 2 and sum(counts) == 5
    assert d["histogram"][9]["to"] == 100


def test_correlation_needs_enough_varied_data():
    assert analytics.correlation([(1, 2)] * 4)["r"] is None
    assert analytics.correlation([(50, 60)] * 6)["r"] is None  # no variation: undefined, not an error
    assert analytics.correlation([(10, 20), (20, 40), (30, 60), (40, 80), (50, 100)])["r"] == 1.0
    assert analytics.correlation([(10, 100), (20, 80), (30, 60), (40, 40), (50, 20)])["r"] == -1.0


def test_question_flags_need_enough_answers():
    hard = [scored_app(30, items=[item(20, "Hard one")]) for _ in range(3)]
    thin = [scored_app(30, items=[item(10, "Thin")]) for _ in range(2)]
    easy = [scored_app(95, items=[item(95, "Easy one")]) for _ in range(3)]
    rows = {q["question"]: q for q in analytics.questions(hard + thin + easy)}
    assert rows["Hard one"]["flag"] == "hard" and rows["Easy one"]["flag"] == "easy"
    assert rows["Thin"]["flag"] is None and rows["Thin"]["average"] == 10


def test_resume_questions_are_grouped_and_follow_ups_counted():
    fu = {"state": "scored", "nlp": 80}
    apps = [scored_app(50, items=[item(40, "About your project A", category="resume"),
                                  item(60, "Explain X", follow_up=fu, state="recovered")]),
            scored_app(50, items=[item(70, "About your project B", category="resume"),
                                  item(None, "Explain X", state="unanswered")])]
    rows = {q["question"]: q for q in analytics.questions(apps)}
    personal = rows["Resume-based questions (personal to each candidate)"]
    assert personal["asked"] == 2 and personal["average"] == 55 and personal["personal"]
    x = rows["Explain X"]
    assert x["asked"] == 2 and x["unanswered"] == 1 and x["follow_up_rate"] == 50 and x["recovered"] == 1


def test_dont_know_rate_and_skills():
    apps = [scored_app(50, items=[item(10, "Q", source="Job description: SQL (basic)", verdict="dont_know")]),
            scored_app(50, items=[item(90, "Q", source="Job description: SQL (basic)"),
                                  item(70, "R", source="Job description: React")])]
    q = {q["question"]: q for q in analytics.questions(apps)}["Q"]
    assert q["dont_know_rate"] == 50
    assert analytics.skills(apps) == [{"skill": "SQL", "answers": 2, "average": 50.0},
                                      {"skill": "React", "answers": 1, "average": 70.0}]


def test_csv_cells_never_start_a_formula():
    for dangerous in ("=1+1", "+1", "-1", "@SUM(A1)", "\tx", "\rx"):
        assert analytics._cell(dangerous).startswith("'")
    assert analytics._cell("Ann Lee") == "Ann Lee" and analytics._cell(None) == "" and analytics._cell(72.456) == "72.5"
