"""Scoring hardening (docs/ENHANCEMENTS.md items 1-8): safe similarity, embedding cache, tolerant evidence matching,
verdicts and human review, validated follow-ups, deadline-aware follow-ups and explicit failures."""
import math
from datetime import timedelta

import numpy as np
import pytest

from app.extensions import db
from app.models import Application, EmbeddingCache, Interview, InterviewAnswer, InterviewEvent, InterviewQuestion, \
    Job, utcnow
from app.services import analysis, gemini, resume, scoring, vectors
from app.services.gemini import GeminiError
from app.services.vectors import VectorError
from tests.test_interview_room import api, room, started  # noqa: F401  (fixtures)
from tests.test_resume_hardening import apply, jobs, the_app  # noqa: F401  (fixtures)
from tests.test_scheduling import booked, get, job_id  # noqa: F401  (fixtures)
from tests.test_scoring import answer_question, answer_row


@pytest.fixture
def ai(interview_ai):
    return interview_ai


WEAK = "Uh, I am not sure, maybe something with servers."


# ---------- 1. safe similarity ----------

@pytest.mark.parametrize("a, b, message", [
    ([], [1.0], "empty"),
    ([0.0, 0.0], [1.0, 0.0], "zero-length"),
    ([1.0, float("nan")], [1.0, 0.0], "NaN or infinite"),
    ([1.0, float("inf")], [1.0, 0.0], "NaN or infinite"),
    ([1.0, 0.0, 0.0], [1.0, 0.0], "sizes differ"),
])
def test_safe_cosine_rejects_unusable_vectors(a, b, message):
    with pytest.raises(VectorError, match=message):
        vectors.safe_cosine(a, b)


def test_safe_cosine_values():
    assert vectors.safe_cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert vectors.safe_cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert vectors.safe_cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)
    value = vectors.safe_cosine([1e-6, 2e-6], [3e-6, 1e-6])  # tiny but valid
    assert math.isfinite(value) and -1.0 <= value <= 1.0


def test_unusable_embedding_fails_scoring_visibly_and_skips_follow_up(app, room, ai, monkeypatch):
    ai.transcript = WEAK
    monkeypatch.setattr(gemini, "embed", lambda text: np.zeros(8, dtype=np.float32))
    started(room)
    answer_question(room, 0)
    state = answer_question(room, 1)  # technical: the decision can't assess it, so the interview just moves on
    assert state["position"] == 2 and state["question"]["follow_up"] is False and ai.calls["follow_up"] == 0
    a = answer_row(app, 1)
    assert a.score_status == "failed" and "Semantic similarity couldn't be computed" in a.score_error
    assert a.nlp_score is None and a.similarity is None  # no silent number
    with app.app_context():
        event = InterviewEvent.query.filter_by(kind="follow_up_skipped").one()
        assert event.position == 1 and event.details["reason"]


def test_resume_analysis_with_unusable_embedding_is_retryable(app, monkeypatch):
    monkeypatch.setattr(gemini, "parse_resume", lambda text, job: {"summary": "Dev", "skills": ["Flask"]})
    monkeypatch.setattr(gemini, "embed", lambda text: np.zeros(8, dtype=np.float32))
    job = Job(title="Dev", skills="Flask", description="d", embedding=np.ones(8, dtype=np.float32).tobytes())
    with app.app_context():
        with pytest.raises(GeminiError, match="unusable embedding") as info:
            resume.analyse("resume text", job)
    assert info.value.retryable and job.embedding is None  # the cached job embedding is rebuilt on the retry


def test_broken_stored_resume_embedding_is_reanalysed_on_rescore(app, candidate_client, jobs, fake_gemini):
    job_id = jobs["Python Backend Developer"]
    apply(candidate_client, job_id)
    with app.app_context():
        a = Application.query.one()
        a.embedding = np.zeros(8, dtype=np.float32).tobytes()
        db.session.commit()
        parses = fake_gemini.parse_calls
        analysis.rescore_job(job_id, skills_changed=False)
    a = the_app(app)
    assert fake_gemini.parse_calls == parses + 1  # analysed again from scratch instead of storing NaN
    assert a.analysis_status == "done" and math.isfinite(a.similarity) and a.match_score is not None


# ---------- 2. reference embedding cache ----------

def test_embed_cached_reuses_and_invalidates(app, ai):
    original_model = app.config["GEMINI_EMBED_MODEL"]
    with app.app_context():
        first = vectors.embed_cached("Reference about Flask")
        again = vectors.embed_cached("Reference about Flask")
        assert ai.calls["embed"] == 1 and np.array_equal(first, again)
        vectors.embed_cached("Reference about Flask, edited by HR")  # edited text: new embedding
        assert ai.calls["embed"] == 2
        app.config["GEMINI_EMBED_MODEL"] = "another-embedding-model"  # model change: new embedding
        try:
            vectors.embed_cached("Reference about Flask")
        finally:
            app.config["GEMINI_EMBED_MODEL"] = original_model
        assert ai.calls["embed"] == 3 and EmbeddingCache.query.count() == 3
        assert {row.model for row in EmbeddingCache.query} == {original_model, "another-embedding-model"}


def test_reference_embedded_once_across_answers(app, room, ai):
    started(room)
    answer_question(room, 0)
    with app.app_context():
        references = {a.reference_answer for a in InterviewAnswer.query}
    embeds = ai.calls["embed"]
    with app.app_context():
        for answer in InterviewAnswer.query.all():
            answer.similarity = None
        db.session.commit()
        for answer in InterviewAnswer.query.all():
            scoring.score(answer.id)
    # re-scoring only embeds the transcripts; the reference comes from the cache
    assert ai.calls["embed"] == embeds + 1 and len(references) == 1


# ---------- 3. evidence quotes ----------

def test_evidence_matching_ignores_punctuation_and_case_but_rejects_invented_text(app, room, ai):
    ai.transcript = "Um, I built REST APIs with Flask and PostgreSQL."
    ai.rubric_overrides = {"evidence": [
        {"quote": "i built rest apis, with flask and postgresql", "criterion": "accuracy", "note": "real"},
        {"quote": "I built REST APIs with Django", "criterion": "relevance", "note": "altered"},
        {"quote": "uilt REST", "criterion": "clarity", "note": "partial word"},
        {"quote": " , ", "criterion": "fluency", "note": "punctuation only"},
    ]}
    started(room)
    answer_question(room, 0)
    a = answer_row(app, 0)
    assert a.feedback["evidence"] == [{"quote": "i built rest apis, with flask and postgresql",
                                       "criterion": "accuracy", "note": "real"}]


def test_normalise_text():
    assert scoring.normalise_text("  Hello,   WORLD! It's  fine. ") == "hello world it's fine"
    assert scoring.normalise_text(None) == ""


# ---------- 4. verdicts, confidence and human review ----------

@pytest.mark.parametrize("verdict, confidence, review", [
    ("correct", "high", False),
    ("partially_correct", "medium", False),
    ("incorrect", "high", False),
    ("off_topic", "high", False),
    ("dont_know", "high", False),
    ("correct", "low", True),
    ("clarification_request", "high", True),
    ("no_answer", "medium", True),  # speech was detected but nothing assessable: a human should look
])
def test_verdicts_and_review_flag(app, room, ai, verdict, confidence, review):
    ai.rubric_overrides = {"verdict": verdict, "confidence": confidence}
    started(room)
    answer_question(room, 0)
    a = answer_row(app, 0)
    assert a.is_scored and a.verdict == verdict and a.confidence == confidence and a.needs_review is review


def test_silent_answer_is_no_answer_without_review(app, room, ai):
    ai.transcript = ""
    started(room)
    answer_question(room, 0)
    a = answer_row(app, 0)
    assert a.verdict == "no_answer" and a.confidence == "high" and a.needs_review is False and a.nlp_score == 0


def test_hr_sees_verdict_review_flag_and_quote_wording(app, room, ai, login):
    ai.rubric_overrides = {"verdict": "partially_correct", "confidence": "low"}
    started(room)
    answer_question(room, 0)
    application_id = get(app, Interview, room.id).application_id
    page = login("hr@test.com").get(f"/hr/applications/{application_id}").get_data(as_text=True)
    assert "Partially correct" in page and "AI certainty: low" in page
    assert "Needs human review" in page and "1 answer needs human review" in page
    assert "Quoted from the transcript" in page and "not that it is correct" in page
    assert "the weights are an implementation choice" in page


# ---------- 8. explicit failure instead of silent zeros ----------

@pytest.mark.parametrize("field, value", [
    ("relevance", 11), ("accuracy", -1), ("fluency", None), ("clarity", float("nan")),
    ("relevance", float("inf")), ("accuracy", True), ("fluency", "8"),
    ("verdict", "great"), ("confidence", "certain"), ("verdict", None),
])
def test_malformed_rubric_is_retried_then_fails_without_numbers(app, room, ai, field, value):
    started(room)
    answer_question(room, 0)
    a = answer_row(app, 0)
    ai.rubric_overrides = {field: value}
    with app.app_context():
        row = db.session.get(InterviewAnswer, a.id)
        row.relevance = row.accuracy = row.fluency = row.clarity = row.nlp_score = None
        row.score_status = "pending"
        db.session.commit()
        with pytest.raises(GeminiError) as info:
            scoring.run_scoring(a.id, retries_left=1)
        assert info.value.retryable
        assert scoring.run_scoring(a.id, retries_left=0) is None
        row = db.session.get(InterviewAnswer, a.id)
        assert row.score_status == "failed" and "invalid" in row.score_error
        assert row.nlp_score is None and row.relevance is None  # nothing clamped or zeroed


def test_boundary_rubric_values_are_valid(app, room, ai):
    ai.rubric_overrides = {"relevance": 0, "accuracy": 10, "fluency": 0.0, "clarity": 10.0}
    started(room)
    answer_question(room, 0)
    a = answer_row(app, 0)
    assert a.is_scored and (a.relevance, a.accuracy, a.fluency, a.clarity) == (0, 100, 0, 100)


# ---------- 5. follow-up validation ----------

QUESTION = "How would you design a REST API for a library system?"
REFERENCE = "Resources such as books and members, HTTP verbs mapped to actions, status codes, pagination and auth."


@pytest.mark.parametrize("follow_up, problem", [
    ({"question": "", "reference_answer": "r"}, "empty question"),
    ({"question": "   ", "reference_answer": "r"}, "empty question"),
    ({"question": "Why?", "reference_answer": "r"}, "question length out of bounds"),
    ({"question": "word " * 41 + "?", "reference_answer": "r"}, "question length out of bounds"),
    ({"question": "Explain how pagination works in your API.", "reference_answer": "r"}, "not phrased as a question"),
    ({"question": "How would pagination work in your API?", "reference_answer": "  "}, "no reference answer"),
    ({"question": "How would you design a REST API for a library system?", "reference_answer": "r"},
     "repeats the original question"),
    ({"question": "How, would you design a REST API for the library system?", "reference_answer": "r"},
     "repeats the original question"),
    ({"question": "Would you use HTTP verbs mapped to actions, status codes and pagination?",
      "reference_answer": "r"}, "reveals the reference answer"),
    ({"question": "Which status codes would a create endpoint return?",
      "reference_answer": "Which status codes would a create endpoint return: 201 Created."},
     "reveals the reference answer"),
])
def test_follow_up_problems(follow_up, problem):
    assert scoring.follow_up_problem(follow_up, QUESTION, REFERENCE) == problem


def test_valid_follow_up_passes():
    follow_up = {"question": "How would you paginate the list of books, and why?",
                 "reference_answer": "Limit/offset or cursor-based pagination; cursors stay stable when data changes."}
    assert scoring.follow_up_problem(follow_up, QUESTION, REFERENCE) is None


def test_invalid_generated_follow_up_is_skipped_and_logged(app, room, ai):
    ai.transcript = WEAK
    ai.follow_up_overrides = {"question": "Tell me more about servers."}  # not a question
    started(room)
    answer_question(room, 0)
    state = answer_question(room, 1)
    assert ai.calls["follow_up"] == 1 and state["question"]["follow_up"] is False and state["total"] == 7
    interview = get(app, Interview, room.id)
    assert interview.follow_up_count == 0 and interview.awaiting_decision_since is None
    with app.app_context():
        event = InterviewEvent.query.filter_by(kind="follow_up_skipped").one()
        assert event.details == {"reason": "not phrased as a question"}


# ---------- 6. duplicates and target concept ----------

def test_follow_up_sees_upcoming_questions_and_records_its_target(app, room, ai, login):
    ai.transcript = WEAK
    started(room)
    answer_question(room, 0)
    plan = get(app, Interview, room.id).question_plan
    state = answer_question(room, 1)
    assert ai.last_upcoming == [q["text"] for q in plan[2:]] and len(ai.last_upcoming) == 5
    assert state["question"]["follow_up"] is True
    assert get(app, Interview, room.id).question_plan[2]["target"] == "REST endpoint design"
    answer_question(room, 2)  # the follow-up itself
    follow_up = answer_row(app, 2)
    assert follow_up.is_follow_up and follow_up.follow_up_target == "REST endpoint design"
    application_id = get(app, Interview, room.id).application_id
    page = login("hr@test.com").get(f"/hr/applications/{application_id}").get_data(as_text=True)
    assert "Probes: REST endpoint design" in page


def test_missing_target_concept_still_allows_follow_up(app, room, ai):
    ai.transcript = WEAK
    ai.follow_up_overrides = {"target_concept": "  "}
    started(room)
    answer_question(room, 0)
    state = answer_question(room, 1)
    assert state["question"]["follow_up"] is True
    assert get(app, Interview, room.id).question_plan[2]["target"] is None


# ---------- 7. remaining-time check ----------

def test_no_follow_up_when_it_would_not_fit_before_the_deadline(app, room, ai):
    ai.transcript = WEAK
    started(room)
    answer_question(room, 0)
    with app.app_context():
        interview = db.session.get(Interview, room.id)
        remaining = sum(20 + q["answer_seconds"] + 15 for q in interview.question_plan[2:])
        # enough for the planned questions, not for an extra follow-up
        interview.hard_deadline = utcnow() + timedelta(seconds=40 + remaining + 60)
        db.session.commit()
    state = answer_question(room, 1)
    assert state["question"]["follow_up"] is False and ai.calls["follow_up"] == 0  # no AI call wasted
    with app.app_context():
        event = InterviewEvent.query.filter_by(kind="follow_up_skipped").one()
        assert event.details == {"reason": "not enough time left"}


def test_time_check_counts_follow_up_and_remaining_questions(app, room):
    started(room)
    with app.app_context():
        interview = db.session.get(Interview, room.id)
        answer = InterviewAnswer(position=1, answer_seconds=120)
        needed = sum(20 + q["answer_seconds"] + 15 for q in interview.question_plan[2:]) + 20 + 120 + 15
        interview.hard_deadline = utcnow() + timedelta(seconds=needed)
        assert scoring.has_time_for_follow_up(interview, answer)
        interview.hard_deadline = utcnow() + timedelta(seconds=needed - 1)
        assert not scoring.has_time_for_follow_up(interview, answer)
        interview.hard_deadline = None
        assert scoring.has_time_for_follow_up(interview, answer)
        db.session.rollback()
