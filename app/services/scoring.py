"""Answer scoring and adaptive follow-ups (paper Sections 3.2-3.4).

Per answer:
  - speech metrics computed locally from the transcript and recording length (pace, filler words)
  - semantic similarity between the answer and the question's reference answer (Gemini embeddings + cosine)
  - Gemini rubric scores (relevance, accuracy, fluency, clarity) with evidence quotes
  - NLP score = weighted mix of relevance, accuracy, similarity and fluency (weights in config.NLP_WEIGHTS)
Adaptive rule (3.3): a technical answer whose similarity score is below FOLLOW_UP_THRESHOLD gets one focused
follow-up question, inserted right after it.
"""
import re

import numpy as np
from flask import current_app

from ..extensions import db
from ..models import Interview, InterviewAnswer, InterviewQuestion, utcnow
from . import gemini
from .gemini import GeminiError

FILLER_PATTERN = re.compile(r"\b(um+|uh+|erm+|er|ah+|hmm+|you know|i mean)\b", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


# ---------- local measurements ----------

def speech_metrics(transcript, duration_seconds):
    words = WORD_PATTERN.findall(transcript or "")
    wpm = None
    if duration_seconds and duration_seconds >= 5 and words:
        wpm = round(len(words) / (duration_seconds / 60), 1)
    return {"word_count": len(words), "words_per_minute": wpm,
            "filler_count": len(FILLER_PATTERN.findall(transcript or ""))}


def similarity_to_score(similarity):
    floor, ceil = current_app.config["ANSWER_SIMILARITY_FLOOR"], current_app.config["ANSWER_SIMILARITY_CEIL"]
    return round(float(np.clip((similarity - floor) / (ceil - floor), 0, 1)) * 100, 1)


def nlp_score(relevance, accuracy, similarity_score, fluency):
    w = current_app.config["NLP_WEIGHTS"]
    return round(w["relevance"] * relevance + w["accuracy"] * accuracy + w["similarity"] * similarity_score
                 + w["fluency"] * fluency, 1)


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------- reference answers ----------

def ensure_reference(answer):
    """The reference answer for this answer's question, generated (and cached on the job's question) if missing."""
    if answer.reference_answer:
        return answer.reference_answer
    job = answer.interview.application.job
    question = InterviewQuestion.query.filter_by(job_id=job.id, text=answer.question_text).first()
    if question is not None and question.reference_answer:
        reference = question.reference_answer
    else:
        reference = gemini.generate_reference(answer.question_text, answer.category, job.title, job.skill_list)
        if question is not None:
            question.reference_answer = reference
    answer.reference_answer = reference
    db.session.commit()
    return reference


def fill_missing_references(job_id):
    """Background: give every question of a job a reference answer (run when questions are created or edited)."""
    count = 0
    for question in InterviewQuestion.query.filter_by(job_id=job_id).all():
        if not question.reference_answer:
            question.reference_answer = gemini.generate_reference(question.text, question.category,
                                                                  question.job.title, question.job.skill_list)
            db.session.commit()
            count += 1
    return count


def _similarity(answer):
    reference = ensure_reference(answer)
    if not (answer.transcript or "").strip():
        return 0.0, 0.0
    sim = _cosine(gemini.embed(answer.transcript), gemini.embed(reference))
    return sim, similarity_to_score(sim)


# ---------- adaptive follow-ups ----------

def needs_decision(interview, answer):
    """Only base technical answers can trigger a follow-up, and only while the interview has follow-ups left."""
    return (answer.category == "technical" and not answer.is_follow_up
            and interview.follow_up_count < current_app.config["MAX_FOLLOW_UPS"])


def decide_follow_up(answer):
    """Called right after transcription while the interview waits. Never raises: on any problem the interview
    simply moves on without a follow-up."""
    from .interview_room import finish, log_event

    interview = answer.interview
    if interview.awaiting_decision_since is None:
        return None
    follow_up = None
    try:
        if answer.transcript_status == InterviewAnswer.TRANSCRIPT_DONE:
            answer.similarity, answer.similarity_score = _similarity(answer)
            db.session.commit()
            if answer.similarity_score < current_app.config["FOLLOW_UP_THRESHOLD"] and answer.speech_detected:
                follow_up = gemini.generate_follow_up(answer.question_text, answer.reference_answer,
                                                      answer.transcript, interview.application.job.title)
    except Exception as e:  # any failure (AI, network, bug) must not keep the candidate waiting
        db.session.rollback()
        current_app.logger.warning("Follow-up decision for answer %s skipped: %s", answer.id, e)
        follow_up = None

    db.session.refresh(interview)
    if interview.awaiting_decision_since is None or interview.status != Interview.STATUS_IN_PROGRESS:
        return None  # the room already timed out the decision and moved on
    if follow_up:
        plan = list(interview.question_plan)
        plan.insert(answer.position + 1, {"text": follow_up["question"], "category": "technical",
                                          "answer_seconds": answer.answer_seconds, "follow_up": True,
                                          "reference": follow_up["reference_answer"],
                                          "follows": answer.position})
        interview.question_plan = plan  # reassign so the JSON change is saved
        interview.follow_up_count += 1
        log_event(interview, "follow_up_added", answer.position,
                  similarity_score=answer.similarity_score)
    interview.awaiting_decision_since = None
    db.session.commit()
    if interview.current_position >= interview.question_count:
        finish(interview, Interview.END_COMPLETED)
    return follow_up


# ---------- scoring ----------

def score(answer_id):
    """Score one transcribed answer. Raises GeminiError (caller decides about retrying)."""
    answer = db.session.get(InterviewAnswer, answer_id)
    if answer is None or answer.transcript_status != InterviewAnswer.TRANSCRIPT_DONE:
        return None
    metrics = speech_metrics(answer.transcript, answer.duration_seconds)
    answer.word_count, answer.words_per_minute, answer.filler_count = (
        metrics["word_count"], metrics["words_per_minute"], metrics["filler_count"])

    if not answer.speech_detected or not (answer.transcript or "").strip():
        answer.similarity, answer.similarity_score = 0.0, 0.0
        answer.relevance = answer.accuracy = answer.fluency = answer.clarity = 0.0
        answer.nlp_score = 0.0
        answer.feedback = {"summary": "No spoken answer was detected.", "strengths": [], "gaps": ["No answer given"],
                           "evidence": []}
    else:
        if answer.similarity is None:
            answer.similarity, answer.similarity_score = _similarity(answer)
        job = answer.interview.application.job
        rubric = gemini.score_answer(answer.question_text, answer.category, answer.reference_answer,
                                     answer.transcript, job.title, metrics)
        to100 = lambda v: float(max(0, min(10, v)) * 10)  # noqa: E731
        answer.relevance, answer.accuracy = to100(rubric["relevance"]), to100(rubric["accuracy"])
        answer.fluency, answer.clarity = to100(rubric["fluency"]), to100(rubric["clarity"])
        answer.nlp_score = nlp_score(answer.relevance, answer.accuracy, answer.similarity_score, answer.fluency)
        # Keep only evidence that really appears in the transcript (guards against invented quotes)
        transcript = " ".join(answer.transcript.lower().split())
        evidence = [e for e in rubric["evidence"] if " ".join(e["quote"].lower().split()) in transcript]
        answer.feedback = {"summary": rubric["summary"], "strengths": rubric["strengths"][:3],
                           "gaps": rubric["gaps"][:3], "evidence": evidence[:3]}
    answer.score_status, answer.score_error, answer.scored_at = InterviewAnswer.SCORE_DONE, None, utcnow()
    db.session.commit()
    return answer.nlp_score


def run_scoring(answer_id, retries_left=0):
    """Task body: score with retry semantics like the other AI jobs."""
    answer = db.session.get(InterviewAnswer, answer_id)
    if answer is None:
        return None
    try:
        return score(answer_id)
    except GeminiError as e:
        if e.retryable and retries_left > 0:
            answer.score_error = f"{e} (retrying automatically)"
            db.session.commit()
            raise
        answer.score_status, answer.score_error = InterviewAnswer.SCORE_FAILED, str(e)
        db.session.commit()
        current_app.logger.error("Scoring answer %s failed: %s", answer_id, e)
        return None


def interview_nlp_summary(interview):
    """Average NLP figures over the scored answers (Phase 9 builds the full scorecard on top of this)."""
    scored = [a for a in interview.answers if a.is_scored]
    if not scored:
        return None
    avg = lambda attr: round(sum(getattr(a, attr) or 0 for a in scored) / len(scored), 1)  # noqa: E731
    paces = [a.words_per_minute for a in scored if a.words_per_minute]
    return {"answers": len(scored), "nlp": avg("nlp_score"), "relevance": avg("relevance"),
            "accuracy": avg("accuracy"), "fluency": avg("fluency"), "similarity": avg("similarity_score"),
            "clarity": avg("clarity"), "wpm": round(sum(paces) / len(paces)) if paces else None,
            "fillers": sum(a.filler_count or 0 for a in scored)}
