"""Answer scoring and adaptive follow-ups (paper Sections 3.2-3.4).

Per answer:
  - speech metrics computed locally from the transcript and recording length (pace, filler words)
  - semantic similarity between the answer and the question's reference answer (Gemini embeddings + cosine)
  - Gemini rubric scores (relevance, accuracy, fluency, clarity) with evidence quotes
  - NLP score = weighted mix of relevance, accuracy, similarity and fluency (weights in config.NLP_WEIGHTS)
Adaptive rule (3.3): a technical answer whose similarity score is below FOLLOW_UP_THRESHOLD gets one focused
follow-up question, inserted right after it, provided it passes validation and fits before the hard deadline.

Integrity rules: a score is either computed from valid inputs or the answer is marked failed with the reason. Zero is
only ever stored for a genuinely empty answer; broken embeddings or malformed rubric values never become numbers.
"""
import math
import re
from datetime import timedelta

import numpy as np
from flask import current_app

from ..extensions import db
from ..models import Interview, InterviewAnswer, InterviewQuestion, as_utc, utcnow
from . import gemini
from .gemini import VERDICTS, GeminiError
from .vectors import VectorError, embed_cached, safe_cosine

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


class ScoringError(RuntimeError):
    """The answer can't be scored from the data we have (not worth retrying automatically)."""


def normalise_text(text):
    """Lower-case words only: punctuation and spacing differences don't matter when matching quotes."""
    return " ".join(re.sub(r"[^\w']+", " ", (text or "").lower()).split())


def rubric_value(rubric, key):
    """A rubric score (0-10) converted to 0-100. Anything missing or out of range is a malformed response, which is
    retried rather than clamped or zeroed."""
    value = rubric.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not 0 <= value <= 10:
        raise GeminiError(f"Gemini returned an invalid {key} score ({value!r}).", retryable=True)
    return float(value) * 10


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
    """(cosine, 0-100 score). The reference embedding is cached; raises VectorError if an embedding is unusable."""
    reference = ensure_reference(answer)
    if not (answer.transcript or "").strip():
        return 0.0, 0.0
    sim = safe_cosine(gemini.embed(answer.transcript), embed_cached(reference))
    return sim, similarity_to_score(sim)


# ---------- adaptive follow-ups ----------

def needs_decision(interview, answer):
    """Only base technical (and resume-based) answers can trigger a follow-up, and only while the interview has
    follow-ups left."""
    return (answer.category in current_app.config["FOLLOW_UP_CATEGORIES"] and not answer.is_follow_up
            and interview.follow_up_count < current_app.config["MAX_FOLLOW_UPS"])


def has_time_for_follow_up(interview, answer):
    """A follow-up (prep + answer + upload grace) plus every question still planned must fit before the deadline."""
    if interview.hard_deadline is None:
        return True
    cfg = current_app.config
    per_question = lambda seconds: cfg["PREP_SECONDS"] + seconds + cfg["ANSWER_GRACE_SECONDS"]  # noqa: E731
    needed = per_question(answer.answer_seconds) + sum(
        per_question(q["answer_seconds"]) for q in interview.question_plan[answer.position + 1:])
    return utcnow() + timedelta(seconds=needed) <= as_utc(interview.hard_deadline)


def follow_up_problem(follow_up, original_question, reference):
    """Why a generated follow-up must not be shown to the candidate, or None if it's fine."""
    question = (follow_up.get("question") or "").strip()
    if not question:
        return "empty question"
    if not 4 <= len(question.split()) <= 40 or len(question) > 300:
        return "question length out of bounds"
    if not question.endswith("?"):
        return "not phrased as a question"
    if not (follow_up.get("reference_answer") or "").strip():
        return "no reference answer"
    asked, original = normalise_text(question), normalise_text(original_question)
    asked_words, original_words = set(asked.split()), set(original.split())
    if asked == original or len(asked_words & original_words) / len(asked_words | original_words) >= 0.8:
        return "repeats the original question"
    padded = f" {asked} "
    for text in (reference, follow_up["reference_answer"]):
        words = normalise_text(text).split()
        if any(f" {' '.join(words[i:i + 6])} " in padded for i in range(len(words) - 5)):
            return "reveals the reference answer"
    return None


def decide_follow_up(answer):
    """Called right after transcription while the interview waits. Never raises: on any problem the interview
    simply moves on without a follow-up."""
    from .interview_room import finish, log_event

    interview = answer.interview
    if interview.awaiting_decision_since is None:
        return None
    follow_up = skipped = None
    try:
        if answer.transcript_status == InterviewAnswer.TRANSCRIPT_DONE:
            answer.similarity, answer.similarity_score = _similarity(answer)
            db.session.commit()
            if answer.similarity_score < current_app.config["FOLLOW_UP_THRESHOLD"] and answer.speech_detected:
                if not has_time_for_follow_up(interview, answer):
                    skipped = "not enough time left"
                else:
                    upcoming = [q["text"] for q in interview.question_plan[answer.position + 1:]]
                    candidate = gemini.generate_follow_up(answer.question_text, answer.reference_answer,
                                                          answer.transcript, interview.application.job.title,
                                                          upcoming=upcoming, category=answer.category)
                    skipped = follow_up_problem(candidate, answer.question_text, answer.reference_answer)
                    follow_up = None if skipped else candidate
    except Exception as e:  # any failure (AI, network, bug) must not keep the candidate waiting
        db.session.rollback()
        current_app.logger.warning("Follow-up decision for answer %s skipped: %s", answer.id, e)
        follow_up, skipped = None, "the answer couldn't be assessed in time"

    db.session.refresh(interview)
    if interview.awaiting_decision_since is None or interview.status != Interview.STATUS_IN_PROGRESS:
        return None  # the room already timed out the decision and moved on
    if follow_up:
        plan = list(interview.question_plan)
        plan.insert(answer.position + 1, {"text": follow_up["question"], "category": answer.category,
                                          "answer_seconds": answer.answer_seconds, "follow_up": True,
                                          "reference": follow_up["reference_answer"].strip(),
                                          "target": (follow_up.get("target_concept") or "").strip()[:200] or None,
                                          "source": answer.question_source, "follows": answer.position})
        interview.question_plan = plan  # reassign so the JSON change is saved
        interview.follow_up_count += 1
        log_event(interview, "follow_up_added", answer.position,
                  similarity_score=answer.similarity_score)
    elif skipped:
        log_event(interview, "follow_up_skipped", answer.position, reason=skipped)
        current_app.logger.info("Follow-up for answer %s skipped: %s", answer.id, skipped)
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
        # The only case where zeros are stored: there is genuinely nothing to assess
        answer.similarity, answer.similarity_score = 0.0, 0.0
        answer.relevance = answer.accuracy = answer.fluency = answer.clarity = 0.0
        answer.nlp_score = 0.0
        answer.verdict, answer.confidence, answer.needs_review = "no_answer", "high", False
        answer.feedback = {"summary": "No spoken answer was detected.", "strengths": [], "gaps": ["No answer given"],
                           "evidence": []}
    else:
        if answer.similarity is None or answer.similarity_score is None \
                or not math.isfinite(answer.similarity) or not math.isfinite(answer.similarity_score):
            try:
                answer.similarity, answer.similarity_score = _similarity(answer)
            except VectorError as e:
                raise ScoringError(f"Semantic similarity couldn't be computed: {e}.") from e
        job = answer.interview.application.job
        rubric = gemini.score_answer(answer.question_text, answer.category, answer.reference_answer,
                                     answer.transcript, job.title, metrics)
        relevance, accuracy = rubric_value(rubric, "relevance"), rubric_value(rubric, "accuracy")
        fluency, clarity = rubric_value(rubric, "fluency"), rubric_value(rubric, "clarity")
        verdict, confidence = rubric.get("verdict"), rubric.get("confidence")
        if verdict not in VERDICTS or confidence not in ("high", "medium", "low"):
            raise GeminiError(f"Gemini returned an invalid verdict/confidence ({verdict!r}, {confidence!r}).",
                              retryable=True)
        total = nlp_score(relevance, accuracy, answer.similarity_score, fluency)
        if not math.isfinite(total):
            raise ScoringError("The NLP score couldn't be computed from the components.")
        answer.relevance, answer.accuracy, answer.fluency, answer.clarity = relevance, accuracy, fluency, clarity
        answer.nlp_score = total
        answer.verdict, answer.confidence = verdict, confidence
        # Uncertain or unassessable results go to a human instead of being presented as certain
        answer.needs_review = confidence == "low" or verdict in ("clarification_request", "no_answer")
        # Keep only evidence that really appears in the transcript (ignoring punctuation and case), so invented
        # quotes are dropped. A real quote shows what was said; it doesn't prove the statement is correct.
        transcript = f" {normalise_text(answer.transcript)} "
        evidence = []
        for e in rubric.get("evidence") or []:
            quote = normalise_text(e.get("quote"))
            if quote and f" {quote} " in transcript:
                evidence.append({"quote": e["quote"].strip(), "criterion": e.get("criterion"),
                                 "note": e.get("note", "")})
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
    except ScoringError as e:
        db.session.rollback()
        answer = db.session.get(InterviewAnswer, answer_id)
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
            "fillers": sum(a.filler_count or 0 for a in scored), "review": sum(bool(a.needs_review) for a in scored)}
