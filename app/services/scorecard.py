"""The interview scorecard (paper §3.4): FinalScore = 0.75 NLP + 0.15 Emotion + 0.10 Context  (paper Eq. 1).

Components (all 0-100):
- NLP: one item per planned question (resume, JD and job questions alike). A follow-up better than its original
  answer recovers FOLLOW_UP_RECOVERY of the gap; a weaker follow-up never lowers the item, so every candidate is
  scored on the same number of items whether or not they were followed up. Unanswered questions count 0.
  Speech measures (clarity, pace, fillers) are shown alongside, as the paper lists them.
- Emotion: the paper's indicators (confidence, engagement, stability), averaged over answers whose facial analysis
  is reliable enough; low-reliability answers are left out.
- Context: consistency (one AI check after the interview: contradictions between answers and with the resume) and
  adjustability during follow-ups (how answers changed when probed).

If emotion or context can't be assessed, their weight is shared out over the remaining components instead of counting
as zero; the scorecard says so. The scorecard is provisional until every answer is scored and the checks have run,
and it is recomputed whenever an answer is re-scored or re-analysed.
"""
import json
import math

from flask import current_app
from sqlalchemy.exc import IntegrityError

from ..extensions import db
from ..models import Interview, InterviewAnswer, Scorecard, utcnow
from . import gemini
from .gemini import GeminiError
from .scoring import normalise_text


def _num(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _mean(values):
    values = [v for v in values if _num(v)]
    return round(sum(values) / len(values), 1) if values else None


# ---------- NLP items ----------

def _answer_info(answer):
    if answer is None or not answer.is_submitted:
        return {"state": "unanswered"}
    if answer.is_scored and _num(answer.nlp_score):
        return {"state": "scored", "nlp": round(answer.nlp_score, 1), "verdict": answer.verdict,
                "answer_id": answer.id}
    if answer.transcript_status == InterviewAnswer.TRANSCRIPT_FAILED or answer.score_status == InterviewAnswer.SCORE_FAILED:
        return {"state": "failed", "answer_id": answer.id}
    return {"state": "processing", "answer_id": answer.id}


def nlp_items(interview, recovery):
    """One item per planned (non follow-up) question, with its follow-up(s) folded in."""
    plan = interview.question_plan or []
    answers = {a.position: a for a in interview.answers}
    items = []
    for position, q in enumerate(plan):
        if q.get("follow_up"):
            continue
        original = _answer_info(answers.get(position))
        follow_ups = [(p, _answer_info(answers.get(p))) for p, f in enumerate(plan)
                      if f.get("follow_up") and f.get("follows") == position]
        item = {"position": position, "question": q.get("text", ""), "category": q.get("category"),
                "source": q.get("source"), "original": original,
                "follow_up": follow_ups[0][1] | {"position": follow_ups[0][0]} if follow_ups else None}
        if original["state"] == "unanswered":
            item["score"], item["state"] = 0.0, "unanswered"
        elif original["state"] != "scored":
            item["score"], item["state"] = None, original["state"]
        else:
            score, state = original["nlp"], "scored"
            fu = item["follow_up"]
            if fu and fu["state"] == "scored" and fu["nlp"] > score:
                score = round(score + recovery * (fu["nlp"] - score), 1)
                state = "recovered"
            elif fu and fu["state"] in ("processing", "failed"):
                state = "follow_up_" + fu["state"]
            item["score"], item["state"] = score, state
        items.append(item)
    return items


def adjustability(items):
    """How answers changed when probed: 50 = same level, above 50 = improved (paper: adjustability during follow-ups)."""
    deltas = [i["follow_up"]["nlp"] - i["original"]["nlp"] for i in items
              if i["follow_up"] and i["follow_up"]["state"] == "scored" and i["original"]["state"] == "scored"]
    if not deltas:
        return None, 0
    return round(sum(max(0.0, min(100.0, 50 + d)) for d in deltas) / len(deltas), 1), len(deltas)


# ---------- emotion ----------

def emotion_component(interview):
    submitted = [a for a in interview.answers if a.is_submitted]
    usable = [a for a in submitted if a.emotion_status == InterviewAnswer.EMOTION_DONE and a.emotion
              and a.emotion.get("reliability") != "low" and _num(a.emotion_score)]
    counts = {"used": len(usable),
              "low_reliability": sum(1 for a in submitted if a.emotion_status == InterviewAnswer.EMOTION_DONE
                                     and a.emotion and a.emotion.get("reliability") == "low"),
              "insufficient": sum(1 for a in submitted if a.emotion_status == InterviewAnswer.EMOTION_INSUFFICIENT),
              "failed": sum(1 for a in submitted if a.emotion_status == InterviewAnswer.EMOTION_FAILED),
              "pending": sum(1 for a in submitted if a.is_emotion_running),
              "not_analysed": sum(1 for a in submitted if a.emotion_status is None)}
    if not usable:
        return None, {"counts": counts}

    def stability(a):
        return a.emotion.get("stability", 100 - (a.emotion_nervousness or 0))

    indicators = {"confidence": _mean(a.emotion_confidence for a in usable),
                  "engagement": _mean(a.emotion_attentiveness for a in usable),
                  "stability": _mean(stability(a) for a in usable)}
    models = sorted({a.emotion.get("model", "DeepFace") for a in usable})
    return _mean(a.emotion_score for a in usable), {"counts": counts, "indicators": indicators, "models": models}


# ---------- the scorecard ----------

def compute(interview, sc=None):
    """All scorecard figures for an interview, as a plain dict (no database writes)."""
    cfg = current_app.config
    items = nlp_items(interview, cfg["FOLLOW_UP_RECOVERY"])
    item_scores = [i["score"] for i in items if i["score"] is not None]
    nlp = _mean(item_scores) if item_scores else None
    scored_answers = [a for a in interview.answers if a.is_scored]
    speech = {"clarity": _mean(a.clarity for a in scored_answers),
              "words_per_minute": _mean(a.words_per_minute for a in scored_answers),
              "fillers": sum(a.filler_count or 0 for a in scored_answers),
              "fluency": _mean(a.fluency for a in scored_answers)}
    emotion, emotion_info = emotion_component(interview)

    consistency = None
    if sc is not None and sc.consistency_status == Scorecard.CHECK_DONE and sc.consistency:
        consistency = round(sc.consistency["score"] * 10, 1)
    adjust, follow_ups = adjustability(items)
    cw = cfg["CONTEXT_WEIGHTS"]
    parts = {k: v for k, v in (("consistency", consistency), ("adjustability", adjust)) if v is not None}
    context = round(sum(cw[k] * v for k, v in parts.items()) / sum(cw[k] for k in parts), 1) if parts else None

    weights = cfg["FINAL_SCORE_WEIGHTS"]
    components = {k: v for k, v in (("nlp", nlp), ("emotion", emotion), ("context", context)) if v is not None}
    final = None
    used_weights = {}
    if "nlp" in components:  # without answer content there is no evaluation
        total = sum(weights[k] for k in components)
        used_weights = {k: round(weights[k] / total, 4) for k in components}
        final = round(sum(used_weights[k] * v for k, v in components.items()), 1)

    notes = []
    if emotion is None:
        notes.append("Emotion not assessed (no answer had reliable facial analysis); its weight was shared out.")
    elif emotion_info["counts"]["low_reliability"]:
        notes.append(f"{emotion_info['counts']['low_reliability']} answer(s) with low-reliability facial analysis "
                     "were left out of the emotion score.")
    if context is None:
        notes.append("Context not assessed yet (consistency check not available and no follow-ups); its weight was "
                     "shared out.")
    elif adjust is None:
        notes.append("No follow-up questions: context is based on consistency only.")
    failed = sum(1 for i in items if i["state"] in ("failed", "follow_up_failed"))
    if failed:
        notes.append(f"{failed} question(s) could not be scored (transcription or scoring failed): re-score them on "
                     "the application page to complete the scorecard.")
    unanswered = sum(1 for i in items if i["state"] == "unanswered")
    if unanswered:
        notes.append(f"{unanswered} question(s) not answered count as 0.")

    waiting = [i for i in items if i["score"] is None or i["state"].startswith("follow_up_")]
    finished = interview.status == Interview.STATUS_COMPLETED
    check_waiting = cfg["CONSISTENCY_CHECK_ENABLED"] and (sc is None or sc.consistency_status in (
        None, Scorecard.CHECK_PENDING))
    if not finished:
        status = Scorecard.STATUS_PENDING
    elif waiting or emotion_info["counts"]["pending"] or check_waiting:
        status = Scorecard.STATUS_PROVISIONAL
    else:
        status = Scorecard.STATUS_COMPLETE
    recommendation = None
    if final is not None:
        recommendation = next(key for minimum, key, _ in cfg["RECOMMENDATION_BANDS"] if final >= minimum)
    return {"status": status, "final": final, "nlp": nlp, "emotion": emotion, "context": context,
            "recommendation": recommendation, "weights": used_weights, "paper_weights": weights,
            "items": items, "speech": speech, "emotion_info": emotion_info,
            "context_parts": {"consistency": consistency, "adjustability": adjust, "follow_ups": follow_ups},
            "notes": notes, "waiting": len(waiting)}


def get_or_create(interview):
    sc = interview.scorecard
    if sc is None:
        sc = Scorecard(interview_id=interview.id)
        db.session.add(sc)
        try:
            db.session.flush()
        except IntegrityError:  # another worker created it at the same moment
            db.session.rollback()
            sc = Scorecard.query.filter_by(interview_id=interview.id).one()
    return sc


def refresh(interview, queue_check=True):
    """Recompute and store the scorecard of a finished interview; start the consistency check once all answers are
    transcribed. Never raises (scorecards must not break scoring or analysis jobs)."""
    try:
        if interview.status != Interview.STATUS_COMPLETED:
            return None
        sc = get_or_create(interview)
        data = compute(interview, sc)
        sc.status, sc.final_score, sc.nlp_score = data["status"], data["final"], data["nlp"]
        sc.emotion_score, sc.context_score, sc.recommendation = data["emotion"], data["context"], data["recommendation"]
        sc.details, sc.computed_at = data, utcnow()
        db.session.commit()
        if queue_check and current_app.config["CONSISTENCY_CHECK_ENABLED"] and sc.consistency_status is None \
                and _transcripts_ready(interview):
            queue_consistency(sc)
        return sc
    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        current_app.logger.error("Scorecard refresh for interview %s failed: %s", interview.id, e)
        return None


def refresh_for_answer(answer_id):
    answer = db.session.get(InterviewAnswer, answer_id)
    if answer is not None:
        refresh(answer.interview)


def _transcripts_ready(interview):
    return all(a.transcript_status in (InterviewAnswer.TRANSCRIPT_DONE, InterviewAnswer.TRANSCRIPT_FAILED)
               for a in interview.answers if a.is_submitted)


def queue_consistency(sc):
    sc.consistency_status, sc.consistency_requested_at, sc.consistency_error = Scorecard.CHECK_PENDING, utcnow(), None
    db.session.commit()
    from ..tasks import check_consistency

    check_consistency(sc.interview_id)


# ---------- consistency check ----------

def _quote_in(quote, transcripts):
    q = normalise_text(quote)
    return bool(q) and any(f" {q} " in f" {t} " for t in transcripts)


def run_consistency(interview_id, retries_left=0):
    """Task body: one AI check of the whole interview. Retryable failures are retried; others are recorded."""
    interview = db.session.get(Interview, interview_id)
    if interview is None:
        return None
    sc = get_or_create(interview)
    answered = [a for a in sorted(interview.answers, key=lambda a: a.position)
                if a.transcript_status == InterviewAnswer.TRANSCRIPT_DONE and a.speech_detected
                and (a.transcript or "").strip()]
    if len(answered) < 2:
        sc.consistency_status, sc.consistency_error = Scorecard.CHECK_SKIPPED, "fewer than two spoken answers"
        sc.consistency_at = utcnow()
        db.session.commit()
        return refresh(interview, queue_check=False)
    labels = {a.id: f"Q{a.position + 1}" for a in answered}
    try:
        result = gemini.check_consistency(interview.application.job.title,
                                          [(labels[a.id], a.question_text, a.transcript) for a in answered],
                                          interview.application.parsed or {})
        score = result.get("consistency")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) \
                or not 0 <= score <= 10 or result.get("confidence") not in ("high", "medium", "low"):
            raise GeminiError(f"Gemini returned an invalid consistency result ({score!r}).", retryable=True)
    except GeminiError as e:
        if e.retryable and retries_left > 0:
            sc.consistency_error = f"{e} (retrying automatically)"
            db.session.commit()
            raise
        sc.consistency_status, sc.consistency_error, sc.consistency_at = Scorecard.CHECK_FAILED, str(e), utcnow()
        db.session.commit()
        current_app.logger.error("Consistency check of interview %s failed: %s", interview_id, e)
        return refresh(interview, queue_check=False)

    # Keep only findings whose quotes really appear in the answers (guards against invented evidence)
    transcripts = [normalise_text(a.transcript) for a in answered]
    resume_text = [normalise_text(json.dumps(interview.application.parsed or {}, ensure_ascii=False))]
    contradictions, mismatches = [], [m for m in result.get("resume_mismatches") or []
                                      if _quote_in(m.get("quote"), transcripts)]
    for c in result.get("contradictions") or []:
        first, second = c.get("first_quote"), c.get("second_quote")
        if _quote_in(first, transcripts) and _quote_in(second, transcripts):
            contradictions.append(c)
        else:  # one side quoted from the resume instead of an answer: it's a resume mismatch
            for said, claim in ((first, second), (second, first)):
                if _quote_in(said, transcripts) and _quote_in(claim, resume_text):
                    mismatches.append({"resume_claim": claim, "quote": said, "explanation": c.get("explanation", "")})
                    break
    dropped = (len(result.get("contradictions") or []) + len(result.get("resume_mismatches") or [])
               - len(contradictions) - len(mismatches))
    sc.consistency = {"score": float(score), "summary": result.get("summary", ""), "confidence": result["confidence"],
                      "contradictions": contradictions[:5], "resume_mismatches": mismatches[:5],
                      "unverified_dropped": dropped}
    sc.consistency_status, sc.consistency_error, sc.consistency_at = Scorecard.CHECK_DONE, None, utcnow()
    db.session.commit()
    return refresh(interview, queue_check=False)


def recheck(interview):
    """HR: run the consistency check again and recompute."""
    sc = get_or_create(interview)
    sc.consistency_status = None
    db.session.commit()
    refresh(interview)
    return sc


def recommendation_label(key):
    return next((label for _, k, label in current_app.config["RECOMMENDATION_BANDS"] if k == key), None)
