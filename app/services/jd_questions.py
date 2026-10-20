"""Technical questions generated from the job description (paper 3.3: dynamic, domain-specific question generation).

Generated once per job, in the background, and stored in the job's question bank, so every candidate for the job is
asked the same technical questions and HR can review, edit, reorder or delete them before interviews. Each carries a
reference answer, so the paper's rule (low semantic similarity to the reference -> follow-up) applies directly.

- Fingerprinted by title, description, skills and count: editing the job regenerates them; questions HR edited become
  HR's own and are never replaced.
- They supersede the generic "Describe a project where you used <skill>" templates (resume-based questions now cover
  projects), which are removed when JD questions are added, unless HR edited them.
- If they aren't ready when an interview starts, the interview uses the job's other questions and logs why.
"""
import hashlib
import json
import re
from datetime import timedelta

from flask import current_app

from ..extensions import db
from ..models import Interview, InterviewQuestion, Job, as_utc, utcnow
from . import gemini
from .gemini import GeminiError
from .resume_questions import SENSITIVE, _norm, _similar

MAX_QUESTIONS = 15  # same limit as the HR question editor


def count_for(job):
    return max(0, min(job.jd_question_count or 0, current_app.config["JD_QUESTIONS_MAX"]))


def fingerprint(job):
    payload = json.dumps({"title": job.title, "description": job.description, "skills": job.skill_list,
                          "count": count_for(job)}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_ready(job):
    return job.jd_questions_status == Job.JD_DONE and job.jd_questions_key == fingerprint(job)


def _template_texts(job):
    """Texts of the generic per-skill template questions (superseded by JD questions)."""
    from .interview_room import default_questions

    return {q["text"] for q in default_questions(job) if q["category"] == "technical"}


def _is_replaceable_template(question, templates):
    return (question.category == "technical" and question.origin in (None, InterviewQuestion.ORIGIN_DEFAULT)
            and question.text in templates)


def question_problem(item, job, others):
    """Why a generated JD question must not be used, or None."""
    text = (item.get("question") or "").strip()
    if not text:
        return "empty question"
    if not 5 <= len(text.split()) <= 45 or len(text) > 350:
        return "question length out of bounds"
    if not text.endswith("?"):
        return "not phrased as a question"
    if len((item.get("reference_answer") or "").split()) < 15:
        return "reference answer too short"
    skill = _norm(item.get("skill"))
    known = [_norm(s) for s in job.skill_list]
    if not skill or not any(skill == k or (len(skill) >= 3 and (skill in k or k in skill)) for k in known):
        return "doesn't test a required skill"
    if SENSITIVE.search(text):
        return "touches a personal or sensitive topic"
    if any(_similar(text, other) for other in others):
        return "repeats another question"
    return None


def _skill_name(job, skill):
    """The job's own spelling of the skill the question tests."""
    wanted = _norm(skill)
    for s in job.skill_list:
        k = _norm(s)
        if k == wanted or (len(wanted) >= 3 and (wanted in k or k in wanted)):
            return s[:80]
    return skill.strip()[:80]


def _remove_jd_questions(job):
    for q in [q for q in job.questions if q.origin == InterviewQuestion.ORIGIN_JD]:
        job.questions.remove(q)


def _renumber(job):
    for i, q in enumerate(job.questions):
        q.position = i


def generate(job_id, retries_left=0):
    """Task body. Raises GeminiError for a retryable failure while retries remain; otherwise records the outcome."""
    from .interview_room import default_questions

    job = db.session.get(Job, job_id)
    if job is None:
        return None
    if count_for(job) == 0:  # switched off: untouched JD questions go away
        _remove_jd_questions(job)
        _renumber(job)
        job.jd_questions_status = job.jd_questions_key = job.jd_questions_error = None
        db.session.commit()
        return 0
    key = fingerprint(job)
    templates = _template_texts(job)
    kept = [q for q in job.questions if q.origin != InterviewQuestion.ORIGIN_JD
            and not _is_replaceable_template(q, templates)]
    existing = [q.text for q in kept] or [q["text"] for q in default_questions(job, skill_questions=False)]

    def fail(message, retryable):
        if retryable and retries_left > 0:
            job.jd_questions_error = f"{message} (retrying automatically)"
            db.session.commit()
            raise GeminiError(message, retryable=True)
        job.jd_questions_status, job.jd_questions_error = Job.JD_FAILED, message
        job.jd_questions_key, job.jd_questions_at = key, utcnow()
        db.session.commit()
        current_app.logger.error("JD questions for job %s failed: %s", job_id, message)
        return None

    try:
        raw = gemini.generate_jd_questions(job.title, job.description, job.skill_list, existing, count_for(job))
    except GeminiError as e:
        return fail(str(e), e.retryable)
    accepted, rejected = [], []
    for item in raw or []:
        problem = question_problem(item, job, existing + [a["question"] for a in accepted])
        (rejected.append(problem) if problem else accepted.append(item))
    if not accepted:
        return fail("None of the generated questions passed the checks"
                    + (f" ({', '.join(sorted(set(rejected)))})." if rejected else "."), True)

    # Replace the previous JD questions and the generic templates; HR-written/edited questions stay where they are
    if not job.questions:  # a new job: start from the non-technical part of the default set
        for q in default_questions(job, skill_questions=False):
            job.questions.append(InterviewQuestion(position=len(job.questions), **q))
    for q in [q for q in job.questions if q.origin == InterviewQuestion.ORIGIN_JD
              or _is_replaceable_template(q, templates)]:
        job.questions.remove(q)
    technical = [i for i, q in enumerate(job.questions) if q.category == "technical"]
    insert_at = technical[-1] + 1 if technical else min(1, len(job.questions))
    room = max(0, MAX_QUESTIONS - len(job.questions))
    wanted = min(count_for(job), len(accepted))
    new = [InterviewQuestion(text=item["question"].strip(), category="technical",
                             answer_seconds=current_app.config["DEFAULT_ANSWER_SECONDS"],
                             reference_answer=item["reference_answer"].strip(), origin=InterviewQuestion.ORIGIN_JD,
                             skill=_skill_name(job, item["skill"]), difficulty=item.get("difficulty"), position=0)
           for item in accepted[:min(wanted, room)]]
    for offset, q in enumerate(new):
        job.questions.insert(insert_at + offset, q)
    _renumber(job)

    notes = []
    if rejected and len(accepted) < count_for(job):
        notes.append(f"{len(accepted)} of {count_for(job)} questions passed the checks "
                     f"({', '.join(sorted(set(rejected)))})")
    if len(new) < min(wanted, len(accepted)) or room < wanted:
        notes.append(f"only {len(new)} added: a job can have at most {MAX_QUESTIONS} questions")
    job.jd_questions_status, job.jd_questions_key, job.jd_questions_at = Job.JD_DONE, key, utcnow()
    note = "; ".join(notes) + "."
    job.jd_questions_error = note[0].upper() + note[1:] if notes else None
    db.session.commit()
    _refresh_resume_questions(job)
    return len(new)


def _refresh_resume_questions(job):
    """The job's questions changed: upcoming interviews need resume questions that don't repeat them."""
    from .resume_questions import queue_if_booked

    for interview in Interview.query.join(Interview.application).filter(
            Interview.status == Interview.STATUS_SCHEDULED).all():
        if interview.application.job_id == job.id:
            queue_if_booked(interview.application)


def queue(job, force=False):
    """Ask the worker for (new) JD questions unless they're up to date or already being prepared. Returns True if
    queued."""
    now = utcnow()
    if not force:
        if count_for(job) == 0:
            if not any(q.origin == InterviewQuestion.ORIGIN_JD for q in job.questions):
                return False
        elif is_ready(job):
            return False
        if job.jd_questions_status == Job.JD_PENDING and job.jd_questions_at is not None \
                and now - as_utc(job.jd_questions_at) < timedelta(minutes=10):
            return False
        if (job.jd_questions_status == Job.JD_FAILED and job.jd_questions_key == fingerprint(job)
                and now - as_utc(job.jd_questions_at) < timedelta(hours=1)):
            return False  # failed for this exact input recently: retried hourly (HR can force it)
    job_id = job.id
    job.jd_questions_status, job.jd_questions_at, job.jd_questions_error = Job.JD_PENDING, now, None
    db.session.commit()
    from ..tasks import prepare_jd_questions

    prepare_jd_questions(job_id)
    db.session.expire(job)
    return True


def safe_queue(job, force=False):
    """queue() for callers that must never fail because of it (job edits, bookings)."""
    try:
        return queue(job, force=force)
    except Exception as e:
        db.session.rollback()
        current_app.logger.warning("Couldn't queue JD questions for job %s: %s", getattr(job, "id", "?"), e)
        return False


def refresh_upcoming():
    """Housekeeping: jobs with interviews coming up get up-to-date JD questions."""
    horizon = utcnow() + timedelta(hours=current_app.config["RESUME_QUESTIONS_LOOKAHEAD_HOURS"])
    jobs = {i.application.job for i in Interview.query.filter(Interview.status == Interview.STATUS_SCHEDULED,
                                                               Interview.starts_at <= horizon).all()}
    return sum(bool(safe_queue(job)) for job in sorted(jobs, key=lambda j: j.id))


def unavailable_reason(job):
    """Why an interview starting now has no JD questions, or None."""
    if count_for(job) == 0 or any(q.origin == InterviewQuestion.ORIGIN_JD for q in job.questions):
        return None
    if job.jd_questions_status == Job.JD_PENDING:
        return "still being prepared"
    if job.jd_questions_status == Job.JD_FAILED:
        return f"preparation failed: {job.jd_questions_error}"
    return "not prepared yet"
