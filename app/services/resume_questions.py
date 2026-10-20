"""Resume-based interview questions, personalised for each candidate.

Questions about the candidate's own experience, projects and skills are generated from the parsed resume in the
background (when the interview is booked, and refreshed by the housekeeping task), validated, and frozen into the
interview plan when it starts, right after the first job question. Weak answers to them get NLP follow-ups like
technical questions do.

Each set is fingerprinted with what it was generated from (resume profile, job, the job's questions, the count), so an
outdated set is never used. If no valid set is ready when the interview starts, the interview runs with the job's
questions only and HR sees why. The job's own questions stay identical for every candidate.
"""
import hashlib
import json
import re
from datetime import timedelta

from flask import current_app

from ..extensions import db
from ..models import Application, Interview, as_utc, utcnow
from . import gemini
from .gemini import GeminiError

SOURCE_LABELS = {"experience": "Experience", "project": "Project", "skill": "Skill"}

# Questions touching personal attributes are never asked (checked with the named item removed, so a project called
# "Health Tracker" is fine)
SENSITIVE = re.compile(
    r"\b(how old|your age|married|marital|spouse|wife|husband|children|kids|pregnan\w*|religio\w*|caste|"
    r"nationality|citizenship|disabilit\w*|your health|medical condition|salary|ctc|gender|"
    r"(career|employment|education|study) gaps?|gaps? (in|between) your)\b", re.IGNORECASE)


def _norm(text):
    return " ".join(re.sub(r"[^\w']+", " ", (text or "").lower()).split())


def count_for(job):
    return max(0, min(job.resume_question_count or 0, current_app.config["RESUME_QUESTIONS_MAX"]))


def job_question_texts(job):
    """The job questions an interview will use (the default set if HR hasn't written any yet)."""
    from .interview_room import default_questions

    return [q.text for q in job.questions] or [q["text"] for q in default_questions(job)]


def wanted(application):
    return (count_for(application.job) > 0 and application.parsed is not None and application.is_analysed
            and not application.is_withdrawn)


def fingerprint(application):
    job = application.job
    payload = json.dumps({"profile": application.parsed, "job": [job.title, job.skills],
                          "questions": job_question_texts(job), "count": count_for(job)},
                         sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_ready(application):
    return (application.resume_questions_status == Application.RQ_DONE and bool(application.resume_questions)
            and application.resume_questions_key == fingerprint(application))


# ---------- validation ----------

def _field(entry, key):
    """Profiles analysed before the structured format store an experience/project as one string ("Name: details")."""
    return entry.get(key) if isinstance(entry, dict) else entry if isinstance(entry, str) else None


def _subjects(profile, job, source):
    if source == "experience":
        items = [_field(e, key) for e in profile.get("experience") or [] for key in ("organisation", "role")]
    elif source == "project":
        items = [_field(p, "name") for p in profile.get("projects") or []]
    else:
        items = (profile.get("skills") or []) + job.skill_list
    return [n for n in (_norm(i) for i in items) if len(n) >= 2]


def _similar(a, b):
    wa, wb = set(_norm(a).split()), set(_norm(b).split())
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.7


def question_problem(item, profile, job, others):
    """Why a generated resume question must not be asked, or None."""
    text = (item.get("question") or "").strip()
    if not text:
        return "empty question"
    if not 5 <= len(text.split()) <= 45 or len(text) > 350:
        return "question length out of bounds"
    if not text.endswith("?"):
        return "not phrased as a question"
    if not (item.get("reference_answer") or "").strip():
        return "no reference answer"
    source, subject = item.get("source"), _norm(item.get("subject"))
    if source not in SOURCE_LABELS or not subject:
        return "no resume item named"
    known = _subjects(profile, job, source)
    if not any(subject == k or (len(subject) >= 3 and (subject in k or k in subject)) for k in known):
        return "asks about something that isn't in the resume"
    without_subject = re.sub(re.escape(item.get("subject") or ""), " ", text, flags=re.IGNORECASE)
    if SENSITIVE.search(without_subject):
        return "touches a personal or sensitive topic"
    if any(_similar(text, other) for other in others):
        return "repeats another question"
    return None


def validate(raw, application, existing):
    """(accepted entries, [reasons for rejected ones]); keeps at most the job's count."""
    job, profile = application.job, application.parsed or {}
    accepted, rejected = [], []
    for item in raw or []:
        problem = question_problem(item, profile, job, existing + [a["text"] for a in accepted])
        if problem:
            rejected.append(problem)
            continue
        accepted.append({"text": item["question"].strip(), "source": item["source"],
                         "subject": item["subject"].strip()[:150], "reference": item["reference_answer"].strip(),
                         "answer_seconds": current_app.config["DEFAULT_ANSWER_SECONDS"]})
    return accepted[:count_for(job)], rejected


# ---------- generation ----------

def generate(application_id, retries_left=0):
    """Task body. Raises GeminiError for a retryable failure while retries remain; otherwise records the outcome."""
    application = db.session.get(Application, application_id)
    if application is None:
        return None
    if not wanted(application):
        application.resume_questions, application.resume_questions_status = None, None
        application.resume_questions_key = application.resume_questions_error = None
        db.session.commit()
        return None
    job, key = application.job, fingerprint(application)
    existing = job_question_texts(job)

    def fail(message, retryable):
        if retryable and retries_left > 0:
            application.resume_questions_error = f"{message} (retrying automatically)"
            db.session.commit()
            raise GeminiError(message, retryable=True)
        application.resume_questions_status, application.resume_questions_error = Application.RQ_FAILED, message
        application.resume_questions_key, application.resume_questions_at = key, utcnow()
        db.session.commit()
        current_app.logger.error("Resume questions for application %s failed: %s", application_id, message)
        return None

    try:
        raw = gemini.generate_resume_questions(application.parsed, job.title, job.skill_list, existing,
                                               count_for(job))
    except GeminiError as e:
        return fail(str(e), e.retryable)
    accepted, rejected = validate(raw, application, existing)
    if not accepted:
        return fail("None of the generated questions passed the checks"
                    + (f" ({', '.join(sorted(set(rejected)))})." if rejected else "."), True)
    application.resume_questions, application.resume_questions_status = accepted, Application.RQ_DONE
    application.resume_questions_key, application.resume_questions_at = key, utcnow()
    wanted_count = count_for(job)
    application.resume_questions_error = (
        f"{len(accepted)} of {wanted_count} questions passed the checks ({', '.join(sorted(set(rejected)))})."
        if len(accepted) < wanted_count and rejected else None)
    db.session.commit()
    return len(accepted)


def queue(application, force=False):
    """Ask the worker for a (new) set unless a valid one exists or one is already being prepared."""
    if not wanted(application):
        return False
    now = utcnow()
    recent = application.resume_questions_at is not None and now - as_utc(application.resume_questions_at) < \
        timedelta(minutes=10)
    if not force:
        if is_ready(application):
            return False
        if application.resume_questions_status == Application.RQ_PENDING and recent:
            return False
        if (application.resume_questions_status == Application.RQ_FAILED
                and application.resume_questions_key == fingerprint(application)
                and now - as_utc(application.resume_questions_at) < timedelta(hours=1)):
            return False  # failed for this exact input recently: retried hourly (HR can force it)
    application_id = application.id
    application.resume_questions_status, application.resume_questions_at = Application.RQ_PENDING, now
    application.resume_questions_error = None
    db.session.commit()
    from ..tasks import prepare_resume_questions

    prepare_resume_questions(application_id)
    db.session.expire(application)  # the worker (or immediate mode) updates the row in another session
    return True


def queue_if_booked(application):
    """After booking or re-analysis: prepare questions for an upcoming interview. Never raises."""
    try:
        interview = application.interview
        if interview is not None and interview.status == Interview.STATUS_SCHEDULED:
            return queue(application)
    except Exception as e:  # preparation problems must never break booking or analysis
        db.session.rollback()
        current_app.logger.warning("Couldn't queue resume questions for application %s: %s", application.id, e)
    return False


def refresh_upcoming():
    """Housekeeping: make sure interviews starting soon have an up-to-date set (covers job/resume edits, interviews
    booked before this feature, and earlier failures)."""
    horizon = utcnow() + timedelta(hours=current_app.config["RESUME_QUESTIONS_LOOKAHEAD_HOURS"])
    upcoming = Interview.query.filter(Interview.status == Interview.STATUS_SCHEDULED,
                                      Interview.starts_at <= horizon).all()
    return sum(bool(queue_if_booked(i.application)) for i in upcoming)


# ---------- interview plan ----------

def plan_entries(application):
    """(plan entries to insert into an interview starting now, reason they're unavailable or None)."""
    if count_for(application.job) == 0:
        return [], None
    if application.parsed is None or not application.is_analysed:
        return [], "the resume hasn't been analysed"
    if is_ready(application):
        return [{"text": q["text"], "category": "resume", "answer_seconds": q["answer_seconds"],
                 "reference": q["reference"], "follow_up": False,
                 "source": f"{SOURCE_LABELS.get(q['source'], q['source'])}: {q['subject']}"}
                for q in application.resume_questions], None
    status = application.resume_questions_status
    if status == Application.RQ_PENDING:
        return [], "still being prepared"
    if status == Application.RQ_FAILED:
        return [], f"preparation failed: {application.resume_questions_error}"
    if status == Application.RQ_DONE:
        return [], "outdated (the resume or the job changed since they were prepared)"
    return [], "not prepared yet"
