"""The live interview (paper Section 3.2): start, ask, record answers, integrity events, finish, time-outs.

All progress lives on the server, so a candidate whose browser crashes or loses the network can rejoin and carry on
from the same question. Recordings are stored encrypted like resumes and transcribed in the background.
"""
from datetime import timedelta

from flask import current_app

from ..extensions import db
from ..models import Interview, InterviewAnswer, InterviewEvent, InterviewQuestion, User, as_utc, utcnow
from . import storage
from .notifications import notify
from .scheduling import join_window
from .security import audit


class RoomError(ValueError):
    """The request doesn't fit the interview's current state; the message is safe to show."""

    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


CLIENT_EVENTS = {"tab_hidden", "tab_visible", "window_blur", "device_error"}


# ---------- questions ----------

def default_questions(job):
    """A sensible starting set built from the job's required skills (HR can edit it)."""
    skills = job.skill_list
    secs = current_app.config["DEFAULT_ANSWER_SECONDS"]
    questions = [("behavioral", f"Tell us about yourself and why you're interested in the {job.title} role.")]
    for skill in skills[:3]:
        questions.append(("technical", f"Describe a project where you used {skill}. What did you build, what was "
                                       f"your role, and what problems did you solve?"))
    if len(skills) > 3:
        questions.append(("technical", f"How would you rate your experience with {', '.join(skills[3:6])}? "
                                       "Give a concrete example for the one you know best."))
    questions.append(("situational", "Imagine a deadline is two days away and a critical bug appears in your work. "
                                     "Walk us through exactly what you would do."))
    questions.append(("behavioral", "Tell us about a time you disagreed with a teammate. How did you handle it, "
                                    "and what was the outcome?"))
    return [{"text": text, "category": cat, "answer_seconds": secs} for cat, text in questions]


def ensure_questions(job):
    """Create the default question bank the first time a job needs one."""
    if not job.questions:
        # Append through the relationship: the (empty) collection is already loaded, so rows added
        # separately would not show up in job.questions until the next query
        for i, q in enumerate(default_questions(job)):
            job.questions.append(InterviewQuestion(position=i, **q))
        db.session.flush()
    return job.questions


# ---------- state ----------

def can_enter(interview):
    """Scheduled interviews open in their join window; started ones stay open until their hard deadline."""
    now = utcnow()
    if interview.status == Interview.STATUS_SCHEDULED:
        opens, closes = join_window(interview)
        return opens <= now < closes
    if interview.status == Interview.STATUS_IN_PROGRESS:
        return interview.hard_deadline is not None and now < as_utc(interview.hard_deadline)
    return False


def _answer(interview, position):
    return InterviewAnswer.query.filter_by(interview_id=interview.id, position=position).first()


def _limits(answer):
    prep = current_app.config["PREP_SECONDS"]
    return prep, answer.answer_seconds


def _deadline(answer):
    prep, limit = _limits(answer)
    return as_utc(answer.asked_at) + timedelta(seconds=prep + limit)


def state(interview):
    """Everything the room page needs to (re)draw itself."""
    data = {"status": interview.status, "position": interview.current_position, "total": interview.question_count,
            "finished": interview.is_finished, "question": None}
    if interview.status == Interview.STATUS_IN_PROGRESS and interview.current_position < interview.question_count:
        q = interview.question_plan[interview.current_position]
        data["question"] = {"position": interview.current_position, "text": q["text"], "category": q["category"],
                            "answer_seconds": q["answer_seconds"], "prep_seconds": current_app.config["PREP_SECONDS"]}
    if interview.hard_deadline is not None:
        data["seconds_left_overall"] = max(0, int((as_utc(interview.hard_deadline) - utcnow()).total_seconds()))
    return data


def log_event(interview, kind, position=None, **details):
    count = interview.events.count()
    if count >= current_app.config["MAX_INTEGRITY_EVENTS"]:
        return None
    event = InterviewEvent(interview_id=interview.id, kind=kind, position=position, details=details)
    db.session.add(event)
    return event


# ---------- lifecycle ----------

def start(interview, consented):
    """Begin (or rejoin) the interview. Freezes the question plan on first start."""
    if interview.status == Interview.STATUS_IN_PROGRESS:
        if not can_enter(interview):
            raise RoomError("This interview's time has run out.", 410)
        log_event(interview, "reconnect", interview.current_position)
        db.session.commit()
        return state(interview)
    if interview.status != Interview.STATUS_SCHEDULED or not can_enter(interview):
        raise RoomError("This interview isn't open right now.", 403)
    if not consented:
        raise RoomError("Please agree to the recording before starting.", 400)

    job = interview.application.job
    plan = [{"text": q.text, "category": q.category, "answer_seconds": q.answer_seconds}
            for q in ensure_questions(job)]
    now = utcnow()
    planned = sum(current_app.config["PREP_SECONDS"] + q["answer_seconds"] for q in plan)
    interview.question_plan = plan
    interview.status = Interview.STATUS_IN_PROGRESS
    interview.started_at = interview.recording_consent_at = now
    interview.current_position = 0
    interview.hard_deadline = max(interview.ends_at, now + timedelta(seconds=planned) +
                                  timedelta(minutes=current_app.config["FINISH_BUFFER_MINUTES"]))
    log_event(interview, "started", questions=len(plan))
    audit("interview.started", target=interview.application.candidate, interview_id=interview.id)
    db.session.commit()
    return state(interview)


def ask(interview, position):
    """The room shows question `position`: records when it was shown (the answer timer starts from here).
    Showing the same question again (after a reconnect) restarts its timer but is flagged for HR."""
    _require_running(interview)
    if position != interview.current_position:
        raise RoomError("That isn't the current question.")
    answer = _answer(interview, position)
    q = interview.question_plan[position]
    if answer is None:
        answer = InterviewAnswer(interview_id=interview.id, position=position, question_text=q["text"],
                                 category=q["category"], answer_seconds=q["answer_seconds"], asked_at=utcnow())
        db.session.add(answer)
    elif not answer.is_submitted:
        answer.attempts += 1
        answer.asked_at = utcnow()
        log_event(interview, "question_restarted", position, attempt=answer.attempts)
    db.session.commit()
    return state(interview)


MEDIA_SIGNATURES = {"video/webm": lambda b: b[:4] == b"\x1a\x45\xdf\xa3",  # EBML header (WebM/Matroska)
                    "video/mp4": lambda b: b[4:8] == b"ftyp"}


def detect_media(data):
    for media_type, check in MEDIA_SIGNATURES.items():
        if len(data) > 16 and check(data):
            return media_type
    return None


def submit(interview, position, data, duration_seconds):
    """Store the recorded answer for `position`. Re-uploading an already accepted answer is a harmless no-op."""
    _require_running(interview, allow_finished_answer=position)
    answer = _answer(interview, position)
    if answer is None:
        raise RoomError("This question hasn't been shown yet.")
    if answer.is_submitted:
        return state(interview)  # the browser retried after a dropped response
    if position != interview.current_position:
        raise RoomError("That isn't the current question.")
    if len(data) > current_app.config["MAX_ANSWER_UPLOAD"]:
        raise RoomError("The recording is too large.", 413)
    media_type = detect_media(data)
    if media_type is None:
        raise RoomError("The recording couldn't be read. Please try again.", 400)

    now = utcnow()
    grace = timedelta(seconds=current_app.config["ANSWER_GRACE_SECONDS"])
    answer.late = now > _deadline(answer) + grace
    answer.media_path = storage.save_resume(data, "webm" if media_type == "video/webm" else "mp4")
    answer.media_type, answer.media_size = media_type, len(data)
    answer.submitted_at = now
    try:
        reported = float(duration_seconds)
    except (TypeError, ValueError):
        reported = None
    limit = answer.answer_seconds + current_app.config["ANSWER_GRACE_SECONDS"]
    answer.duration_seconds = None if reported is None else max(0.0, min(reported, float(limit)))
    answer.transcript_status = InterviewAnswer.TRANSCRIPT_PENDING
    if answer.late:
        log_event(interview, "late_answer", position)
    interview.current_position = position + 1
    db.session.commit()

    from ..tasks import transcribe_answer

    transcribe_answer(answer.id)
    if interview.current_position >= interview.question_count:
        finish(interview, Interview.END_COMPLETED)
    return state(interview)


def record_client_event(interview, kind, position=None, details=None):
    if kind not in CLIENT_EVENTS:
        raise RoomError("Unknown event.", 400)
    if interview.status != Interview.STATUS_IN_PROGRESS:
        return False
    safe = {k: str(v)[:200] for k, v in (details or {}).items()} if isinstance(details, dict) else {}
    event = log_event(interview, kind, position if isinstance(position, int) else None, **dict(list(safe.items())[:5]))
    db.session.commit()
    return event is not None


def finish(interview, reason):
    if interview.status != Interview.STATUS_IN_PROGRESS:
        return False
    interview.status = Interview.STATUS_COMPLETED
    interview.completed_at, interview.end_reason = utcnow(), reason
    log_event(interview, "finished" if reason == Interview.END_COMPLETED else "timed_out",
              answered=sum(a.is_submitted for a in interview.answers), total=interview.question_count)
    audit("interview.finished", target=interview.application.candidate, interview_id=interview.id, reason=reason)
    db.session.commit()
    application = interview.application
    candidate, job = application.candidate, application.job
    answered = sum(a.is_submitted for a in interview.answers)
    notify(candidate, "interview.completed", f"Interview completed: {job.title}",
           "Thank you! The recruiter will review your interview." if reason == Interview.END_COMPLETED
           else f"Your interview ended after {answered} of {interview.question_count} answers.",
           f"/candidate/interviews/{interview.id}")
    for staff in User.query.filter(User.role.in_([User.ROLE_HR, User.ROLE_ADMIN]), User.active.is_(True)):
        notify(staff, "interview.completed_hr", f"Interview finished: {candidate.name} · {job.title}",
               f"{answered} of {interview.question_count} answers recorded"
               + ("" if reason == Interview.END_COMPLETED else " (timed out)") + ".",
               f"/hr/applications/{application.id}")
    return True


def expire_stale():
    """Housekeeping: interviews left in progress past their hard deadline are closed as timed out."""
    count = 0
    for interview in Interview.query.filter(Interview.status == Interview.STATUS_IN_PROGRESS,
                                            Interview.hard_deadline < utcnow()).all():
        if finish(interview, Interview.END_TIMED_OUT):
            count += 1
    return count


def _require_running(interview, allow_finished_answer=None):
    if interview.status == Interview.STATUS_COMPLETED and allow_finished_answer is not None:
        answer = _answer(interview, allow_finished_answer)
        if answer is not None and answer.is_submitted:
            return  # idempotent retry of the final answer after the interview finished
    if interview.status != Interview.STATUS_IN_PROGRESS:
        raise RoomError("This interview isn't in progress.", 403)
    if not can_enter(interview):
        raise RoomError("This interview's time has run out.", 410)
