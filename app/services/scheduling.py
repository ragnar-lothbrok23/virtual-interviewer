"""Interview scheduling (paper Section 3.6): slots, booking, rescheduling, cancellation, reminders, no-shows."""
from datetime import datetime, timedelta, timezone

from flask import current_app
from sqlalchemy import update

from ..extensions import db
from ..models import Application, Interview, InterviewSlot, as_utc, utcnow
from .ics import interview_ics
from .mail import external_url, send_email
from .notifications import notify
from .security import audit
from .timeutil import app_tz, format_local


class SchedulingError(ValueError):
    """A booking rule was broken; the message is safe to show to the user."""


# ---------- rules ----------

def _minutes(key):
    return timedelta(minutes=current_app.config[key])


def can_book(application):
    return (application.status == Application.STATUS_SHORTLISTED and application.job.is_open
            and (application.interview is None or not application.interview.is_active))


def change_deadline(interview):
    return as_utc(interview.starts_at) - _minutes("CHANGE_CUTOFF_MINUTES")


def can_change(interview):
    return interview.status == Interview.STATUS_SCHEDULED and utcnow() < change_deadline(interview)


def can_reschedule(interview):
    return can_change(interview) and interview.reschedule_count < current_app.config["MAX_RESCHEDULES"]


def join_window(interview):
    return as_utc(interview.starts_at) - _minutes("JOIN_EARLY_MINUTES"), interview.ends_at


def can_join(interview):
    opens, closes = join_window(interview)
    return interview.status in Interview.ACTIVE and opens <= utcnow() < closes


def bookable_slots(job):
    """Future, open slots with free seats, at least BOOKING_MIN_NOTICE_MINUTES away."""
    earliest = utcnow() + _minutes("BOOKING_MIN_NOTICE_MINUTES")
    return (InterviewSlot.query.filter(InterviewSlot.job_id == job.id, InterviewSlot.cancelled_at.is_(None),
                                       InterviewSlot.starts_at >= earliest,
                                       InterviewSlot.booked_count < InterviewSlot.capacity)
            .order_by(InterviewSlot.starts_at).all())


# ---------- slot creation ----------

def create_slot(job, starts_at_utc, duration, capacity, created_by):
    if as_utc(starts_at_utc) <= utcnow():
        raise SchedulingError("The slot must start in the future.")
    if InterviewSlot.query.filter_by(job_id=job.id, starts_at=starts_at_utc).first():
        raise SchedulingError("There is already a slot for this job at that time.")
    slot = InterviewSlot(job_id=job.id, starts_at=starts_at_utc, duration_minutes=duration, capacity=capacity,
                         created_by_id=created_by.id)
    db.session.add(slot)
    return slot


def plan_bulk_slots(first_day, last_day, weekdays, day_start, day_end, duration, gap):
    """Local (app timezone) start times for every slot that fits fully inside [day_start, day_end] on the chosen
    weekdays. Returns aware UTC datetimes."""
    if last_day < first_day:
        raise SchedulingError("The end date must be on or after the start date.")
    if day_end <= day_start:
        raise SchedulingError("The daily end time must be after the start time.")
    tz = app_tz()
    step = timedelta(minutes=duration + gap)
    starts, day = [], first_day
    while day <= last_day:
        if day.weekday() in weekdays:
            cursor = datetime.combine(day, day_start, tzinfo=tz)
            day_close = datetime.combine(day, day_end, tzinfo=tz)
            while cursor + timedelta(minutes=duration) <= day_close:
                starts.append(cursor.astimezone(timezone.utc))
                cursor += step
        day += timedelta(days=1)
    if len(starts) > current_app.config["MAX_SLOTS_PER_BULK"]:
        raise SchedulingError(f"That would create {len(starts)} slots; the limit is "
                              f"{current_app.config['MAX_SLOTS_PER_BULK']} at once. Choose a shorter period.")
    return starts


def create_bulk_slots(job, starts, duration, capacity, created_by):
    """Creates every planned slot that is in the future and not already taken. Returns (created, skipped)."""
    existing = {as_utc(s.starts_at) for s in job.slots}
    created = skipped = 0
    now = utcnow()
    for start in starts:
        if start <= now or start in existing:
            skipped += 1
            continue
        db.session.add(InterviewSlot(job_id=job.id, starts_at=start, duration_minutes=duration, capacity=capacity,
                                     created_by_id=created_by.id))
        existing.add(start)
        created += 1
    return created, skipped


# ---------- seats (atomic) ----------

def _take_seat(slot_id):
    """Atomically claim a seat: the UPDATE only succeeds while seats remain, so two simultaneous bookings for the
    last seat can't both win (works the same on SQLite and PostgreSQL)."""
    result = db.session.execute(
        update(InterviewSlot)
        .where(InterviewSlot.id == slot_id, InterviewSlot.cancelled_at.is_(None),
               InterviewSlot.booked_count < InterviewSlot.capacity)
        .values(booked_count=InterviewSlot.booked_count + 1)
        .execution_options(synchronize_session=False))
    return result.rowcount == 1


def _release_seat(slot_id):
    if slot_id is None:
        return
    db.session.execute(update(InterviewSlot)
                       .where(InterviewSlot.id == slot_id, InterviewSlot.booked_count > 0)
                       .values(booked_count=InterviewSlot.booked_count - 1)
                       .execution_options(synchronize_session=False))


def _check_slot_for(slot, application):
    if slot is None or slot.job_id != application.job_id or slot.is_cancelled:
        raise SchedulingError("That slot isn't available. Please choose another.")
    if as_utc(slot.starts_at) < utcnow() + _minutes("BOOKING_MIN_NOTICE_MINUTES"):
        raise SchedulingError("That slot starts too soon to book. Please choose a later one.")


# ---------- booking lifecycle ----------

def book(application, slot):
    if not can_book(application):
        raise SchedulingError("You can't book an interview for this application.")
    _check_slot_for(slot, application)
    if not _take_seat(slot.id):
        db.session.rollback()
        raise SchedulingError("Sorry, that slot was just taken. Please choose another.")
    interview = application.interview
    if interview is None:
        interview = Interview(application=application)
        db.session.add(interview)
    else:  # re-booking after a cancellation or a missed interview
        interview.calendar_sequence += 1
        interview.cancelled_at = interview.cancelled_by = interview.cancel_reason = None
        interview.reminder_24h_sent_at = interview.reminder_1h_sent_at = None
        interview.started_at = interview.completed_at = None
    interview.slot_id, interview.starts_at, interview.duration_minutes = slot.id, slot.starts_at, slot.duration_minutes
    interview.status, interview.booked_at = Interview.STATUS_SCHEDULED, utcnow()
    db.session.flush()
    audit("interview.booked", target=application.candidate, interview_id=interview.id,
          job=application.job.title, starts_at=as_utc(slot.starts_at).isoformat())
    db.session.commit()
    _send(interview, "booked")
    from .jd_questions import safe_queue
    from .resume_questions import queue_if_booked

    # Questions are prepared in the background, well before the start. The job's JD questions come first: once
    # they're ready they refresh the resume questions (which must not repeat them).
    if not safe_queue(application.job):
        queue_if_booked(application)
    return interview


def reschedule(interview, new_slot):
    if not can_reschedule(interview):
        raise SchedulingError("This interview can no longer be rescheduled.")
    _check_slot_for(new_slot, interview.application)
    if new_slot.id == interview.slot_id:
        raise SchedulingError("That's the slot you already have.")
    if not _take_seat(new_slot.id):
        db.session.rollback()
        raise SchedulingError("Sorry, that slot was just taken. Please choose another.")
    old_start = as_utc(interview.starts_at)
    _release_seat(interview.slot_id)
    interview.slot_id, interview.starts_at = new_slot.id, new_slot.starts_at
    interview.duration_minutes = new_slot.duration_minutes
    interview.reschedule_count += 1
    interview.calendar_sequence += 1
    interview.reminder_24h_sent_at = interview.reminder_1h_sent_at = None
    audit("interview.rescheduled", target=interview.application.candidate, interview_id=interview.id,
          old_start=old_start.isoformat(), new_start=as_utc(new_slot.starts_at).isoformat())
    db.session.commit()
    _send(interview, "rescheduled", old_start=old_start)
    return interview


def cancel(interview, by, reason="", notify_candidate=True):
    """Cancel an active interview and free its seat. `by` is candidate / hr / system."""
    if not interview.is_active:
        return False
    if by == "candidate" and not can_change(interview):
        raise SchedulingError("It's too close to the start to cancel this interview.")
    _release_seat(interview.slot_id)
    interview.status = Interview.STATUS_CANCELLED
    interview.cancelled_at, interview.cancelled_by, interview.cancel_reason = utcnow(), by, (reason or "")[:255]
    interview.calendar_sequence += 1
    audit("interview.cancelled", target=interview.application.candidate, interview_id=interview.id, by=by,
          reason=reason)
    db.session.commit()
    if notify_candidate:
        _send(interview, "cancelled", reason=reason, by=by)
    return True


def cancel_slot(slot, actor, reason):
    """HR cancels a whole slot: every booked candidate is told and may book again."""
    if slot.is_cancelled:
        return 0
    affected = [i for i in slot.interviews if i.is_active]
    for interview in affected:
        cancel(interview, by="hr", reason=reason or "The interview slot was cancelled by the recruiter.")
    slot.cancelled_at = utcnow()
    audit("slot.cancelled", actor=actor, slot_id=slot.id, job=slot.job.title, affected=len(affected))
    db.session.commit()
    return len(affected)


def sync_with_status(application):
    """Only shortlisted applications may hold an interview: if the application was withdrawn, rejected or moved
    back to review (by HR or by re-scoring), its scheduled interview is cancelled and the seat freed."""
    interview = application.interview
    if interview is None or interview.status != Interview.STATUS_SCHEDULED:
        return False
    if application.status == Application.STATUS_SHORTLISTED:
        return False
    reasons = {Application.STATUS_WITHDRAWN: "You withdrew your application.",
               Application.STATUS_REJECTED: "The application is no longer being considered.",
               Application.STATUS_REVIEW: "The application is back under review."}
    return cancel(interview, by="system", reason=reasons.get(application.status, "The application changed."))


# ---------- notifications ----------

def _send(interview, event, old_start=None, reason="", by=""):
    application = interview.application
    candidate, job = application.candidate, application.job
    page = f"/candidate/interviews/{interview.id}"
    join_url = external_url("candidate.interview", interview_id=interview.id)
    when = format_local(interview.starts_at)
    titles = {
        "booked": (f"Interview booked: {job.title}", f"Your interview is on {when}."),
        "rescheduled": (f"Interview moved: {job.title}", f"New time: {when} (was {format_local(old_start)})."),
        "cancelled": (f"Interview cancelled: {job.title}",
                      "You can book a new time from your application page." if by != "candidate" else "You cancelled it."),
    }
    title, body = titles[event]
    notify(candidate, f"interview.{event}", title, body, page)
    cancelled = event == "cancelled"
    ics = interview_ics(interview, join_url, cancelled=cancelled)
    # Booking changes are transactional: always emailed, regardless of the "interview reminders" preference
    send_email(candidate.email, title, f"interview_{event}", user=candidate, interview=interview, job=job,
               when=when, old_when=format_local(old_start) if old_start else "", reason=reason, by=by,
               link=external_url("candidate.interview", interview_id=interview.id),
               book_link=external_url("candidate.application", app_id=application.id),
               attachments=[{"filename": "interview.ics", "data": ics,
                             "content_type": f"text/calendar; method={'CANCEL' if cancelled else 'REQUEST'}"}])


# ---------- periodic jobs ----------

def send_due_reminders():
    """Called every few minutes: 24-hour and 1-hour reminders, each sent at most once per booking."""
    now = utcnow()
    sent = 0
    upcoming = Interview.query.filter(Interview.status == Interview.STATUS_SCHEDULED,
                                      Interview.starts_at > now, Interview.starts_at <= now + timedelta(hours=24)).all()
    for interview in upcoming:
        until = as_utc(interview.starts_at) - now
        kind = None
        if until <= timedelta(hours=1) and interview.reminder_1h_sent_at is None:
            kind = "1h"
        elif until > timedelta(hours=1) and interview.reminder_24h_sent_at is None:
            kind = "24h"
        if kind is None:
            continue
        # Mark first and commit, so a crash mid-send never causes a duplicate reminder
        if kind == "1h":
            interview.reminder_1h_sent_at = now
            interview.reminder_24h_sent_at = interview.reminder_24h_sent_at or now  # no 24h reminder after the 1h one
        else:
            interview.reminder_24h_sent_at = now
        db.session.commit()
        candidate, job = interview.application.candidate, interview.application.job
        when = format_local(interview.starts_at)
        label = "in about an hour" if kind == "1h" else "tomorrow" if until > timedelta(hours=12) else "soon"
        email = dict(category="interviews", subject=f"Reminder: your {job.title} interview is {label}",
                     template="interview_reminder", interview=interview, job=job, when=when, kind=kind,
                     link=external_url("candidate.interview", interview_id=interview.id))
        notify(candidate, f"interview.reminder_{kind}", f"Interview {label}: {job.title}", f"Starts {when}.",
               f"/candidate/interviews/{interview.id}", email=email)
        sent += 1
    return sent


def mark_no_shows():
    """Interviews whose window has passed without the candidate joining become 'missed'."""
    now = utcnow()
    grace = _minutes("NO_SHOW_GRACE_MINUTES")
    marked = 0
    for interview in Interview.query.filter(Interview.status == Interview.STATUS_SCHEDULED,
                                            Interview.starts_at < now).all():
        if interview.ends_at + grace > now:
            continue
        interview.status = Interview.STATUS_NO_SHOW
        audit("interview.no_show", target=interview.application.candidate, interview_id=interview.id)
        db.session.commit()
        job = interview.application.job
        notify(interview.application.candidate, "interview.no_show", f"Missed interview: {job.title}",
               "You didn't join your interview. Contact the recruiter if something went wrong.",
               f"/candidate/interviews/{interview.id}")
        marked += 1
    return marked

