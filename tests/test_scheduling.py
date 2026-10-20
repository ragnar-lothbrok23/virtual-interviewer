import io
import re
from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest
from freezegun import freeze_time

from app.extensions import db
from app.models import Application, AuditLog, Interview, InterviewSlot, Job, Notification, User, as_utc, utcnow
from app.services import mail, scheduling
from app.services.ics import interview_ics
from tests.conftest import PASSWORD
from tests.test_jobs_and_applications import BACKEND_RESUME

IST = ZoneInfo("Asia/Kolkata")


# ---------- fixtures and helpers ----------

@pytest.fixture
def job_id(app):
    app.test_cli_runner().invoke(args=["seed-jobs"])
    with app.app_context():
        return Job.query.filter_by(title="Python Backend Developer").one().id


def shortlist(app, client, job_id):
    """Apply with the backend resume (auto-shortlisted by the fake AI). Returns the application id."""
    client.post(f"/candidate/jobs/{job_id}/apply", data={"resume": (io.BytesIO(BACKEND_RESUME), "cv.docx")},
                content_type="multipart/form-data")
    with app.app_context():
        a = Application.query.filter_by(job_id=job_id).order_by(Application.id.desc()).first()
        assert a.status == "shortlisted"
        return a.id


def add_slot(app, job_id, hours_ahead=48, capacity=1, duration=30):
    with app.app_context():
        hr = User.query.filter(User.role.in_(["hr", "admin"])).first()
        if hr is None:
            hr = User(name="Slot Maker", email="slots@test.com", role=User.ROLE_HR)
            hr.set_password(PASSWORD)
            db.session.add(hr)
            db.session.flush()
        start = (utcnow() + timedelta(hours=hours_ahead)).replace(second=0, microsecond=0)
        slot = scheduling.create_slot(db.session.get(Job, job_id), start, duration, capacity, hr)
        db.session.commit()
        return slot.id


def get(app, model, ident):
    with app.app_context():
        obj = db.session.get(model, ident)
        db.session.expunge(obj)
        return obj


def ics_of(message):
    attachment = next(a for a in message["attachments"] if a["filename"] == "interview.ics")
    return attachment["data"].decode()


@pytest.fixture
def booked(app, candidate_client, hr_client, job_id, fake_gemini, outbox):
    """A shortlisted candidate with an interview booked 48h ahead. Returns (app_id, interview_id, slot_id)."""
    app_id = shortlist(app, candidate_client, job_id)
    slot_id = add_slot(app, job_id)
    candidate_client.post(f"/candidate/applications/{app_id}/book", data={"slot_id": slot_id})
    with app.app_context():
        interview_id = Interview.query.one().id
    return app_id, interview_id, slot_id


# ---------- HR slot creation ----------

def test_hr_creates_single_slot_in_local_time(app, hr_client, job_id):
    local = (utcnow() + timedelta(days=2)).astimezone(IST).replace(hour=10, minute=30, second=0, microsecond=0)
    response = hr_client.post(f"/hr/jobs/{job_id}/slots", data={
        "one-starts_at": local.strftime("%Y-%m-%dT%H:%M"), "one-duration": "30", "one-capacity": "3",
        "one-submit_single": "y"}, follow_redirects=True)
    assert b"Slot added for" in response.data and b"10:30 AM IST" in response.data
    with app.app_context():
        slot = InterviewSlot.query.one()
        assert as_utc(slot.starts_at) == local and slot.capacity == 3  # stored in UTC
        assert (as_utc(slot.starts_at).hour, as_utc(slot.starts_at).minute) == (5, 0)  # 10:30 IST == 05:00 UTC


def test_single_slot_validation(app, hr_client, job_id):
    past = (utcnow() - timedelta(hours=1)).astimezone(IST).strftime("%Y-%m-%dT%H:%M")
    response = hr_client.post(f"/hr/jobs/{job_id}/slots", data={"one-starts_at": past, "one-duration": "30",
                                                               "one-capacity": "1", "one-submit_single": "y"})
    assert b"must start in the future" in response.data
    future = (utcnow() + timedelta(days=1)).astimezone(IST).strftime("%Y-%m-%dT%H:%M")
    data = {"one-starts_at": future, "one-duration": "30", "one-capacity": "1", "one-submit_single": "y"}
    hr_client.post(f"/hr/jobs/{job_id}/slots", data=data)
    assert b"already a slot" in hr_client.post(f"/hr/jobs/{job_id}/slots", data=data).data
    data["one-capacity"] = "0"
    assert b"Number must be between 1 and 50" in hr_client.post(f"/hr/jobs/{job_id}/slots", data=data).data


def bulk(hr_client, job_id, first, last, start="10:00", end="12:00", weekdays=(0, 1, 2, 3, 4), duration=30, gap=0):
    data = {"bulk-first_day": first.isoformat(), "bulk-last_day": last.isoformat(), "bulk-day_start": start,
            "bulk-day_end": end, "bulk-duration": str(duration), "bulk-gap": str(gap), "bulk-capacity": "2",
            "bulk-weekdays": [str(w) for w in weekdays], "bulk-submit_bulk": "y"}
    return hr_client.post(f"/hr/jobs/{job_id}/slots", data=data, follow_redirects=True)


def test_bulk_slots(app, hr_client, job_id):
    today = utcnow().astimezone(IST).date()
    monday = today + timedelta(days=(7 - today.weekday()))  # next Monday
    response = bulk(hr_client, job_id, monday, monday + timedelta(days=6))  # Mon-Sun, weekdays only
    assert b"Created 20 slot(s)" in response.data  # 5 days x 4 x 30-minute slots (10:00-12:00)
    response = bulk(hr_client, job_id, monday, monday + timedelta(days=6))
    assert b"Created 0 slot(s) (20 skipped" in response.data  # idempotent
    # 20-minute slots with 10-minute breaks start at 10:00 and 10:30 too: both already exist
    response = bulk(hr_client, job_id, monday, monday, start="10:00", end="11:00", duration=20, gap=10)
    assert b"Created 0 slot(s) (2 skipped" in response.data
    response = bulk(hr_client, job_id, monday, monday, start="13:00", end="14:00", duration=20, gap=10)
    assert b"Created 2 slot(s)" in response.data  # 13:00-13:20 and 13:30-13:50
    with app.app_context():
        assert InterviewSlot.query.count() == 22
        starts = sorted(as_utc(s.starts_at).astimezone(IST).strftime("%a %H:%M") for s in InterviewSlot.query)
    assert "Mon 10:00" in starts and "Mon 13:30" in starts and not any(s.startswith(("Sat", "Sun")) for s in starts)


def test_bulk_slot_validation(app, hr_client, job_id):
    today = utcnow().astimezone(IST).date()
    assert b"end date must be on or after" in bulk(hr_client, job_id, today + timedelta(days=3), today + timedelta(days=1)).data
    assert b"end time must be after" in bulk(hr_client, job_id, today + timedelta(days=1), today + timedelta(days=2),
                                             start="12:00", end="10:00").data
    app.config["MAX_SLOTS_PER_BULK"] = 5
    try:
        assert b"the limit is 5" in bulk(hr_client, job_id, today + timedelta(days=1), today + timedelta(days=30)).data
    finally:
        app.config["MAX_SLOTS_PER_BULK"] = 300
    with app.app_context():
        assert InterviewSlot.query.count() == 0


# ---------- booking ----------

def test_book_sends_confirmation_with_calendar_invite(app, booked, candidate_client, outbox):
    app_id, interview_id, slot_id = booked
    interview, slot = get(app, Interview, interview_id), get(app, InterviewSlot, slot_id)
    assert interview.status == "scheduled" and slot.booked_count == 1
    message = next(m for m in outbox if m["subject"].startswith("Interview booked"))
    ics = ics_of(message)
    assert "METHOD:REQUEST" in ics and f"UID:interview-{interview_id}@" in ics and "SEQUENCE:0" in ics
    assert as_utc(slot.starts_at).strftime("DTSTART:%Y%m%dT%H%M%SZ") in ics and "STATUS:CONFIRMED" in ics
    assert all(len(line.encode()) <= 75 for line in ics.split("\r\n"))  # RFC 5545 line folding
    mime = bytes(mail.to_mime(message)).decode(errors="replace")
    assert "text/calendar; method=\"REQUEST\"" in mime or "method=REQUEST" in mime
    page = candidate_client.get(f"/candidate/interviews/{interview_id}").get_data(as_text=True)
    assert "Join interview" in page and "disabled" in page and "Reschedule" in page
    with app.app_context():
        assert Notification.query.filter_by(kind="interview.booked").count() == 1
        assert AuditLog.query.filter_by(action="interview.booked").count() == 1
    # booking again just redirects to the existing interview
    response = candidate_client.get(f"/candidate/applications/{app_id}/book")
    assert response.status_code == 302 and f"/interviews/{interview_id}" in response.headers["Location"]


def test_only_shortlisted_candidates_can_book(app, candidate_client, hr_client, job_id, fake_gemini):
    app_id = shortlist(app, candidate_client, job_id)
    slot_id = add_slot(app, job_id)
    hr_client.post(f"/hr/applications/{app_id}/status", data={"status": "under_review"})
    response = candidate_client.post(f"/candidate/applications/{app_id}/book", data={"slot_id": slot_id},
                                     follow_redirects=True)
    assert b"opens once you" in response.data
    assert get(app, InterviewSlot, slot_id).booked_count == 0


def test_last_seat_cannot_be_taken_twice(app, candidate_client, make_user, login, job_id, fake_gemini):
    first = shortlist(app, candidate_client, job_id)
    make_user("second@test.com")
    second_client = login("second@test.com")
    second = shortlist(app, second_client, job_id)
    slot_id = add_slot(app, job_id, capacity=1)
    candidate_client.post(f"/candidate/applications/{first}/book", data={"slot_id": slot_id})
    response = second_client.post(f"/candidate/applications/{second}/book", data={"slot_id": slot_id},
                                  follow_redirects=True)
    assert b"just taken" in response.data
    assert get(app, InterviewSlot, slot_id).booked_count == 1
    with app.app_context():
        assert Interview.query.count() == 1
        # the atomic claim refuses a full slot even if called directly
        assert scheduling._take_seat(slot_id) is False


def test_shared_slot_capacity(app, candidate_client, make_user, login, job_id, fake_gemini):
    slot_id = add_slot(app, job_id, capacity=2)
    clients = [candidate_client]
    for email in ("b@test.com", "c@test.com"):
        make_user(email)
        clients.append(login(email))
    results = []
    for client in clients:
        app_id = shortlist(app, client, job_id)
        results.append(client.post(f"/candidate/applications/{app_id}/book", data={"slot_id": slot_id},
                                   follow_redirects=True))
    assert b"just taken" in results[2].data
    assert get(app, InterviewSlot, slot_id).booked_count == 2


@pytest.mark.parametrize("problem", ["too_soon", "other_job", "cancelled", "missing"])
def test_invalid_slot_choices_are_refused(app, candidate_client, hr_client, job_id, fake_gemini, problem):
    app_id = shortlist(app, candidate_client, job_id)
    with app.app_context():
        other_job = Job.query.filter(Job.id != job_id).first().id
    if problem == "too_soon":
        slot_id = add_slot(app, job_id, hours_ahead=0.5)
    elif problem == "other_job":
        slot_id = add_slot(app, other_job)
    elif problem == "cancelled":
        slot_id = add_slot(app, job_id)
        with app.app_context():
            db.session.get(InterviewSlot, slot_id).cancelled_at = utcnow()
            db.session.commit()
    else:
        slot_id = 9999
    response = candidate_client.post(f"/candidate/applications/{app_id}/book", data={"slot_id": slot_id},
                                     follow_redirects=True)
    assert b"starts too soon" in response.data or b"isn&#39;t available" in response.data
    with app.app_context():
        assert Interview.query.count() == 0


def test_booking_page_lists_only_bookable_slots(app, candidate_client, job_id, fake_gemini):
    app_id = shortlist(app, candidate_client, job_id)
    good = add_slot(app, job_id, hours_ahead=30)
    add_slot(app, job_id, hours_ahead=0.5)  # too soon
    page = candidate_client.get(f"/candidate/applications/{app_id}/book").get_data(as_text=True)
    assert page.count('name="slot_id"') == 1 and f'value="{good}"' in page


# ---------- reschedule and cancel ----------

def test_reschedule_moves_seat_and_updates_invite(app, booked, candidate_client, job_id, outbox):
    app_id, interview_id, old_slot = booked
    new_slot = add_slot(app, job_id, hours_ahead=72)
    response = candidate_client.post(f"/candidate/interviews/{interview_id}/reschedule", data={"slot_id": new_slot},
                                     follow_redirects=True)
    assert b"has been moved" in response.data
    interview = get(app, Interview, interview_id)
    assert interview.slot_id == new_slot and interview.reschedule_count == 1 and interview.calendar_sequence == 1
    assert get(app, InterviewSlot, old_slot).booked_count == 0 and get(app, InterviewSlot, new_slot).booked_count == 1
    ics = ics_of(next(m for m in outbox if m["subject"].startswith("Interview moved")))
    assert f"UID:interview-{interview_id}@" in ics and "SEQUENCE:1" in ics  # same entry, updated


def test_reschedule_limit(app, booked, candidate_client, job_id):
    _, interview_id, _ = booked
    for hours in (60, 72):
        candidate_client.post(f"/candidate/interviews/{interview_id}/reschedule",
                              data={"slot_id": add_slot(app, job_id, hours_ahead=hours)})
    assert get(app, Interview, interview_id).reschedule_count == 2
    response = candidate_client.get(f"/candidate/interviews/{interview_id}/reschedule", follow_redirects=True)
    assert b"can no longer be rescheduled" in response.data


def test_cancel_frees_seat_sends_cancellation_and_allows_rebooking(app, booked, candidate_client, job_id, outbox):
    app_id, interview_id, slot_id = booked
    response = candidate_client.post(f"/candidate/interviews/{interview_id}/cancel", data={"reason": "Exam clash"},
                                     follow_redirects=True)
    assert b"has been cancelled" in response.data
    interview = get(app, Interview, interview_id)
    assert interview.status == "cancelled" and interview.cancelled_by == "candidate" and interview.cancel_reason == "Exam clash"
    assert get(app, InterviewSlot, slot_id).booked_count == 0
    ics = ics_of(next(m for m in outbox if m["subject"].startswith("Interview cancelled")))
    assert "METHOD:CANCEL" in ics and "STATUS:CANCELLED" in ics and "SEQUENCE:1" in ics
    # rebook reuses the same interview row (one per application)
    candidate_client.post(f"/candidate/applications/{app_id}/book", data={"slot_id": slot_id})
    with app.app_context():
        rows = Interview.query.all()
        assert len(rows) == 1 and rows[0].status == "scheduled" and rows[0].calendar_sequence == 2


def test_changes_blocked_close_to_start(app, booked, login):
    _, interview_id, _ = booked
    start = as_utc(get(app, Interview, interview_id).starts_at)
    with freeze_time(start - timedelta(minutes=90)):  # inside the 2-hour cutoff
        candidate_client = login("cand@test.com")  # sessions expire after 12 h, so sign in "at" that time
        page = candidate_client.get(f"/candidate/interviews/{interview_id}").get_data(as_text=True)
        assert "too close to the start" in page
        response = candidate_client.post(f"/candidate/interviews/{interview_id}/cancel", data={}, follow_redirects=True)
        assert b"too close to the start" in response.data
    assert get(app, Interview, interview_id).status == "scheduled"


# ---------- HR cancellations and status changes ----------

def test_hr_cancels_slot_and_candidates_are_told(app, booked, hr_client, outbox):
    _, interview_id, slot_id = booked
    response = hr_client.post(f"/hr/slots/{slot_id}/cancel", data={"reason": "Panel unavailable"}, follow_redirects=True)
    assert b"1 candidate(s) were told" in response.data
    assert get(app, InterviewSlot, slot_id).is_cancelled
    interview = get(app, Interview, interview_id)
    assert interview.status == "cancelled" and interview.cancelled_by == "hr"
    assert any(m["subject"].startswith("Interview cancelled") and "Panel unavailable" in m["text"] for m in outbox)


def test_hr_cancels_single_interview(app, booked, hr_client):
    app_id, interview_id, slot_id = booked
    hr_client.post(f"/hr/interviews/{interview_id}/cancel", data={"reason": "Role on hold"})
    assert get(app, Interview, interview_id).cancelled_by == "hr" and get(app, InterviewSlot, slot_id).booked_count == 0


@pytest.mark.parametrize("action", ["reject", "back_to_review", "withdraw"])
def test_leaving_shortlist_cancels_interview(app, booked, hr_client, candidate_client, action):
    app_id, interview_id, slot_id = booked
    if action == "withdraw":
        candidate_client.post(f"/candidate/applications/{app_id}/withdraw", data={"confirm": "y"})
    else:
        hr_client.post(f"/hr/applications/{app_id}/status",
                       data={"status": "rejected" if action == "reject" else "under_review"})
    interview = get(app, Interview, interview_id)
    assert interview.status == "cancelled" and interview.cancelled_by == "system"
    assert get(app, InterviewSlot, slot_id).booked_count == 0


# ---------- reminders and no-shows ----------

def run_housekeeping(app):
    with app.app_context():
        return scheduling.send_due_reminders(), scheduling.mark_no_shows()


def test_reminders_are_sent_once_each(app, booked, outbox):
    _, interview_id, _ = booked
    start = as_utc(get(app, Interview, interview_id).starts_at)
    outbox.clear()
    with freeze_time(start - timedelta(hours=30)):
        assert run_housekeeping(app) == (0, 0)  # too early
    with freeze_time(start - timedelta(hours=23)):
        assert run_housekeeping(app) == (1, 0)
        assert run_housekeeping(app) == (0, 0)  # not twice
    with freeze_time(start - timedelta(minutes=50)):
        assert run_housekeeping(app) == (1, 0)
        assert run_housekeeping(app) == (0, 0)
    subjects = [m["subject"] for m in outbox]
    assert subjects == ["Reminder: your Python Backend Developer interview is tomorrow",
                        "Reminder: your Python Backend Developer interview is in about an hour"]


def test_late_booking_gets_only_the_one_hour_reminder(app, booked, outbox):
    _, interview_id, _ = booked
    start = as_utc(get(app, Interview, interview_id).starts_at)
    outbox.clear()
    with freeze_time(start - timedelta(minutes=40)):
        assert run_housekeeping(app) == (1, 0)
    with freeze_time(start - timedelta(minutes=20)):
        assert run_housekeeping(app) == (0, 0)
    assert len(outbox) == 1 and "about an hour" in outbox[0]["subject"]


def test_reminder_respects_email_preference_but_notifies_in_app(app, booked, candidate_client, outbox):
    _, interview_id, _ = booked
    candidate_client.post("/account/preferences", data={"prefs-email_applications": "y"})  # interviews off
    start = as_utc(get(app, Interview, interview_id).starts_at)
    outbox.clear()
    with freeze_time(start - timedelta(hours=20)):
        run_housekeeping(app)
    assert outbox == []
    with app.app_context():
        assert Notification.query.filter_by(kind="interview.reminder_24h").count() == 1


def test_no_show_marked_after_window_plus_grace(app, booked, hr_client):
    app_id, interview_id, _ = booked
    interview = get(app, Interview, interview_id)
    end = interview.ends_at
    with freeze_time(end + timedelta(minutes=5)):
        assert run_housekeeping(app)[1] == 0  # still within the grace period
    with freeze_time(end + timedelta(minutes=16)):
        assert run_housekeeping(app)[1] == 1
    assert get(app, Interview, interview_id).status == "no_show"
    with app.app_context():
        assert Notification.query.filter_by(kind="interview.no_show").count() == 1
    assert b"Missed" in hr_client.get(f"/hr/applications/{app_id}").data


def test_cancelled_interviews_get_no_reminders(app, booked, candidate_client, outbox):
    _, interview_id, _ = booked
    candidate_client.post(f"/candidate/interviews/{interview_id}/cancel", data={})
    start = as_utc(get(app, Interview, interview_id).starts_at)
    outbox.clear()
    with freeze_time(start - timedelta(hours=20)):
        assert run_housekeeping(app) == (0, 0)


# ---------- join window, calendar download, access control ----------

def test_join_window(app, booked, candidate_client, login):
    _, interview_id, _ = booked
    start = as_utc(get(app, Interview, interview_id).starts_at)
    response = candidate_client.get(f"/candidate/interviews/{interview_id}/join", follow_redirects=True)
    assert b"You can join from" in response.data
    with freeze_time(start - timedelta(minutes=5)):
        candidate_client = login("cand@test.com")
        response = candidate_client.get(f"/candidate/interviews/{interview_id}/join")
        assert response.status_code == 200 and b"Before we start" in response.data  # the interview room
        page = candidate_client.get(f"/candidate/interviews/{interview_id}").get_data(as_text=True)
        assert "disabled" not in page.split("Join interview")[0][-200:]
    with freeze_time(start + timedelta(minutes=31)):
        candidate_client = login("cand@test.com")
        response = candidate_client.get(f"/candidate/interviews/{interview_id}/join", follow_redirects=True)
        assert b"isn&#39;t open for joining" in response.data


def test_calendar_download(app, booked, candidate_client):
    _, interview_id, _ = booked
    response = candidate_client.get(f"/candidate/interviews/{interview_id}/calendar.ics")
    assert response.mimetype == "text/calendar" and b"BEGIN:VCALENDAR" in response.data


def test_interview_pages_are_private(app, booked, make_user, login, hr_client):
    app_id, interview_id, slot_id = booked
    make_user("other@test.com")
    other = login("other@test.com")
    for path in (f"/candidate/interviews/{interview_id}", f"/candidate/interviews/{interview_id}/calendar.ics",
                 f"/candidate/interviews/{interview_id}/join", f"/candidate/applications/{app_id}/book"):
        assert other.get(path).status_code == 404
    assert other.post(f"/candidate/interviews/{interview_id}/cancel").status_code == 404
    assert other.get(f"/hr/jobs/1/slots").status_code == 403
    assert other.post(f"/hr/slots/{slot_id}/cancel").status_code == 403
    assert hr_client.get(f"/candidate/interviews/{interview_id}").status_code == 403


def test_hr_views(app, booked, hr_client, job_id):
    _, interview_id, _ = booked
    slots_page = hr_client.get(f"/hr/jobs/{job_id}/slots").get_data(as_text=True)
    assert "Arjun" in slots_page and "1/1" in slots_page
    overview = hr_client.get("/hr/interviews").get_data(as_text=True)
    assert "Arjun" in overview and "Scheduled" in overview
    assert "Arjun" not in hr_client.get("/hr/interviews?view=cancelled").get_data(as_text=True)
    assert "Scheduled" in hr_client.get(f"/hr/jobs/{job_id}").get_data(as_text=True)


# ---------- calendar file details ----------

def test_ics_escapes_and_folds(app, booked):
    _, interview_id, _ = booked
    with app.app_context():
        interview = db.session.get(Interview, interview_id)
        interview.application.job.title = "Dev, Ops; and\nmore " + "x" * 120
        data = interview_ics(interview, "https://example.com/i/1").decode()
        db.session.rollback()
    assert "Dev\\, Ops\\; and\\nmore" in data.replace("\r\n ", "")
    assert all(len(line.encode()) <= 75 for line in data.split("\r\n"))
    assert data.endswith("\r\n")


def test_housekeeping_task_runs(app):
    from app.tasks import interview_housekeeping

    with app.app_context():
        assert interview_housekeeping.call_local() == {"reminders": 0, "no_shows": 0, "timed_out": 0, "jd_questions": 0,
                                                       "resume_questions": 0}
