import io
import smtplib
from datetime import timedelta
from email import message_from_bytes, policy
from types import SimpleNamespace

import pytest

from app.extensions import db
from app.models import Application, EmailLog, Job, Notification, User, utcnow
from app.services import mail
from app.tasks import deliver_email
from tests.conftest import PASSWORD
from tests.test_jobs_and_applications import BACKEND_RESUME, FRONTEND_RESUME


def logs(app):
    with app.app_context():
        return [(e.to_address, e.template, e.status, e.attempts) for e in EmailLog.query.order_by(EmailLog.id)]


def notes(app, email):
    with app.app_context():
        u = User.query.filter_by(email=email).one()
        return [(n.kind, n.title, n.is_read) for n in u.notifications.order_by(Notification.id)]


@pytest.fixture
def jobs(app):
    app.test_cli_runner().invoke(args=["seed-jobs"])
    with app.app_context():
        return {j.title: j.id for j in Job.query.all()}


def apply(client, job_id, content=BACKEND_RESUME):
    return client.post(f"/candidate/jobs/{job_id}/apply", data={"resume": (io.BytesIO(content), "cv.docx")},
                       content_type="multipart/form-data", follow_redirects=True)


# ---------- email log and delivery job ----------

def test_every_email_is_logged_and_marked_sent(app, make_user, outbox):
    make_user("a@test.com")
    app.test_client().post("/forgot-password", data={"email": "a@test.com"})
    assert logs(app) == [("a@test.com", "reset_password", "sent", 1)]
    with app.app_context():
        entry = EmailLog.query.one()
        assert entry.user_id is not None and entry.sent_at is not None and entry.message_id == outbox[0]["message_id"]


def message_for(app, to="a@test.com"):
    with app.app_context():
        log = EmailLog(to_address=to, subject="S", template="t", backend="memory")
        db.session.add(log)
        db.session.commit()
        return {"to": to, "from": "x", "subject": "S", "text": "t", "html": "<p>t</p>", "message_id": "<m@x>",
                "log_id": log.id}


def run_job(app, message, retries_left):
    with app.app_context():
        return deliver_email.func(message, task=SimpleNamespace(retries=retries_left))


def test_permanent_failure_is_not_retried(app, monkeypatch):
    msg = message_for(app)
    monkeypatch.setattr(mail, "deliver", lambda m: (_ for _ in ()).throw(mail.PermanentMailError("bad password")))
    assert run_job(app, msg, retries_left=3) == "failed"
    assert logs(app) == [("a@test.com", "t", "failed", 1)]


def test_temporary_failure_retries_then_fails(app, monkeypatch):
    msg = message_for(app)
    monkeypatch.setattr(mail, "deliver", lambda m: (_ for _ in ()).throw(mail.TemporaryMailError("network down")))
    with pytest.raises(mail.TemporaryMailError):
        run_job(app, msg, retries_left=2)  # Huey sees the exception and schedules a retry
    assert logs(app)[0][2:] == ("retrying", 1)
    assert run_job(app, msg, retries_left=0) == "failed"
    assert logs(app)[0][2:] == ("failed", 2)
    with app.app_context():
        assert "network down" in EmailLog.query.one().last_error


def test_daily_limit_defers_mail(app, monkeypatch):
    app.config.update(MAIL_BACKEND="gmail_smtp", MAIL_DAILY_LIMIT=2)
    scheduled = []
    monkeypatch.setattr(deliver_email, "schedule", lambda args, delay: scheduled.append((args, delay)))
    monkeypatch.setattr(mail, "deliver", lambda m: pytest.fail("must not send over the limit"))
    try:
        with app.app_context():
            for _ in range(2):
                db.session.add(EmailLog(to_address="x@test.com", subject="s", template="t", status="sent", sent_at=utcnow()))
            db.session.add(EmailLog(to_address="old@test.com", subject="s", template="t", status="sent",
                                    sent_at=utcnow() - timedelta(hours=25)))  # outside the 24h window
            db.session.commit()
        msg = message_for(app)
        assert run_job(app, msg, retries_left=3) == "deferred"
    finally:
        app.config.update(MAIL_BACKEND="memory", MAIL_DAILY_LIMIT=450)
    assert scheduled and scheduled[0][1] == app.config["MAIL_DEFER_SECONDS"]
    with app.app_context():
        assert db.session.get(EmailLog, msg["log_id"]).status == "deferred"


# ---------- Gmail SMTP backend (smtplib faked; nothing is really sent) ----------

class FakeSMTP:
    instances = []
    fail_with = None

    def __init__(self, host, port, timeout, context):
        self.host, self.port, self.timeout, self.sent, self.login_args = host, port, timeout, [], None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        if isinstance(FakeSMTP.fail_with, OSError):
            raise FakeSMTP.fail_with
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        self.login_args = (user, password)
        if isinstance(FakeSMTP.fail_with, smtplib.SMTPAuthenticationError):
            raise FakeSMTP.fail_with

    def send_message(self, msg):
        if isinstance(FakeSMTP.fail_with, smtplib.SMTPResponseException) and not isinstance(
                FakeSMTP.fail_with, smtplib.SMTPAuthenticationError):
            raise FakeSMTP.fail_with
        self.sent.append(msg)
        return {}


@pytest.fixture
def gmail(app, monkeypatch):
    FakeSMTP.instances, FakeSMTP.fail_with = [], None
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    app.config.update(MAIL_BACKEND="gmail_smtp", MAIL_USERNAME="virtualinterviewer.test@gmail.com",
                      MAIL_APP_PASSWORD="abcd efgh ijkl mnop")
    yield FakeSMTP
    app.config.update(MAIL_BACKEND="memory", MAIL_USERNAME="", MAIL_APP_PASSWORD="")


def test_gmail_smtp_sends_with_correct_headers(app, gmail):
    with app.app_context():
        msg = mail.build_message("cand@test.com", "Hello", "test_email", reply_to="hr@company.com",
                                 sender="s", backend="gmail_smtp")
        mail.deliver(msg)
    smtp = gmail.instances[0]
    assert (smtp.host, smtp.port) == ("smtp.gmail.com", 465)
    assert smtp.login_args == ("virtualinterviewer.test@gmail.com", "abcdefghijklmnop")  # spaces removed
    sent = smtp.sent[0]
    assert sent["From"] == "Virtual Interviewer <virtualinterviewer.test@gmail.com>"
    assert sent["To"] == "cand@test.com" and sent["Reply-To"] == "hr@company.com"
    assert sent["Auto-Submitted"] == "auto-generated"
    assert sent.get_body(("html",)) is not None and sent.get_body(("plain",)) is not None


@pytest.mark.parametrize("error, kind", [
    (smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted"), mail.PermanentMailError),
    (smtplib.SMTPResponseException(421, b"4.7.0 Try again later"), mail.TemporaryMailError),
    (smtplib.SMTPResponseException(550, b"5.1.1 No such user"), mail.PermanentMailError),
    (ConnectionRefusedError("refused"), mail.TemporaryMailError),
])
def test_gmail_smtp_error_classification(app, gmail, error, kind):
    gmail.fail_with = error
    with app.app_context(), pytest.raises(kind):
        mail.deliver(mail.build_message("a@test.com", "S", "test_email", sender="s", backend="b"))


def test_gmail_smtp_without_credentials_fails_permanently(app, gmail):
    app.config.update(MAIL_APP_PASSWORD="")
    with app.app_context(), pytest.raises(mail.PermanentMailError, match="not set"):
        mail.deliver(mail.build_message("a@test.com", "S", "test_email", sender="s", backend="b"))


def test_mail_test_command(app, gmail):
    result = app.test_cli_runner().invoke(args=["mail-test", "--to", "me@test.com"])
    assert result.exit_code == 0 and "Test email sent to me@test.com via gmail_smtp" in result.output
    assert gmail.instances[0].sent[0]["Subject"] == "Test email from Virtual Interviewer"
    gmail.fail_with = smtplib.SMTPAuthenticationError(535, b"bad")
    result = app.test_cli_runner().invoke(args=["mail-test", "--to", "me@test.com"])
    assert result.exit_code != 0 and "Gmail rejected the login" in result.output
    assert [s for (_, _, s, _) in logs(app)] == ["sent", "failed"]


def test_invitation_reply_to_is_the_inviting_admin(app, make_user, login, outbox):
    make_user("root@company.com", role=User.ROLE_ADMIN, name="Root")
    login("root@company.com").post("/admin/users/invite", data={"name": "Hira", "email": "hira@test.com", "role": "hr"})
    assert outbox[0]["reply_to"] == "root@company.com"


# ---------- application notifications ----------

def test_apply_creates_notification_and_email(app, candidate_client, jobs, fake_gemini, outbox):
    apply(candidate_client, jobs["Python Backend Developer"])
    assert notes(app, "cand@test.com") == [("application.received", "Application received: Python Backend Developer", False)]
    assert outbox[-1]["subject"] == "Application received: Python Backend Developer"
    assert "shortlisted" in outbox[-1]["text"] and "account settings" in outbox[-1]["html"]


def test_status_changes_notify_candidate(app, hr_client, candidate_client, jobs, fake_gemini, outbox):
    apply(candidate_client, jobs["Frontend Developer"], content=BACKEND_RESUME)  # poor fit -> under review
    with app.app_context():
        app_id = Application.query.one().id
    outbox.clear()
    hr_client.post(f"/hr/applications/{app_id}/status", data={"status": "shortlisted"})
    assert outbox[-1]["subject"] == "Good news about your application for Frontend Developer"
    hr_client.post(f"/hr/applications/{app_id}/status", data={"status": "shortlisted"})  # unchanged: nothing new
    assert len(outbox) == 1
    hr_client.post(f"/hr/applications/{app_id}/status", data={"status": "under_review"})  # in-app only
    assert len(outbox) == 1
    hr_client.post(f"/hr/applications/{app_id}/status", data={"status": "rejected"})
    assert outbox[-1]["subject"] == "Update on your application for Frontend Developer"
    kinds = [k for k, _, _ in notes(app, "cand@test.com")]
    assert kinds == ["application.received", "application.status", "application.status", "application.status"]
    with app.app_context():
        from app.models import AuditLog
        assert AuditLog.query.filter_by(action="application.status_changed").count() == 3


def test_email_preferences_are_respected(app, candidate_client, hr_client, jobs, fake_gemini, outbox):
    response = candidate_client.post("/account/preferences", data={"prefs-email_interviews": "y"}, follow_redirects=True)
    assert b"email preferences have been saved" in response.data
    with app.app_context():
        u = User.query.filter_by(email="cand@test.com").one()
        assert not u.wants_email("applications") and u.wants_email("interviews") and u.wants_email("security")
    apply(candidate_client, jobs["Python Backend Developer"])
    assert outbox == []  # opted out of application emails...
    assert len(notes(app, "cand@test.com")) == 1  # ...but still notified in-app
    candidate_client.post("/forgot-password", data={"email": "cand@test.com"})
    assert outbox[-1]["subject"] == "Reset your password"  # security emails always go out


def test_preferences_form_shows_current_values(candidate_client):
    candidate_client.post("/account/preferences", data={})
    page = candidate_client.get("/account/").get_data(as_text=True)
    assert 'id="prefs-email_applications"' in page and 'id="prefs-email_applications" checked' not in page


# ---------- bell and notifications page ----------

def test_bell_shows_unread_count_and_opening_marks_read(app, candidate_client, jobs, fake_gemini):
    apply(candidate_client, jobs["Python Backend Developer"])
    page = candidate_client.get("/candidate/").get_data(as_text=True)
    assert "1 unread" in page and "Application received: Python Backend Developer" in page
    with app.app_context():
        note = Notification.query.one()
        note_id, link = note.id, note.link
    response = candidate_client.get(f"/notifications/{note_id}/open")
    assert response.status_code == 302 and response.headers["Location"].endswith(link)
    assert notes(app, "cand@test.com")[0][2] is True
    assert "unread" not in candidate_client.get("/candidate/").get_data(as_text=True).split("Notifications")[0][-50:]


def test_cannot_open_someone_elses_notification(app, candidate_client, make_user, login, jobs, fake_gemini):
    apply(candidate_client, jobs["Python Backend Developer"])
    with app.app_context():
        note_id = Notification.query.one().id
    make_user("other@test.com")
    assert login("other@test.com").get(f"/notifications/{note_id}/open").status_code == 404


def test_mark_all_read_and_pagination(app, candidate_client):
    with app.app_context():
        uid = User.query.filter_by(email="cand@test.com").one().id
        for i in range(25):
            db.session.add(Notification(user_id=uid, kind="test", title=f"Note {i}", body=""))
        db.session.commit()
    page1 = candidate_client.get("/notifications/").get_data(as_text=True)
    assert "Note 24" in page1 and "Note 0<" not in page1 and "Mark all as read" in page1
    assert "Note 0" in candidate_client.get("/notifications/?page=2").get_data(as_text=True)
    candidate_client.post("/notifications/read-all")
    with app.app_context():
        assert Notification.query.filter(Notification.read_at.is_(None)).count() == 0


def test_notification_links_must_be_local(app, make_user):
    from app.services.notifications import notify

    uid = make_user("a@test.com")
    with app.app_context(), pytest.raises(ValueError):
        notify(db.session.get(User, uid), "x", "Title", link="https://evil.example")


def test_deleting_user_removes_notifications_and_unlinks_email_logs(app, candidate_client, jobs, fake_gemini):
    apply(candidate_client, jobs["Python Backend Developer"])
    candidate_client.post("/account/delete", data={"delete-current_password": PASSWORD, "delete-confirm_text": "DELETE"})
    with app.app_context():
        assert Notification.query.count() == 0
        assert EmailLog.query.count() >= 1 and EmailLog.query.filter(EmailLog.user_id.isnot(None)).count() == 0


# ---------- admin email log ----------

def test_admin_email_log_page(app, make_user, login, outbox):
    make_user("root@test.com", role=User.ROLE_ADMIN)
    make_user("a@test.com")
    admin = login("root@test.com")
    app.test_client().post("/forgot-password", data={"email": "a@test.com"})
    with app.app_context():
        db.session.add(EmailLog(to_address="bounce@test.com", subject="X", template="t", status="failed",
                                last_error="SMTP 550: No such user"))
        db.session.commit()
    page = admin.get("/admin/emails").get_data(as_text=True)
    assert "a@test.com" in page and "Reset your password" in page and "SMTP 550" in page
    failed = admin.get("/admin/emails?status=failed").get_data(as_text=True)
    assert "bounce@test.com" in failed and "a@test.com" not in failed
    assert "bounce@test.com" not in admin.get("/admin/emails?to=a%40test").get_data(as_text=True)


def test_email_log_is_admin_only(hr_client, candidate_client):
    assert hr_client.get("/admin/emails").status_code == 403
    assert candidate_client.get("/admin/emails").status_code == 403


def test_dev_redirect_sends_everything_to_one_inbox(app, make_user, outbox):
    make_user("stranger@test.com")
    app.config["MAIL_REDIRECT_TO"] = "dev-inbox@gmail.com"
    try:
        app.test_client().post("/forgot-password", data={"email": "stranger@test.com"})
    finally:
        app.config["MAIL_REDIRECT_TO"] = ""
    assert outbox[-1]["to"] == "dev-inbox@gmail.com"
    assert outbox[-1]["subject"] == "[to stranger@test.com] Reset your password"
    assert logs(app)[-1][0] == "stranger@test.com"  # the log keeps the intended recipient


def test_production_rejects_mail_redirect(monkeypatch):
    from app.config import ProductionConfig

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    for key, value in dict(SECRET_KEY="x" * 64, REDIS_URL="redis://r", GEMINI_API_KEY="k", APP_BASE_URL="https://x.example",
                           MAIL_BACKEND="gmail_smtp", MAIL_USERNAME="a@gmail.com", MAIL_APP_PASSWORD="p",
                           MAIL_REDIRECT_TO="dev@gmail.com").items():
        monkeypatch.setattr(ProductionConfig, key, value)
    with pytest.raises(RuntimeError, match="MAIL_REDIRECT_TO"):
        ProductionConfig.validate()
