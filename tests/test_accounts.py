import io
import re
import zipfile
from datetime import timedelta

import pytest
from freezegun import freeze_time

from app.extensions import db
from app.models import Application, AuditLog, Job, User, utcnow
from app.services.security import make_token, password_problems
from tests.conftest import PASSWORD

NEW_PASSWORD = "Brand-New-Secret-7"


def link_in(message):
    match = re.search(r"http://localhost(/\S+)", message["text"])
    assert match, f"no link in email: {message['text']}"
    return match.group(1)


def user(app, email):
    with app.app_context():
        u = User.query.filter_by(email=email).one()
        db.session.expunge(u)
        return u


def actions(app):
    with app.app_context():
        return [a.action for a in AuditLog.query.order_by(AuditLog.id)]


def register(client, email="new@test.com", **extra):
    data = {"name": "New Person", "email": email, "password": PASSWORD, "confirm": PASSWORD, "consent": "y", **extra}
    return client.post("/register", data=data, follow_redirects=True)


# ---------- registration and email verification ----------

def test_registration_requires_consent(app, outbox):
    response = register(app.test_client(), consent="")
    assert b"must accept the Privacy Policy" in response.data
    with app.app_context():
        assert User.query.count() == 0
    assert outbox == []


def test_registration_sends_verification_email_and_link_verifies(app, outbox):
    client = app.test_client()
    response = register(client)
    assert b"Please verify your email address" in response.data  # banner
    assert len(outbox) == 1 and outbox[0]["to"] == "new@test.com"
    assert "Confirm your email" in outbox[0]["subject"] and outbox[0]["html"].startswith("<!doctype html>")
    response = client.get(link_in(outbox[0]), follow_redirects=True)
    assert b"Your email address is verified" in response.data
    assert user(app, "new@test.com").is_verified
    assert client.get(link_in(outbox[0]), follow_redirects=True).data.count(b"already verified") == 1
    assert {"account.registered", "email.verified"} <= set(actions(app))


def test_unverified_candidate_cannot_apply(app, outbox, fake_gemini):
    client = app.test_client()
    register(client)
    app.test_cli_runner().invoke(args=["seed-jobs"])
    with app.app_context():
        job_id = Job.query.first().id
    response = client.get(f"/candidate/jobs/{job_id}/apply", follow_redirects=True)
    assert b"verify your email address before applying" in response.data
    response = client.post(f"/candidate/jobs/{job_id}/apply", data={"resume": (io.BytesIO(b"x"), "cv.pdf")},
                           content_type="multipart/form-data", follow_redirects=True)
    assert fake_gemini.parse_calls == 0
    with app.app_context():
        assert Application.query.count() == 0


def test_resend_verification_is_rate_limited(app, outbox):
    client = app.test_client()
    register(client)
    outbox.clear()
    codes = [client.post("/verify-email/resend").status_code for _ in range(4)]
    assert codes == [302, 302, 302, 429]
    assert len(outbox) == 3


def test_verification_link_tampered_or_expired(app, outbox):
    client = app.test_client()
    register(client)
    link = link_in(outbox[0])
    assert b"link is invalid" in client.get(link[:-3] + "xyz", follow_redirects=True).data
    with freeze_time(utcnow() + timedelta(hours=49)):
        assert b"link has expired" in client.get(link, follow_redirects=True).data
    assert not user(app, "new@test.com").is_verified


# ---------- login protection ----------

def test_lockout_after_repeated_failures(app, make_user, outbox):
    make_user("a@test.com")
    client = app.test_client()
    for _ in range(4):
        assert b"Invalid email or password" in client.post("/login", data={"email": "a@test.com", "password": "nope-nope-1"}).data
    response = client.post("/login", data={"email": "a@test.com", "password": "nope-nope-1"})
    assert b"locked for 15 minutes" in response.data
    # even the right password is refused while locked
    assert b"is locked for" in client.post("/login", data={"email": "a@test.com", "password": PASSWORD}).data
    with freeze_time(utcnow() + timedelta(minutes=16)):
        assert client.post("/login", data={"email": "a@test.com", "password": PASSWORD}).status_code == 302
    u = user(app, "a@test.com")
    assert u.failed_login_count == 0 and not u.is_locked and u.last_login_at is not None
    assert "account.locked" in actions(app) and "login.blocked" in actions(app)


def test_successful_login_resets_failure_count(app, make_user):
    make_user("a@test.com")
    client = app.test_client()
    for _ in range(3):
        client.post("/login", data={"email": "a@test.com", "password": "nope-nope-1"})
    client.post("/login", data={"email": "a@test.com", "password": PASSWORD})
    assert user(app, "a@test.com").failed_login_count == 0


def test_deactivated_user_cannot_log_in_and_is_logged_out(app, make_user, login):
    uid = make_user("a@test.com")
    client = login("a@test.com")
    with app.app_context():
        u = db.session.get(User, uid)
        u.active = False
        db.session.commit()
    assert client.get("/candidate/").status_code == 302  # existing session rejected
    response = app.test_client().post("/login", data={"email": "a@test.com", "password": PASSWORD})
    assert b"has been deactivated" in response.data
    # wrong password on a deactivated account gives the generic message (no information leak)
    response = app.test_client().post("/login", data={"email": "a@test.com", "password": "nope-nope-1"})
    assert b"Invalid email or password" in response.data


def test_unknown_email_login_is_audited(app):
    app.test_client().post("/login", data={"email": "ghost@test.com", "password": PASSWORD})
    with app.app_context():
        entry = AuditLog.query.one()
        assert entry.action == "login.failed" and entry.target_email == "ghost@test.com" and entry.ip_address


def test_session_from_old_cookie_format_is_rejected(app, make_user):
    uid = make_user("a@test.com")
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["_user_id"] = str(uid)  # pre-Phase-2 format without session token
        sess["_fresh"] = True
    assert client.get("/candidate/").status_code == 302


# ---------- password reset ----------

def test_forgot_password_does_not_reveal_accounts(app, make_user, outbox):
    make_user("a@test.com")
    client = app.test_client()
    known = client.post("/forgot-password", data={"email": "a@test.com"}, follow_redirects=True)
    unknown = client.post("/forgot-password", data={"email": "nobody@test.com"}, follow_redirects=True)
    assert b"If an account exists" in known.data and b"If an account exists" in unknown.data
    assert [m["to"] for m in outbox] == ["a@test.com"]


def test_reset_password_full_flow(app, make_user, login, outbox):
    make_user("a@test.com")
    other_device = login("a@test.com")
    client = app.test_client()
    client.post("/forgot-password", data={"email": "a@test.com"})
    link = link_in(outbox[0])
    assert b"Set a new password" in client.get(link).data
    response = client.post(link, data={"password": NEW_PASSWORD, "confirm": NEW_PASSWORD}, follow_redirects=True)
    assert b"Your password has been set" in response.data
    assert other_device.get("/candidate/").status_code == 302  # signed out everywhere
    assert login("a@test.com", NEW_PASSWORD)
    assert outbox[-1]["subject"] == "Your password was changed"
    # link is single-use
    assert b"already been used" in client.get(link, follow_redirects=True).data
    assert "password.reset" in actions(app)


def test_reset_password_unlocks_and_verifies(app, make_user, outbox):
    uid = make_user("a@test.com", verified=False)
    with app.app_context():
        u = db.session.get(User, uid)
        u.locked_until = utcnow() + timedelta(minutes=10)
        db.session.commit()
    client = app.test_client()
    client.post("/forgot-password", data={"email": "a@test.com"})
    client.post(link_in(outbox[0]), data={"password": NEW_PASSWORD, "confirm": NEW_PASSWORD})
    u = user(app, "a@test.com")
    assert not u.is_locked and u.is_verified


def test_reset_link_expires(app, make_user, outbox):
    make_user("a@test.com")
    client = app.test_client()
    client.post("/forgot-password", data={"email": "a@test.com"})
    with freeze_time(utcnow() + timedelta(minutes=61)):
        assert b"link has expired" in client.get(link_in(outbox[0]), follow_redirects=True).data


def test_reset_rejects_weak_password(app, make_user, outbox):
    make_user("asha.k@test.com", name="Asha Kumar")
    client = app.test_client()
    client.post("/forgot-password", data={"email": "asha.k@test.com"})
    link = link_in(outbox[0])
    assert b"too common" in client.post(link, data={"password": "password123", "confirm": "password123"}).data
    assert b"must not contain your name" in client.post(link, data={"password": "Kumar-is-great1", "confirm": "Kumar-is-great1"}).data


def test_deactivated_account_gets_no_reset_email(app, make_user, outbox):
    make_user("a@test.com", active=False)
    app.test_client().post("/forgot-password", data={"email": "a@test.com"})
    assert outbox == []


def test_token_purposes_are_not_interchangeable(app, make_user):
    uid = make_user("a@test.com", verified=False)
    with app.app_context():
        reset_token = make_token(db.session.get(User, uid), "reset-password")
    response = app.test_client().get(f"/verify-email/{reset_token}", follow_redirects=True)
    assert b"link is invalid" in response.data
    assert not user(app, "a@test.com").is_verified


# ---------- consent ----------

def test_candidate_without_current_consent_is_asked_again(app, make_user, login):
    make_user("a@test.com", consent=False)
    client = login("a@test.com")
    response = client.get("/candidate/jobs")
    assert response.status_code == 302 and "/consent" in response.headers["Location"]
    assert client.get("/privacy").status_code == 200  # policy readable without consenting
    assert client.get("/account/").status_code == 200  # can still manage / delete data
    assert b"Please accept" in client.post("/consent", data={}).data
    response = client.post("/consent?next=/candidate/jobs", data={"consent": "y"})
    assert response.headers["Location"].endswith("/candidate/jobs")
    assert client.get("/candidate/jobs").status_code == 200
    assert user(app, "a@test.com").consent_version == app.config["PRIVACY_POLICY_VERSION"]


def test_policy_version_bump_requires_reconsent(app, make_user, login):
    make_user("a@test.com")
    client = login("a@test.com")
    assert client.get("/candidate/").status_code == 200
    original = app.config["PRIVACY_POLICY_VERSION"]
    app.config["PRIVACY_POLICY_VERSION"] = "2099-01-01"
    try:
        response = client.get("/candidate/")
        assert response.status_code == 302 and "/consent" in response.headers["Location"]
        assert b"has been updated" in client.get("/consent").data
    finally:
        app.config["PRIVACY_POLICY_VERSION"] = original


def test_staff_are_not_asked_for_candidate_consent(hr_client):
    assert hr_client.get("/hr/").status_code == 200


# ---------- password policy ----------

@pytest.mark.parametrize("password, fragment", [
    ("short1", "at least 8"),
    ("x" * 129, "at most 128"),
    ("Password123", "too common"),
    ("aaaaaaaaaa", "too repetitive"),
    ("asha.k-rocks-99", "email address"),
    ("Kumar-forever-1", "your name"),
])
def test_password_policy(password, fragment):
    assert any(fragment in p for p in password_problems(password, email="asha.k@test.com", name="Asha Kumar"))


def test_password_policy_accepts_good_password():
    assert password_problems("Correct-Horse-42", email="asha.k@test.com", name="Asha Kumar") == []


# ---------- account settings ----------

def test_update_profile(app, candidate_client):
    response = candidate_client.post("/account/profile", data={"profile-name": "  Arjun M  "}, follow_redirects=True)
    assert b"Your name has been updated" in response.data
    assert user(app, "cand@test.com").name == "Arjun M"
    assert b"This field is required" in candidate_client.post("/account/profile", data={"profile-name": "   "}).data
    assert user(app, "cand@test.com").name == "Arjun M"


def test_change_password(app, candidate_client, login, outbox):
    other = login("cand@test.com")
    data = {"password-current_password": "wrong-one-1", "password-password": NEW_PASSWORD, "password-confirm": NEW_PASSWORD}
    assert b"Incorrect password" in candidate_client.post("/account/password", data=data).data
    data["password-current_password"] = PASSWORD
    data["password-password"] = data["password-confirm"] = PASSWORD
    assert b"must be different" in candidate_client.post("/account/password", data=data).data
    data["password-password"] = data["password-confirm"] = NEW_PASSWORD
    response = candidate_client.post("/account/password", data=data, follow_redirects=True)
    assert b"Your password has been changed" in response.data
    assert candidate_client.get("/candidate/").status_code == 200  # this browser stays signed in
    assert other.get("/candidate/").status_code == 302  # other devices signed out
    assert outbox[-1]["subject"] == "Your password was changed"


def test_change_email_flow(app, candidate_client, outbox):
    bad = candidate_client.post("/account/email", data={"email-new_email": "fresh@test.com", "email-current_password": "nope-nope-1"})
    assert b"Incorrect password" in bad.data
    response = candidate_client.post("/account/email", data={"email-new_email": "Fresh@Test.com", "email-current_password": PASSWORD},
                                     follow_redirects=True)
    assert b"confirmation link to fresh@test.com" in response.data
    assert {m["to"] for m in outbox} == {"fresh@test.com", "cand@test.com"}  # new address + notice to old
    confirm = next(m for m in outbox if m["to"] == "fresh@test.com")
    assert user(app, "cand@test.com").pending_email == "fresh@test.com"
    response = candidate_client.get(link_in(confirm), follow_redirects=True)
    assert b"Your email address is now fresh@test.com" in response.data
    u = user(app, "fresh@test.com")
    assert u.pending_email is None and u.is_verified
    assert candidate_client.get("/candidate/").status_code == 200  # still signed in here


def test_change_email_to_taken_address(app, candidate_client, make_user):
    make_user("taken@test.com")
    response = candidate_client.post("/account/email", data={"email-new_email": "taken@test.com", "email-current_password": PASSWORD})
    assert b"already exists" in response.data


def test_cancel_email_change_invalidates_link(app, candidate_client, outbox):
    candidate_client.post("/account/email", data={"email-new_email": "fresh@test.com", "email-current_password": PASSWORD})
    confirm = next(m for m in outbox if m["to"] == "fresh@test.com")
    candidate_client.post("/account/email/cancel")
    candidate_client.get(link_in(confirm), follow_redirects=True)
    assert user(app, "cand@test.com").email == "cand@test.com"


def test_data_export(app, candidate_client, fake_gemini):
    from tests.test_jobs_and_applications import BACKEND_RESUME

    app.test_cli_runner().invoke(args=["seed-jobs"])
    with app.app_context():
        job_id = Job.query.filter_by(title="Python Backend Developer").one().id
    candidate_client.post(f"/candidate/jobs/{job_id}/apply", data={"resume": (io.BytesIO(BACKEND_RESUME), "cv.docx")},
                          content_type="multipart/form-data")
    response = candidate_client.post("/account/export")
    assert response.status_code == 200 and response.mimetype == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.data)) as zf:
        names = zf.namelist()
        data = __import__("json").loads(zf.read("my_data.json"))
    response.close()
    assert "my_data.json" in names and any(n.startswith("resumes/") for n in names)
    assert data["account"]["email"] == "cand@test.com"
    assert data["applications"][0]["job_title"] == "Python Backend Developer"
    assert "data.exported" in actions(app)


def test_delete_account_removes_everything(app, candidate_client, fake_gemini):
    from tests.test_jobs_and_applications import BACKEND_RESUME

    app.test_cli_runner().invoke(args=["seed-jobs"])
    with app.app_context():
        job_id = Job.query.filter_by(title="Python Backend Developer").one().id
    candidate_client.post(f"/candidate/jobs/{job_id}/apply", data={"resume": (io.BytesIO(BACKEND_RESUME), "cv.docx")},
                          content_type="multipart/form-data")
    assert len(list(app.config["UPLOAD_FOLDER"].iterdir())) == 1

    bad = candidate_client.post("/account/delete", data={"delete-current_password": PASSWORD, "delete-confirm_text": "delete"})
    assert b"type DELETE" in bad.data
    response = candidate_client.post("/account/delete", data={"delete-current_password": PASSWORD, "delete-confirm_text": "DELETE"},
                                     follow_redirects=True)
    assert b"permanently deleted" in response.data
    with app.app_context():
        assert User.query.filter_by(email="cand@test.com").first() is None
        assert Application.query.count() == 0
        entry = AuditLog.query.filter_by(action="account.deleted").one()
        assert entry.target_email == "cand@test.com" and entry.target_user_id is None and entry.actor_email == "cand@test.com"
    assert list(app.config["UPLOAD_FOLDER"].iterdir()) == []
    assert candidate_client.get("/candidate/").status_code == 302


def test_last_admin_cannot_self_delete(app, make_user, login):
    make_user("root@test.com", role=User.ROLE_ADMIN)
    client = login("root@test.com")
    assert b"only active admin" in client.get("/account/").data
    response = client.post("/account/delete", data={"delete-current_password": PASSWORD, "delete-confirm_text": "DELETE"},
                           follow_redirects=True)
    assert b"only active admin" in response.data
    assert user(app, "root@test.com")


def test_account_pages_require_login(app):
    client = app.test_client()
    for path in ("/account/", "/account/export", "/account/delete"):
        response = client.get(path) if path == "/account/" else client.post(path)
        assert response.status_code == 302 and "/login" in response.headers["Location"]
