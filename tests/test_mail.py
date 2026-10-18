import email
from email import policy

import pytest

from app.models import User
from app.services import mail


@pytest.fixture
def recipient():
    return User(name="Asha & Co", email="asha@test.com", role=User.ROLE_CANDIDATE)


def test_console_backend_saves_eml(app, tmp_path, recipient):
    app.config.update(MAIL_BACKEND="console", MAIL_OUTBOX_DIR=tmp_path)
    try:
        with app.app_context():
            mail.send_email("asha@test.com", "Reset your password", "reset_password", user=recipient,
                            link="http://localhost/reset-password/abc", minutes=60)
    finally:
        app.config["MAIL_BACKEND"] = "memory"
    files = list(tmp_path.glob("*.eml"))
    assert len(files) == 1
    msg = email.message_from_bytes(files[0].read_bytes(), policy=policy.default)
    assert msg["To"] == "asha@test.com" and msg["Subject"] == "Reset your password" and msg["Message-ID"]
    text = msg.get_body(("plain",)).get_content()
    html = msg.get_body(("html",)).get_content()
    assert "Hi Asha & Co," in text  # plain text is not HTML-escaped
    assert "Asha &amp; Co" in html  # HTML is escaped
    assert "http://localhost/reset-password/abc" in text and "http://localhost/reset-password/abc" in html


def test_unknown_backend_raises(app):
    app.config["MAIL_BACKEND"] = "carrier-pigeon"
    try:
        with app.app_context(), pytest.raises(RuntimeError, match="Unknown MAIL_BACKEND"):
            mail.deliver({"to": "a@test.com"})
    finally:
        app.config["MAIL_BACKEND"] = "memory"


def test_external_url_uses_base_url_and_works_without_request(app):
    with app.app_context():
        assert mail.external_url("auth.reset_password", token="tok") == "http://localhost/reset-password/tok"
        app.config["APP_BASE_URL"] = "https://interviews.example.com/portal"
        try:
            assert mail.external_url("auth.login", next="/x") == "https://interviews.example.com/portal/login?next=/x"
        finally:
            app.config["APP_BASE_URL"] = "http://localhost"


def test_every_email_template_renders(app, recipient):
    inviter = User(name="Root", email="root@test.com", role=User.ROLE_ADMIN)
    cases = {
        "verify_email": dict(link="L", hours=48), "reset_password": dict(link="L", minutes=60),
        "invite": dict(link="L", days=7, inviter=inviter), "change_email": dict(link="L", hours=48, new_email="n@test.com"),
        "email_change_notice": dict(link="L", new_email="n@test.com"), "password_changed": dict(link="L"),
    }
    with app.app_context():
        for template, ctx in cases.items():
            msg = mail.build_message("asha@test.com", "Subject", template, user=recipient, **ctx)
            assert "L" in msg["text"] and msg["html"].lstrip().startswith("<!doctype html>"), template
