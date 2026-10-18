"""Outgoing email.

send_email() renders the text + HTML templates in the request (or job) that wants to send, then hands the
finished message to a background job, so a slow or failing mail server never blocks a page.

Backends (MAIL_BACKEND):
  console  - development: logs the message and saves it as an .eml file in instance/outbox/
  memory   - tests: keeps messages in app.extensions["mail_outbox"]
  (Gmail SMTP / Gmail API backends are added in Phase 3)
"""
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from urllib.parse import urlsplit

from flask import current_app, render_template


def external_url(endpoint, **values):
    """Absolute link for emails, based on APP_BASE_URL. Built straight from the URL map, so it works the same
    inside a request and inside a background job (no request context needed)."""
    base = urlsplit(current_app.config["APP_BASE_URL"])
    adapter = current_app.url_map.bind(base.netloc, script_name=base.path or "/", url_scheme=base.scheme)
    return adapter.build(endpoint, values, force_external=True)


def build_message(to, subject, template, **context):
    context.setdefault("app_name", current_app.config["APP_NAME"])
    context.setdefault("base_url", current_app.config["APP_BASE_URL"])
    text = render_template(f"email/{template}.txt", **context)
    html = render_template(f"email/{template}.html", subject=subject, **context)
    sender = current_app.config["MAIL_USERNAME"] or "no-reply@localhost"
    return {
        "to": to,
        "from": formataddr((current_app.config["MAIL_SENDER_NAME"], sender)),
        "subject": subject,
        "text": text,
        "html": html,
        "message_id": make_msgid(domain=sender.split("@")[-1] or "localhost"),
    }


def send_email(to, subject, template, **context):
    """Queue an email. Returns the queued message dict."""
    from ..tasks import deliver_email

    message = build_message(to, subject, template, **context)
    deliver_email(message)
    return message


def to_mime(message):
    mime = EmailMessage()
    mime["From"] = message["from"]
    mime["To"] = message["to"]
    mime["Subject"] = message["subject"]
    mime["Message-ID"] = message["message_id"]
    mime["Date"] = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    mime.set_content(message["text"])
    mime.add_alternative(message["html"], subtype="html")
    return mime


def deliver(message):
    """Actually send a message with the configured backend (called by the background job)."""
    backend = current_app.config["MAIL_BACKEND"]
    if backend == "memory":
        current_app.extensions.setdefault("mail_outbox", []).append(message)
    elif backend == "console":
        outbox = current_app.config["MAIL_OUTBOX_DIR"]
        outbox.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = outbox / f"{stamp}.eml"
        path.write_bytes(bytes(to_mime(message)))
        current_app.logger.info("EMAIL (console backend) to=%s subject=%r saved=%s\n%s",
                                message["to"], message["subject"], path, message["text"])
    else:
        raise RuntimeError(f"Unknown MAIL_BACKEND {backend!r}")
