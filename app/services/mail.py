"""Outgoing email.

send_email() renders the text + HTML templates where the email is triggered, records it in EmailLog, then hands
the finished message to a background job, so a slow or failing mail server never blocks a page.

Backends (MAIL_BACKEND):
  gmail_smtp - production: Gmail over implicit-TLS SMTP with an app password
  console    - development: logs the message and saves it as an .eml file in instance/outbox/
  memory     - tests: keeps messages in app.extensions["mail_outbox"]
"""
import smtplib
import socket
import ssl
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import formataddr, format_datetime, make_msgid
from urllib.parse import urlsplit

from flask import current_app, render_template

from ..extensions import db


class PermanentMailError(Exception):
    """Retrying won't help (bad credentials, rejected recipient...)."""


class TemporaryMailError(Exception):
    """Worth retrying later (network trouble, server busy...)."""


def external_url(endpoint, **values):
    """Absolute link for emails, based on APP_BASE_URL. Built straight from the URL map, so it works the same
    inside a request and inside a background job (no request context needed)."""
    base = urlsplit(current_app.config["APP_BASE_URL"])
    adapter = current_app.url_map.bind(base.netloc, script_name=base.path or "/", url_scheme=base.scheme)
    return adapter.build(endpoint, values, force_external=True)


def _sender_address():
    return current_app.config["MAIL_USERNAME"] or "no-reply@localhost"


def build_message(to, subject, template, reply_to=None, **context):
    context.setdefault("app_name", current_app.config["APP_NAME"])
    context.setdefault("base_url", current_app.config["APP_BASE_URL"])
    text = render_template(f"email/{template}.txt", **context)
    html = render_template(f"email/{template}.html", subject=subject, **context)
    sender = _sender_address()
    return {
        "to": to,
        "from": formataddr((current_app.config["MAIL_SENDER_NAME"], sender)),
        "reply_to": reply_to,
        "subject": subject,
        "template": template,
        "text": text,
        "html": html,
        "message_id": make_msgid(domain=sender.split("@")[-1] or "localhost"),
    }


def send_email(to, subject, template, reply_to=None, **context):
    """Queue an email and record it in the email log. Returns the message dict.

    Call this after committing your own changes: it commits the log row so the worker can find it."""
    from ..models import EmailLog, User
    from ..tasks import deliver_email

    message = build_message(to, subject, template, reply_to=reply_to, **context)
    recipient = context.get("user")
    log = EmailLog(to_address=to, subject=subject, template=template, message_id=message["message_id"],
                   backend=current_app.config["MAIL_BACKEND"],
                   user_id=recipient.id if isinstance(recipient, User) and recipient.id else None)
    db.session.add(log)
    db.session.commit()
    message["log_id"] = log.id
    deliver_email(message)
    return message


def to_mime(message):
    mime = EmailMessage()
    mime["From"] = message["from"]
    mime["To"] = message["to"]
    if message.get("reply_to"):
        mime["Reply-To"] = message["reply_to"]
    mime["Subject"] = message["subject"]
    mime["Message-ID"] = message["message_id"]
    mime["Date"] = format_datetime(datetime.now(timezone.utc))
    mime["Auto-Submitted"] = "auto-generated"  # tells mail servers not to send out-of-office replies
    mime.set_content(message["text"])
    mime.add_alternative(message["html"], subtype="html")
    return mime


# ---------- backends ----------

def _send_memory(message):
    current_app.extensions.setdefault("mail_outbox", []).append(message)


def _send_console(message):
    outbox = current_app.config["MAIL_OUTBOX_DIR"]
    outbox.mkdir(parents=True, exist_ok=True)
    path = outbox / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.eml"
    path.write_bytes(bytes(to_mime(message)))
    current_app.logger.info("EMAIL (console backend) to=%s subject=%r saved=%s\n%s",
                            message["to"], message["subject"], path, message["text"])


def _send_gmail_smtp(message):
    cfg = current_app.config
    if not (cfg["MAIL_USERNAME"] and cfg["MAIL_APP_PASSWORD"]):
        raise PermanentMailError("MAIL_USERNAME / MAIL_APP_PASSWORD are not set")
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(cfg["MAIL_SMTP_HOST"], cfg["MAIL_SMTP_PORT"], timeout=cfg["MAIL_SMTP_TIMEOUT"],
                              context=context) as smtp:
            smtp.login(cfg["MAIL_USERNAME"], cfg["MAIL_APP_PASSWORD"].replace(" ", ""))
            refused = smtp.send_message(to_mime(message))
            if refused:
                raise PermanentMailError(f"Recipient refused: {refused}")
    except smtplib.SMTPAuthenticationError as e:
        raise PermanentMailError(f"Gmail rejected the login (check MAIL_USERNAME / MAIL_APP_PASSWORD): {e.smtp_code} "
                                 f"{_decode(e.smtp_error)}") from e
    except smtplib.SMTPRecipientsRefused as e:
        raise PermanentMailError(f"Recipient refused: {e.recipients}") from e
    except smtplib.SMTPResponseException as e:
        # 4xx = temporary (greylisting, busy); 5xx = permanent (policy, invalid address)
        error = f"SMTP {e.smtp_code}: {_decode(e.smtp_error)}"
        raise (TemporaryMailError(error) if 400 <= e.smtp_code < 500 else PermanentMailError(error)) from e
    except (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError, socket.timeout, OSError) as e:
        raise TemporaryMailError(f"Could not reach {cfg['MAIL_SMTP_HOST']}: {e}") from e


def _decode(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


BACKENDS = {"memory": _send_memory, "console": _send_console, "gmail_smtp": _send_gmail_smtp}


def deliver(message):
    """Send one message with the configured backend. Raises PermanentMailError / TemporaryMailError."""
    backend = current_app.config["MAIL_BACKEND"]
    if backend not in BACKENDS:
        raise RuntimeError(f"Unknown MAIL_BACKEND {backend!r}")
    redirect_to = current_app.config.get("MAIL_REDIRECT_TO")
    if redirect_to and message["to"].lower() != redirect_to.lower():
        message = {**message, "to": redirect_to, "subject": f"[to {message['to']}] {message['subject']}"}
    BACKENDS[backend](message)


def sent_in_last_24h():
    from ..models import EmailLog

    since = datetime.now(timezone.utc) - timedelta(hours=24)
    return EmailLog.query.filter(EmailLog.status == EmailLog.STATUS_SENT, EmailLog.sent_at >= since).count()


def over_daily_limit():
    """Only real sending counts against Gmail's daily quota."""
    if current_app.config["MAIL_BACKEND"] != "gmail_smtp":
        return False
    return sent_in_last_24h() >= current_app.config["MAIL_DAILY_LIMIT"]
