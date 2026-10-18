"""Signed one-time tokens, password policy and the audit log."""
import hashlib

from flask import current_app, has_request_context, request
from flask_login import current_user
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ..extensions import db
from ..models import AuditLog

# ---------- signed tokens (email verification, password reset, invitations, email change) ----------

PURPOSES = {"verify-email", "reset-password", "invite", "change-email"}


def _serializer(purpose):
    if purpose not in PURPOSES:
        raise ValueError(f"Unknown token purpose {purpose!r}")
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt=f"vi-{purpose}")


def _fingerprint(user):
    """Changes whenever the password or session token changes, which makes reset/invite links single-use."""
    return hashlib.sha256(f"{user.password_hash}|{user.session_token}".encode()).hexdigest()[:16]


def make_token(user, purpose, email=None):
    payload = {"uid": user.id, "fp": _fingerprint(user), "email": email or user.email}
    return _serializer(purpose).dumps(payload)


class TokenError(Exception):
    pass


def read_token(token, purpose, max_age):
    """Returns (user, email) or raises TokenError with a message safe to show to the user."""
    from ..models import User

    try:
        payload = _serializer(purpose).loads(token, max_age=max_age)
    except SignatureExpired:
        raise TokenError("This link has expired. Please request a new one.")
    except BadSignature:
        raise TokenError("This link is invalid. Please check you copied the whole link, or request a new one.")
    user = db.session.get(User, payload.get("uid"))
    if user is None or payload.get("fp") != _fingerprint(user):
        raise TokenError("This link has already been used or is no longer valid. Please request a new one.")
    if purpose != "change-email" and payload.get("email") != user.email:
        raise TokenError("This link was issued for a different email address. Please request a new one.")
    return user, payload.get("email")


# ---------- password policy ----------

# The most common leaked passwords (all >= 8 characters); rejected outright.
COMMON_PASSWORDS = {
    "password", "password1", "password12", "password123", "password1234", "passw0rd", "p@ssw0rd", "p@ssword",
    "12345678", "123456789", "1234567890", "123123123", "11111111", "00000000", "87654321", "12341234",
    "qwertyui", "qwerty123", "qwertyuiop", "1q2w3e4r", "1qaz2wsx", "zaq12wsx", "asdfghjk", "asdf1234",
    "iloveyou", "sunshine", "princess", "football", "baseball", "superman", "trustno1", "letmein1",
    "welcome1", "welcome123", "admin123", "administrator", "changeme", "abc12345", "abcd1234", "aa123456",
    "computer", "internet", "starwars", "whatever", "michelle", "jennifer", "corvette", "mercedes",
    "india123", "chennai123", "test1234", "testtest", "secret123", "monkey123", "dragon123", "master123",
}
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128


def password_problems(password, email="", name=""):
    """Returns a list of reasons the password is unacceptable (empty list = OK)."""
    problems = []
    if len(password) < MIN_PASSWORD_LENGTH:
        problems.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        problems.append(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")
    lowered = password.lower()
    if lowered in COMMON_PASSWORDS:
        problems.append("This password is too common. Please choose something harder to guess.")
    elif len(set(lowered)) <= 2:
        problems.append("Password is too repetitive.")
    local = email.split("@")[0].lower() if email else ""
    if len(local) >= 4 and local in lowered:
        problems.append("Password must not contain your email address.")
    for part in (name or "").lower().split():
        if len(part) >= 4 and part in lowered:
            problems.append("Password must not contain your name.")
            break
    return problems


# ---------- audit log ----------

def client_ip():
    return request.remote_addr if has_request_context() else None


def audit(action, target=None, actor=None, commit=False, **details):
    """Record a security event. `actor` defaults to the logged-in user (if any)."""
    if actor is None and has_request_context() and current_user.is_authenticated:
        actor = current_user._get_current_object()
    entry = AuditLog(
        action=action,
        actor_id=actor.id if actor is not None else None,
        actor_email=actor.email if actor is not None else details.pop("actor_email", None),
        target_user_id=target.id if target is not None else None,
        target_email=target.email if target is not None else details.pop("target_email", None),
        ip_address=client_ip(),
        details=details,
    )
    db.session.add(entry)
    if commit:
        db.session.commit()
    current_app.logger.info("audit %s actor=%s target=%s %s", action, entry.actor_email, entry.target_email, details or "")
    return entry
