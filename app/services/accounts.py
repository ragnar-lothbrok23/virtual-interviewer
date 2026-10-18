"""Account emails and account-level operations shared by the auth, account and admin pages."""
import json
import tempfile
import zipfile
from pathlib import Path

from flask import current_app

from ..extensions import db
from ..models import AuditLog
from .mail import external_url, send_email
from .security import audit, make_token


def _hours(seconds):
    return max(1, seconds // 3600)


def _real(user):
    """Unwrap Flask-Login's current_user proxy so templates and tokens get a plain User object."""
    return user._get_current_object() if hasattr(user, "_get_current_object") else user


def send_verification_email(user):
    user = _real(user)
    token = make_token(user, "verify-email")
    send_email(user.email, f"Confirm your email for {current_app.config['APP_NAME']}", "verify_email",
               user=user, link=external_url("auth.verify_email", token=token),
               hours=_hours(current_app.config["EMAIL_VERIFY_MAX_AGE"]))


def send_password_reset_email(user):
    user = _real(user)
    token = make_token(user, "reset-password")
    send_email(user.email, "Reset your password", "reset_password",
               user=user, link=external_url("auth.reset_password", token=token),
               minutes=current_app.config["PASSWORD_RESET_MAX_AGE"] // 60)


def send_invitation_email(user, inviter):
    user, inviter = _real(user), _real(inviter)
    token = make_token(user, "invite")
    send_email(user.email, f"You've been invited to {current_app.config['APP_NAME']}", "invite",
               user=user, inviter=inviter, link=external_url("auth.accept_invite", token=token),
               days=current_app.config["INVITE_MAX_AGE"] // 86400)


def send_email_change_confirmation(user, new_email):
    user = _real(user)
    token = make_token(user, "change-email", email=new_email)
    send_email(new_email, "Confirm your new email address", "change_email",
               user=user, new_email=new_email, link=external_url("account.confirm_email_change", token=token),
               hours=_hours(current_app.config["EMAIL_VERIFY_MAX_AGE"]))
    # Tell the old address too, so a hijacked session can't silently move the account
    send_email(user.email, "Your email address is being changed", "email_change_notice",
               user=user, new_email=new_email, link=external_url("auth.forgot_password"))


def send_password_changed_notice(user):
    user = _real(user)
    send_email(user.email, "Your password was changed", "password_changed",
               user=user, link=external_url("auth.forgot_password"))


def export_user_data(user):
    """Builds a ZIP with the user's data (JSON) and their uploaded resumes. Returns the ZIP path (temp file)."""
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    data = {
        "account": {
            "name": user.name, "email": user.email, "role": user.role,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "email_verified_at": user.email_verified_at.isoformat() if user.email_verified_at else None,
            "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
            "consent_version": user.consent_version,
            "consent_at": user.consent_at.isoformat() if user.consent_at else None,
        },
        "applications": [
            {
                "job_title": a.job.title, "applied_at": a.created_at.isoformat(), "status": a.status,
                "match_score": a.match_score, "similarity": a.similarity, "resume_file": f"resumes/{a.id}-{a.resume_original_name}",
                "parsed_resume": a.parsed, "resume_text": a.resume_text,
            }
            for a in user.applications
        ],
    }
    with tempfile.NamedTemporaryFile(prefix="vi-export-", suffix=".zip", delete=False) as tmp:
        path = tmp.name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("my_data.json", json.dumps(data, indent=2, ensure_ascii=False))
        for a in user.applications:
            file = upload_dir / a.resume_path
            if file.exists():
                zf.write(file, f"resumes/{a.id}-{a.resume_original_name}")
    return path


def delete_user_account(user, actor=None, reason="self-service"):
    """Permanently deletes the user, their applications and uploaded files. The audit log keeps the email."""
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    files = [upload_dir / a.resume_path for a in user.applications]
    audit("account.deleted", target=user, actor=actor, reason=reason, applications=len(files))
    # Keep audit entries (they store the email) but drop the link to the deleted row
    AuditLog.query.filter(AuditLog.actor_id == user.id).update({"actor_id": None}, synchronize_session=False)
    AuditLog.query.filter(AuditLog.target_user_id == user.id).update({"target_user_id": None}, synchronize_session=False)
    db.session.flush()
    db.session.delete(user)
    db.session.commit()
    for file in files:
        file.unlink(missing_ok=True)
