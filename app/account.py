"""Self-service account settings: profile, email, password, data export and deletion."""
import os

from flask import (Blueprint, after_this_request, current_app, flash, redirect, render_template, send_file, url_for)
from flask_login import current_user, login_required, login_user, logout_user

from .extensions import db, limiter
from .forms import ChangeEmailForm, ChangePasswordForm, DeleteAccountForm, ProfileForm, email_is_taken, normalise_email
from .models import User
from .services.accounts import (delete_user_account, export_user_data, send_email_change_confirmation,
                                send_password_changed_notice)
from .services.security import TokenError, audit, read_token

bp = Blueprint("account", __name__, url_prefix="/account")


def _forms():
    profile = ProfileForm(prefix="profile", data={"name": current_user.name})
    email = ChangeEmailForm(prefix="email")
    password = ChangePasswordForm(prefix="password")
    password.policy_email, password.policy_name = current_user.email, current_user.name
    delete = DeleteAccountForm(prefix="delete")
    return profile, email, password, delete


def _render(profile, email, password, delete):
    return render_template("account/settings.html", profile_form=profile, email_form=email,
                           password_form=password, delete_form=delete, can_delete=_can_self_delete())


def _can_self_delete():
    """Admins can't delete themselves if they are the last active admin (someone must be able to manage users)."""
    if not current_user.is_admin:
        return True
    return User.query.filter_by(role=User.ROLE_ADMIN, active=True).count() > 1


@bp.route("/", methods=["GET"])
@login_required
def settings():
    return _render(*_forms())


@bp.post("/profile")
@login_required
def update_profile():
    profile, email, password, delete = _forms()
    if profile.validate_on_submit():
        old = current_user.name
        current_user.name = profile.name.data.strip()
        audit("profile.updated", target=current_user, old_name=old, new_name=current_user.name)
        db.session.commit()
        flash("Your name has been updated.", "success")
        return redirect(url_for("account.settings"))
    return _render(profile, email, password, delete)


@bp.post("/email")
@login_required
@limiter.limit("5 per hour")
def change_email():
    profile, email, password, delete = _forms()
    if email.validate_on_submit():
        new_email = normalise_email(email.new_email.data)
        if not current_user.check_password(email.current_password.data):
            email.current_password.errors.append("Incorrect password.")
        elif new_email == current_user.email:
            email.new_email.errors.append("That is already your email address.")
        elif email_is_taken(new_email, exclude_user_id=current_user.id):
            email.new_email.errors.append("An account with this email already exists.")
        else:
            current_user.pending_email = new_email
            audit("email.change_requested", target=current_user, new_email=new_email)
            db.session.commit()
            send_email_change_confirmation(current_user, new_email)
            flash(f"We've sent a confirmation link to {new_email}. Your email changes once you click it.", "info")
            return redirect(url_for("account.settings"))
    return _render(profile, email, password, delete)


@bp.route("/email/confirm/<token>")
def confirm_email_change(token):
    try:
        user, new_email = read_token(token, "change-email", current_app.config["EMAIL_VERIFY_MAX_AGE"])
    except TokenError as e:
        flash(str(e), "danger")
        return redirect(url_for("account.settings") if current_user.is_authenticated else url_for("auth.login"))
    if user.pending_email != new_email:
        flash("This link is no longer valid. Please request the email change again.", "danger")
    elif email_is_taken(new_email, exclude_user_id=user.id):
        flash("That email address is now used by another account.", "danger")
    else:
        old = user.email
        user.email, user.pending_email = new_email, None
        user.mark_verified()
        user.end_all_sessions()  # sign out other devices; the new address is the login from now on
        audit("email.changed", target=user, actor=user, old_email=old, new_email=new_email)
        db.session.commit()
        if current_user.is_authenticated and current_user.id == user.id:
            login_user(user)  # keep this browser signed in with the new session token
        flash(f"Your email address is now {new_email}.", "success")
    return redirect(url_for("account.settings") if current_user.is_authenticated else url_for("auth.login"))


@bp.post("/email/cancel")
@login_required
def cancel_email_change():
    if current_user.pending_email:
        current_user.pending_email = None
        current_user.end_all_sessions()  # also invalidates the emailed link
        db.session.commit()
        login_user(current_user._get_current_object())
        flash("The pending email change was cancelled.", "info")
    return redirect(url_for("account.settings"))


@bp.post("/password")
@login_required
@limiter.limit("10 per hour")
def change_password():
    profile, email, password, delete = _forms()
    if password.validate_on_submit():
        if not current_user.check_password(password.current_password.data):
            password.current_password.errors.append("Incorrect password.")
        elif current_user.check_password(password.password.data):
            password.password.errors.append("Your new password must be different from the current one.")
        else:
            current_user.set_password(password.password.data)  # ends every other session
            audit("password.changed", target=current_user)
            db.session.commit()
            login_user(current_user._get_current_object())  # keep this browser signed in (pass the real object, not the proxy)
            send_password_changed_notice(current_user)
            flash("Your password has been changed. Other devices have been signed out.", "success")
            return redirect(url_for("account.settings"))
    return _render(profile, email, password, delete)


@bp.post("/export")
@login_required
@limiter.limit("5 per hour")
def export_data():
    path = export_user_data(current_user)
    audit("data.exported", target=current_user, commit=True)

    @after_this_request
    def _cleanup(response):
        response.call_on_close(lambda: os.path.exists(path) and os.remove(path))
        return response

    return send_file(path, as_attachment=True, download_name="virtual-interviewer-my-data.zip", mimetype="application/zip")


@bp.post("/delete")
@login_required
@limiter.limit("5 per hour")
def delete_account():
    profile, email, password, delete = _forms()
    if not _can_self_delete():
        flash("You are the only active admin. Make someone else an admin before deleting your account.", "danger")
        return redirect(url_for("account.settings"))
    if delete.validate_on_submit():
        if not current_user.check_password(delete.current_password.data):
            delete.current_password.errors.append("Incorrect password.")
        else:
            user = current_user._get_current_object()
            delete_user_account(user, actor=user)
            logout_user()
            flash("Your account and all your data have been permanently deleted.", "info")
            return redirect(url_for("main.index"))
    return _render(profile, email, password, delete)
