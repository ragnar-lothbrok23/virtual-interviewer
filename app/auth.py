from urllib.parse import urlsplit

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db, limiter
from .forms import (ConsentForm, ForgotPasswordForm, LoginForm, RegisterForm, ResendVerificationForm,
                    ResetPasswordForm, normalise_email)
from .models import User
from .services.accounts import send_password_changed_notice, send_password_reset_email, send_verification_email
from .services.security import TokenError, audit, read_token

bp = Blueprint("auth", __name__)

_DUMMY_HASH = None


def _equalise_timing(password):
    """Check against a throwaway hash for unknown emails so response time doesn't reveal which emails exist."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = generate_password_hash("not-a-real-password")
    check_password_hash(_DUMMY_HASH, password)


def _safe_next(default):
    target = request.args.get("next")
    if not target or urlsplit(target).netloc or urlsplit(target).scheme or not target.startswith("/") or target.startswith("//"):
        return default
    return target


# ---------- sign up / log in / log out ----------

@bp.route("/register", methods=["GET", "POST"])
@limiter.limit("10 per hour", methods=["POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = RegisterForm()
    if form.validate_on_submit():
        # Public sign-up is for candidates only; staff accounts are created by an admin invitation
        user = User(name=form.name.data.strip(), email=normalise_email(form.email.data), role=User.ROLE_CANDIDATE)
        user.set_password(form.password.data)
        user.give_consent(current_app.config["PRIVACY_POLICY_VERSION"])
        db.session.add(user)
        db.session.flush()
        audit("account.registered", target=user, actor=user)
        db.session.commit()
        send_verification_email(user)
        login_user(user)
        flash(f"Welcome! We've sent a confirmation link to {user.email}. Please verify your email to apply for jobs.", "success")
        return redirect(url_for("main.dashboard"))
    return render_template("auth/register.html", form=form)


@bp.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute", methods=["POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = LoginForm()
    if form.validate_on_submit():
        email = normalise_email(form.email.data)
        user = User.query.filter_by(email=email).first()
        cfg = current_app.config

        if user is None:
            _equalise_timing(form.password.data)
            audit("login.failed", target_email=email, reason="unknown email", commit=True)
            flash("Invalid email or password.", "danger")
            return render_template("auth/login.html", form=form)

        if user.is_locked:
            audit("login.blocked", target=user, reason="locked", commit=True)
            flash(f"Too many failed attempts. This account is locked for {user.lock_minutes_left} more minute(s). "
                  "You can also reset your password.", "danger")
            return render_template("auth/login.html", form=form)

        if not user.check_password(form.password.data):
            just_locked = user.register_failed_login(cfg["LOCKOUT_THRESHOLD"], cfg["LOCKOUT_MINUTES"])
            audit("login.failed", target=user, reason="wrong password")
            if just_locked:
                audit("account.locked", target=user, minutes=cfg["LOCKOUT_MINUTES"])
            db.session.commit()
            if just_locked:
                flash(f"Too many failed attempts. This account is locked for {cfg['LOCKOUT_MINUTES']} minutes. "
                      "You can also reset your password.", "danger")
            else:
                flash("Invalid email or password.", "danger")
            return render_template("auth/login.html", form=form)

        if not user.active:
            audit("login.blocked", target=user, reason="deactivated", commit=True)
            flash("This account has been deactivated. Please contact the site administrator.", "danger")
            return render_template("auth/login.html", form=form)

        user.register_successful_login()
        audit("login.success", target=user, actor=user)
        db.session.commit()
        login_user(user, remember=form.remember.data)
        return redirect(_safe_next(url_for("main.dashboard")))
    return render_template("auth/login.html", form=form)


@bp.post("/logout")
@login_required
def logout():
    audit("logout", target=current_user, commit=True)
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("main.index"))


# ---------- email verification ----------

@bp.route("/verify-email/<token>")
def verify_email(token):
    try:
        user, _ = read_token(token, "verify-email", current_app.config["EMAIL_VERIFY_MAX_AGE"])
    except TokenError as e:
        flash(str(e), "danger")
        return redirect(url_for("main.dashboard") if current_user.is_authenticated else url_for("auth.login"))
    if user.is_verified:
        flash("Your email is already verified.", "info")
    else:
        user.mark_verified()
        audit("email.verified", target=user, actor=user)
        db.session.commit()
        flash("Thanks! Your email address is verified.", "success")
    return redirect(url_for("main.dashboard") if current_user.is_authenticated else url_for("auth.login"))


@bp.post("/verify-email/resend")
@login_required
@limiter.limit("3 per 15 minutes", key_func=lambda: f"resend-verify:{current_user.id}")
def resend_verification():
    form = ResendVerificationForm()
    if form.validate_on_submit():
        if current_user.is_verified:
            flash("Your email is already verified.", "info")
        else:
            send_verification_email(current_user)
            flash(f"A new confirmation link has been sent to {current_user.email}.", "success")
    return redirect(request.referrer if _same_site_referrer() else url_for("main.dashboard"))


def _same_site_referrer():
    ref = request.referrer
    return bool(ref) and urlsplit(ref).netloc == request.host


# ---------- password reset ----------

@bp.route("/forgot-password", methods=["GET", "POST"])
@limiter.limit("5 per hour", methods=["POST"])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        email = normalise_email(form.email.data)
        user = User.query.filter_by(email=email).first()
        if user is not None and user.active:
            send_password_reset_email(user)
            audit("password.reset_requested", target=user, commit=True)
        # Same answer whether or not the account exists, so this page can't be used to discover emails
        flash(f"If an account exists for {email}, we've sent a password reset link. It expires in "
              f"{current_app.config['PASSWORD_RESET_MAX_AGE'] // 60} minutes.", "info")
        return redirect(url_for("auth.login"))
    return render_template("auth/forgot_password.html", form=form)


@bp.route("/reset-password/<token>", methods=["GET", "POST"])
@limiter.limit("10 per hour", methods=["POST"])
def reset_password(token):
    return _set_password_from_token(token, "reset-password", current_app.config["PASSWORD_RESET_MAX_AGE"],
                                    template="auth/reset_password.html")


@bp.route("/invite/<token>", methods=["GET", "POST"])
@limiter.limit("10 per hour", methods=["POST"])
def accept_invite(token):
    return _set_password_from_token(token, "invite", current_app.config["INVITE_MAX_AGE"],
                                    template="auth/accept_invite.html")


def _set_password_from_token(token, purpose, max_age, template):
    try:
        user, _ = read_token(token, purpose, max_age)
    except TokenError as e:
        flash(str(e), "danger")
        return redirect(url_for("auth.forgot_password"))
    if not user.active:
        flash("This account has been deactivated. Please contact the site administrator.", "danger")
        return redirect(url_for("auth.login"))

    form = ResetPasswordForm()
    form.policy_email, form.policy_name = user.email, user.name
    if form.validate_on_submit():
        user.set_password(form.password.data)  # also ends every existing session
        user.unlock()
        user.mark_verified()  # they proved they control the inbox
        audit("password.reset" if purpose == "reset-password" else "invite.accepted", target=user, actor=user)
        db.session.commit()
        if purpose == "reset-password":
            send_password_changed_notice(user)
        if current_user.is_authenticated:
            logout_user()
        flash("Your password has been set. Please log in.", "success")
        return redirect(url_for("auth.login"))
    return render_template(template, form=form, user=user)


# ---------- consent ----------

@bp.route("/consent", methods=["GET", "POST"])
@login_required
def consent():
    version = current_app.config["PRIVACY_POLICY_VERSION"]
    if current_user.has_current_consent(version):
        return redirect(url_for("main.dashboard"))
    form = ConsentForm()
    if form.validate_on_submit():
        current_user.give_consent(version)
        audit("consent.given", target=current_user, version=version)
        db.session.commit()
        flash("Thank you.", "success")
        return redirect(_safe_next(url_for("main.dashboard")))
    return render_template("auth/consent.html", form=form, updated=current_user.consent_version is not None)
