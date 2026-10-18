"""Admin panel: user management and the audit log."""
import secrets

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import or_

from .extensions import db
from .forms import InviteStaffForm, normalise_email
from .main import role_required
from .models import AuditLog, EmailLog, User
from .services.accounts import delete_user_account, send_invitation_email
from .services.mail import sent_in_last_24h
from .services.security import audit

bp = Blueprint("admin", __name__, url_prefix="/admin")
admin_only = role_required(User.ROLE_ADMIN)
PER_PAGE = 25


def _active_admin_count():
    return User.query.filter_by(role=User.ROLE_ADMIN, active=True).count()


def _would_remove_last_admin(user):
    return user.is_admin and user.active and _active_admin_count() <= 1


def _get_user(user_id):
    return db.get_or_404(User, user_id)


def _back(user):
    return redirect(url_for("admin.user_detail", user_id=user.id))


# ---------- user list / invite ----------

@bp.route("/")
@admin_only
def users():
    q = request.args.get("q", "").strip()
    role = request.args.get("role", "")
    status = request.args.get("status", "")
    query = User.query
    if q:
        needle = q.lower()
        query = query.filter(or_(db.func.lower(User.name).contains(needle, autoescape=True),
                                 User.email.contains(needle, autoescape=True)))
    if role in User.ROLES:
        query = query.filter(User.role == role)
    if status == "active":
        query = query.filter(User.active.is_(True))
    elif status == "inactive":
        query = query.filter(User.active.is_(False))
    elif status == "unverified":
        query = query.filter(User.email_verified_at.is_(None))
    page = query.order_by(User.created_at.desc()).paginate(per_page=PER_PAGE, error_out=False)
    counts = {r: User.query.filter_by(role=r).count() for r in User.ROLES}
    return render_template("admin/users.html", page=page, q=q, role=role, status=status, counts=counts,
                           roles=User.ROLE_LABELS)


@bp.route("/users/invite", methods=["GET", "POST"])
@admin_only
def invite():
    form = InviteStaffForm()
    if form.validate_on_submit():
        user = User(name=form.name.data.strip(), email=normalise_email(form.email.data), role=form.role.data)
        user.set_password(secrets.token_urlsafe(32))  # unusable until they set their own via the invite link
        db.session.add(user)
        db.session.flush()
        audit("staff.invited", target=user, role=user.role)
        db.session.commit()
        send_invitation_email(user, current_user)
        flash(f"Invitation sent to {user.email}.", "success")
        return _back(user)
    return render_template("admin/invite.html", form=form)


# ---------- user detail and actions ----------

@bp.route("/users/<int:user_id>")
@admin_only
def user_detail(user_id):
    user = _get_user(user_id)
    events = (AuditLog.query.filter(or_(AuditLog.target_user_id == user.id, AuditLog.actor_id == user.id))
              .order_by(AuditLog.created_at.desc(), AuditLog.id.desc()).limit(30).all())
    return render_template("admin/user_detail.html", user=user, events=events, is_self=user.id == current_user.id,
                           last_admin=_would_remove_last_admin(user))


@bp.post("/users/<int:user_id>/deactivate")
@admin_only
def deactivate(user_id):
    user = _get_user(user_id)
    if user.id == current_user.id:
        flash("You can't deactivate your own account.", "danger")
    elif _would_remove_last_admin(user):
        flash("You can't deactivate the last active admin.", "danger")
    elif user.active:
        user.active = False
        user.end_all_sessions()
        audit("account.deactivated", target=user)
        db.session.commit()
        flash(f"{user.email} has been deactivated and signed out everywhere.", "success")
    return _back(user)


@bp.post("/users/<int:user_id>/activate")
@admin_only
def activate(user_id):
    user = _get_user(user_id)
    if not user.active:
        user.active = True
        audit("account.activated", target=user)
        db.session.commit()
        flash(f"{user.email} has been reactivated.", "success")
    return _back(user)


@bp.post("/users/<int:user_id>/unlock")
@admin_only
def unlock(user_id):
    user = _get_user(user_id)
    user.unlock()
    audit("account.unlocked", target=user)
    db.session.commit()
    flash(f"{user.email} has been unlocked.", "success")
    return _back(user)


@bp.post("/users/<int:user_id>/sign-out")
@admin_only
def force_sign_out(user_id):
    user = _get_user(user_id)
    if user.id == current_user.id:
        flash("Use Log out to end your own session.", "warning")
        return _back(user)
    user.end_all_sessions()
    audit("account.sessions_ended", target=user)
    db.session.commit()
    flash(f"{user.email} has been signed out of all devices.", "success")
    return _back(user)


@bp.post("/users/<int:user_id>/role")
@admin_only
def change_role(user_id):
    user = _get_user(user_id)
    new_role = request.form.get("role")
    if new_role not in (User.ROLE_HR, User.ROLE_ADMIN):
        abort(400)
    if user.role == User.ROLE_CANDIDATE:
        flash("Candidate accounts can't be turned into staff accounts. Invite the person with a work email instead.", "danger")
    elif user.id == current_user.id:
        flash("You can't change your own role.", "danger")
    elif new_role != User.ROLE_ADMIN and _would_remove_last_admin(user):
        flash("You can't demote the last active admin.", "danger")
    elif user.role != new_role:
        old = user.role
        user.role = new_role
        user.end_all_sessions()  # permissions changed: make them sign in again
        audit("account.role_changed", target=user, old_role=old, new_role=new_role)
        db.session.commit()
        flash(f"{user.email} is now {user.role_label}.", "success")
    return _back(user)


@bp.post("/users/<int:user_id>/resend-invite")
@admin_only
def resend_invite(user_id):
    user = _get_user(user_id)
    if user.role == User.ROLE_CANDIDATE or user.last_login_at is not None:
        flash("Only staff who haven't signed in yet can be re-invited. Others can use “Forgot password”.", "warning")
    else:
        send_invitation_email(user, current_user)
        audit("staff.reinvited", target=user, commit=True)
        flash(f"A new invitation was sent to {user.email}.", "success")
    return _back(user)


@bp.post("/users/<int:user_id>/delete")
@admin_only
def delete(user_id):
    user = _get_user(user_id)
    if user.id == current_user.id:
        flash("Delete your own account from Account settings.", "danger")
        return _back(user)
    if _would_remove_last_admin(user):
        flash("You can't delete the last active admin.", "danger")
        return _back(user)
    if request.form.get("confirm_email", "").strip().lower() != user.email:
        flash("Type the user's email address exactly to confirm deletion.", "danger")
        return _back(user)
    email = user.email
    delete_user_account(user, actor=current_user._get_current_object(), reason="deleted by admin")
    flash(f"{email} and all their data have been permanently deleted.", "success")
    return redirect(url_for("admin.users"))


# ---------- email log ----------

@bp.route("/emails")
@admin_only
def emails():
    status = request.args.get("status", "")
    to = request.args.get("to", "").strip().lower()
    query = EmailLog.query
    if status in EmailLog.STATUS_COLORS:
        query = query.filter(EmailLog.status == status)
    if to:
        query = query.filter(EmailLog.to_address.contains(to, autoescape=True))
    page = query.order_by(EmailLog.created_at.desc(), EmailLog.id.desc()).paginate(per_page=50, error_out=False)
    stats = {s: EmailLog.query.filter_by(status=s).count() for s in EmailLog.STATUS_COLORS}
    return render_template("admin/emails.html", page=page, status=status, to=to, stats=stats,
                           statuses=list(EmailLog.STATUS_COLORS), sent_24h=sent_in_last_24h(),
                           daily_limit=current_app.config["MAIL_DAILY_LIMIT"], backend=current_app.config["MAIL_BACKEND"])


# ---------- audit log ----------

@bp.route("/audit")
@admin_only
def audit_log():
    action = request.args.get("action", "").strip()
    email = request.args.get("email", "").strip().lower()
    query = AuditLog.query
    if action:
        query = query.filter(AuditLog.action.startswith(action, autoescape=True))
    if email:
        query = query.filter(or_(AuditLog.actor_email.contains(email, autoescape=True),
                                 AuditLog.target_email.contains(email, autoescape=True)))
    page = query.order_by(AuditLog.created_at.desc(), AuditLog.id.desc()).paginate(per_page=50, error_out=False)
    actions = [a for (a,) in db.session.query(AuditLog.action).distinct().order_by(AuditLog.action)]
    return render_template("admin/audit.html", page=page, action=action, email=email, actions=actions)
