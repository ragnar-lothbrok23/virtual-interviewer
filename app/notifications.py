"""The bell icon: list, open and mark in-app notifications as read."""
from flask import Blueprint, abort, has_request_context, redirect, render_template, url_for
from flask_login import current_user, login_required

from .extensions import db
from .models import Notification, utcnow

bp = Blueprint("notifications", __name__, url_prefix="/notifications")


def unread_count(user):
    return user.notifications.filter(Notification.read_at.is_(None)).count()


@bp.app_context_processor
def inject_notifications():
    """Unread count and the latest few notifications for the bell dropdown on every page.
    Skipped outside requests (emails rendered by background jobs) and for anonymous visitors."""
    if not has_request_context() or current_user is None or not current_user.is_authenticated:
        return {}
    latest = current_user.notifications.order_by(Notification.created_at.desc(), Notification.id.desc()).limit(5).all()
    return {"unread_notifications": unread_count(current_user), "latest_notifications": latest}


@bp.route("/")
@login_required
def index():
    page = (current_user.notifications.order_by(Notification.created_at.desc(), Notification.id.desc())
            .paginate(per_page=20, error_out=False))
    return render_template("notifications/index.html", page=page)


@bp.route("/<int:note_id>/open")
@login_required
def open_notification(note_id):
    note = db.get_or_404(Notification, note_id)
    if note.user_id != current_user.id:
        abort(404)
    if note.read_at is None:
        note.read_at = utcnow()
        db.session.commit()
    # Only local paths are ever stored, but double-check before redirecting
    if note.link and note.link.startswith("/") and not note.link.startswith("//"):
        return redirect(note.link)
    return redirect(url_for("notifications.index"))


@bp.post("/read-all")
@login_required
def mark_all_read():
    current_user.notifications.filter(Notification.read_at.is_(None)).update(
        {"read_at": utcnow()}, synchronize_session=False)
    db.session.commit()
    return redirect(url_for("notifications.index"))
