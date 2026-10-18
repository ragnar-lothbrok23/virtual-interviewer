"""In-app notifications, optionally paired with an email (respecting the user's email preferences)."""
from flask import current_app

from ..extensions import db
from ..models import Application, Notification
from .mail import external_url, send_email


def notify(user, kind, title, body="", link=None, email=None):
    """Create an in-app notification and, if `email` is given and the user wants that category, send it.

    email: dict(category=..., subject=..., template=..., **template_context) or None.
    `link` is an endpoint-free local path (e.g. "/candidate/applications/3")."""
    if link and not link.startswith("/"):
        raise ValueError("Notification links must be local paths")
    note = Notification(user_id=user.id, kind=kind, title=title[:200], body=body, link=link)
    db.session.add(note)
    db.session.commit()
    if email:
        email = dict(email)
        category = email.pop("category")
        if user.wants_email(category):
            send_email(user.email, email.pop("subject"), email.pop("template"), user=user,
                       preferences_link=external_url("account.settings"), **email)
    return note


def notify_application_received(application):
    a = application
    shortlisted = a.status == Application.STATUS_SHORTLISTED
    link = f"/candidate/applications/{a.id}"
    title = f"Application received: {a.job.title}"
    body = (f"Your resume matched {a.match_score:.0f}/100 and you've been shortlisted."
            if shortlisted else f"Your resume matched {a.match_score:.0f}/100. A recruiter will review it.")
    notify(a.candidate, "application.received", title, body, link,
           email=dict(category="applications", subject=f"Application received: {a.job.title}",
                      template="application_received", application=a, shortlisted=shortlisted,
                      link=external_url("candidate.application", app_id=a.id)))


def notify_application_status(application, old_status):
    """Tell the candidate their application status changed (no-op if it didn't)."""
    a = application
    if a.status == old_status:
        return
    link = f"/candidate/applications/{a.id}"
    if a.status == Application.STATUS_SHORTLISTED:
        title, subject = f"You've been shortlisted: {a.job.title}", f"Good news about your application for {a.job.title}"
        body = "The recruiter has shortlisted you. Interview scheduling will open soon."
    elif a.status == Application.STATUS_REJECTED:
        title, subject = f"Update on your application: {a.job.title}", f"Update on your application for {a.job.title}"
        body = "Thank you for applying. The recruiter has decided not to move forward with this application."
    else:
        title, subject = f"Application back under review: {a.job.title}", None
        body = "Your application is being reviewed again."
    email = None
    if subject:
        email = dict(category="applications", subject=subject, template="application_status", application=a,
                     link=external_url("candidate.application", app_id=a.id))
    notify(a.candidate, "application.status", title, body, link, email=email)
    current_app.logger.info("Notified %s: application %s %s -> %s", a.candidate.email, a.id, old_status, a.status)
