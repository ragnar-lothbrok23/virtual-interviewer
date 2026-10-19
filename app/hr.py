import io
from urllib.parse import urlsplit

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_file, url_for
from sqlalchemy import case

from .extensions import db, limiter
from .forms import JobForm
from .main import role_required
from .models import Application, Job, User, utcnow
from .services import storage
from .services.analysis import queue_analysis, resume_file_name
from .services.notifications import notify_application_status
from .services.resume import skill_summary
from .services.security import audit
from .services.timeutil import from_local_naive, to_local

bp = Blueprint("hr", __name__, url_prefix="/hr")
hr_only = role_required(User.ROLE_HR, User.ROLE_ADMIN)


def _back(default):
    """Return to the page the action came from, but only if it is on this site."""
    ref = request.referrer
    if ref and urlsplit(ref).netloc == request.host:
        return redirect(ref)
    return redirect(default)


def _clean_skills(raw):
    unique = {}
    for s in (s.strip() for s in raw.split(",")):
        if s:
            unique.setdefault(s.lower(), s)  # case-insensitive de-duplication, keeps first spelling
    return ", ".join(unique.values())


def _deadline_from_form(form, current=None):
    """Convert the form's local deadline to UTC; a new deadline must be in the future. Returns (value, error)."""
    if not form.closes_at.data:
        return None, None
    value = from_local_naive(form.closes_at.data)
    unchanged = current is not None and to_local(current).replace(second=0, microsecond=0) == to_local(value)
    if not unchanged and value <= utcnow():
        return None, "The deadline must be in the future."
    return (current if unchanged else value), None


@bp.route("/")
@hr_only
def dashboard():
    jobs = Job.query.order_by(Job.created_at.desc()).all()
    candidates = User.query.filter_by(role=User.ROLE_CANDIDATE).order_by(User.created_at.desc()).all()
    stats = {
        "jobs": sum(j.is_accepting for j in jobs),
        "candidates": len(candidates),
        "applications": Application.query.filter(Application.status != Application.STATUS_WITHDRAWN).count(),
        "shortlisted": Application.query.filter_by(status=Application.STATUS_SHORTLISTED).count(),
    }
    return render_template("hr/dashboard.html", jobs=jobs, candidates=candidates, stats=stats)


@bp.route("/jobs", methods=["GET", "POST"])
@hr_only
def jobs():
    form = JobForm()
    if form.validate_on_submit():
        closes_at, error = _deadline_from_form(form)
        if error:
            form.closes_at.errors.append(error)
        else:
            job = Job(title=form.title.data.strip(), skills=_clean_skills(form.skills.data),
                      description=form.description.data.strip(), closes_at=closes_at)
            db.session.add(job)
            db.session.flush()
            audit("job.created", job_id=job.id, title=job.title)
            db.session.commit()
            flash(f"Job “{job.title}” created.", "success")
            return redirect(url_for("hr.jobs"))
    jobs = Job.query.order_by(Job.created_at.desc()).all()
    return render_template("hr/jobs.html", form=form, jobs=jobs)


@bp.route("/jobs/<int:job_id>/edit", methods=["GET", "POST"])
@hr_only
def edit_job(job_id):
    job = db.get_or_404(Job, job_id)
    form = JobForm(obj=job)
    if request.method == "GET":
        form.closes_at.data = to_local(job.closes_at).replace(tzinfo=None) if job.closes_at else None
    if form.validate_on_submit():
        closes_at, error = _deadline_from_form(form, current=job.closes_at)
        if error:
            form.closes_at.errors.append(error)
        else:
            new_title, new_skills = form.title.data.strip(), _clean_skills(form.skills.data)
            new_description = form.description.data.strip()
            skills_changed = [s.lower() for s in Job(skills=new_skills).skill_list] != [s.lower() for s in job.skill_list]
            content_changed = skills_changed or new_title != job.title or new_description != job.description
            job.title, job.skills, job.description, job.closes_at = new_title, new_skills, new_description, closes_at
            if content_changed:
                job.embedding = None  # re-embedded on next use
            audit("job.updated", job_id=job.id, title=job.title, skills_changed=skills_changed,
                  content_changed=content_changed)
            db.session.commit()
            active = sum(not a.is_withdrawn for a in job.applications)
            if content_changed and active:
                from .tasks import rescore_job_applications

                rescore_job_applications(job.id, skills_changed)
                what = ("re-analysing" if skills_changed else "re-scoring")
                flash(f"Job saved. {what.capitalize()} {active} applicant(s) in the background"
                      + (f" (uses {active} AI request(s))." if skills_changed else "."), "success")
            else:
                flash("Job saved.", "success")
            return redirect(url_for("hr.job_detail", job_id=job.id))
    active = sum(not a.is_withdrawn for a in job.applications)
    return render_template("hr/job_edit.html", form=form, job=job, active=active)


@bp.route("/jobs/<int:job_id>")
@hr_only
def job_detail(job_id):
    job = db.get_or_404(Job, job_id)
    status = request.args.get("status")
    query = Application.query.filter_by(job_id=job.id)
    if status in Application.STATUS_LABELS:
        query = query.filter_by(status=status)
    # Analysed applicants by score first, then ones still being analysed, then failed analyses
    order = case((Application.analysis_status == Application.ANALYSIS_DONE, 0),
                 (Application.analysis_status == Application.ANALYSIS_FAILED, 2), else_=1)
    applicants = query.order_by(order, Application.match_score.desc(), Application.created_at).all()
    return render_template("hr/job_detail.html", job=job, applicants=applicants, status=status,
                           labels=Application.STATUS_LABELS, threshold=current_app.config["SHORTLIST_THRESHOLD"])


@bp.post("/jobs/<int:job_id>/toggle")
@hr_only
def toggle_job(job_id):
    job = db.get_or_404(Job, job_id)
    job.is_open = not job.is_open
    audit("job.opened" if job.is_open else "job.closed", job_id=job.id, title=job.title)
    db.session.commit()
    message = f"“{job.title}” is now {'open' if job.is_open else 'closed'} for applications."
    if job.is_open and job.deadline_passed:
        message += " Note: its deadline has passed, so candidates still can't apply. Edit the job to extend it."
    flash(message, "info")
    return _back(url_for("hr.jobs"))


@bp.route("/applications/<int:app_id>")
@hr_only
def application(app_id):
    app_row = db.get_or_404(Application, app_id)
    summary = skill_summary(app_row.parsed, app_row.job.skill_list) if app_row.parsed else None
    return render_template("hr/application.html", a=app_row, labels=Application.STATUS_LABELS,
                           hr_statuses=Application.HR_STATUSES, summary=summary)


@bp.post("/applications/<int:app_id>/status")
@hr_only
def set_status(app_id):
    app_row = db.get_or_404(Application, app_id)
    status = request.form.get("status")
    if status not in Application.HR_STATUSES:
        abort(400)
    if app_row.is_withdrawn:
        flash("The candidate has withdrawn this application.", "warning")
        return _back(url_for("hr.application", app_id=app_row.id))
    old_status = app_row.status
    app_row.status = status
    app_row.status_source = Application.SOURCE_HR  # HR decisions are never overwritten by re-scoring
    if status != old_status:
        audit("application.status_changed", target=app_row.candidate, application_id=app_row.id,
              job=app_row.job.title, old_status=old_status, new_status=status)
    db.session.commit()
    notify_application_status(app_row, old_status)
    flash(f"{app_row.candidate.name} marked as {app_row.status_label.lower()}.", "success")
    return _back(url_for("hr.application", app_id=app_row.id))


@bp.post("/applications/<int:app_id>/reanalyse")
@hr_only
@limiter.limit("30 per hour")
def reanalyse(app_id):
    app_row = db.get_or_404(Application, app_id)
    if app_row.is_analysing:
        flash("This resume is already being analysed.", "info")
    elif app_row.is_withdrawn:
        flash("The candidate has withdrawn this application.", "warning")
    else:
        app_row.analysis_status, app_row.analysis_attempts = Application.ANALYSIS_PENDING, 0
        audit("application.reanalysis_requested", target=app_row.candidate, application_id=app_row.id)
        db.session.commit()
        queue_analysis(app_row)
        flash("The resume is being analysed again.", "info")
    return redirect(url_for("hr.application", app_id=app_row.id))


@bp.route("/applications/<int:app_id>/resume")
@hr_only
def resume(app_id):
    app_row = db.get_or_404(Application, app_id)
    try:
        data = storage.read_resume(app_row.resume_path)
    except storage.StorageError as e:
        current_app.logger.error("Could not read resume for application %s: %s", app_row.id, e)
        flash("The resume file could not be opened. It may be missing or damaged.", "danger")
        return redirect(url_for("hr.application", app_id=app_row.id))
    audit("resume.downloaded", target=app_row.candidate, application_id=app_row.id, commit=True)
    mimetype = "application/pdf" if app_row.resume_kind == "pdf" else \
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return send_file(io.BytesIO(data), as_attachment=True, download_name=resume_file_name(app_row), mimetype=mimetype)
