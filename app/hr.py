from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_from_directory, url_for

from .extensions import db
from .forms import JobForm
from .main import role_required
from .models import Application, Job, User

bp = Blueprint("hr", __name__, url_prefix="/hr")
hr_only = role_required(User.ROLE_HR)


@bp.route("/")
@hr_only
def dashboard():
    jobs = Job.query.order_by(Job.created_at.desc()).all()
    candidates = User.query.filter_by(role=User.ROLE_CANDIDATE).order_by(User.created_at.desc()).all()
    stats = {
        "jobs": sum(j.is_open for j in jobs),
        "candidates": len(candidates),
        "applications": Application.query.count(),
        "shortlisted": Application.query.filter_by(status=Application.STATUS_SHORTLISTED).count(),
    }
    return render_template("hr/dashboard.html", jobs=jobs, candidates=candidates, stats=stats)


@bp.route("/jobs", methods=["GET", "POST"])
@hr_only
def jobs():
    form = JobForm()
    if form.validate_on_submit():
        unique = {}
        for s in (s.strip() for s in form.skills.data.split(",")):
            if s:
                unique.setdefault(s.lower(), s)  # case-insensitive de-duplication, keeps first spelling
        skills = ", ".join(unique.values())
        job = Job(title=form.title.data.strip(), skills=skills, description=form.description.data.strip())
        db.session.add(job)
        db.session.commit()
        flash(f"Job “{job.title}” created.", "success")
        return redirect(url_for("hr.jobs"))
    jobs = Job.query.order_by(Job.created_at.desc()).all()
    return render_template("hr/jobs.html", form=form, jobs=jobs)


@bp.route("/jobs/<int:job_id>")
@hr_only
def job_detail(job_id):
    job = db.get_or_404(Job, job_id)
    status = request.args.get("status")
    query = Application.query.filter_by(job_id=job.id)
    if status in Application.STATUS_LABELS:
        query = query.filter_by(status=status)
    applicants = query.order_by(Application.match_score.desc()).all()
    return render_template("hr/job_detail.html", job=job, applicants=applicants, status=status,
                           labels=Application.STATUS_LABELS, threshold=current_app.config["SHORTLIST_THRESHOLD"])


@bp.post("/jobs/<int:job_id>/toggle")
@hr_only
def toggle_job(job_id):
    job = db.get_or_404(Job, job_id)
    job.is_open = not job.is_open
    db.session.commit()
    flash(f"“{job.title}” is now {'open' if job.is_open else 'closed'} for applications.", "info")
    return redirect(request.referrer or url_for("hr.jobs"))


@bp.route("/applications/<int:app_id>")
@hr_only
def application(app_id):
    app_row = db.get_or_404(Application, app_id)
    return render_template("hr/application.html", a=app_row, labels=Application.STATUS_LABELS)


@bp.post("/applications/<int:app_id>/status")
@hr_only
def set_status(app_id):
    app_row = db.get_or_404(Application, app_id)
    status = request.form.get("status")
    if status not in Application.STATUS_LABELS:
        abort(400)
    app_row.status = status
    db.session.commit()
    flash(f"{app_row.candidate.name} marked as {app_row.status_label.lower()}.", "success")
    return redirect(request.referrer or url_for("hr.application", app_id=app_row.id))


@bp.route("/applications/<int:app_id>/resume")
@hr_only
def resume(app_id):
    app_row = db.get_or_404(Application, app_id)
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], app_row.resume_path,
                               as_attachment=True, download_name=app_row.resume_original_name)
