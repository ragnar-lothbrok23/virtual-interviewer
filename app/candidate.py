import uuid
from pathlib import Path

from flask import Blueprint, abort, current_app, flash, redirect, render_template, url_for
from flask_login import current_user
from werkzeug.utils import secure_filename

from .extensions import db, limiter
from .forms import ApplyForm
from .main import role_required
from .models import Application, Job, User
from .services.gemini import GeminiError
from .services.notifications import notify_application_received
from .services.resume import ResumeError, analyse, extract_text, initial_status

bp = Blueprint("candidate", __name__, url_prefix="/candidate")
candidate_only = role_required(User.ROLE_CANDIDATE)


@bp.route("/")
@candidate_only
def dashboard():
    applications = (Application.query.filter_by(candidate_id=current_user.id)
                    .order_by(Application.created_at.desc()).all())
    return render_template("candidate/dashboard.html", applications=applications)


@bp.route("/jobs")
@candidate_only
def jobs():
    open_jobs = Job.query.filter_by(is_open=True).order_by(Job.created_at.desc()).all()
    applied = {a.job_id: a for a in current_user.applications}
    return render_template("candidate/jobs.html", jobs=open_jobs, applied=applied)


@bp.route("/jobs/<int:job_id>/apply", methods=["GET", "POST"])
@candidate_only
@limiter.limit("20 per hour", methods=["POST"])
def apply(job_id):
    job = db.get_or_404(Job, job_id)
    existing = Application.query.filter_by(candidate_id=current_user.id, job_id=job.id).first()
    if existing:
        return redirect(url_for("candidate.application", app_id=existing.id))
    if not job.is_open:
        flash("This job is no longer accepting applications.", "warning")
        return redirect(url_for("candidate.jobs"))
    if not current_user.is_verified:
        flash("Please verify your email address before applying. Check your inbox, or resend the link below.", "warning")
        return redirect(url_for("candidate.dashboard"))

    form = ApplyForm()
    if form.validate_on_submit():
        upload = form.resume.data
        original = secure_filename(upload.filename) or "resume"
        stored = f"{uuid.uuid4().hex}{Path(original).suffix.lower()}"
        path = Path(current_app.config["UPLOAD_FOLDER"]) / stored
        upload.save(path)
        try:
            text = extract_text(path)
            profile, similarity, score = analyse(text, job)
        except (ResumeError, GeminiError) as e:
            path.unlink(missing_ok=True)
            flash(str(e), "danger")
            return render_template("candidate/apply.html", job=job, form=form)

        app_row = Application(candidate=current_user, job=job, resume_path=stored, resume_original_name=original,
                              resume_text=text, parsed=profile, similarity=similarity, match_score=score,
                              status=initial_status(score))
        db.session.add(app_row)
        db.session.commit()
        notify_application_received(app_row)
        flash("Your resume was analysed and your application submitted.", "success")
        return redirect(url_for("candidate.application", app_id=app_row.id))
    return render_template("candidate/apply.html", job=job, form=form)


@bp.route("/applications/<int:app_id>")
@candidate_only
def application(app_id):
    app_row = db.get_or_404(Application, app_id)
    if app_row.candidate_id != current_user.id:
        abort(404)
    return render_template("candidate/application.html", a=app_row)
