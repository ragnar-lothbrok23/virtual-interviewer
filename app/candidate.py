from flask import Blueprint, abort, flash, jsonify, redirect, render_template, url_for
from flask_login import current_user

from .extensions import db, limiter
from .forms import ApplyForm, WithdrawForm
from .main import role_required
from .models import Application, Job, User, utcnow
from .services import storage
from .services.analysis import UploadError, attach_resume, prepare_upload, queue_analysis
from .services.security import audit

bp = Blueprint("candidate", __name__, url_prefix="/candidate")
candidate_only = role_required(User.ROLE_CANDIDATE)


def _own_application(app_id):
    app_row = db.get_or_404(Application, app_id)
    if app_row.candidate_id != current_user.id:
        abort(404)
    return app_row


@bp.route("/")
@candidate_only
def dashboard():
    applications = (Application.query.filter_by(candidate_id=current_user.id)
                    .order_by(Application.created_at.desc()).all())
    return render_template("candidate/dashboard.html", applications=applications)


@bp.route("/jobs")
@candidate_only
def jobs():
    open_jobs = [j for j in Job.query.filter_by(is_open=True).order_by(Job.created_at.desc()) if j.is_accepting]
    applied = {a.job_id: a for a in current_user.applications}
    return render_template("candidate/jobs.html", jobs=open_jobs, applied=applied)


@bp.route("/jobs/<int:job_id>/apply", methods=["GET", "POST"])
@candidate_only
@limiter.limit("20 per hour", methods=["POST"])
def apply(job_id):
    job = db.get_or_404(Job, job_id)
    existing = Application.query.filter_by(candidate_id=current_user.id, job_id=job.id).first()
    if existing and not existing.is_withdrawn:
        return redirect(url_for("candidate.application", app_id=existing.id))
    if not job.is_accepting:
        flash("This job is no longer accepting applications.", "warning")
        return redirect(url_for("candidate.jobs"))
    if not current_user.is_verified:
        flash("Please verify your email address before applying. Check your inbox, or resend the link below.", "warning")
        return redirect(url_for("candidate.dashboard"))

    form = ApplyForm()
    if form.validate_on_submit():
        try:
            upload = prepare_upload(form.resume.data)
        except UploadError as e:
            form.resume.errors.append(str(e))
            return render_template("candidate/apply.html", job=job, form=form)

        if existing:  # re-applying after withdrawing: reuse the row (one application per job)
            app_row = existing
            app_row.status, app_row.status_source, app_row.withdrawn_at = (
                Application.STATUS_REVIEW, Application.SOURCE_AUTO, None)
            app_row.parsed = app_row.embedding = app_row.similarity = app_row.match_score = None
        else:
            app_row = Application(candidate=current_user, job=job, status=Application.STATUS_REVIEW,
                                  status_source=Application.SOURCE_AUTO)
            db.session.add(app_row)
        old_file = attach_resume(app_row, upload)
        db.session.flush()
        audit("application.submitted", target=current_user, application_id=app_row.id, job=job.title,
              reapplied=bool(existing))
        db.session.commit()
        if old_file:
            storage.delete_resume(old_file)
        queue_analysis(app_row)
        flash("Your application was submitted. We're analysing your resume now.", "success")
        return redirect(url_for("candidate.application", app_id=app_row.id))
    return render_template("candidate/apply.html", job=job, form=form)


@bp.route("/applications/<int:app_id>")
@candidate_only
def application(app_id):
    app_row = _own_application(app_id)
    return render_template("candidate/application.html", a=app_row, upload_form=ApplyForm(),
                           withdraw_form=WithdrawForm())


@bp.route("/applications/<int:app_id>/status")
@candidate_only
@limiter.exempt
def application_status(app_id):
    """Polled by the application page while the resume is being analysed."""
    a = _own_application(app_id)
    return jsonify(analysis_status=a.analysis_status, status=a.status, match_score=a.match_score,
                   error=a.analysis_error if a.analysis_status == Application.ANALYSIS_FAILED else None)


@bp.post("/applications/<int:app_id>/resume")
@candidate_only
@limiter.limit("10 per hour")
def update_resume(app_id):
    app_row = _own_application(app_id)
    if not app_row.can_update_resume:
        flash("You can't update your resume for this application right now.", "warning")
        return redirect(url_for("candidate.application", app_id=app_row.id))
    form = ApplyForm()
    if not form.validate_on_submit():
        for error in form.resume.errors:
            flash(error, "danger")
        return redirect(url_for("candidate.application", app_id=app_row.id))
    try:
        upload = prepare_upload(form.resume.data)
    except UploadError as e:
        flash(str(e), "danger")
        return redirect(url_for("candidate.application", app_id=app_row.id))
    old_file = attach_resume(app_row, upload)
    audit("application.resume_updated", target=current_user, application_id=app_row.id, job=app_row.job.title)
    db.session.commit()
    storage.delete_resume(old_file)
    queue_analysis(app_row)
    flash("Your new resume was uploaded and is being analysed.", "success")
    return redirect(url_for("candidate.application", app_id=app_row.id))


@bp.post("/applications/<int:app_id>/retry")
@candidate_only
@limiter.limit("5 per hour")
def retry_analysis(app_id):
    app_row = _own_application(app_id)
    if app_row.analysis_status != Application.ANALYSIS_FAILED:
        return redirect(url_for("candidate.application", app_id=app_row.id))
    app_row.analysis_status, app_row.analysis_attempts = Application.ANALYSIS_PENDING, 0
    db.session.commit()
    queue_analysis(app_row)
    flash("We're trying to analyse your resume again.", "info")
    return redirect(url_for("candidate.application", app_id=app_row.id))


@bp.post("/applications/<int:app_id>/withdraw")
@candidate_only
def withdraw(app_id):
    app_row = _own_application(app_id)
    form = WithdrawForm()
    if app_row.is_withdrawn or app_row.status == Application.STATUS_REJECTED:
        flash("This application can't be withdrawn.", "warning")
    elif form.validate_on_submit():
        app_row.status, app_row.withdrawn_at = Application.STATUS_WITHDRAWN, utcnow()
        audit("application.withdrawn", target=current_user, application_id=app_row.id, job=app_row.job.title)
        db.session.commit()
        flash("Your application has been withdrawn. You can apply again while the job is open.", "info")
    else:
        flash("Please tick the box to confirm you want to withdraw.", "warning")
    return redirect(url_for("candidate.application", app_id=app_row.id))
