import io
from datetime import timedelta
from urllib.parse import urlsplit

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_file, url_for
from flask_login import current_user
from sqlalchemy import case

from .extensions import db, limiter
from .forms import BulkSlotForm, CancelForm, JobForm, QuestionForm, SlotForm
from .main import role_required
from .models import Application, Interview, InterviewAnswer, InterviewQuestion, InterviewSlot, Job, User, utcnow
from .services import emotion, interview_room, jd_questions, resume_questions, scheduling, scoring, storage
from .services.analysis import queue_analysis, resume_file_name
from .services.notifications import notify_application_status
from .services.resume import skill_summary
from .services.scheduling import sync_with_status
from .services.security import audit
from .services.timeutil import format_local, from_local_naive, to_local

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
                      description=form.description.data.strip(), closes_at=closes_at,
                      resume_question_count=current_app.config["RESUME_QUESTIONS_DEFAULT"])
            db.session.add(job)
            db.session.flush()
            audit("job.created", job_id=job.id, title=job.title)
            db.session.commit()
            queued = jd_questions.safe_queue(job)
            flash(f"Job “{job.title}” created." + (" Technical questions are being written from the job "
                                                   "description; review them under Questions." if queued else ""),
                  "success")
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
            if content_changed:
                jd_questions.safe_queue(job)  # JD-based questions follow the description (HR-edited ones are kept)
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
    nlp = scoring.interview_nlp_summary(app_row.interview) if app_row.interview else None
    emo = emotion.interview_emotion_summary(app_row.interview) if app_row.interview else None
    rq = None
    if resume_questions.count_for(app_row.job):
        _, reason = resume_questions.plan_entries(app_row)
        rq = {"count": resume_questions.count_for(app_row.job), "ready": resume_questions.is_ready(app_row),
              "reason": reason, "can_prepare": resume_questions.wanted(app_row),
              "labels": resume_questions.SOURCE_LABELS}
    return render_template("hr/application.html", a=app_row, labels=Application.STATUS_LABELS,
                           hr_statuses=Application.HR_STATUSES, summary=summary, nlp=nlp, rq=rq, emo=emo)


@bp.post("/applications/<int:app_id>/resume-questions")
@hr_only
def prepare_resume_questions(app_id):
    app_row = db.get_or_404(Application, app_id)
    if not resume_questions.wanted(app_row):
        flash("Resume-based questions aren't available for this application.", "warning")
    else:
        resume_questions.queue(app_row, force=True)
        audit("application.resume_questions_requested", target=app_row.candidate, application_id=app_row.id)
        db.session.commit()
        flash("Preparing new resume-based questions (one AI request). Interviews that already started keep "
              "their questions.", "success")
    return redirect(url_for("hr.application", app_id=app_row.id))


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
    if sync_with_status(app_row):
        flash("Their scheduled interview was cancelled and the candidate informed.", "info")
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


# ---------- interview slots and interviews ----------

@bp.route("/jobs/<int:job_id>/slots", methods=["GET", "POST"])
@hr_only
def slots(job_id):
    job = db.get_or_404(Job, job_id)
    single, bulk = SlotForm(prefix="one"), BulkSlotForm(prefix="bulk")
    actor = current_user._get_current_object()
    if single.submit_single.data and single.validate_on_submit():
        try:
            slot = scheduling.create_slot(job, from_local_naive(single.starts_at.data), single.duration.data,
                                          single.capacity.data, actor)
            db.session.flush()
            audit("slot.created", slot_id=slot.id, job=job.title, starts_at=slot.starts_at.isoformat())
            db.session.commit()
            flash(f"Slot added for {format_local(slot.starts_at)}.", "success")
            return redirect(url_for("hr.slots", job_id=job.id))
        except scheduling.SchedulingError as e:
            db.session.rollback()
            single.starts_at.errors.append(str(e))
    elif bulk.submit_bulk.data and bulk.validate_on_submit():
        try:
            starts = scheduling.plan_bulk_slots(bulk.first_day.data, bulk.last_day.data, set(bulk.weekdays.data),
                                                bulk.day_start.data, bulk.day_end.data, bulk.duration.data, bulk.gap.data)
            created, skipped = scheduling.create_bulk_slots(job, starts, bulk.duration.data, bulk.capacity.data, actor)
            audit("slot.bulk_created", job=job.title, created=created, skipped=skipped)
            db.session.commit()
            note = f" ({skipped} skipped: already existed or in the past)" if skipped else ""
            flash(f"Created {created} slot(s){note}.", "success" if created else "warning")
            return redirect(url_for("hr.slots", job_id=job.id))
        except scheduling.SchedulingError as e:
            db.session.rollback()
            bulk.first_day.errors.append(str(e))
    upcoming = (job.slots.filter(InterviewSlot.starts_at >= utcnow() - timedelta(hours=12))
                .order_by(InterviewSlot.starts_at).all())
    return render_template("hr/slots.html", job=job, slots=upcoming, single=single, bulk=bulk,
                           cancel_form=CancelForm())


@bp.post("/slots/<int:slot_id>/cancel")
@hr_only
def cancel_slot(slot_id):
    slot = db.get_or_404(InterviewSlot, slot_id)
    form = CancelForm()
    if form.validate_on_submit():
        affected = scheduling.cancel_slot(slot, current_user._get_current_object(), form.reason.data or "")
        flash(f"Slot cancelled. {affected} candidate(s) were told and can book another time."
              if affected else "Slot cancelled.", "info")
    return redirect(url_for("hr.slots", job_id=slot.job_id))


@bp.post("/interviews/<int:interview_id>/cancel")
@hr_only
def cancel_interview(interview_id):
    interview = db.get_or_404(Interview, interview_id)
    form = CancelForm()
    if form.validate_on_submit() and scheduling.cancel(interview, by="hr", reason=form.reason.data or ""):
        flash("Interview cancelled. The candidate was told and can book another time.", "info")
    return redirect(url_for("hr.application", app_id=interview.application_id))


@bp.route("/interviews")
@hr_only
def interviews():
    view = request.args.get("view", "upcoming")
    query = Interview.query
    now = utcnow()
    if view == "upcoming":
        query = query.filter(Interview.status.in_(Interview.ACTIVE), Interview.starts_at >= now - timedelta(hours=3))
        query = query.order_by(Interview.starts_at)
    elif view in Interview.STATUS_LABELS:
        query = query.filter(Interview.status == view).order_by(Interview.starts_at.desc())
    else:
        query = query.order_by(Interview.starts_at.desc())
    page = query.paginate(per_page=50, error_out=False)
    return render_template("hr/interviews.html", page=page, view=view, labels=Interview.STATUS_LABELS)


# ---------- interview questions ----------

MAX_QUESTIONS = 15


@bp.route("/jobs/<int:job_id>/questions", methods=["GET", "POST"])
@hr_only
def questions(job_id):
    job = db.get_or_404(Job, job_id)
    form = QuestionForm()
    if form.validate_on_submit():
        if len(job.questions) >= MAX_QUESTIONS:
            flash(f"A job can have at most {MAX_QUESTIONS} questions.", "warning")
        else:
            db.session.add(InterviewQuestion(job_id=job.id, position=len(job.questions), text=form.text.data.strip(),
                                             category=form.category.data, answer_seconds=form.answer_seconds.data,
                                             reference_answer=(form.reference_answer.data or "").strip() or None))
            audit("question.added", job=job.title)
            db.session.commit()
            interview_room.queue_missing_references(job)
            flash("Question added.", "success")
        return redirect(url_for("hr.questions", job_id=job.id))
    edit_forms = {q.id: QuestionForm(prefix=f"q{q.id}", obj=q) for q in job.questions}
    return render_template("hr/questions.html", job=job, form=form, edit_forms=edit_forms, max_questions=MAX_QUESTIONS)


def _question(question_id):
    return db.get_or_404(InterviewQuestion, question_id)


@bp.post("/questions/<int:question_id>/edit")
@hr_only
def edit_question(question_id):
    q = _question(question_id)
    form = QuestionForm(prefix=f"q{q.id}")
    if form.validate_on_submit():
        new_text = form.text.data.strip()
        new_reference = (form.reference_answer.data or "").strip() or None
        if new_text != q.text and new_reference == q.reference_answer:
            new_reference = None  # the question changed: its old reference answer no longer fits
        if new_text != q.text or new_reference != q.reference_answer or form.category.data != q.category:
            q.origin = None  # edited by HR: it's theirs now and is never replaced by regeneration
        q.text, q.category, q.answer_seconds = new_text, form.category.data, form.answer_seconds.data
        q.reference_answer = new_reference
        audit("question.edited", job=q.job.title, question_id=q.id)
        db.session.commit()
        interview_room.queue_missing_references(q.job)
        flash("Question saved. Interviews that already started keep their original questions.", "success")
    else:
        flash(" ".join(e for errors in form.errors.values() for e in errors), "danger")
    return redirect(url_for("hr.questions", job_id=q.job_id))


@bp.post("/questions/<int:question_id>/delete")
@hr_only
def delete_question(question_id):
    q = _question(question_id)
    job = q.job
    db.session.delete(q)
    db.session.flush()
    for i, other in enumerate(InterviewQuestion.query.filter_by(job_id=job.id).order_by(InterviewQuestion.position)):
        other.position = i
    audit("question.deleted", job=job.title)
    db.session.commit()
    flash("Question deleted.", "info")
    return redirect(url_for("hr.questions", job_id=job.id))


@bp.post("/questions/<int:question_id>/move")
@hr_only
def move_question(question_id):
    q = _question(question_id)
    ordered = InterviewQuestion.query.filter_by(job_id=q.job_id).order_by(InterviewQuestion.position).all()
    i = ordered.index(q)
    j = i - 1 if request.form.get("direction") == "up" else i + 1
    if 0 <= j < len(ordered):
        ordered[i], ordered[j] = ordered[j], ordered[i]
        for pos, item in enumerate(ordered):
            item.position = pos
        db.session.commit()
    return redirect(url_for("hr.questions", job_id=q.job_id))


@bp.post("/jobs/<int:job_id>/resume-question-count")
@hr_only
def resume_question_count(job_id):
    job = db.get_or_404(Job, job_id)
    try:
        count = int(request.form.get("count", ""))
    except ValueError:
        count = -1
    maximum = current_app.config["RESUME_QUESTIONS_MAX"]
    if not 0 <= count <= maximum:
        flash(f"Choose between 0 and {maximum} resume-based questions.", "warning")
    elif count != job.resume_question_count:
        old, job.resume_question_count = job.resume_question_count, count
        audit("job.resume_questions_changed", job_id=job.id, old=old, new=count)
        db.session.commit()
        flash("Saved. Upcoming interviews get an updated set automatically; interviews that already started keep "
              "their questions." if count else "Saved. Candidates will get the job's questions only.", "success")
    return redirect(url_for("hr.questions", job_id=job.id))


@bp.post("/jobs/<int:job_id>/jd-question-count")
@hr_only
def jd_question_count(job_id):
    job = db.get_or_404(Job, job_id)
    try:
        count = int(request.form.get("count", ""))
    except ValueError:
        count = -1
    maximum = current_app.config["JD_QUESTIONS_MAX"]
    if not 0 <= count <= maximum:
        flash(f"Choose between 0 and {maximum} technical questions from the job description.", "warning")
    elif count != job.jd_question_count:
        old, job.jd_question_count = job.jd_question_count, count
        audit("job.jd_questions_changed", job_id=job.id, old=old, new=count)
        db.session.commit()
        jd_questions.safe_queue(job)
        flash("Saved. The technical questions are being updated from the job description (questions you edited "
              "are kept)." if count else "Saved. Questions generated from the job description were removed "
              "(questions you edited are kept).", "success")
    return redirect(url_for("hr.questions", job_id=job.id))


@bp.post("/jobs/<int:job_id>/jd-questions")
@hr_only
def regenerate_jd_questions(job_id):
    job = db.get_or_404(Job, job_id)
    if not jd_questions.count_for(job):
        flash("Choose how many technical questions to generate first.", "warning")
    else:
        jd_questions.safe_queue(job, force=True)
        audit("job.jd_questions_requested", job_id=job.id)
        db.session.commit()
        flash("Writing new technical questions from the job description (one AI request). Questions you edited are "
              "kept; interviews that already started keep their questions.", "success")
    return redirect(url_for("hr.questions", job_id=job.id))


@bp.post("/jobs/<int:job_id>/questions/defaults")
@hr_only
def default_questions(job_id):
    job = db.get_or_404(Job, job_id)
    if job.questions:
        flash("This job already has questions.", "info")
    else:
        interview_room.ensure_questions(job)
        audit("question.defaults_created", job=job.title)
        db.session.commit()
        interview_room.queue_missing_references(job)
        flash("Default questions created from the job's required skills. Review and edit them.", "success")
    return redirect(url_for("hr.questions", job_id=job.id))


# ---------- interview recordings ----------

@bp.route("/answers/<int:answer_id>/media")
@hr_only
def answer_media(answer_id):
    answer = db.get_or_404(InterviewAnswer, answer_id)
    if not answer.media_path:
        abort(404)
    try:
        data = storage.read_resume(answer.media_path)
    except storage.StorageError as e:
        current_app.logger.error("Could not read recording %s: %s", answer.id, e)
        abort(404)
    # conditional=True answers Range requests, so the video player can seek
    return send_file(io.BytesIO(data), mimetype=answer.media_type or "video/webm", conditional=True,
                     download_name=f"answer-{answer.position + 1}.{'mp4' if answer.media_type == 'video/mp4' else 'webm'}")


@bp.post("/answers/<int:answer_id>/emotion")
@hr_only
@limiter.limit("60 per hour")
def reanalyse_emotion(answer_id):
    answer = db.get_or_404(InterviewAnswer, answer_id)
    if not answer.media_path:
        flash("This answer has no recording.", "warning")
    elif not current_app.config["EMOTION_ENABLED"]:
        flash("Facial expression analysis is switched off on this server.", "warning")
    else:
        audit("answer.emotion_requested", target=answer.interview.application.candidate, answer_id=answer.id)
        interview_room.queue_emotion(answer)
        flash("The facial expressions are being analysed again.", "info")
    return redirect(url_for("hr.application", app_id=answer.interview.application_id) + "#interview-results")


@bp.post("/answers/<int:answer_id>/rescore")
@hr_only
@limiter.limit("60 per hour")
def rescore_answer(answer_id):
    answer = db.get_or_404(InterviewAnswer, answer_id)
    if answer.transcript_status != InterviewAnswer.TRANSCRIPT_DONE:
        flash("This answer has no transcript to score yet.", "warning")
    else:
        answer.score_status, answer.score_error, answer.similarity = InterviewAnswer.SCORE_PENDING, None, None
        answer.score_requested_at = utcnow()
        audit("answer.rescore_requested", target=answer.interview.application.candidate, answer_id=answer.id)
        db.session.commit()
        from .tasks import score_answer

        score_answer(answer.id)
        flash("The answer is being scored again.", "info")
    return redirect(url_for("hr.application", app_id=answer.interview.application_id) + "#interview-results")
