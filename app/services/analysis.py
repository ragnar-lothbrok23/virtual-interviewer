"""Background resume analysis and re-scoring (paper Section 3.1), shared by the candidate, HR and task code."""
from pathlib import Path

from flask import current_app
from werkzeug.utils import secure_filename

from ..extensions import db
from ..models import Application, utcnow
from . import resume as resume_service
from . import storage
from .gemini import GeminiError
from .notifications import notify, notify_application_received, notify_application_status
from .scheduling import sync_with_status
from .vectors import VectorError


class UploadError(ValueError):
    """Problem with the uploaded file itself; the message is safe to show to the candidate."""


def prepare_upload(file_storage):
    """Validate an uploaded resume and extract its text. Returns dict(data, kind, text, original_name, sha256)."""
    data = file_storage.read()
    original = secure_filename(file_storage.filename or "") or "resume"
    try:
        kind = storage.detect_resume_type(data, original)
        text = resume_service.extract_text(data, kind)
    except (storage.FileTypeError, resume_service.ResumeError) as e:
        raise UploadError(str(e)) from e
    return {"data": data, "kind": kind, "text": text, "original_name": original, "sha256": storage.sha256(data)}


def attach_resume(application, upload):
    """Store the (encrypted) file on the application and reset it for a fresh analysis. Returns the old file name."""
    old_file = application.resume_path
    application.resume_path = storage.save_resume(upload["data"], upload["kind"])
    application.resume_original_name = upload["original_name"]
    application.resume_kind = upload["kind"]
    application.resume_sha256 = upload["sha256"]
    application.resume_text = upload["text"]
    application.analysis_status = Application.ANALYSIS_PENDING
    application.analysis_error = None
    application.analysis_attempts = 0
    return old_file


def queue_analysis(application):
    """Send the application to the background worker (runs immediately in tests)."""
    from ..tasks import analyse_application

    analyse_application(application.id)


def apply_auto_status(application):
    """Auto-shortlist by match score, unless HR has already decided (HR decisions are never overwritten)."""
    if application.status_source != Application.SOURCE_AUTO or application.is_withdrawn:
        return
    threshold = current_app.config["SHORTLIST_THRESHOLD"]
    application.status = (Application.STATUS_SHORTLISTED if application.match_score >= threshold
                          else Application.STATUS_REVIEW)


def run_analysis(application_id, retries_left=0):
    """Analyse one application. Raises GeminiError for a retryable failure while retries remain;
    otherwise records the failure on the application. Returns the final analysis status."""
    application = db.session.get(Application, application_id)
    if application is None or application.is_withdrawn:
        return None
    first_time = application.parsed is None
    old_status = application.status
    application.analysis_status = Application.ANALYSIS_RUNNING
    application.analysis_attempts += 1
    db.session.commit()

    try:
        profile, embedding, similarity, score = resume_service.analyse(application.resume_text, application.job)
    except GeminiError as e:
        if e.retryable and retries_left > 0:
            application.analysis_status = Application.ANALYSIS_PENDING
            application.analysis_error = f"{e} (retrying automatically)"
            db.session.commit()
            raise
        application.analysis_status = Application.ANALYSIS_FAILED
        application.analysis_error = str(e)
        db.session.commit()
        current_app.logger.error("Analysis of application %s failed: %s", application.id, e)
        notify(application.candidate, "application.analysis_failed",
               f"We couldn't analyse your resume for {application.job.title}",
               "The AI service was unavailable. You can try again from your application page.",
               f"/candidate/applications/{application.id}")
        return application.analysis_status

    application.parsed = profile
    application.embedding = embedding
    application.similarity = similarity
    application.match_score = score
    application.analysis_status = Application.ANALYSIS_DONE
    application.analysis_error = None
    application.analysed_at = utcnow()
    apply_auto_status(application)
    db.session.commit()

    if first_time:
        notify_application_received(application)
    else:
        notify_application_status(application, old_status)
        sync_with_status(application)
        from .resume_questions import queue_if_booked

        queue_if_booked(application)  # the resume was re-analysed: an upcoming interview needs a fresh set
    return application.analysis_status


def rescore_job(job_id, skills_changed):
    """After HR edits a job: if the required skills changed, every active application is fully re-analysed
    (skill evidence depends on the skills); otherwise only the similarity is recomputed from stored embeddings."""
    from ..models import Job

    job = db.session.get(Job, job_id)
    if job is None:
        return 0
    active = [a for a in job.applications if not a.is_withdrawn]
    if skills_changed:
        for a in active:
            a.analysis_status = Application.ANALYSIS_PENDING
            a.analysis_attempts = 0
        db.session.commit()
        for a in active:
            queue_analysis(a)
        return len(active)

    count = 0
    for a in active:
        if a.embedding is None or not a.is_analysed:
            continue
        old_status = a.status
        try:
            a.similarity, a.match_score = resume_service.score_against(a.embedding, job)
        except VectorError as e:  # unusable stored embedding: analyse this application again from scratch
            current_app.logger.warning("Application %s needs re-analysis: %s", a.id, e)
            a.analysis_status, a.analysis_attempts = Application.ANALYSIS_PENDING, 0
            db.session.commit()
            queue_analysis(a)
            continue
        apply_auto_status(a)
        db.session.commit()
        notify_application_status(a, old_status)
        sync_with_status(a)
        count += 1
    db.session.commit()
    return count


def resume_file_name(application):
    """Download name: the candidate's original file name, with the verified extension."""
    stem = Path(application.resume_original_name).stem or "resume"
    return f"{stem}.{application.resume_kind or Path(application.resume_original_name).suffix.lstrip('.') or 'pdf'}"
