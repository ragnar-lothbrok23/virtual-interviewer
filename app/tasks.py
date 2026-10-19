"""Background jobs (Huey).

Run the worker with:   venv\\Scripts\\huey_consumer app.tasks.huey -k thread -w 2
Queue backend: Redis when REDIS_URL is set (production), otherwise a SQLite file (works natively on Windows).
In testing (APP_ENV=testing) jobs run immediately in-process.
"""
import os
import socket
from datetime import datetime, timezone

from flask import current_app, has_app_context
from huey import MemoryHuey, RedisHuey, SqliteHuey, crontab

from .config import get_config


def _make_huey():
    config = get_config()
    if config.ENV_NAME == "testing":
        return MemoryHuey("virtual-interviewer", immediate=True)
    if config.REDIS_URL:
        return RedisHuey("virtual-interviewer", url=config.REDIS_URL)
    config.HUEY_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return SqliteHuey("virtual-interviewer", filename=str(config.HUEY_SQLITE_PATH))


huey = _make_huey()
_worker_app = None


def app_context():
    """Jobs need the Flask app (database, config). Reuse the current one, or build one once per worker process."""
    global _worker_app
    if has_app_context():
        return current_app.app_context()
    if _worker_app is None:
        from . import create_app

        _worker_app = create_app()
    return _worker_app.app_context()


@huey.task(retries=3, retry_delay=60, context=True)
def deliver_email(message, task=None):
    """Send one email and record the outcome in EmailLog.

    Temporary failures are retried up to 3 times, a minute apart; permanent ones (bad credentials, rejected
    recipient) fail immediately. Mail over the daily Gmail limit is deferred by MAIL_DEFER_SECONDS."""
    from .extensions import db
    from .models import EmailLog, utcnow
    from .services.mail import PermanentMailError, deliver, over_daily_limit

    with app_context():
        log = db.session.get(EmailLog, message.get("log_id")) if message.get("log_id") else None
        retries_left = task.retries if task is not None else 0

        def record(status, error=None):
            if log is not None:
                log.status, log.last_error = status, (error[:2000] if error else None)
                if status == EmailLog.STATUS_SENT:
                    log.sent_at = utcnow()
                db.session.commit()

        if over_daily_limit():
            delay = current_app.config["MAIL_DEFER_SECONDS"]
            record(EmailLog.STATUS_DEFERRED, f"Daily sending limit reached; retrying in {delay // 60} minutes")
            current_app.logger.warning("Email to %s deferred: daily limit reached", message["to"])
            deliver_email.schedule(args=(message,), delay=delay)
            return "deferred"

        if log is not None:
            log.attempts += 1
            db.session.commit()
        try:
            deliver(message)
        except PermanentMailError as e:
            current_app.logger.error("Email to %s failed permanently: %s", message["to"], e)
            record(EmailLog.STATUS_FAILED, str(e))
            return "failed"
        except Exception as e:
            if retries_left > 0:
                current_app.logger.warning("Email to %s failed (%s); %d retries left", message["to"], e, retries_left)
                record(EmailLog.STATUS_RETRYING, str(e))
                raise  # Huey schedules the retry
            current_app.logger.error("Email to %s failed after all retries: %s", message["to"], e)
            record(EmailLog.STATUS_FAILED, str(e))
            return "failed"
        record(EmailLog.STATUS_SENT)
        return "sent"


_cfg = get_config()


@huey.task(retries=_cfg.ANALYSIS_RETRIES, retry_delay=_cfg.ANALYSIS_RETRY_DELAY, context=True)
def analyse_application(application_id, task=None):
    """Parse and score one resume with Gemini. Temporary AI errors are retried; others fail visibly."""
    from .services.analysis import run_analysis

    with app_context():
        return run_analysis(application_id, retries_left=task.retries if task is not None else 0)


@huey.task(retries=_cfg.ANALYSIS_RETRIES, retry_delay=_cfg.ANALYSIS_RETRY_DELAY)
def rescore_job_applications(job_id, skills_changed):
    """Re-score a job's applicants after HR edits the job."""
    from .services.analysis import rescore_job

    with app_context():
        return rescore_job(job_id, skills_changed)


@huey.task(retries=_cfg.ANALYSIS_RETRIES, retry_delay=_cfg.ANALYSIS_RETRY_DELAY, context=True)
def transcribe_answer(answer_id, task=None):
    """Transcribe one recorded interview answer with Gemini. Temporary AI errors are retried."""
    from .services.transcription import run_transcription

    with app_context():
        return run_transcription(answer_id, retries_left=task.retries if task is not None else 0)


@huey.periodic_task(crontab(minute="*/5"))
def interview_housekeeping():
    """Every 5 minutes: send due interview reminders, mark missed interviews, close abandoned ones."""
    from .services.interview_room import expire_stale
    from .services.scheduling import mark_no_shows, send_due_reminders

    with app_context():
        return {"reminders": send_due_reminders(), "no_shows": mark_no_shows(), "timed_out": expire_stale()}


@huey.task()
def ping(message):
    """Health check job: proves a worker is consuming the queue."""
    with app_context():
        return {"message": message, "worker": socket.gethostname(), "pid": os.getpid(),
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
