import logging
import sys

import click
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import get_config
from .extensions import csrf, db, limiter, login_manager, migrate, talisman

# Content Security Policy: only our own files plus the Bootstrap CDN; inline <script> needs the per-request nonce
CSP = {
    "default-src": "'self'",
    "script-src": ["'self'", "https://cdn.jsdelivr.net"],
    "style-src": ["'self'", "https://cdn.jsdelivr.net", "'unsafe-inline'"],  # Bootstrap progress bars use style=""
    "font-src": ["'self'", "https://cdn.jsdelivr.net"],
    "img-src": ["'self'", "data:", "blob:"],
    "media-src": ["'self'", "blob:"],
    "connect-src": "'self'",
    "frame-ancestors": "'none'",
    "form-action": "'self'",
    "base-uri": "'self'",
    "object-src": "'none'",
}
PERMISSIONS_POLICY = {"camera": "(self)", "microphone": "(self)", "geolocation": "()", "payment": "()"}


def create_app(config_class=None):
    config_class = config_class or get_config()
    config_class.validate()

    app = Flask(__name__)
    app.config.from_object(config_class)
    configure_logging(app)

    if app.config["TRUST_PROXY"]:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    for folder in (app.config["UPLOAD_FOLDER"], config_class.HUEY_SQLITE_PATH.parent):
        folder.mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    migrate.init_app(app, db, render_as_batch=True)  # batch mode lets SQLite handle ALTER TABLE
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)
    talisman.init_app(
        app,
        force_https=app.config["FORCE_HTTPS"],
        strict_transport_security=app.config["FORCE_HTTPS"],
        session_cookie_secure=app.config["SESSION_COOKIE_SECURE"],
        content_security_policy=CSP,
        content_security_policy_nonce_in=["script-src"],
        permissions_policy=PERMISSIONS_POLICY,
        referrer_policy="strict-origin-when-cross-origin",
        frame_options="DENY",
    )

    from . import models  # noqa: F401  (registers tables with SQLAlchemy)
    from .account import bp as account_bp
    from .admin import bp as admin_bp
    from .auth import bp as auth_bp
    from .candidate import bp as candidate_bp
    from .hr import bp as hr_bp
    from .main import bp as main_bp
    from .notifications import bp as notifications_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(candidate_bp)
    app.register_blueprint(hr_bp)
    app.register_blueprint(account_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(notifications_bp)

    from .services.timeutil import register_filters

    register_filters(app)
    register_cli(app)
    app.logger.info("App started (env=%s, db=%s)", config_class.ENV_NAME, db_kind(app))
    return app


def db_kind(app):
    uri = app.config["SQLALCHEMY_DATABASE_URI"]
    return "postgresql" if uri.startswith("postgresql") else uri.split(":", 1)[0]


def configure_logging(app):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s"))
    app.logger.handlers = [handler]
    app.logger.setLevel(app.config["LOG_LEVEL"])
    app.logger.propagate = False


def register_cli(app):
    def create_staff(role, name, email, password):
        from .models import User
        from .services.security import audit, password_problems

        name, email = name.strip(), email.strip().lower()
        if not name or "@" not in email:
            raise click.ClickException("A name and a valid email are required.")
        problems = password_problems(password, email=email, name=name)
        if problems:
            raise click.ClickException(" ".join(problems))
        if User.query.filter_by(email=email).first():
            raise click.ClickException(f"An account with {email} already exists.")
        user = User(name=name, email=email, role=role)
        user.set_password(password)
        user.mark_verified()  # created by someone with server access
        db.session.add(user)
        db.session.flush()
        audit("staff.created_cli", target=user, role=role)
        db.session.commit()
        click.echo(f"{user.role_label} account created for {email}.")

    @app.cli.command("create-admin")
    @click.option("--name", prompt=True)
    @click.option("--email", prompt=True)
    @click.password_option()
    def create_admin(name, email, password):
        """Create an Admin account (manages users; also has HR access). Use this for the first account."""
        from .models import User

        create_staff(User.ROLE_ADMIN, name, email, password)

    @app.cli.command("create-hr")
    @click.option("--name", prompt=True)
    @click.option("--email", prompt=True)
    @click.password_option()
    def create_hr(name, email, password):
        """Create an HR (recruiter) account. Admins can also invite HR users from the Admin panel."""
        from .models import User

        create_staff(User.ROLE_HR, name, email, password)

    @app.cli.command("seed-jobs")
    def seed_jobs():
        """Add three sample job roles for testing (skips ones that already exist)."""
        from .models import Job
        from .seed_data import SAMPLE_JOBS

        added = 0
        for data in SAMPLE_JOBS:
            if not Job.query.filter_by(title=data["title"]).first():
                db.session.add(Job(**data))
                added += 1
        db.session.commit()
        click.echo(f"Added {added} sample job(s).")

    @app.cli.command("mail-test")
    @click.option("--to", "to", required=True, help="Address to send the test email to.")
    def mail_test(to):
        """Send a test email right now (bypassing the queue) and report exactly what happened."""
        from .models import EmailLog, utcnow
        from .services.mail import build_message, deliver

        backend = app.config["MAIL_BACKEND"]
        message = build_message(to.strip(), f"Test email from {app.config['APP_NAME']}", "test_email",
                                sender=app.config["MAIL_USERNAME"] or "(not set)", backend=backend)
        log = EmailLog(to_address=message["to"], subject=message["subject"], template="test_email",
                       message_id=message["message_id"], backend=backend, attempts=1)
        db.session.add(log)
        try:
            deliver(message)
        except Exception as e:
            log.status, log.last_error = EmailLog.STATUS_FAILED, str(e)[:2000]
            db.session.commit()
            raise click.ClickException(f"Sending failed ({backend}): {e}")
        log.status, log.sent_at = EmailLog.STATUS_SENT, utcnow()
        db.session.commit()
        click.echo(f"Test email sent to {message['to']} via {backend}.")

    @app.cli.command("generate-encryption-key")
    @click.option("--key-id", default=None, help="Short id for the key (default: random).")
    def generate_encryption_key(key_id):
        """Print a new FILE_ENCRYPTION_KEYS entry. To rotate, put it FIRST and keep the old entries after it."""
        from .services.storage import generate_key_entry

        click.echo(generate_key_entry(key_id))

    @app.cli.command("encrypt-existing-resumes")
    def encrypt_existing_resumes():
        """Encrypt resume files uploaded before encryption existed (safe to run more than once)."""
        from pathlib import Path

        from .models import Application
        from .services import storage

        folder = Path(app.config["UPLOAD_FOLDER"])
        done = skipped = missing = 0
        for a in Application.query.all():
            path = folder / Path(a.resume_path).name
            if not path.exists():
                missing += 1
                continue
            blob = path.read_bytes()
            if storage.is_encrypted(blob):
                skipped += 1
                continue
            kind = a.resume_kind or path.suffix.lstrip(".").lower() or "pdf"
            a.resume_path = storage.save_resume(blob, kind)
            a.resume_kind = kind
            a.resume_sha256 = a.resume_sha256 or storage.sha256(blob)
            db.session.commit()
            path.unlink()
            done += 1
        click.echo(f"Encrypted {done} file(s); {skipped} already encrypted; {missing} missing.")

    @app.cli.command("dev-interview-soon")
    @click.argument("interview_id", type=int)
    @click.option("--minutes", default=5, show_default=True, help="Start this many minutes from now.")
    def dev_interview_soon(interview_id, minutes):
        """DEVELOPMENT ONLY: move a scheduled interview to start soon, so the interview room can be tried now."""
        from datetime import timedelta

        from .models import Interview, utcnow
        from .services.timeutil import format_local

        if app.config.get("ENV_NAME") == "production":
            raise click.ClickException("This command is disabled in production.")
        interview = db.session.get(Interview, interview_id)
        if interview is None or interview.status != Interview.STATUS_SCHEDULED:
            raise click.ClickException("No scheduled interview with that id.")
        interview.starts_at = (utcnow() + timedelta(minutes=minutes)).replace(second=0, microsecond=0)
        interview.reminder_24h_sent_at = interview.reminder_1h_sent_at = utcnow()  # don't fire reminders for it
        db.session.commit()
        click.echo(f"Interview {interview_id} now starts {format_local(interview.starts_at)}; the room opens "
                   f"{app.config['JOIN_EARLY_MINUTES']} minutes before that.")

    @app.cli.command("queue-check")
    @click.option("--timeout", default=15, help="Seconds to wait for the worker.")
    def queue_check(timeout):
        """Send a test job to the background worker and wait for its reply."""
        from .tasks import ping

        result = ping("queue-check")
        try:
            reply = result.get(blocking=True, timeout=timeout)
        except Exception:
            raise click.ClickException(
                f"No reply within {timeout}s. Is the worker running?  venv\\Scripts\\huey_consumer app.tasks.huey -k thread -w 2"
            )
        click.echo(f"Worker OK: {reply}")
