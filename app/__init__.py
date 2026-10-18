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
    from .auth import bp as auth_bp
    from .candidate import bp as candidate_bp
    from .hr import bp as hr_bp
    from .main import bp as main_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(candidate_bp)
    app.register_blueprint(hr_bp)

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
    @app.cli.command("create-hr")
    @click.option("--name", prompt=True)
    @click.option("--email", prompt=True)
    @click.password_option()
    def create_hr(name, email, password):
        """Create an HR (recruiter) account. Candidates sign up themselves."""
        from .models import User

        name, email = name.strip(), email.strip().lower()
        if not name or "@" not in email:
            raise click.ClickException("A name and a valid email are required.")
        if len(password) < 8:
            raise click.ClickException("Password must be at least 8 characters.")
        if User.query.filter_by(email=email).first():
            raise click.ClickException(f"An account with {email} already exists.")
        user = User(name=name, email=email, role=User.ROLE_HR)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        click.echo(f"HR account created for {email}.")

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
