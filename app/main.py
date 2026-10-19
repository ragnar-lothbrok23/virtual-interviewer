from functools import wraps

from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from flask_wtf.csrf import CSRFError
from sqlalchemy import text

from .extensions import csrf, db, limiter

bp = Blueprint("main", __name__)


def role_required(*roles):
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            if current_user.role not in roles:
                abort(403)
            return view(*args, **kwargs)

        return wrapped

    return decorator


# Pages a candidate may use before accepting the current privacy policy
CONSENT_EXEMPT = {"auth.consent", "auth.logout", "auth.verify_email", "main.privacy", "main.terms", "main.healthz",
                  "static", "account.settings", "account.export_data", "account.delete_account",
                  "account.confirm_email_change"}


@bp.before_app_request
def require_current_consent():
    """Candidates must accept the current privacy policy (re-asked whenever its version changes)."""
    if (current_user.is_authenticated and current_user.is_candidate and request.endpoint not in CONSENT_EXEMPT
            and not current_user.has_current_consent(current_app.config["PRIVACY_POLICY_VERSION"])):
        return redirect(url_for("auth.consent", next=request.full_path if request.method == "GET" else None))


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/dashboard")
@login_required
def dashboard():
    if current_user.is_admin:
        return redirect(url_for("admin.users"))
    return redirect(url_for("hr.dashboard" if current_user.is_hr else "candidate.dashboard"))


@bp.route("/privacy")
def privacy():
    return render_template("legal/privacy.html", version=current_app.config["PRIVACY_POLICY_VERSION"])


@bp.route("/terms")
def terms():
    return render_template("legal/terms.html", version=current_app.config["PRIVACY_POLICY_VERSION"])


@bp.route("/favicon.ico")
@limiter.exempt
def favicon():
    """Browsers request /favicon.ico regardless of the <link rel="icon"> tag."""
    return redirect(url_for("static", filename="favicon.svg"), code=301)


@bp.route("/healthz")
@limiter.exempt
@csrf.exempt
def healthz():
    """Liveness/readiness probe for load balancers and monitoring."""
    try:
        db.session.execute(text("SELECT 1"))
        return jsonify(status="ok", database="ok"), 200
    except Exception as e:
        current_app.logger.error("Health check failed: %s", e)
        return jsonify(status="error", database="unreachable"), 503


def _error(code, message):
    if request.accept_mimetypes.best == "application/json":
        return jsonify(error=message), code
    return render_template("error.html", code=code, message=message), code


@bp.app_errorhandler(400)
def bad_request(_):
    return _error(400, "The request could not be understood.")


@bp.app_errorhandler(CSRFError)
def csrf_error(_):
    return _error(400, "This form expired or was submitted from another site. Please go back, refresh the page and try again.")


@bp.app_errorhandler(403)
def forbidden(_):
    return _error(403, "You don't have access to this page.")


@bp.app_errorhandler(404)
def not_found(_):
    return _error(404, "Page not found.")


@bp.app_errorhandler(405)
def method_not_allowed(_):
    return _error(405, "That action isn't allowed here.")


@bp.app_errorhandler(413)
def too_large(_):
    return _error(413, "That file is too large. The limit is 5 MB.")


@bp.app_errorhandler(429)
def too_many_requests(_):
    return _error(429, "Too many attempts. Please wait a minute and try again.")


@bp.app_errorhandler(500)
def server_error(e):
    db.session.rollback()
    current_app.logger.exception("Unhandled error on %s %s", request.method, request.path)
    return _error(500, "Something went wrong on our side. Please try again.")
