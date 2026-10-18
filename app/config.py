"""Settings for each environment. Pick one with APP_ENV=development|testing|production (default: development)."""
import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
INSTANCE_DIR = BASE_DIR / "instance"
load_dotenv(BASE_DIR / ".env")

DEV_SECRET = "dev-only-insecure-secret"


def _bool(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def database_url(default):
    """Read DATABASE_URL, normalising postgres:// URLs (as given by most hosts) to the psycopg 3 driver."""
    url = os.getenv("DATABASE_URL", "").strip() or default
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


class BaseConfig:
    ENV_NAME = "base"
    SECRET_KEY = os.getenv("SECRET_KEY", DEV_SECRET)

    SQLALCHEMY_DATABASE_URI = database_url(f"sqlite:///{INSTANCE_DIR / 'interviewer.db'}")
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", BASE_DIR / "uploads"))
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB resume uploads
    SHORTLIST_THRESHOLD = float(os.getenv("SHORTLIST_THRESHOLD", "60"))  # match score (0-100) for auto-shortlisting

    # Sessions and cookies
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_SECURE = False
    REMEMBER_COOKIE_DURATION = timedelta(days=14)
    PERMANENT_SESSION_LIFETIME = timedelta(hours=12)
    WTF_CSRF_TIME_LIMIT = 4 * 3600  # forms stay valid for 4 hours

    # Security headers (flask-talisman)
    FORCE_HTTPS = False
    TRUST_PROXY = _bool("TRUST_PROXY")  # set when running behind a reverse proxy (Caddy/Nginx) that sets X-Forwarded-*

    # Rate limiting (Flask-Limiter)
    RATELIMIT_ENABLED = True
    RATELIMIT_STORAGE_URI = "memory://"
    RATELIMIT_HEADERS_ENABLED = True

    # Background jobs (Huey). Without REDIS_URL a SQLite-backed queue is used, which works natively on Windows.
    REDIS_URL = os.getenv("REDIS_URL", "").strip()
    HUEY_SQLITE_PATH = INSTANCE_DIR / "huey.db"

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2")
    GEMINI_PARSE_THINKING = os.getenv("GEMINI_PARSE_THINKING", "LOW")  # LOW / MEDIUM / HIGH

    # Public address of the site, used for links in emails sent from background jobs (no request available there)
    APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
    APP_NAME = "Virtual Interviewer"

    # Email. "console" logs messages and saves them to instance/outbox (development); Gmail backends come in Phase 3.
    MAIL_BACKEND = os.getenv("MAIL_BACKEND", "console")
    MAIL_OUTBOX_DIR = INSTANCE_DIR / "outbox"
    MAIL_USERNAME = os.getenv("MAIL_USERNAME", "")
    MAIL_APP_PASSWORD = os.getenv("MAIL_APP_PASSWORD", "")
    MAIL_SENDER_NAME = os.getenv("MAIL_SENDER_NAME", "Virtual Interviewer")

    # Accounts
    EMAIL_VERIFY_MAX_AGE = 48 * 3600      # seconds a verification link stays valid
    PASSWORD_RESET_MAX_AGE = 3600         # seconds a password reset link stays valid
    INVITE_MAX_AGE = 7 * 24 * 3600        # seconds an HR/admin invitation stays valid
    LOCKOUT_THRESHOLD = 5                 # failed logins before the account is locked
    LOCKOUT_MINUTES = 15
    PRIVACY_POLICY_VERSION = "2026-10-18"  # bump when the policy changes; candidates re-consent on next visit

    @classmethod
    def validate(cls):
        """Raise if the configuration is unsafe or incomplete for this environment."""


class DevelopmentConfig(BaseConfig):
    ENV_NAME = "development"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()


class TestingConfig(BaseConfig):
    ENV_NAME = "testing"
    TESTING = True
    SECRET_KEY = "testing-secret"
    SQLALCHEMY_DATABASE_URI = "sqlite://"  # in-memory; the test suite can swap in PostgreSQL (see tests/conftest.py)
    UPLOAD_FOLDER = INSTANCE_DIR / "test-uploads"
    HUEY_SQLITE_PATH = INSTANCE_DIR / "test-huey.db"
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = True  # same as production; counters are reset between tests
    REDIS_URL = ""
    LOG_LEVEL = "WARNING"
    GEMINI_API_KEY = "test-key-never-used"
    MAIL_BACKEND = "memory"
    APP_BASE_URL = "http://localhost"


class ProductionConfig(BaseConfig):
    ENV_NAME = "production"
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    FORCE_HTTPS = _bool("FORCE_HTTPS", True)
    RATELIMIT_STORAGE_URI = os.getenv("REDIS_URL", "").strip() or "memory://"

    @classmethod
    def validate(cls):
        problems = []
        if cls.SECRET_KEY in ("", DEV_SECRET, "change-me") or len(cls.SECRET_KEY) < 32:
            problems.append("SECRET_KEY must be set to a random value of at least 32 characters")
        if not os.getenv("DATABASE_URL"):
            problems.append("DATABASE_URL must be set (PostgreSQL)")
        if not cls.REDIS_URL:
            problems.append("REDIS_URL must be set (background jobs and rate limiting)")
        if not cls.GEMINI_API_KEY:
            problems.append("GEMINI_API_KEY must be set")
        if not cls.APP_BASE_URL.startswith("https://"):
            problems.append("APP_BASE_URL must be the public https:// address of the site")
        if cls.MAIL_BACKEND in ("console", "memory"):
            problems.append("MAIL_BACKEND must be a real email backend in production (Gmail)")
        if problems:
            raise RuntimeError("Invalid production configuration:\n  - " + "\n  - ".join(problems))


CONFIGS = {"development": DevelopmentConfig, "testing": TestingConfig, "production": ProductionConfig}


def get_config(name=None):
    name = (name or os.getenv("APP_ENV", "development")).strip().lower()
    if name not in CONFIGS:
        raise RuntimeError(f"Unknown APP_ENV '{name}'. Use one of: {', '.join(CONFIGS)}")
    return CONFIGS[name]


# Backwards-compatible alias
Config = DevelopmentConfig
