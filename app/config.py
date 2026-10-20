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
    MAX_DOCX_UNCOMPRESSED = 50 * 1024 * 1024  # reject "zip bombs" disguised as DOCX
    SHORTLIST_THRESHOLD = float(os.getenv("SHORTLIST_THRESHOLD", "60"))  # [paper] match score (0-100) to shortlist
    ANALYSIS_RETRIES = 3  # automatic retries of a failed resume analysis (temporary AI errors only)
    ANALYSIS_RETRY_DELAY = 30  # seconds

    # Stored resumes are encrypted with AES-256-GCM. Format: "<id>:<base64 32-byte key>", comma-separated;
    # the FIRST key encrypts new files, the others can still decrypt older files (key rotation).
    # Generate one:  venv\Scripts\flask --app run generate-encryption-key
    FILE_ENCRYPTION_KEYS = os.getenv("FILE_ENCRYPTION_KEYS", "").strip()

    # Times are stored in UTC and shown / entered in this timezone
    APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Kolkata")

    # Interview scheduling
    BOOKING_MIN_NOTICE_MINUTES = 60   # candidates can't book a slot starting sooner than this
    CHANGE_CUTOFF_MINUTES = 120       # reschedule / cancel allowed until this long before the start
    MAX_RESCHEDULES = 2               # per interview
    JOIN_EARLY_MINUTES = 10           # the join button opens this long before the start
    NO_SHOW_GRACE_MINUTES = 15        # after the slot ends + this, an interview never started is a no-show
    MAX_SLOTS_PER_BULK = 300          # safety limit for bulk slot creation

    # Interview room
    PREP_SECONDS = 20                 # reading time before recording starts automatically
    DEFAULT_ANSWER_SECONDS = 120      # answer time limit for generated questions
    ANSWER_GRACE_SECONDS = 15         # uploads this much after the limit are still on time (network delay)
    MAX_ANSWER_UPLOAD = 60 * 1024 * 1024  # bytes per recorded answer
    FINISH_BUFFER_MINUTES = 20        # once started, an interview may run this long past its planned length
    MAX_INTEGRITY_EVENTS = 500        # per interview, so a misbehaving client can't flood the table
    GEMINI_TRANSCRIBE_MODEL = os.getenv("GEMINI_TRANSCRIBE_MODEL", "").strip()  # empty = use GEMINI_MODEL

    # Facial emotion analysis (paper 3.2: DeepFace; confidence, nervousness, attentiveness; alignment, histogram
    # normalisation and temporal smoothing are [paper]; the numbers below are [impl]). Runs in its own worker:
    #   venv\Scripts\huey_consumer app.tasks.emotion_huey -k thread -w 1
    EMOTION_ENABLED = os.getenv("EMOTION_ENABLED", "1") == "1"
    EMOTION_SAMPLE_FPS = 2            # [impl] frames analysed per second of recording (~27 ms each on CPU)
    EMOTION_MAX_FRAMES = 1200         # [impl] safety cap per answer (10 minutes at 2 fps)
    EMOTION_DETECTOR_THRESHOLD = 0.7  # [impl] YuNet; 0.9 (default) missed faces in a dim room, 0.7 had no false faces
    EMOTION_SECOND_FACE_CONFIDENCE = 0.85  # [impl] a second face counts only if clearly detected
    EMOTION_SMOOTHING = 0.3           # [impl] weight of the newest frame in the moving average
    EMOTION_FACING_YAW = 0.35         # [impl] nose offset (eye distances) still counted as facing the camera
    EMOTION_MIN_FRAMES = 4            # [impl] fewer sampled frames -> no emotion score
    EMOTION_MIN_FACE_RATIO = 0.5      # [impl] face visible in fewer frames -> "insufficient", not a low score
    EMOTION_WEIGHTS = {"confidence": 0.5, "attentiveness": 0.5}  # [impl] per-answer emotion score
    EMOTION_UNSTABLE_FLIP_RATE = 0.5  # [impl] top emotion changing this often between frames -> "low reliability"
    EMOTION_MULTI_FACE_MIN_FRAMES = 2  # [impl] integrity flag: another person in view
    EMOTION_ABSENCE_FLAG_SECONDS = 5  # [impl] integrity flag: face out of view this long

    # JD-based technical questions: generated once per job from its description and required skills, so every
    # candidate for the job gets the same ones (HR can review and edit them). [impl] realises the paper's "dynamic and
    # domain specific question generation"; 0 per job = only HR-written / template questions.
    JD_QUESTIONS_DEFAULT = 4          # [impl] per job, for new jobs
    JD_QUESTIONS_MAX = 8              # [impl]

    # Resume-based questions: generated per candidate from their parsed resume (experience, projects, skills).
    # [impl] extension of the paper's "individualization" (3.3); 0 per job = paper-baseline interview.
    RESUME_QUESTIONS_DEFAULT = 3      # [impl] per candidate, for new jobs (HR can change it per job)
    RESUME_QUESTIONS_MAX = 5          # [impl]
    RESUME_QUESTIONS_LOOKAHEAD_HOURS = 72  # housekeeping prepares missing/outdated sets for interviews this soon

    # Answer scoring and adaptive questions (paper Sections 3.2-3.4).
    # [paper] = value or rule stated in the paper; [impl] = implementation decision where the paper gives no number
    # (kept configurable so a paper-baseline configuration stays identifiable in evaluation; docs/ENHANCEMENTS.md).
    # The rule "follow up a technical answer with low semantic similarity" is [paper]; its numbers below are [impl].
    FOLLOW_UP_THRESHOLD = 55          # [impl] technical answers below this similarity score (0-100) get a follow-up
    FOLLOW_UP_CATEGORIES = ("technical", "resume")  # [paper] technical; [impl] resume-based questions too
    # What triggers a follow-up. "similarity" = [paper] only (low semantic similarity). "similarity_or_verdict" =
    # [impl, deviation] also when the rubric judges the answer wrong/incomplete: a real interview showed confidently
    # wrong answers that use the right vocabulary score high similarity (68-93) and never got one. The rubric then
    # runs before the decision and is reused for scoring (no extra AI calls; ~3-5 s more wait after such answers).
    FOLLOW_UP_TRIGGER = os.getenv("FOLLOW_UP_TRIGGER", "similarity_or_verdict")
    FOLLOW_UP_VERDICTS = ("incorrect", "partially_correct", "off_topic", "dont_know")  # [impl]
    MAX_FOLLOW_UPS = 3                # [impl] per interview; follow-ups never chain
    DECISION_TIMEOUT_SECONDS = 45     # [impl] if deciding on a follow-up takes longer, the interview just moves on
    ANSWER_SIMILARITY_FLOOR = 0.55    # [impl] cosine similarity (answer vs reference answer) mapped onto 0-100
    ANSWER_SIMILARITY_CEIL = 0.85     # [impl]
    # [paper] components: content relevance, domain accuracy, fluency + semantic similarity; [impl] their weights
    NLP_WEIGHTS = {"relevance": 0.30, "accuracy": 0.30, "similarity": 0.20, "fluency": 0.20}
    GEMINI_INLINE_MEDIA_LIMIT = 18 * 1024 * 1024  # larger recordings go through the Gemini Files API

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

    # Email backends: "gmail_smtp" sends through Gmail with an app password; "console" logs messages and saves
    # them to instance/outbox (development); "memory" keeps them in a list (tests).
    MAIL_BACKEND = os.getenv("MAIL_BACKEND", "console").strip().lower()
    MAIL_OUTBOX_DIR = INSTANCE_DIR / "outbox"
    MAIL_USERNAME = os.getenv("MAIL_USERNAME", "").strip()
    MAIL_APP_PASSWORD = os.getenv("MAIL_APP_PASSWORD", "").replace(" ", "")  # Google shows it as "abcd efgh ijkl mnop"
    MAIL_SENDER_NAME = os.getenv("MAIL_SENDER_NAME", "Virtual Interviewer")
    MAIL_SMTP_HOST = os.getenv("MAIL_SMTP_HOST", "smtp.gmail.com")
    MAIL_SMTP_PORT = int(os.getenv("MAIL_SMTP_PORT", "465"))  # implicit TLS
    MAIL_SMTP_TIMEOUT = 20  # seconds
    MAIL_DAILY_LIMIT = int(os.getenv("MAIL_DAILY_LIMIT", "450"))  # stay under Gmail's ~500/day; extra mail is deferred
    MAIL_DEFER_SECONDS = 3600  # how long to wait before retrying mail deferred by the daily limit
    # Development safety net: deliver EVERY email to this one address instead of the real recipient
    # (the intended recipient is shown in the subject). Not allowed in production.
    MAIL_REDIRECT_TO = os.getenv("MAIL_REDIRECT_TO", "").strip()

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
    APP_BASE_URL = "http://localhost"
    # Pin every mail setting so a developer's .env (real Gmail, redirect...) can never leak into tests
    MAIL_BACKEND = "memory"
    MAIL_USERNAME = ""
    MAIL_APP_PASSWORD = ""
    MAIL_REDIRECT_TO = ""
    MAIL_SENDER_NAME = "Virtual Interviewer"
    MAIL_DAILY_LIMIT = 450
    SHORTLIST_THRESHOLD = 60.0
    GEMINI_MODEL = "gemini-test-model"
    RESUME_QUESTIONS_DEFAULT = 0  # tests that cover resume-based questions switch them on per job
    JD_QUESTIONS_DEFAULT = 0      # likewise for JD-based questions
    EMOTION_ENABLED = True        # the model itself is replaced by a fake in tests (see conftest)
    GEMINI_EMBED_MODEL = "gemini-test-embedding"
    FILE_ENCRYPTION_KEYS = "t1:dGVzdC1rZXktMzItYnl0ZXMtZm9yLWFlcy1nY20hISE="  # fixed 32-byte test key (b'test-key-32-bytes-for-aes-gcm!!!')
    APP_TIMEZONE = "Asia/Kolkata"
    ANALYSIS_RETRY_DELAY = 0


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
            problems.append("MAIL_BACKEND must be a real email backend in production (gmail_smtp)")
        if cls.MAIL_BACKEND == "gmail_smtp" and not (cls.MAIL_USERNAME and cls.MAIL_APP_PASSWORD):
            problems.append("MAIL_USERNAME and MAIL_APP_PASSWORD must be set for gmail_smtp")
        if cls.MAIL_REDIRECT_TO:
            problems.append("MAIL_REDIRECT_TO must be empty in production (it would hide every real email)")
        if not cls.FILE_ENCRYPTION_KEYS:
            problems.append("FILE_ENCRYPTION_KEYS must be set (stored resumes are encrypted)")
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
