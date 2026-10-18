import secrets
from datetime import datetime, timedelta, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db, login_manager


def utcnow():
    return datetime.now(timezone.utc)


def as_utc(value):
    """SQLite hands back naive datetimes even for timezone-aware columns; treat those as UTC."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def new_session_token():
    return secrets.token_hex(16)


class User(UserMixin, db.Model):
    ROLE_CANDIDATE = "candidate"
    ROLE_HR = "hr"
    ROLE_ADMIN = "admin"
    ROLES = (ROLE_CANDIDATE, ROLE_HR, ROLE_ADMIN)
    ROLE_LABELS = {ROLE_CANDIDATE: "Candidate", ROLE_HR: "HR", ROLE_ADMIN: "Admin"}

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default=ROLE_CANDIDATE, index=True)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)

    # Account state
    active = db.Column(db.Boolean, nullable=False, default=True)  # admins can deactivate accounts
    email_verified_at = db.Column(db.DateTime(timezone=True))
    pending_email = db.Column(db.String(255))  # new address awaiting confirmation
    session_token = db.Column(db.String(32), nullable=False, default=new_session_token)  # rotated to end all sessions
    failed_login_count = db.Column(db.Integer, nullable=False, default=0)
    locked_until = db.Column(db.DateTime(timezone=True))
    last_login_at = db.Column(db.DateTime(timezone=True))
    password_changed_at = db.Column(db.DateTime(timezone=True))
    consent_version = db.Column(db.String(20))  # privacy policy version the user accepted
    consent_at = db.Column(db.DateTime(timezone=True))
    notification_prefs = db.Column(db.JSON)  # {"email_applications": bool, "email_interviews": bool}; missing = on

    applications = db.relationship("Application", back_populates="candidate", cascade="all, delete-orphan")

    # --- passwords and sessions ---

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        self.password_changed_at = utcnow()
        self.end_all_sessions()

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def end_all_sessions(self):
        """Invalidates every existing login (including 'remember me' cookies) for this user."""
        self.session_token = new_session_token()

    def get_id(self):
        # The session stores id + token, so rotating the token logs the user out everywhere
        return f"{self.id}:{self.session_token}"

    @property
    def is_active(self):
        return self.active

    # --- lockout ---

    @property
    def is_locked(self):
        return self.locked_until is not None and as_utc(self.locked_until) > utcnow()

    @property
    def lock_minutes_left(self):
        if not self.is_locked:
            return 0
        return max(1, int((as_utc(self.locked_until) - utcnow()).total_seconds() // 60) + 1)

    def register_failed_login(self, threshold, minutes):
        self.failed_login_count = (self.failed_login_count or 0) + 1
        if self.failed_login_count >= threshold:
            self.locked_until = utcnow() + timedelta(minutes=minutes)
            self.failed_login_count = 0
            return True  # just locked
        return False

    def register_successful_login(self):
        self.failed_login_count = 0
        self.locked_until = None
        self.last_login_at = utcnow()

    def unlock(self):
        self.failed_login_count = 0
        self.locked_until = None

    # --- status helpers ---

    @property
    def is_verified(self):
        return self.email_verified_at is not None

    def mark_verified(self):
        if self.email_verified_at is None:
            self.email_verified_at = utcnow()

    @property
    def is_hr(self):
        """HR pages are open to HR users and admins."""
        return self.role in (self.ROLE_HR, self.ROLE_ADMIN)

    @property
    def is_admin(self):
        return self.role == self.ROLE_ADMIN

    @property
    def is_candidate(self):
        return self.role == self.ROLE_CANDIDATE

    @property
    def role_label(self):
        return self.ROLE_LABELS.get(self.role, self.role)

    # --- notification preferences ---

    # Categories a user may switch off. Security emails (password reset, password changed...) are always sent.
    EMAIL_CATEGORIES = {"applications": "Application updates (received, shortlisted, decisions)",
                        "interviews": "Interview scheduling and reminders"}

    def wants_email(self, category):
        if category not in self.EMAIL_CATEGORIES:
            return True
        return (self.notification_prefs or {}).get(f"email_{category}", True)

    def set_email_preferences(self, **enabled):
        prefs = dict(self.notification_prefs or {})
        for category, value in enabled.items():
            if category in self.EMAIL_CATEGORIES:
                prefs[f"email_{category}"] = bool(value)
        self.notification_prefs = prefs  # reassign so SQLAlchemy sees the JSON change

    def has_current_consent(self, version):
        return self.consent_version == version

    def give_consent(self, version):
        self.consent_version = version
        self.consent_at = utcnow()


class AuditLog(db.Model):
    """Append-only record of security-relevant events (logins, password changes, role changes, deletions...)."""

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    actor_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="SET NULL"), index=True)
    actor_email = db.Column(db.String(255))  # kept so the log stays readable after the account is deleted
    action = db.Column(db.String(64), nullable=False, index=True)
    target_user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="SET NULL"), index=True)
    target_email = db.Column(db.String(255))
    ip_address = db.Column(db.String(64))
    details = db.Column(db.JSON, nullable=False, default=dict)

    actor = db.relationship("User", foreign_keys=[actor_id])
    target_user = db.relationship("User", foreign_keys=[target_user_id])


class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(160), nullable=False)
    description = db.Column(db.Text, nullable=False)
    skills = db.Column(db.Text, nullable=False)  # comma-separated required skills
    is_open = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)
    embedding = db.Column(db.LargeBinary)  # float32 bytes of the job's semantic embedding (cached)

    applications = db.relationship("Application", back_populates="job", cascade="all, delete-orphan")

    @property
    def skill_list(self):
        return [s.strip() for s in self.skills.split(",") if s.strip()]

    @property
    def embedding_text(self):
        return f"Job title: {self.title}\nRequired skills: {', '.join(self.skill_list)}\n{self.description}"


class Application(db.Model):
    STATUS_SHORTLISTED = "shortlisted"
    STATUS_REVIEW = "under_review"
    STATUS_REJECTED = "rejected"
    STATUS_LABELS = {STATUS_SHORTLISTED: "Shortlisted", STATUS_REVIEW: "Under review", STATUS_REJECTED: "Rejected"}
    STATUS_COLORS = {STATUS_SHORTLISTED: "success", STATUS_REVIEW: "warning", STATUS_REJECTED: "secondary"}

    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    job_id = db.Column(db.Integer, db.ForeignKey("job.id"), nullable=False, index=True)
    resume_path = db.Column(db.String(255), nullable=False)  # file name inside UPLOAD_FOLDER
    resume_original_name = db.Column(db.String(255), nullable=False)
    resume_text = db.Column(db.Text, nullable=False)
    parsed = db.Column(db.JSON, nullable=False)  # structured profile extracted by Gemini
    similarity = db.Column(db.Float, nullable=False)  # raw cosine similarity resume <-> job
    match_score = db.Column(db.Float, nullable=False)  # 0-100 score shown to users
    status = db.Column(db.String(20), nullable=False, default=STATUS_REVIEW, index=True)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)

    candidate = db.relationship("User", back_populates="applications")
    job = db.relationship("Job", back_populates="applications")

    __table_args__ = (db.UniqueConstraint("candidate_id", "job_id", name="one_application_per_job"),)

    @property
    def status_label(self):
        return self.STATUS_LABELS.get(self.status, self.status)

    @property
    def status_color(self):
        return self.STATUS_COLORS.get(self.status, "light")


class EmailLog(db.Model):
    """One row per outgoing email, so delivery problems are visible to admins. Bodies are not stored
    (they contain single-use login links)."""

    STATUS_QUEUED = "queued"
    STATUS_SENT = "sent"
    STATUS_RETRYING = "retrying"
    STATUS_DEFERRED = "deferred"  # held back by the daily sending limit
    STATUS_FAILED = "failed"
    STATUS_COLORS = {STATUS_QUEUED: "secondary", STATUS_SENT: "success", STATUS_RETRYING: "warning",
                     STATUS_DEFERRED: "info", STATUS_FAILED: "danger"}

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="SET NULL"), index=True)
    to_address = db.Column(db.String(255), nullable=False, index=True)
    subject = db.Column(db.String(255), nullable=False)
    template = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(16), nullable=False, default=STATUS_QUEUED, index=True)
    attempts = db.Column(db.Integer, nullable=False, default=0)
    backend = db.Column(db.String(32))
    message_id = db.Column(db.String(255))
    last_error = db.Column(db.Text)
    sent_at = db.Column(db.DateTime(timezone=True), index=True)

    @property
    def status_color(self):
        return self.STATUS_COLORS.get(self.status, "light")


class Notification(db.Model):
    """In-app notification shown under the bell icon."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    kind = db.Column(db.String(40), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    body = db.Column(db.Text, nullable=False, default="")
    link = db.Column(db.String(500))  # local path only, e.g. /candidate/applications/3
    read_at = db.Column(db.DateTime(timezone=True))

    user = db.relationship("User", backref=db.backref("notifications", cascade="all, delete-orphan", lazy="dynamic"))

    @property
    def is_read(self):
        return self.read_at is not None


@login_manager.user_loader
def load_user(session_id):
    """Session ids look like '<user id>:<session token>'. Old-format, stale or deactivated sessions are rejected."""
    user_id, _, token = session_id.partition(":")
    if not user_id.isdigit() or not token:
        return None
    user = db.session.get(User, int(user_id))
    if user is None or not user.active or not secrets.compare_digest(user.session_token, token):
        return None
    return user
