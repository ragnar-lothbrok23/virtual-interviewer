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
    closes_at = db.Column(db.DateTime(timezone=True))  # application deadline (UTC); None = no deadline
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    embedding = db.Column(db.LargeBinary)  # float32 bytes of the job's semantic embedding (cached)

    applications = db.relationship("Application", back_populates="job", cascade="all, delete-orphan")

    @property
    def skill_list(self):
        return [s.strip() for s in self.skills.split(",") if s.strip()]

    @property
    def embedding_text(self):
        return f"Job title: {self.title}\nRequired skills: {', '.join(self.skill_list)}\n{self.description}"

    @property
    def deadline_passed(self):
        return self.closes_at is not None and as_utc(self.closes_at) <= utcnow()

    @property
    def is_accepting(self):
        """Open for new applications: not closed by HR and the deadline (if any) hasn't passed."""
        return self.is_open and not self.deadline_passed

    @property
    def state_label(self):
        if not self.is_open:
            return "Closed"
        return "Deadline passed" if self.deadline_passed else "Open"


class Application(db.Model):
    STATUS_SHORTLISTED = "shortlisted"
    STATUS_REVIEW = "under_review"
    STATUS_REJECTED = "rejected"
    STATUS_WITHDRAWN = "withdrawn"
    STATUS_LABELS = {STATUS_SHORTLISTED: "Shortlisted", STATUS_REVIEW: "Under review", STATUS_REJECTED: "Rejected",
                     STATUS_WITHDRAWN: "Withdrawn"}
    STATUS_COLORS = {STATUS_SHORTLISTED: "success", STATUS_REVIEW: "warning", STATUS_REJECTED: "secondary",
                     STATUS_WITHDRAWN: "light border text-dark"}
    HR_STATUSES = (STATUS_SHORTLISTED, STATUS_REVIEW, STATUS_REJECTED)  # statuses HR may set

    # Who decided the status: "auto" (match-score threshold) can be re-evaluated; "hr" is never overwritten
    SOURCE_AUTO = "auto"
    SOURCE_HR = "hr"

    # Resume analysis lifecycle (runs in the background)
    ANALYSIS_PENDING = "pending"
    ANALYSIS_RUNNING = "running"
    ANALYSIS_DONE = "done"
    ANALYSIS_FAILED = "failed"

    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    job_id = db.Column(db.Integer, db.ForeignKey("job.id"), nullable=False, index=True)
    resume_path = db.Column(db.String(255), nullable=False)  # encrypted file name inside UPLOAD_FOLDER
    resume_original_name = db.Column(db.String(255), nullable=False)
    resume_kind = db.Column(db.String(8))  # "pdf" / "docx" (verified from the file's bytes)
    resume_sha256 = db.Column(db.String(64), index=True)  # spots identical resumes from different candidates
    resume_text = db.Column(db.Text, nullable=False)
    parsed = db.Column(db.JSON)  # structured profile extracted by Gemini (None until analysed)
    embedding = db.Column(db.LargeBinary)  # float32 resume embedding, so editing a job can re-score cheaply
    similarity = db.Column(db.Float)  # raw cosine similarity resume <-> job
    match_score = db.Column(db.Float)  # 0-100 score shown to users
    analysis_status = db.Column(db.String(10), nullable=False, default=ANALYSIS_PENDING, index=True)
    analysis_error = db.Column(db.Text)
    analysis_attempts = db.Column(db.Integer, nullable=False, default=0)
    analysed_at = db.Column(db.DateTime(timezone=True))
    status = db.Column(db.String(20), nullable=False, default=STATUS_REVIEW, index=True)
    status_source = db.Column(db.String(8), nullable=False, default=SOURCE_AUTO)
    withdrawn_at = db.Column(db.DateTime(timezone=True))
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    candidate = db.relationship("User", back_populates="applications")
    job = db.relationship("Job", back_populates="applications")

    __table_args__ = (db.UniqueConstraint("candidate_id", "job_id", name="one_application_per_job"),)

    @property
    def status_label(self):
        return self.STATUS_LABELS.get(self.status, self.status)

    @property
    def status_color(self):
        return self.STATUS_COLORS.get(self.status, "light")

    @property
    def is_analysed(self):
        return self.analysis_status == self.ANALYSIS_DONE and self.match_score is not None

    @property
    def is_analysing(self):
        return self.analysis_status in (self.ANALYSIS_PENDING, self.ANALYSIS_RUNNING)

    @property
    def is_withdrawn(self):
        return self.status == self.STATUS_WITHDRAWN

    @property
    def can_update_resume(self):
        """Candidates may replace their resume while the job accepts applications and HR hasn't rejected them."""
        return self.job.is_accepting and self.status != self.STATUS_REJECTED and not self.is_analysing

    @property
    def duplicates(self):
        """Other candidates' applications that uploaded a byte-identical resume."""
        if not self.resume_sha256:
            return []
        return (Application.query.filter(Application.resume_sha256 == self.resume_sha256,
                                         Application.candidate_id != self.candidate_id).all())


class InterviewSlot(db.Model):
    """A time window HR opens for a job. AI interviews run in parallel, so a slot can hold `capacity` candidates."""

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey("job.id", ondelete="CASCADE"), nullable=False, index=True)
    starts_at = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    duration_minutes = db.Column(db.Integer, nullable=False)
    capacity = db.Column(db.Integer, nullable=False, default=1)
    booked_count = db.Column(db.Integer, nullable=False, default=0)  # kept in step atomically (see scheduling.book)
    cancelled_at = db.Column(db.DateTime(timezone=True))
    created_by_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="SET NULL"))
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)

    job = db.relationship("Job", backref=db.backref("slots", cascade="all, delete-orphan", lazy="dynamic"))
    interviews = db.relationship("Interview", back_populates="slot")

    __table_args__ = (
        db.UniqueConstraint("job_id", "starts_at", name="uq_interview_slot_job_start"),
        db.CheckConstraint("capacity >= 1", name="capacity_positive"),
        db.CheckConstraint("booked_count >= 0 AND booked_count <= capacity", name="booked_within_capacity"),
        db.CheckConstraint("duration_minutes BETWEEN 5 AND 240", name="duration_range"),
    )

    @property
    def ends_at(self):
        return as_utc(self.starts_at) + timedelta(minutes=self.duration_minutes)

    @property
    def seats_left(self):
        return max(0, self.capacity - self.booked_count)

    @property
    def is_cancelled(self):
        return self.cancelled_at is not None

    @property
    def is_past(self):
        return as_utc(self.starts_at) <= utcnow()


class Interview(db.Model):
    """One candidate's booking for an application. Rescheduling moves it to another slot; cancelling keeps the row
    (with history) so the candidate can book again later."""

    STATUS_SCHEDULED = "scheduled"
    STATUS_IN_PROGRESS = "in_progress"
    STATUS_COMPLETED = "completed"
    STATUS_CANCELLED = "cancelled"
    STATUS_NO_SHOW = "no_show"
    STATUS_LABELS = {STATUS_SCHEDULED: "Scheduled", STATUS_IN_PROGRESS: "In progress", STATUS_COMPLETED: "Completed",
                     STATUS_CANCELLED: "Cancelled", STATUS_NO_SHOW: "Missed"}
    STATUS_COLORS = {STATUS_SCHEDULED: "primary", STATUS_IN_PROGRESS: "info", STATUS_COMPLETED: "success",
                     STATUS_CANCELLED: "secondary", STATUS_NO_SHOW: "danger"}
    ACTIVE = (STATUS_SCHEDULED, STATUS_IN_PROGRESS)

    id = db.Column(db.Integer, primary_key=True)
    application_id = db.Column(db.Integer, db.ForeignKey("application.id", ondelete="CASCADE"), nullable=False,
                               unique=True)
    slot_id = db.Column(db.Integer, db.ForeignKey("interview_slot.id", ondelete="SET NULL"), index=True)
    status = db.Column(db.String(16), nullable=False, default=STATUS_SCHEDULED, index=True)
    starts_at = db.Column(db.DateTime(timezone=True), nullable=False, index=True)  # copied from the slot
    duration_minutes = db.Column(db.Integer, nullable=False)
    booked_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)
    reschedule_count = db.Column(db.Integer, nullable=False, default=0)
    cancelled_at = db.Column(db.DateTime(timezone=True))
    cancelled_by = db.Column(db.String(12))  # candidate / hr / system
    cancel_reason = db.Column(db.String(255))
    reminder_24h_sent_at = db.Column(db.DateTime(timezone=True))
    reminder_1h_sent_at = db.Column(db.DateTime(timezone=True))
    calendar_sequence = db.Column(db.Integer, nullable=False, default=0)  # .ics SEQUENCE, bumped on every change
    started_at = db.Column(db.DateTime(timezone=True))
    completed_at = db.Column(db.DateTime(timezone=True))
    updated_at = db.Column(db.DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Interview room (Phase 6)
    recording_consent_at = db.Column(db.DateTime(timezone=True))
    question_plan = db.Column(db.JSON)  # questions frozen at start: [{text, category, answer_seconds}]
    current_position = db.Column(db.Integer, nullable=False, default=0)  # index of the next unanswered question
    hard_deadline = db.Column(db.DateTime(timezone=True))  # once started, the latest moment answers are accepted
    end_reason = db.Column(db.String(16))  # completed / timed_out
    follow_up_count = db.Column(db.Integer, nullable=False, default=0, server_default=db.text("0"))
    awaiting_decision_since = db.Column(db.DateTime(timezone=True))  # set while deciding on a follow-up question

    END_COMPLETED = "completed"
    END_TIMED_OUT = "timed_out"

    application = db.relationship("Application", backref=db.backref("interview", uselist=False,
                                                                    cascade="all, delete-orphan"))
    slot = db.relationship("InterviewSlot", back_populates="interviews")
    answers = db.relationship("InterviewAnswer", back_populates="interview", cascade="all, delete-orphan",
                              order_by="InterviewAnswer.position")
    events = db.relationship("InterviewEvent", back_populates="interview", cascade="all, delete-orphan",
                             order_by="InterviewEvent.id", lazy="dynamic")

    @property
    def question_count(self):
        return len(self.question_plan or [])

    @property
    def is_finished(self):
        return self.status in (self.STATUS_COMPLETED, self.STATUS_NO_SHOW, self.STATUS_CANCELLED)

    @property
    def ends_at(self):
        return as_utc(self.starts_at) + timedelta(minutes=self.duration_minutes)

    @property
    def is_active(self):
        return self.status in self.ACTIVE

    @property
    def status_label(self):
        return self.STATUS_LABELS.get(self.status, self.status)

    @property
    def status_color(self):
        return self.STATUS_COLORS.get(self.status, "light")


class InterviewQuestion(db.Model):
    """The job's question bank (fixed order). Phase 7 makes follow-ups adaptive."""

    CATEGORIES = {"technical": "Technical", "behavioral": "Behavioural", "situational": "Situational"}

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey("job.id", ondelete="CASCADE"), nullable=False, index=True)
    position = db.Column(db.Integer, nullable=False)
    text = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(16), nullable=False, default="technical")
    answer_seconds = db.Column(db.Integer, nullable=False, default=120)
    reference_answer = db.Column(db.Text)  # what a strong answer covers; answers are compared with it (paper 3.3)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)

    job = db.relationship("Job", backref=db.backref("questions", cascade="all, delete-orphan",
                                                    order_by="InterviewQuestion.position"))

    __table_args__ = (db.CheckConstraint("answer_seconds BETWEEN 30 AND 600", name="answer_seconds_range"),)

    @property
    def category_label(self):
        return self.CATEGORIES.get(self.category, self.category)


class InterviewAnswer(db.Model):
    """One recorded answer. Created when the question is shown, completed when the recording is uploaded."""

    TRANSCRIPT_PENDING = "pending"
    TRANSCRIPT_DONE = "done"
    TRANSCRIPT_FAILED = "failed"

    id = db.Column(db.Integer, primary_key=True)
    interview_id = db.Column(db.Integer, db.ForeignKey("interview.id", ondelete="CASCADE"), nullable=False, index=True)
    position = db.Column(db.Integer, nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(16), nullable=False)
    answer_seconds = db.Column(db.Integer, nullable=False)
    asked_at = db.Column(db.DateTime(timezone=True), nullable=False)
    attempts = db.Column(db.Integer, nullable=False, default=1)  # >1 = the question was shown again after a reconnect
    submitted_at = db.Column(db.DateTime(timezone=True))
    duration_seconds = db.Column(db.Float)  # recording length reported by the browser (clamped)
    late = db.Column(db.Boolean, nullable=False, default=False)
    media_path = db.Column(db.String(255))  # encrypted recording in UPLOAD_FOLDER
    media_type = db.Column(db.String(40))  # video/webm, video/mp4
    media_size = db.Column(db.Integer)
    transcript = db.Column(db.Text)
    transcript_status = db.Column(db.String(10))
    transcript_error = db.Column(db.Text)
    speech_detected = db.Column(db.Boolean)

    # Adaptive questioning
    is_follow_up = db.Column(db.Boolean, nullable=False, default=False, server_default=db.false())
    reference_answer = db.Column(db.Text)

    # Scoring (Phase 7): Gemini rubric + semantic similarity + speech metrics computed locally
    SCORE_PENDING = "pending"
    SCORE_DONE = "done"
    SCORE_FAILED = "failed"
    score_status = db.Column(db.String(10))
    score_error = db.Column(db.Text)
    scored_at = db.Column(db.DateTime(timezone=True))
    score_requested_at = db.Column(db.DateTime(timezone=True))  # when (re-)scoring was last queued
    similarity = db.Column(db.Float)  # cosine similarity to the reference answer
    similarity_score = db.Column(db.Float)  # 0-100
    relevance = db.Column(db.Float)  # 0-100 each, from the Gemini rubric
    accuracy = db.Column(db.Float)
    fluency = db.Column(db.Float)
    clarity = db.Column(db.Float)
    nlp_score = db.Column(db.Float)  # weighted per-answer NLP score (0-100)
    word_count = db.Column(db.Integer)
    words_per_minute = db.Column(db.Float)
    filler_count = db.Column(db.Integer)
    feedback = db.Column(db.JSON)  # {summary, strengths: [], gaps: [], evidence: [{quote, note}]}

    @property
    def is_scored(self):
        return self.score_status == self.SCORE_DONE and self.nlp_score is not None

    @property
    def is_scoring(self):
        """Scoring is genuinely in progress (queued recently). Answers recorded before scoring existed have no status,
        and a job lost to a worker restart stays 'pending' forever; neither should show a spinner."""
        requested = self.score_requested_at or self.submitted_at
        if self.score_status != self.SCORE_PENDING or requested is None:
            return False
        return utcnow() - as_utc(requested) < timedelta(minutes=10)

    interview = db.relationship("Interview", back_populates="answers")

    __table_args__ = (db.UniqueConstraint("interview_id", "position", name="uq_interview_answer_position"),)

    @property
    def is_submitted(self):
        return self.submitted_at is not None


class InterviewEvent(db.Model):
    """Integrity and lifecycle signals during an interview (tab switches, reconnects, ...)."""

    LABELS = {"tab_hidden": "Left the interview tab", "tab_visible": "Returned to the interview tab",
              "window_blur": "Interview window lost focus", "reconnect": "Rejoined after a disconnect",
              "question_restarted": "Question shown again after reconnecting", "started": "Interview started",
              "finished": "Interview finished", "timed_out": "Interview timed out", "device_error": "Camera/mic problem",
              "late_answer": "Answer submitted after the time limit",
              "follow_up_added": "Follow-up question added (weak technical answer)",
              "decision_timeout": "Follow-up decision timed out; moved on"}
    FLAGGED = {"tab_hidden", "window_blur", "question_restarted", "late_answer", "device_error"}

    id = db.Column(db.Integer, primary_key=True)
    interview_id = db.Column(db.Integer, db.ForeignKey("interview.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)
    kind = db.Column(db.String(32), nullable=False)
    position = db.Column(db.Integer)  # question index, if relevant
    details = db.Column(db.JSON, nullable=False, default=dict)

    interview = db.relationship("Interview", back_populates="events")

    @property
    def label(self):
        return self.LABELS.get(self.kind, self.kind)

    @property
    def flagged(self):
        return self.kind in self.FLAGGED


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
