from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db, login_manager


def utcnow():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    ROLE_CANDIDATE = "candidate"
    ROLE_HR = "hr"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default=ROLE_CANDIDATE)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)

    applications = db.relationship("Application", back_populates="candidate", cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_hr(self):
        return self.role == self.ROLE_HR


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


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))
