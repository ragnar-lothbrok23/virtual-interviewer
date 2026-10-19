from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import (BooleanField, DateField, DateTimeLocalField, EmailField, IntegerField, PasswordField,
                     SelectField, SelectMultipleField, StringField, SubmitField, TextAreaField, TimeField)
from wtforms.validators import (DataRequired, Email, EqualTo, InputRequired, Length, NumberRange, Optional,
                                ValidationError)

from .models import User
from .services.security import MAX_PASSWORD_LENGTH, password_problems


def normalise_email(value):
    return (value or "").strip().lower()


def email_is_taken(email, exclude_user_id=None):
    email = normalise_email(email)
    query = User.query.filter((User.email == email) | (User.pending_email == email))
    if exclude_user_id is not None:
        query = query.filter(User.id != exclude_user_id)
    return query.first() is not None


class StrongPassword:
    """Applies the password policy; reads email/name from sibling fields when present."""

    def __init__(self, email_field="email", name_field="name"):
        self.email_field, self.name_field = email_field, name_field

    def __call__(self, form, field):
        # Forms without email/name fields (reset, change password) get them via form.policy_email / form.policy_name
        email = getattr(getattr(form, self.email_field, None), "data", None) or getattr(form, "policy_email", "") or ""
        name = getattr(getattr(form, self.name_field, None), "data", None) or getattr(form, "policy_name", "") or ""
        problems = password_problems(field.data or "", email=email, name=name)
        if problems:
            raise ValidationError(" ".join(problems))


def new_password_fields(label="Password"):
    password = PasswordField(label, validators=[DataRequired(), Length(max=MAX_PASSWORD_LENGTH), StrongPassword()])
    confirm = PasswordField(f"Confirm {label.lower()}", validators=[DataRequired(), EqualTo("password", "Passwords must match.")])
    return password, confirm


# ---------- authentication ----------

class LoginForm(FlaskForm):
    email = EmailField("Email", validators=[DataRequired(), Email(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired(), Length(max=MAX_PASSWORD_LENGTH)])
    remember = BooleanField("Remember me")
    submit = SubmitField("Log in")


class RegisterForm(FlaskForm):
    name = StringField("Full name", validators=[DataRequired(), Length(max=120)])
    email = EmailField("Email", validators=[DataRequired(), Email(), Length(max=255)])
    password, confirm = new_password_fields()
    consent = BooleanField("I have read and agree to the Privacy Policy and Terms of Use",
                           validators=[DataRequired("You must accept the Privacy Policy and Terms to create an account.")])
    submit = SubmitField("Create account")

    def validate_name(self, field):
        if not field.data.strip():
            raise ValidationError("Please enter your name.")

    def validate_email(self, field):
        if email_is_taken(field.data):
            raise ValidationError("An account with this email already exists.")


class ForgotPasswordForm(FlaskForm):
    email = EmailField("Email", validators=[DataRequired(), Email(), Length(max=255)])
    submit = SubmitField("Send reset link")


class ResetPasswordForm(FlaskForm):
    password, confirm = new_password_fields("New password")
    submit = SubmitField("Set password")


class ResendVerificationForm(FlaskForm):
    submit = SubmitField("Resend verification email")


class ConsentForm(FlaskForm):
    consent = BooleanField("I have read and agree to the Privacy Policy and Terms of Use",
                           validators=[DataRequired("Please accept to continue.")])
    submit = SubmitField("Continue")


# ---------- account settings ----------

class ProfileForm(FlaskForm):
    name = StringField("Full name", validators=[DataRequired(), Length(max=120)])
    submit_profile = SubmitField("Save name")

    def validate_name(self, field):
        if not field.data.strip():
            raise ValidationError("Please enter your name.")


class ChangeEmailForm(FlaskForm):
    new_email = EmailField("New email", validators=[DataRequired(), Email(), Length(max=255)])
    current_password = PasswordField("Current password", validators=[DataRequired(), Length(max=MAX_PASSWORD_LENGTH)])
    submit_email = SubmitField("Send confirmation link")


class ChangePasswordForm(FlaskForm):
    current_password = PasswordField("Current password", validators=[DataRequired(), Length(max=MAX_PASSWORD_LENGTH)])
    password, confirm = new_password_fields("New password")
    submit_password = SubmitField("Change password")


class EmailPreferencesForm(FlaskForm):
    email_applications = BooleanField("Application updates (received, shortlisted, decisions)")
    email_interviews = BooleanField("Interview scheduling and reminders")
    submit_prefs = SubmitField("Save email preferences")


class DeleteAccountForm(FlaskForm):
    current_password = PasswordField("Current password", validators=[DataRequired(), Length(max=MAX_PASSWORD_LENGTH)])
    confirm_text = StringField('Type DELETE to confirm', validators=[DataRequired()])
    submit_delete = SubmitField("Permanently delete my account")

    def validate_confirm_text(self, field):
        if field.data.strip() != "DELETE":
            raise ValidationError("Please type DELETE in capital letters.")


# ---------- admin ----------

class InviteStaffForm(FlaskForm):
    name = StringField("Full name", validators=[DataRequired(), Length(max=120)])
    email = EmailField("Email", validators=[DataRequired(), Email(), Length(max=255)])
    role = SelectField("Role", choices=[(User.ROLE_HR, "HR"), (User.ROLE_ADMIN, "Admin")], validators=[DataRequired()])
    submit = SubmitField("Send invitation")

    def validate_name(self, field):
        if not field.data.strip():
            raise ValidationError("Please enter a name.")

    def validate_email(self, field):
        if email_is_taken(field.data):
            raise ValidationError("An account with this email already exists.")


# ---------- recruiting ----------

class JobForm(FlaskForm):
    title = StringField("Job title", validators=[DataRequired(), Length(max=160)])
    skills = StringField("Required skills (comma-separated)", validators=[DataRequired(), Length(max=1000)],
                         render_kw={"placeholder": "Python, Flask, SQL, REST APIs, Git"})
    description = TextAreaField("Job description", validators=[DataRequired(), Length(min=50, max=8000)],
                                render_kw={"rows": 7})
    closes_at = DateTimeLocalField("Application deadline (optional)", format="%Y-%m-%dT%H:%M",
                                   validators=[Optional()])
    submit = SubmitField("Save job")

    def validate_skills(self, field):
        skills = [s.strip() for s in field.data.split(",") if s.strip()]
        if not skills:
            raise ValidationError("Enter at least one skill.")
        if len(skills) > 25:
            raise ValidationError("Please list at most 25 required skills.")
        if any(len(s) > 60 for s in skills):
            raise ValidationError("Each skill must be at most 60 characters.")


class ApplyForm(FlaskForm):
    resume = FileField("Resume (PDF or DOCX, max 5 MB)",
                       validators=[FileRequired("Please choose a file."), FileAllowed(["pdf", "docx"], "Only PDF or DOCX files.")])
    submit = SubmitField("Upload and apply")


class WithdrawForm(FlaskForm):
    confirm = BooleanField("I understand my application will be withdrawn",
                           validators=[DataRequired("Please confirm you want to withdraw.")])
    submit_withdraw = SubmitField("Withdraw application")


# ---------- interview scheduling ----------

DURATION_CHOICES = [(15, "15 minutes"), (20, "20 minutes"), (30, "30 minutes"), (45, "45 minutes"), (60, "60 minutes")]
WEEKDAY_CHOICES = [(0, "Mon"), (1, "Tue"), (2, "Wed"), (3, "Thu"), (4, "Fri"), (5, "Sat"), (6, "Sun")]


class SlotForm(FlaskForm):
    starts_at = DateTimeLocalField("Start", format="%Y-%m-%dT%H:%M", validators=[DataRequired()])
    duration = SelectField("Length", coerce=int, choices=DURATION_CHOICES, default=30)
    capacity = IntegerField("Candidates at once", default=1, validators=[InputRequired(), NumberRange(1, 50)])
    submit_single = SubmitField("Add slot")


class BulkSlotForm(FlaskForm):
    first_day = DateField("From", validators=[DataRequired()])
    last_day = DateField("To", validators=[DataRequired()])
    weekdays = SelectMultipleField("Days", coerce=int, choices=WEEKDAY_CHOICES, default=[0, 1, 2, 3, 4],
                                   validators=[DataRequired("Choose at least one day.")])
    day_start = TimeField("Daily from", validators=[DataRequired()])
    day_end = TimeField("Daily until", validators=[DataRequired()])
    duration = SelectField("Length", coerce=int, choices=DURATION_CHOICES, default=30)
    gap = SelectField("Break between slots", coerce=int, default=0,
                      choices=[(0, "None"), (5, "5 minutes"), (10, "10 minutes"), (15, "15 minutes")])
    capacity = IntegerField("Candidates at once", default=1, validators=[InputRequired(), NumberRange(1, 50)])
    submit_bulk = SubmitField("Create slots")


class CancelForm(FlaskForm):
    reason = StringField("Reason (shown to the candidate)", validators=[Optional(), Length(max=255)])
    submit_cancel = SubmitField("Cancel")


class QuestionForm(FlaskForm):
    text = TextAreaField("Question", validators=[DataRequired(), Length(min=10, max=1000)], render_kw={"rows": 2})
    category = SelectField("Type", choices=[("technical", "Technical"), ("behavioral", "Behavioural"),
                                            ("situational", "Situational")])
    answer_seconds = IntegerField("Answer time (seconds)", default=120, validators=[InputRequired(), NumberRange(30, 600)])
    submit_question = SubmitField("Save")
