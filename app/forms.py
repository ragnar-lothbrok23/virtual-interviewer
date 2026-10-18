from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import BooleanField, EmailField, PasswordField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError

from .models import User


class LoginForm(FlaskForm):
    email = EmailField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    remember = BooleanField("Remember me")
    submit = SubmitField("Log in")


class RegisterForm(FlaskForm):
    name = StringField("Full name", validators=[DataRequired(), Length(max=120)])
    email = EmailField("Email", validators=[DataRequired(), Email(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8)])
    confirm = PasswordField("Confirm password", validators=[DataRequired(), EqualTo("password", "Passwords must match.")])
    submit = SubmitField("Create account")

    def validate_email(self, field):
        if User.query.filter_by(email=field.data.strip().lower()).first():
            raise ValidationError("An account with this email already exists.")


class JobForm(FlaskForm):
    title = StringField("Job title", validators=[DataRequired(), Length(max=160)])
    skills = StringField("Required skills (comma-separated)", validators=[DataRequired(), Length(max=1000)],
                         render_kw={"placeholder": "Python, Flask, SQL, REST APIs, Git"})
    description = TextAreaField("Job description", validators=[DataRequired(), Length(min=50, max=8000)],
                                render_kw={"rows": 7})
    submit = SubmitField("Create job")


class ApplyForm(FlaskForm):
    resume = FileField("Resume (PDF or DOCX, max 5 MB)",
                       validators=[FileRequired("Please choose a file."), FileAllowed(["pdf", "docx"], "Only PDF or DOCX files.")])
    submit = SubmitField("Upload and apply")
