# AI-Based Virtual Interviewer

A web platform that screens resumes by meaning, runs AI-led video interviews, and gives recruiters
ranked, evidence-backed scorecards. It implements *"AI-Based Virtual Interviewer System Using NLP and
Emotion Detection"* (Bershika & Nancy, SCIS 2025).

**Stack:** Flask · SQLAlchemy (PostgreSQL in production, SQLite in development) · Alembic migrations ·
Huey background jobs (Redis in production) · Gemini API (parsing, scoring, embeddings) · DeepFace
(facial emotion) · Bootstrap 5.

## Local development (Windows)

Requires Python 3.11 (TensorFlow does not support 3.12+ on Windows yet).

```powershell
py -3.11 -m venv venv
venv\Scripts\pip install -r requirements-dev.txt
copy .env.example .env          # then fill in SECRET_KEY and GEMINI_API_KEY
venv\Scripts\flask --app run db upgrade
venv\Scripts\flask --app run create-admin    # first admin; invite HR users from the Admin panel
venv\Scripts\flask --app run seed-jobs       # optional: three sample job roles
```

In development, emails (verification, password reset, invitations) are not sent: the worker prints them
in its terminal and saves them as `.eml` files in `instance/outbox/`, so you can open the links.

Run the app and the background worker in two terminals:

```powershell
venv\Scripts\python run.py                                   # http://127.0.0.1:5000
venv\Scripts\huey_consumer app.tasks.huey -k thread -w 2     # background jobs
venv\Scripts\flask --app run queue-check                     # verify the worker responds
```

## Tests

```powershell
venv\Scripts\pytest                     # SQLite
venv\Scripts\pytest --postgres          # real PostgreSQL (embedded, no install needed)
venv\Scripts\pytest --cov=app           # with coverage
```

Tests never call the real Gemini API; it is replaced by a deterministic fake.

## Database changes

After changing a model:

```powershell
venv\Scripts\flask --app run db migrate -m "describe the change"
venv\Scripts\flask --app run db upgrade
```

`tests/test_migrations.py` fails if models and migrations ever drift apart.

## Configuration

All settings come from environment variables (see `.env.example`). `APP_ENV=production` refuses to
start unless `SECRET_KEY` (32+ chars), `DATABASE_URL`, `REDIS_URL` and `GEMINI_API_KEY` are set, and
enables secure cookies and HTTPS enforcement.

| Endpoint | Purpose |
|---|---|
| `/healthz` | Liveness/readiness probe (checks the database) |

## Roles and accounts

| Role | How the account is created | Can do |
|---|---|---|
| Candidate | Signs up on the site (must accept the privacy policy and verify email to apply) | Apply to jobs, see own results, export or delete own data |
| HR | Invited by an admin (or `flask create-hr`) | Manage jobs, review and rank applicants |
| Admin | `flask create-admin`, or invited by another admin | Everything HR can do, plus user management and the audit log |

Security: 5 failed logins lock an account for 15 minutes; changing or resetting a password signs the user
out everywhere; reset and invitation links are single-use and expire; every security event is in the audit log.
After changing the privacy policy, bump `PRIVACY_POLICY_VERSION` in `app/config.py` so candidates re-consent.

## Project layout

```
app/
  __init__.py      app factory, security headers, CLI commands
  config.py        per-environment settings
  extensions.py    db, migrations, login, CSRF, rate limiter, security headers
  tasks.py         background jobs (Huey)
  models.py        User, AuditLog, Job, Application
  auth.py          sign up, log in/out, email verification, password reset, invitations, consent
  account.py       profile, email and password changes, data export, account deletion
  admin.py         user management and audit log
  candidate.py     job browsing, resume upload, results
  hr.py            job management, applicant ranking and review
  services/        Gemini, resume matching, mail, tokens/password policy/audit, account operations
  templates/       Jinja2 + Bootstrap pages
migrations/        Alembic migration scripts
tests/             pytest suite
sample_resumes/    fictional resumes for testing
```
