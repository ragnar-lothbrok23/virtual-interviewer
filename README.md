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
venv\Scripts\flask --app run create-hr
venv\Scripts\flask --app run seed-jobs       # optional: three sample job roles
```

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

## Project layout

```
app/
  __init__.py      app factory, security headers, CLI commands
  config.py        per-environment settings
  extensions.py    db, migrations, login, CSRF, rate limiter, security headers
  tasks.py         background jobs (Huey)
  models.py        User, Job, Application
  auth.py          sign up / log in / log out
  candidate.py     job browsing, resume upload, results
  hr.py            job management, applicant ranking and review
  services/        Gemini wrapper, resume extraction and matching
  templates/       Jinja2 + Bootstrap pages
migrations/        Alembic migration scripts
tests/             pytest suite
sample_resumes/    fictional resumes for testing
```
