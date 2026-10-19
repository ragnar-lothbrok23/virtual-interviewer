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
copy .env.example .env          # then fill in SECRET_KEY, GEMINI_API_KEY and FILE_ENCRYPTION_KEYS
venv\Scriptslask --app run generate-encryption-key   # paste the output into FILE_ENCRYPTION_KEYS
venv\Scripts\flask --app run db upgrade
venv\Scripts\flask --app run create-admin    # first admin; invite HR users from the Admin panel
venv\Scripts\flask --app run seed-jobs       # optional: three sample job roles
```

### Email

Emails are sent by the background worker, so it must be running.

- `MAIL_BACKEND=console` (default): nothing is sent; the worker prints each email and saves it as an `.eml`
  file in `instance/outbox/`.
- `MAIL_BACKEND=gmail_smtp`: real sending through the Gmail account in `MAIL_USERNAME` with its app password.
  In development, also set `MAIL_REDIRECT_TO` to your own inbox so test accounts (e.g. `@test.com`) never
  email strangers; the intended recipient appears in the subject.

Check delivery with `venv\Scripts\flask --app run mail-test --to you@example.com`. Admins can see every
email's status (sent, retrying, deferred, failed) under **Admin → Emails**.

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
venv\Scripts\pytest --e2e               # also run real-browser tests (Chrome, simulated camera/mic)
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

## Resumes

- Uploads are checked by their real contents (PDF or DOCX; macro-enabled and oversized "zip bomb" files are
  refused), then stored **encrypted with AES-256-GCM** (`FILE_ENCRYPTION_KEYS`). Back up the key: without it
  the stored resumes cannot be read. To rotate, put a new key first and keep the old ones after it.
- Gemini analysis runs in the background worker; the candidate's page updates when it finishes. Temporary AI
  errors are retried automatically; HR can also press **Re-analyse**.
- The match score is the paper's cosine similarity between the resume and the job. Skill evidence
  (proven in projects/experience, only listed, missing) is shown alongside it for HR.
- Editing a job's description re-scores its applicants; editing its required skills re-analyses them
  (one AI request each). HR decisions are never overwritten.
- After upgrading from a version without encryption: `venv\Scriptslask --app run encrypt-existing-resumes`.

## Interview scheduling

- HR opens interview slots per job (Job → **Interview slots**), one at a time or in bulk (date range, weekdays,
  daily hours, length, break). AI interviews run in parallel, so a slot can hold several candidates.
- Shortlisted candidates book from their application page (at least 1 hour ahead), and can reschedule up to
  2 times or cancel until 2 hours before. Booking is race-safe: the last seat can't be taken twice.
- Booking, rescheduling and cancellation emails carry a calendar invite (.ics); updates replace the same entry.
- The worker's periodic job (every 5 minutes) sends 24-hour and 1-hour reminders and marks missed interviews.
  **The worker must be running** for reminders to go out.
- If an application stops being shortlisted (withdrawn, rejected, back to review), its interview is cancelled
  automatically and the seat freed.

## Interview room

- Each job has an ordered question list (Job → **Questions**); if none is set, defaults are generated from the
  job's skills when the first interview starts. An interview keeps the questions it started with.
- Candidates join from 10 minutes before their slot: consent → camera/microphone/browser/network check → for each
  question, reading time then a timed answer recorded on camera. Answers upload as they go and are stored
  encrypted; the worker transcribes them with Gemini (one request per answer).
- Progress is kept on the server: after a crash or dropped connection the candidate rejoins and continues.
  Tab switches, focus loss, reconnects, restarted questions and late answers are recorded for HR.
- HR sees each answer's video, transcript and timing, plus an integrity timeline, on the applicant's page.
- Browsers only allow camera and microphone on `https://` sites or `localhost`.
- Development only: `venv\Scriptslask --app run dev-interview-soon <interview id>` moves a booked interview to
  start in 5 minutes so the room can be tried immediately.

## Answer scoring and adaptive questions

- Every question has a reference answer (written by Gemini in the background, editable by HR in Job → Questions).
- After each answer is transcribed, the worker scores it: semantic similarity to the reference answer (Gemini
  embeddings + cosine), a Gemini rubric (relevance, accuracy, fluency, clarity) with evidence quotes checked
  against the transcript, and locally measured speaking pace and filler words.
  Per-answer NLP score = 30% relevance + 30% accuracy + 20% similarity + 20% fluency (`NLP_WEIGHTS`).
- Adaptive rule (paper §3.3): a technical answer whose similarity score is below `FOLLOW_UP_THRESHOLD` (55)
  gets one focused follow-up question (at most `MAX_FOLLOW_UPS` = 3 per interview). The room shows
  "Preparing your next question…" meanwhile; if the decision takes over 45 s the interview simply moves on.
- HR sees every score, the quotes and strengths/gaps per answer, an interview NLP summary, and can re-score.
- Calibration with real Gemini: strong / medium / weak spoken answers scored 87.5 / 53.5 / 30.9.

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
  models.py        User, AuditLog, EmailLog, Notification, Job, Application, InterviewSlot, Interview
  auth.py          sign up, log in/out, email verification, password reset, invitations, consent
  account.py       profile, email and password changes, data export, account deletion
  admin.py         user management, email log and audit log
  notifications.py in-app notifications (bell icon)
  candidate.py     job browsing, resume upload, results
  hr.py            job management, applicant ranking and review
  services/        Gemini, resume matching and storage, scheduling and calendar invites, mail (Gmail SMTP),
                   notifications, tokens/password policy/audit, account operations
  templates/       Jinja2 + Bootstrap pages
migrations/        Alembic migration scripts
tests/             pytest suite
sample_resumes/    fictional resumes for testing
```
