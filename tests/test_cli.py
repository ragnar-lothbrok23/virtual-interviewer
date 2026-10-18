from app.models import Job, User


def run(app, *args):
    return app.test_cli_runner().invoke(args=list(args))


def test_create_hr(app):
    result = run(app, "create-hr", "--name", "HR Admin", "--email", " HR@Test.com ", "--password", "Recruit-2026!")
    assert result.exit_code == 0 and "HR account created" in result.output
    with app.app_context():
        user = User.query.one()
        assert user.email == "hr@test.com" and user.role == User.ROLE_HR and user.check_password("Recruit-2026!")
        assert user.is_verified


def test_create_admin(app):
    result = run(app, "create-admin", "--name", "Root", "--email", "root@test.com", "--password", "Admin-Pass-2026")
    assert result.exit_code == 0 and "Admin account created" in result.output
    with app.app_context():
        assert User.query.one().is_admin


def test_create_hr_validation(app, make_user):
    make_user("taken@test.com")
    assert "at least 8" in run(app, "create-hr", "--name", "A", "--email", "a@test.com", "--password", "short").output
    assert "valid email" in run(app, "create-hr", "--name", "A", "--email", "nope", "--password", "Recruit-2026!").output
    assert "too common" in run(app, "create-hr", "--name", "A", "--email", "a@test.com", "--password", "password123").output
    assert "already exists" in run(app, "create-hr", "--name", "A", "--email", "taken@test.com", "--password", "Recruit-2026!").output
    with app.app_context():
        assert User.query.filter_by(role=User.ROLE_HR).count() == 0


def test_seed_jobs_is_idempotent(app):
    assert "Added 3" in run(app, "seed-jobs").output
    assert "Added 0" in run(app, "seed-jobs").output
    with app.app_context():
        assert Job.query.count() == 3


def test_queue_check_runs_job(app):
    result = run(app, "queue-check")
    assert result.exit_code == 0 and "Worker OK" in result.output


def test_queue_check_reports_missing_worker(app, monkeypatch):
    import app.tasks as tasks

    class NoReply:
        def get(self, blocking, timeout):
            raise TimeoutError

    monkeypatch.setattr(tasks, "ping", lambda message: NoReply())
    result = run(app, "queue-check", "--timeout", "1")
    assert result.exit_code != 0 and "Is the worker running?" in result.output
