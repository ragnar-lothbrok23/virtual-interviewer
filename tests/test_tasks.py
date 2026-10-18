from app.tasks import huey, ping


def test_testing_queue_runs_jobs_immediately(app):
    assert huey.immediate
    result = ping("hello")
    reply = result.get()
    assert reply["message"] == "hello" and "pid" in reply and "at" in reply


def test_job_runs_outside_request_with_its_own_app_context():
    # Called with no app context: the task must build/obtain one itself (as it does inside the worker)
    assert ping("no-context").get()["message"] == "no-context"
