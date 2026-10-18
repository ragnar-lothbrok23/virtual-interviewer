import io
from pathlib import Path

import pytest

from app.extensions import db
from app.models import Application, Job
from app.services.gemini import GeminiError

SAMPLES = Path(__file__).resolve().parent.parent / "sample_resumes"
BACKEND_RESUME = (SAMPLES / "arjun_backend_developer.docx").read_bytes()
FRONTEND_RESUME = (SAMPLES / "karthik_frontend_developer.docx").read_bytes()
PDF_RESUME = (SAMPLES / "priya_data_analyst.pdf").read_bytes()


@pytest.fixture
def jobs(app):
    app.test_cli_runner().invoke(args=["seed-jobs"])
    with app.app_context():
        return {j.title: j.id for j in Job.query.all()}


def apply(client, job_id, content=BACKEND_RESUME, filename="cv.docx"):
    return client.post(f"/candidate/jobs/{job_id}/apply", data={"resume": (io.BytesIO(content), filename)},
                       content_type="multipart/form-data", follow_redirects=True)


# ---------- HR: jobs ----------

def test_hr_creates_job_with_case_insensitive_skill_dedupe(app, hr_client):
    response = hr_client.post("/hr/jobs", data={"title": " ML Engineer ", "skills": "Python, PyTorch, python, , SQL",
                                                "description": "Build and deploy machine learning models " * 3},
                              follow_redirects=True)
    assert b"ML Engineer" in response.data
    with app.app_context():
        job = Job.query.one()
        assert job.title == "ML Engineer" and job.skills == "Python, PyTorch, SQL" and job.is_open


def test_job_form_validation(app, hr_client):
    response = hr_client.post("/hr/jobs", data={"title": "", "skills": "", "description": "too short"})
    assert response.status_code == 200 and b"This field is required" in response.data
    with app.app_context():
        assert Job.query.count() == 0


def test_hr_toggles_job_open_closed(app, hr_client, jobs):
    job_id = jobs["Data Analyst"]
    hr_client.post(f"/hr/jobs/{job_id}/toggle")
    with app.app_context():
        assert not db.session.get(Job, job_id).is_open
    hr_client.post(f"/hr/jobs/{job_id}/toggle")
    with app.app_context():
        assert db.session.get(Job, job_id).is_open


def test_toggle_requires_post(hr_client, jobs):
    assert hr_client.get(f"/hr/jobs/{jobs['Data Analyst']}/toggle").status_code == 405


def test_missing_job_and_application_return_404(hr_client, candidate_client):
    assert hr_client.get("/hr/jobs/9999").status_code == 404
    assert hr_client.get("/hr/applications/9999").status_code == 404
    assert candidate_client.get("/candidate/jobs/9999/apply").status_code == 404


# ---------- Candidate: applying ----------

def test_candidate_sees_only_open_jobs(app, hr_client, candidate_client, jobs):
    hr_client.post(f"/hr/jobs/{jobs['Frontend Developer']}/toggle")
    page = candidate_client.get("/candidate/jobs").data
    assert b"Python Backend Developer" in page and b"Frontend Developer" not in page


def test_apply_scores_and_auto_shortlists_good_fit(app, candidate_client, jobs, fake_gemini):
    response = apply(candidate_client, jobs["Python Backend Developer"], filename="My CV (final).docx")
    assert b"Match score for" in response.data and b"Shortlisted" in response.data
    with app.app_context():
        a = Application.query.one()
        assert a.match_score == 100.0 and a.status == Application.STATUS_SHORTLISTED
        assert "Python" in a.parsed["matched_skills"]
        assert a.resume_original_name == "My_CV_final.docx"  # sanitised
        assert (app.config["UPLOAD_FOLDER"] / a.resume_path).exists()
        assert ".." not in a.resume_path and "/" not in a.resume_path


def test_poor_fit_goes_to_review_and_ranks_lower(app, candidate_client, jobs, fake_gemini):
    apply(candidate_client, jobs["Python Backend Developer"])
    response = apply(candidate_client, jobs["Frontend Developer"])
    assert b"Under review" in response.data
    with app.app_context():
        scores = {a.job.title: a.match_score for a in Application.query.all()}
    assert scores["Python Backend Developer"] > scores["Frontend Developer"]


def test_pdf_resume_is_accepted(app, candidate_client, jobs, fake_gemini):
    response = apply(candidate_client, jobs["Data Analyst"], content=PDF_RESUME, filename="priya.pdf")
    assert b"Match score for" in response.data


@pytest.mark.parametrize("content, filename, message", [
    (b"plain text", "cv.txt", b"Only PDF or DOCX"),
    (b"MZ\x90\x00 fake exe", "cv.exe", b"Only PDF or DOCX"),
    (b"not really a pdf", "cv.pdf", b"Could not read the file"),
    (b"", "empty.docx", b"Could not read the file"),
])
def test_bad_uploads_are_rejected_cleanly(app, candidate_client, jobs, fake_gemini, content, filename, message):
    response = apply(candidate_client, jobs["Python Backend Developer"], content=content, filename=filename)
    assert message in response.data
    with app.app_context():
        assert Application.query.count() == 0
    assert list(app.config["UPLOAD_FOLDER"].iterdir()) == []
    assert fake_gemini.parse_calls == 0


def test_upload_without_file(candidate_client, jobs):
    response = candidate_client.post(f"/candidate/jobs/{jobs['Data Analyst']}/apply", data={}, content_type="multipart/form-data")
    assert b"Please choose a file" in response.data


def test_oversized_upload_rejected(candidate_client, jobs):
    big = b"%PDF-1.4" + b"0" * (5 * 1024 * 1024 + 10)
    response = apply(candidate_client, jobs["Data Analyst"], content=big, filename="big.pdf")
    assert response.status_code == 413


def test_gemini_failure_shows_message_and_saves_nothing(app, candidate_client, jobs, fake_gemini):
    fake_gemini.error = GeminiError("The AI service has reached its usage limit for now. Please try again later.")
    response = apply(candidate_client, jobs["Python Backend Developer"])
    assert b"usage limit" in response.data
    with app.app_context():
        assert Application.query.count() == 0
    assert list(app.config["UPLOAD_FOLDER"].iterdir()) == []


def test_reapplying_redirects_to_existing_application(app, candidate_client, jobs, fake_gemini):
    apply(candidate_client, jobs["Python Backend Developer"])
    response = candidate_client.get(f"/candidate/jobs/{jobs['Python Backend Developer']}/apply")
    assert response.status_code == 302 and "/candidate/applications/" in response.headers["Location"]
    apply(candidate_client, jobs["Python Backend Developer"])  # a second POST must not create a duplicate
    with app.app_context():
        assert Application.query.count() == 1
    assert fake_gemini.parse_calls == 1


def test_closed_job_refuses_applications(app, hr_client, candidate_client, jobs, fake_gemini):
    hr_client.post(f"/hr/jobs/{jobs['Data Analyst']}/toggle")
    response = apply(candidate_client, jobs["Data Analyst"])
    assert b"no longer accepting" in response.data
    assert fake_gemini.parse_calls == 0


def test_job_embedding_is_cached(app, candidate_client, make_user, login, jobs, fake_gemini):
    apply(candidate_client, jobs["Python Backend Developer"])
    make_user("second@test.com")
    apply(login("second@test.com"), jobs["Python Backend Developer"])
    assert fake_gemini.embed_calls == 3  # job once + two resumes


def test_candidate_cannot_see_others_application(app, candidate_client, make_user, login, jobs, fake_gemini):
    apply(candidate_client, jobs["Python Backend Developer"])
    with app.app_context():
        app_id = Application.query.one().id
    make_user("other@test.com")
    assert login("other@test.com").get(f"/candidate/applications/{app_id}").status_code == 404


# ---------- HR: reviewing ----------

@pytest.fixture
def application_id(app, candidate_client, jobs, fake_gemini):
    apply(candidate_client, jobs["Python Backend Developer"])
    with app.app_context():
        return Application.query.one().id


def test_hr_sees_ranked_applicants_and_filters(hr_client, jobs, application_id):
    job_id = jobs["Python Backend Developer"]
    assert b"Arjun" in hr_client.get(f"/hr/jobs/{job_id}").data
    assert b"Arjun" in hr_client.get(f"/hr/jobs/{job_id}?status=shortlisted").data
    assert b"No applicants with this status" in hr_client.get(f"/hr/jobs/{job_id}?status=rejected").data
    assert b"Arjun" in hr_client.get(f"/hr/jobs/{job_id}?status=nonsense").data  # unknown filter -> all


def test_hr_reviews_application(hr_client, application_id):
    page = hr_client.get(f"/hr/applications/{application_id}").data
    assert b"Parsed resume profile" in page and b"Anna University" in page


def test_hr_changes_status(app, hr_client, application_id):
    response = hr_client.post(f"/hr/applications/{application_id}/status", data={"status": "rejected"}, follow_redirects=True)
    assert b"marked as rejected" in response.data
    with app.app_context():
        assert db.session.get(Application, application_id).status == Application.STATUS_REJECTED
    assert hr_client.post(f"/hr/applications/{application_id}/status", data={"status": "hired!"}).status_code == 400


def test_hr_downloads_resume(hr_client, application_id):
    response = hr_client.get(f"/hr/applications/{application_id}/resume")
    assert response.status_code == 200 and response.data[:2] == b"PK"
    assert "attachment" in response.headers["Content-Disposition"]
    response.close()


def test_candidate_cannot_use_hr_actions(candidate_client, application_id):
    assert candidate_client.post(f"/hr/applications/{application_id}/status", data={"status": "shortlisted"}).status_code == 403
    assert candidate_client.get(f"/hr/applications/{application_id}/resume").status_code == 403


def test_dashboards_render_with_data(hr_client, candidate_client, application_id):
    assert b"Applications" in hr_client.get("/hr/").data
    assert b"Python Backend Developer" in candidate_client.get("/candidate/").data
