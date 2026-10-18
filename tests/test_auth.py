from app.extensions import db
from app.models import User


def register(client, **overrides):
    data = {"name": "Asha", "email": "Asha@Test.com", "password": "secret123", "confirm": "secret123", **overrides}
    return client.post("/register", data=data, follow_redirects=True)


def test_home_page_loads(app):
    assert app.test_client().get("/").status_code == 200


def test_candidate_signup_lands_on_dashboard(app):
    response = register(app.test_client())
    assert b"Hi, Asha" in response.data
    with app.app_context():
        user = User.query.one()
        assert user.email == "asha@test.com"  # stored lower-case
        assert user.role == User.ROLE_CANDIDATE
        assert user.password_hash.startswith(("scrypt", "pbkdf2")) and "secret123" not in user.password_hash


def test_signup_rejects_duplicate_email_case_insensitively(app, make_user):
    make_user("asha@test.com")
    response = register(app.test_client(), email="ASHA@test.com")
    assert b"already exists" in response.data


def test_signup_validation_errors(app):
    client = app.test_client()
    assert b"at least 8" in register(client, password="short", confirm="short").data
    assert b"Passwords must match" in register(client, confirm="different1").data
    assert b"Invalid email" in register(client, email="not-an-email").data
    with app.app_context():
        assert User.query.count() == 0


def test_login_success_and_failure(app, make_user):
    make_user("a@test.com")
    client = app.test_client()
    assert b"Invalid email or password" in client.post("/login", data={"email": "a@test.com", "password": "wrong-pass"}).data
    assert b"Invalid email or password" in client.post("/login", data={"email": "nobody@test.com", "password": "password123"}).data
    response = client.post("/login", data={"email": "A@TEST.COM", "password": "password123"})
    assert response.status_code == 302 and response.headers["Location"].endswith("/dashboard")


def test_login_next_parameter_only_allows_local_redirects(app, make_user):
    make_user("a@test.com")
    client = app.test_client()
    evil = client.post("/login?next=https://evil.example/phish", data={"email": "a@test.com", "password": "password123"})
    assert "evil.example" not in evil.headers["Location"]
    client.get("/logout")
    local = client.post("/login?next=/candidate/jobs", data={"email": "a@test.com", "password": "password123"})
    assert local.headers["Location"].endswith("/candidate/jobs")


def test_logout_requires_login_and_logs_out(candidate_client):
    assert candidate_client.get("/logout").status_code == 302
    assert candidate_client.get("/candidate/").status_code == 302  # back to login


def test_logged_in_user_visiting_login_is_redirected(candidate_client):
    assert candidate_client.get("/login").status_code == 302
    assert candidate_client.get("/register").status_code == 302


def test_role_separation(app, hr_client, candidate_client):
    assert app.test_client().get("/hr/").status_code == 302  # anonymous -> login
    assert candidate_client.get("/hr/").status_code == 403
    assert candidate_client.get("/hr/jobs").status_code == 403
    assert hr_client.get("/candidate/").status_code == 403
    assert hr_client.get("/candidate/jobs").status_code == 403
    assert hr_client.get("/dashboard").headers["Location"].endswith("/hr/")
    assert candidate_client.get("/dashboard").headers["Location"].endswith("/candidate/")


def test_deleted_user_session_is_logged_out(app, make_user, login):
    user_id = make_user("gone@test.com")
    client = login("gone@test.com")
    with app.app_context():
        db.session.delete(db.session.get(User, user_id))
        db.session.commit()
    assert client.get("/candidate/").status_code == 302
