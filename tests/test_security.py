def test_security_headers_present(app):
    response = app.test_client().get("/")
    csp = response.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
    assert "'nonce-" in csp  # inline scripts need the per-request nonce
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "camera=(self)" in response.headers["Permissions-Policy"]


def test_session_cookie_is_httponly_and_samesite(app, make_user):
    make_user("a@test.com")
    response = app.test_client().post("/login", data={"email": "a@test.com", "password": "password123"})
    cookie = response.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie


def test_csrf_is_enforced_when_enabled(app, make_user):
    make_user("a@test.com")
    app.config["WTF_CSRF_ENABLED"] = True
    response = app.test_client().post("/login", data={"email": "a@test.com", "password": "password123"})
    assert response.status_code == 400
    assert b"form expired" in response.data


def test_login_rate_limit(app):
    client = app.test_client()
    codes = [client.post("/login", data={"email": "x@test.com", "password": "wrongpass1"}).status_code for _ in range(11)]
    assert codes[:10] == [200] * 10
    assert codes[10] == 429


def test_rate_limit_does_not_apply_to_page_views(app):
    client = app.test_client()
    codes = {client.get("/login").status_code for _ in range(15)}
    assert codes == {200}


def test_healthz(app):
    response = app.test_client().get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "database": "ok"}


def test_error_pages_html_and_json(app):
    client = app.test_client()
    page = client.get("/does-not-exist")
    assert page.status_code == 404 and b"Page not found" in page.data
    api = client.get("/does-not-exist", headers={"Accept": "application/json"})
    assert api.status_code == 404 and api.get_json() == {"error": "Page not found."}
    assert client.post("/").status_code == 405


def test_unhandled_exception_returns_friendly_500():
    # A fresh app: Flask doesn't allow adding routes to the shared app after it has served requests
    from app import create_app
    from app.config import TestingConfig

    class Config(TestingConfig):
        PROPAGATE_EXCEPTIONS = False

    fresh = create_app(Config)

    @fresh.route("/_boom")
    def boom():
        raise RuntimeError("kaboom")

    response = fresh.test_client().get("/_boom")
    assert response.status_code == 500
    assert b"Something went wrong" in response.data and b"kaboom" not in response.data


def test_healthz_reports_database_outage(app, monkeypatch):
    from app.extensions import db

    def down(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(db.session, "execute", down)
    response = app.test_client().get("/healthz")
    assert response.status_code == 503
    assert response.get_json() == {"status": "error", "database": "unreachable"}


def test_trust_proxy_uses_forwarded_client_ip():
    from flask import request

    from app import create_app
    from app.config import TestingConfig

    class Config(TestingConfig):
        TRUST_PROXY = True

    fresh = create_app(Config)

    @fresh.route("/_ip")
    def ip():
        return f"{request.remote_addr}|{request.scheme}"

    body = fresh.test_client().get("/_ip", headers={"X-Forwarded-For": "203.0.113.7", "X-Forwarded-Proto": "https"}).data
    assert body == b"203.0.113.7|https"
