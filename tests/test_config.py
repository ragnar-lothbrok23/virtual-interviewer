import pytest

from app.config import DEV_SECRET, ProductionConfig, database_url, get_config


def test_get_config_by_name_and_env(monkeypatch):
    assert get_config("production") is ProductionConfig
    monkeypatch.setenv("APP_ENV", "testing")
    assert get_config().ENV_NAME == "testing"
    with pytest.raises(RuntimeError, match="Unknown APP_ENV"):
        get_config("staging")


@pytest.mark.parametrize("given, expected", [
    ("postgres://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
    ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
    ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
    ("sqlite:///x.db", "sqlite:///x.db"),
])
def test_database_url_normalisation(monkeypatch, given, expected):
    monkeypatch.setenv("DATABASE_URL", given)
    assert database_url("sqlite://") == expected


def test_database_url_default_when_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert database_url("sqlite:///default.db") == "sqlite:///default.db"


def _prod(monkeypatch, **attrs):
    for key, value in attrs.items():
        monkeypatch.setattr(ProductionConfig, key, value)


def test_production_rejects_unsafe_config(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _prod(monkeypatch, SECRET_KEY=DEV_SECRET, REDIS_URL="", GEMINI_API_KEY="")
    with pytest.raises(RuntimeError) as err:
        ProductionConfig.validate()
    message = str(err.value)
    for expected in ("SECRET_KEY", "DATABASE_URL", "REDIS_URL", "GEMINI_API_KEY"):
        assert expected in message


def test_production_rejects_short_secret(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    _prod(monkeypatch, SECRET_KEY="short", REDIS_URL="redis://localhost:6379/0", GEMINI_API_KEY="k")
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        ProductionConfig.validate()


def test_production_accepts_complete_config(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    _prod(monkeypatch, SECRET_KEY="x" * 64, REDIS_URL="redis://localhost:6379/0", GEMINI_API_KEY="k",
          APP_BASE_URL="https://interviews.example.com", MAIL_BACKEND="gmail_smtp",
          MAIL_USERNAME="system@gmail.com", MAIL_APP_PASSWORD="abcdefghijklmnop", MAIL_REDIRECT_TO="",
          FILE_ENCRYPTION_KEYS="k1:" + "A" * 43 + "=")
    ProductionConfig.validate()


def test_production_requires_https_base_url_and_real_mail(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    _prod(monkeypatch, SECRET_KEY="x" * 64, REDIS_URL="redis://r", GEMINI_API_KEY="k",
          APP_BASE_URL="http://insecure.example.com", MAIL_BACKEND="console")
    with pytest.raises(RuntimeError) as err:
        ProductionConfig.validate()
    assert "APP_BASE_URL" in str(err.value) and "MAIL_BACKEND" in str(err.value)


def test_production_cookie_and_https_settings():
    assert ProductionConfig.SESSION_COOKIE_SECURE and ProductionConfig.REMEMBER_COOKIE_SECURE
    assert ProductionConfig.SESSION_COOKIE_HTTPONLY
