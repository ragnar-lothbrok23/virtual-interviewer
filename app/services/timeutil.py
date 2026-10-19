"""Times are stored in UTC and shown / entered in APP_TIMEZONE (default Asia/Kolkata)."""
from datetime import timezone
from zoneinfo import ZoneInfo

from flask import current_app


def app_tz():
    return ZoneInfo(current_app.config["APP_TIMEZONE"])


def to_local(value):
    """UTC (aware or naive-from-SQLite) -> aware datetime in the app timezone."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(app_tz())


def from_local_naive(value):
    """A naive datetime typed into a form (meaning app-timezone wall time) -> aware UTC."""
    if value is None:
        return None
    return value.replace(tzinfo=app_tz()).astimezone(timezone.utc)


def format_local(value, fmt="%d %b %Y, %I:%M %p"):
    local = to_local(value)
    return f"{local.strftime(fmt)} {local.tzname()}" if local else ""


def register_filters(app):
    app.jinja_env.filters["localtime"] = format_local
    app.jinja_env.filters["localinput"] = lambda v: to_local(v).strftime("%Y-%m-%dT%H:%M") if v else ""
