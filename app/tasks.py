"""Background jobs (Huey).

Run the worker with:   venv\\Scripts\\huey_consumer app.tasks.huey -k thread -w 2
Queue backend: Redis when REDIS_URL is set (production), otherwise a SQLite file (works natively on Windows).
In testing (APP_ENV=testing) jobs run immediately in-process.
"""
import os
import socket
from datetime import datetime, timezone

from flask import current_app, has_app_context
from huey import MemoryHuey, RedisHuey, SqliteHuey

from .config import get_config


def _make_huey():
    config = get_config()
    if config.ENV_NAME == "testing":
        return MemoryHuey("virtual-interviewer", immediate=True)
    if config.REDIS_URL:
        return RedisHuey("virtual-interviewer", url=config.REDIS_URL)
    config.HUEY_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return SqliteHuey("virtual-interviewer", filename=str(config.HUEY_SQLITE_PATH))


huey = _make_huey()
_worker_app = None


def app_context():
    """Jobs need the Flask app (database, config). Reuse the current one, or build one once per worker process."""
    global _worker_app
    if has_app_context():
        return current_app.app_context()
    if _worker_app is None:
        from . import create_app

        _worker_app = create_app()
    return _worker_app.app_context()


@huey.task()
def ping(message):
    """Health check job: proves a worker is consuming the queue."""
    with app_context():
        return {"message": message, "worker": socket.gethostname(), "pid": os.getpid(),
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
