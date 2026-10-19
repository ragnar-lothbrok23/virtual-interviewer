import io
import re
import zipfile
from datetime import timedelta
from types import SimpleNamespace

import pytest
from freezegun import freeze_time

from app.extensions import db
from app.models import (AuditLog, Interview, InterviewAnswer, InterviewEvent, InterviewQuestion, Job, Notification,
                        User, as_utc)
from app.services import gemini, interview_room, storage, transcription
from app.services.gemini import GeminiError
from tests.conftest import PASSWORD
from tests.test_scheduling import add_slot, booked, get, job_id, shortlist  # noqa: F401  (fixtures)

WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 200  # EBML header + body
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 200
ORIGINAL_TRANSCRIBE = gemini.transcribe


@pytest.fixture(autouse=True)
def fake_transcribe(monkeypatch):
    calls = []

    def fake(media, mime_type):
        calls.append((len(media), mime_type))
        return {"speech_detected": True, "language": "en", "transcript": f"Answer of {len(media)} bytes"}

    monkeypatch.setattr(gemini, "transcribe", fake)
    return calls


def start_of(app, interview_id):
    return as_utc(get(app, Interview, interview_id).starts_at)


@pytest.fixture
def room(app, booked, login):
    """Clock frozen 5 minutes before the start, candidate signed in at that moment. Yields (client, interview_id)."""
    _, interview_id, _ = booked
    with freeze_time(start_of(app, interview_id) - timedelta(minutes=5)) as frozen:
        client = login("cand@test.com")
        yield SimpleNamespace(client=client, id=interview_id, frozen=frozen, base=f"/candidate/interviews/{interview_id}")


def api(room, path, body=None):
    return room.client.post(f"{room.base}/room/{path}", json=body or {})


def upload(room, position, data=WEBM, duration="42.5"):
    return room.client.post(f"{room.base}/room/answer", content_type="multipart/form-data",
                            data={"position": str(position), "duration": duration, "media": (io.BytesIO(data), "a.webm")})


def started(room):
    response = api(room, "start", {"consent": True})
    assert response.status_code == 200, response.get_json()
    return response.get_json()


# ---------- entering and starting ----------

def test_room_page_and_window(app, booked, candidate_client, room):
    page = room.client.get(f"{room.base}/join").get_data(as_text=True)
    assert "Before we start" in page and 'name="csrf-token"' in page and "interview_room.js" in page
    data = re.search(r"data-state='([^']*)'", page).group(1)
    assert '"status": "scheduled"' in data.replace("&#34;", '"') or "scheduled" in data


def test_start_requires_consent_and_freezes_questions(app, room):
    response = api(room, "start", {"consent": False})
    assert response.status_code == 400 and "agree to the recording" in response.get_json()["error"]
    state = started(room)
    assert state["status"] == "in_progress" and state["position"] == 0 and state["total"] == 7
    assert state["question"]["prep_seconds"] == 20 and "Tell us about yourself" in state["question"]["text"]
    interview = get(app, Interview, room.id)
    assert interview.recording_consent_at is not None and interview.hard_deadline is not None
    with app.app_context():
        assert InterviewQuestion.query.count() == 7  # defaults created from the job's skills
        assert InterviewEvent.query.filter_by(kind="started").count() == 1
        assert AuditLog.query.filter_by(action="interview.started").count() == 1


def test_start_outside_window_is_refused(app, booked, login):
    _, interview_id, _ = booked
    client = login("cand@test.com")  # now: 48 h before the interview
    response = client.post(f"/candidate/interviews/{interview_id}/room/start", json={"consent": True})
    assert response.status_code == 403


# ---------- the question loop ----------

def test_full_interview(app, room, fake_transcribe):
    started(room)
    for position in range(7):
        assert api(room, "ask", {"position": position}).status_code == 200
        room.frozen.tick(60)
        response = upload(room, position, data=WEBM if position % 2 == 0 else MP4)
        assert response.status_code == 200
    state = response.get_json()
    assert state["finished"] and state["status"] == "completed"
    interview = get(app, Interview, room.id)
    assert interview.end_reason == "completed" and interview.current_position == 7
    with app.app_context():
        answers = InterviewAnswer.query.order_by(InterviewAnswer.position).all()
        assert [a.transcript_status for a in answers] == ["done"] * 7
        assert answers[1].media_type == "video/mp4" and answers[0].media_type == "video/webm"
        assert answers[0].duration_seconds == 42.5 and not answers[0].late
        on_disk = (app.config["UPLOAD_FOLDER"] / answers[0].media_path).read_bytes()
        assert on_disk.startswith(b"VIE1") and WEBM not in on_disk  # encrypted at rest
        kinds = {n.kind for n in Notification.query}
        assert {"interview.completed", "interview.completed_hr"} <= kinds
    assert len(fake_transcribe) == 7
    # nothing more can be asked; re-sending the final answer is a harmless no-op
    assert api(room, "ask", {"position": 6}).status_code == 403
    assert upload(room, 6).status_code == 200


def test_order_and_validation(app, room):
    started(room)
    assert api(room, "ask", {"position": 1}).status_code == 409  # not the current question
    assert api(room, "ask", {}).status_code == 400
    assert upload(room, 0).status_code == 409  # not shown yet
    api(room, "ask", {"position": 0})
    response = upload(room, 0, data=b"definitely not a video file" * 10)
    assert response.status_code == 400 and "couldn't be read" in response.get_json()["error"]
    app.config["MAX_ANSWER_UPLOAD"] = 100
    try:
        assert upload(room, 0).status_code == 413
    finally:
        app.config["MAX_ANSWER_UPLOAD"] = 60 * 1024 * 1024
    assert get(app, Interview, room.id).current_position == 0
    assert upload(room, 0).status_code == 200
    assert upload(room, 0).status_code == 200  # duplicate (retry after a dropped response): no-op
    assert get(app, Interview, room.id).current_position == 1


def test_reported_duration_is_clamped(app, room):
    started(room)
    api(room, "ask", {"position": 0})
    upload(room, 0, duration="99999")
    with app.app_context():
        assert InterviewAnswer.query.one().duration_seconds == 120 + 15
    api(room, "ask", {"position": 1})
    upload(room, 1, duration="not a number")
    with app.app_context():
        assert InterviewAnswer.query.filter_by(position=1).one().duration_seconds is None


def test_late_answer_is_flagged(app, room):
    started(room)
    api(room, "ask", {"position": 0})
    room.frozen.tick(20 + 120 + 16)  # prep + limit + just over the grace period
    upload(room, 0)
    with app.app_context():
        assert InterviewAnswer.query.one().late is True
        assert InterviewEvent.query.filter_by(kind="late_answer").count() == 1


def test_reconnect_resumes_and_flags_restarted_question(app, room):
    started(room)
    api(room, "ask", {"position": 0})
    upload(room, 0)
    api(room, "ask", {"position": 1})
    # browser crashes mid-answer; the candidate comes back
    page = room.client.get(f"{room.base}/join").get_data(as_text=True)
    assert "in_progress" in page
    state = api(room, "start", {}).get_json()  # rejoin needs no new consent
    assert state["position"] == 1 and state["question"]["position"] == 1
    api(room, "ask", {"position": 1})
    with app.app_context():
        answer = InterviewAnswer.query.filter_by(position=1).one()
        assert answer.attempts == 2
        kinds = [e.kind for e in InterviewEvent.query.order_by(InterviewEvent.id)]
        assert "reconnect" in kinds and "question_restarted" in kinds


def test_can_continue_after_join_window_until_hard_deadline(app, room):
    started(room)
    interview = get(app, Interview, room.id)
    hard = as_utc(interview.hard_deadline)
    assert hard > interview.ends_at
    room.frozen.move_to(interview.ends_at + (hard - interview.ends_at) / 2)  # slot over, interview still running
    assert room.client.get(f"{room.base}/join").status_code == 200
    assert api(room, "ask", {"position": 0}).status_code == 200
    room.frozen.move_to(as_utc(interview.hard_deadline) + timedelta(seconds=1))
    assert api(room, "ask", {"position": 0}).status_code == 410
    with app.app_context():
        assert interview_room.expire_stale() == 1
    timed_out = get(app, Interview, room.id)
    assert timed_out.status == "completed" and timed_out.end_reason == "timed_out"


# ---------- integrity events ----------

def test_client_events(app, room):
    started(room)
    for kind in ("tab_hidden", "tab_visible", "window_blur"):
        assert api(room, "event", {"kind": kind, "position": 0, "details": {"x": "y" * 500}}).get_json() == {"recorded": True}
    assert api(room, "event", {"kind": "finished"}).status_code == 400  # clients can't fake lifecycle events
    with app.app_context():
        event = InterviewEvent.query.filter_by(kind="tab_hidden").one()
        assert event.flagged and event.position == 0 and len(event.details["x"]) == 200
    app.config["MAX_INTEGRITY_EVENTS"] = 4  # 'started' + the 3 events above
    try:
        assert api(room, "event", {"kind": "tab_hidden"}).get_json() == {"recorded": False}  # capped
    finally:
        app.config["MAX_INTEGRITY_EVENTS"] = 500


def test_events_ignored_when_not_running(app, room):
    assert api(room, "event", {"kind": "tab_hidden"}).get_json() == {"recorded": False}


# ---------- security ----------

def test_room_api_is_private(app, room, make_user, login, hr_client):
    started(room)
    make_user("other@test.com")
    other = login("other@test.com")
    assert other.post(f"{room.base}/room/start", json={"consent": True}).status_code == 404
    assert other.get(f"{room.base}/room/state").status_code == 404
    hr = login("hr@test.com")  # signed in at the frozen time (sessions expire after 12 h)
    assert hr.post(f"{room.base}/room/ask", json={"position": 0}).status_code == 403


def test_room_api_requires_csrf_header(app, room):
    app.config["WTF_CSRF_ENABLED"] = True
    try:
        assert api(room, "start", {"consent": True}).status_code == 400  # no token
        page = room.client.get(f"{room.base}/join").get_data(as_text=True)
        token = re.search(r'name="csrf-token" content="([^"]+)"', page).group(1)
        response = room.client.post(f"{room.base}/room/start", json={"consent": True}, headers={"X-CSRFToken": token})
        assert response.status_code == 200
    finally:
        app.config["WTF_CSRF_ENABLED"] = False


# ---------- transcription ----------

def run_transcription_for_first_answer(app, room, monkeypatch, fake):
    monkeypatch.setattr(gemini, "transcribe", fake)
    import app.services.interview_room as ir
    monkeypatch.setattr("app.tasks.transcribe_answer", lambda answer_id: None)  # don't auto-run
    started(room)
    api(room, "ask", {"position": 0})
    upload(room, 0)
    with app.app_context():
        return InterviewAnswer.query.one().id


def test_transcription_failure_and_retry(app, room, monkeypatch):
    def broken(media, mime_type):
        raise GeminiError("network hiccup", retryable=True)

    answer_id = run_transcription_for_first_answer(app, room, monkeypatch, broken)
    with app.app_context():
        with pytest.raises(GeminiError):
            transcription.run_transcription(answer_id, retries_left=2)
        assert db.session.get(InterviewAnswer, answer_id).transcript_status == "pending"
        assert transcription.run_transcription(answer_id, retries_left=0) == "failed"


def test_silence_gives_empty_transcript(app, room, monkeypatch):
    answer_id = run_transcription_for_first_answer(
        app, room, monkeypatch, lambda m, t: {"speech_detected": False, "language": "", "transcript": ""})
    with app.app_context():
        assert transcription.run_transcription(answer_id) == "done"
        answer = db.session.get(InterviewAnswer, answer_id)
        assert answer.speech_detected is False and answer.transcript == ""


def test_large_media_uses_files_api_and_cleans_up(app, monkeypatch):
    monkeypatch.setattr(gemini, "transcribe", ORIGINAL_TRANSCRIBE)
    deleted, uploaded = [], []

    class Files:
        def upload(self, file, config):
            uploaded.append(config.mime_type)
            return SimpleNamespace(name="files/abc")

        def delete(self, name):
            deleted.append(name)

    class Models:
        def generate_content(self, **kw):
            assert kw["contents"][0].name == "files/abc"
            return SimpleNamespace(parsed=gemini.Transcript(speech_detected=True, language="en", transcript="hi"))

    with app.app_context():
        app.extensions["gemini_client"] = SimpleNamespace(files=Files(), models=Models())
        app.config["GEMINI_INLINE_MEDIA_LIMIT"] = 10
        try:
            assert gemini.transcribe(WEBM, "video/webm")["transcript"] == "hi"
        finally:
            app.config["GEMINI_INLINE_MEDIA_LIMIT"] = 18 * 1024 * 1024
            app.extensions.pop("gemini_client")
    assert uploaded == ["video/webm"] and deleted == ["files/abc"]


# ---------- HR side ----------

def complete_two(app, room):
    started(room)
    for position in range(2):
        api(room, "ask", {"position": position})
        upload(room, position)
    api(room, "event", {"kind": "tab_hidden", "position": 1})


def test_hr_sees_answers_transcripts_and_flags(app, room, hr_client, booked, login):
    app_id, _, _ = booked
    complete_two(app, room)
    hr_client = login("hr@test.com")
    page = hr_client.get(f"/hr/applications/{app_id}").get_data(as_text=True)
    assert "Interview answers" in page and "2 of 7 answered" in page
    assert "Answer of 204 bytes" in page and "<video" in page
    assert "Left the interview tab" in page and "1 flag(s)" in page


def test_hr_streams_decrypted_recording_with_seeking(app, room, hr_client, login):
    complete_two(app, room)
    hr_client = login("hr@test.com")
    with app.app_context():
        answer_id = InterviewAnswer.query.filter_by(position=0).one().id
    full = hr_client.get(f"/hr/answers/{answer_id}/media")
    assert full.status_code == 200 and full.data == WEBM and full.mimetype == "video/webm"
    part = hr_client.get(f"/hr/answers/{answer_id}/media", headers={"Range": "bytes=0-3"})
    assert part.status_code == 206 and part.data == WEBM[:4]
    assert room.client.get(f"/hr/answers/{answer_id}/media").status_code == 403


def test_question_editor(app, hr_client, job_id):
    base = f"/hr/jobs/{job_id}/questions"
    assert b"Create default questions" in hr_client.get(base).data
    hr_client.post(f"{base}/defaults")
    with app.app_context():
        qs = InterviewQuestion.query.filter_by(job_id=job_id).order_by(InterviewQuestion.position).all()
        first, second = qs[0].id, qs[1].id
        assert len(qs) == 7
    response = hr_client.post(base, data={"text": "What is a Python decorator?", "category": "technical",
                                          "answer_seconds": "90"}, follow_redirects=True)
    assert b"Question added" in response.data
    bad = hr_client.post(f"/hr/questions/{first}/edit", data={f"q{first}-text": "Short", f"q{first}-category": "technical",
                                                             f"q{first}-answer_seconds": "10"}, follow_redirects=True)
    assert b"between 30 and 600" in bad.data
    hr_client.post(f"/hr/questions/{second}/move", data={"direction": "up"})
    hr_client.post(f"/hr/questions/{first}/delete")
    with app.app_context():
        qs = InterviewQuestion.query.filter_by(job_id=job_id).order_by(InterviewQuestion.position).all()
        assert qs[0].id == second and [q.position for q in qs] == list(range(7)) and qs[-1].answer_seconds == 90


def test_editing_questions_does_not_change_started_interview(app, room, hr_client, job_id):
    started(room)
    with app.app_context():
        q = InterviewQuestion.query.filter_by(job_id=job_id, position=0).one()
        q.text = "A completely different question?"
        db.session.commit()
    state = api(room, "ask", {"position": 0}).get_json()
    assert "Tell us about yourself" in state["question"]["text"]


# ---------- privacy ----------

def test_export_and_delete_cover_recordings(app, room):
    complete_two(app, room)
    response = room.client.post("/account/export")
    with zipfile.ZipFile(io.BytesIO(response.data)) as zf:
        recordings = [n for n in zf.namelist() if n.startswith("recordings/")]
        assert len(recordings) == 2 and zf.read(recordings[0]) == WEBM
        assert "Answer of 204 bytes" in zf.read("my_data.json").decode()
    response.close()
    files_before = len(list(app.config["UPLOAD_FOLDER"].iterdir()))
    assert files_before == 3  # resume + 2 recordings
    room.client.post("/account/delete", data={"delete-current_password": PASSWORD, "delete-confirm_text": "DELETE"})
    assert list(app.config["UPLOAD_FOLDER"].iterdir()) == []
    with app.app_context():
        assert Interview.query.count() == 0 and InterviewAnswer.query.count() == 0


def test_dev_interview_soon_command(app, booked):
    _, interview_id, _ = booked
    result = app.test_cli_runner().invoke(args=["dev-interview-soon", str(interview_id), "--minutes", "3"])
    assert result.exit_code == 0 and "now starts" in result.output
    with app.app_context():
        assert interview_room.can_enter(db.session.get(Interview, interview_id))
    assert "No scheduled interview" in app.test_cli_runner().invoke(args=["dev-interview-soon", "999"]).output
