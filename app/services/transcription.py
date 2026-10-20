"""Background transcription of recorded interview answers."""
from flask import current_app

from ..extensions import db
from ..models import InterviewAnswer
from . import gemini, media as media_service, scoring, storage
from .gemini import GeminiError


def run_transcription(answer_id, retries_left=0):
    """Transcribe one answer. Raises GeminiError for retryable failures while retries remain; otherwise records the
    outcome on the answer. Returns the final transcript status."""
    answer = db.session.get(InterviewAnswer, answer_id)
    if answer is None or not answer.media_path:
        return None
    try:
        media = storage.read_resume(answer.media_path)
    except storage.StorageError as e:
        answer.transcript_status, answer.transcript_error = InterviewAnswer.TRANSCRIPT_FAILED, str(e)
        answer.score_status, answer.score_error = InterviewAnswer.SCORE_FAILED, "Recording unreadable"
        db.session.commit()
        scoring.decide_follow_up(answer)  # never leave the candidate waiting
        return answer.transcript_status
    audio = media_service.extract_audio(media)  # the sound alone transcribes 3-5x faster than the video
    content, mime_type = audio if audio else (media, answer.media_type or "video/webm")
    try:
        result = gemini.transcribe(content, mime_type)
    except GeminiError as e:
        if e.retryable and retries_left > 0:
            answer.transcript_error = f"{e} (retrying automatically)"
            db.session.commit()
            raise
        current_app.logger.error("Transcription of answer %s failed: %s", answer.id, e)
        answer.transcript_status, answer.transcript_error = InterviewAnswer.TRANSCRIPT_FAILED, str(e)
        answer.score_status, answer.score_error = InterviewAnswer.SCORE_FAILED, "No transcript to score"
        db.session.commit()
        scoring.decide_follow_up(answer)  # releases a waiting interview (no follow-up without a transcript)
        return answer.transcript_status
    answer.transcript = result["transcript"]
    answer.speech_detected = result["speech_detected"]
    answer.transcript_status, answer.transcript_error = InterviewAnswer.TRANSCRIPT_DONE, None
    db.session.commit()
    scoring.decide_follow_up(answer)  # time-critical: the candidate is waiting for the next question

    db.session.refresh(answer)
    if not answer.is_scored:  # the decision may already have scored it (verdict-based follow-ups)
        from ..tasks import score_answer

        score_answer(answer.id)
    return answer.transcript_status
