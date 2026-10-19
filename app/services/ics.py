"""Minimal RFC 5545 calendar invites (.ics) for interviews.

The same UID is reused for every version of an interview's invite and SEQUENCE increases on each change, so
calendar apps update or remove the existing entry instead of adding duplicates. Times are UTC ("Z"), so every
calendar shows them in the viewer's own timezone.
"""
from datetime import datetime, timezone

from flask import current_app

from ..models import as_utc


def _escape(text):
    return (str(text).replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
            .replace("\r\n", "\\n").replace("\n", "\\n"))


def _fold(line):
    """Lines longer than 75 octets are folded with CRLF + space (RFC 5545 section 3.1)."""
    raw = line.encode("utf-8")
    if len(raw) <= 75:
        return line
    parts, current = [], b""
    for ch in line:
        encoded = ch.encode("utf-8")
        limit = 75 if not parts else 74  # continuation lines start with a space
        if len(current) + len(encoded) > limit:
            parts.append(current.decode("utf-8"))
            current = b""
        current += encoded
    parts.append(current.decode("utf-8"))
    return "\r\n ".join(parts)


def _utc(dt):
    return as_utc(dt).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def interview_uid(interview):
    domain = (current_app.config["MAIL_USERNAME"].split("@")[-1] if current_app.config["MAIL_USERNAME"]
              else "virtual-interviewer.local")
    return f"interview-{interview.id}@{domain}"


def interview_ics(interview, join_url, cancelled=False):
    """Returns the .ics file as bytes. `cancelled=True` produces a cancellation that removes the entry."""
    app_name = current_app.config["APP_NAME"]
    job = interview.application.job
    organizer = current_app.config["MAIL_USERNAME"] or "no-reply@localhost"
    attendee = interview.application.candidate.email
    description = (f"AI-led interview for {job.title}.\n"
                   f"Join from {join_url} (opens {current_app.config['JOIN_EARLY_MINUTES']} minutes before the start).\n"
                   "Use Chrome or Edge on a computer with a webcam and microphone, in a quiet, well-lit room.")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:-//{app_name}//Interview Scheduling//EN",
        "CALSCALE:GREGORIAN",
        f"METHOD:{'CANCEL' if cancelled else 'REQUEST'}",
        "BEGIN:VEVENT",
        f"UID:{interview_uid(interview)}",
        f"SEQUENCE:{interview.calendar_sequence}",
        f"DTSTAMP:{_utc(datetime.now(timezone.utc))}",
        f"DTSTART:{_utc(interview.starts_at)}",
        f"DTEND:{_utc(interview.ends_at)}",
        f"SUMMARY:{_escape(f'Interview: {job.title} ({app_name})')}",
        f"DESCRIPTION:{_escape(description)}",
        f"URL:{join_url}",
        f"LOCATION:{_escape('Online - ' + join_url)}",
        f"ORGANIZER;CN={_escape(app_name)}:mailto:{organizer}",
        f"ATTENDEE;CN={_escape(interview.application.candidate.name)};ROLE=REQ-PARTICIPANT:mailto:{attendee}",
        f"STATUS:{'CANCELLED' if cancelled else 'CONFIRMED'}",
    ]
    if not cancelled:
        lines += ["BEGIN:VALARM", "ACTION:DISPLAY", "DESCRIPTION:Interview in 30 minutes", "TRIGGER:-PT30M",
                  "END:VALARM"]
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return ("\r\n".join(_fold(line) for line in lines) + "\r\n").encode("utf-8")
