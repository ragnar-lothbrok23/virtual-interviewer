"""HR analytics (Phase 10): the hiring pipeline, interview results, questions and data quality across jobs.

Everything is computed for a cohort: the applications made in the chosen period (by application date), optionally for
one job, and what happened to them afterwards. Counting by cohort keeps every figure on the page about the same
people (a funnel mixing "applied this month" with "interviewed this month" would not add up).

Only stored results are read (no AI calls). Scores come from the scorecards (paper Eq. 1); an application's interview
is its latest one. Emotion figures are indicative only, as on the scorecard.
"""
import csv
import io
import re
import statistics
from collections import Counter, defaultdict
from datetime import timedelta

from sqlalchemy.orm import selectinload

from ..models import (Application, Interview, InterviewAnswer, InterviewEvent, Job, Scorecard, as_utc,
                      utcnow)

PERIODS = {"30": ("Last 30 days", 30), "90": ("Last 90 days", 90), "365": ("Last 12 months", 365),
           "all": ("All time", None)}
SCORE_BINS = 10  # histogram bins of 10 points
MIN_FOR_STATS = 3  # fewer data points: no question flags / averages are marked as thin
MIN_FOR_CORRELATION = 5
HARD_QUESTION = 40  # average item score below this (with enough answers): possibly too hard or unclear
EASY_QUESTION = 90  # above this: possibly too easy to tell candidates apart
JD_SOURCE = re.compile(r"^Job description: (?P<skill>.+?)(?: \((?P<difficulty>[^)]*)\))?$")


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 1) if values else None


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


def _pct(part, whole):
    return round(100 * part / whole) if whole else None


def latest_interview(application):
    return application.interview  # one interview per application (rescheduling moves it)


def cohort(job_id=None, period="all", now=None):
    """The applications the page is about, with their interviews and scorecards loaded."""
    query = Application.query.options(
        selectinload(Application.interview).selectinload(Interview.scorecard),
        selectinload(Application.candidate), selectinload(Application.job))
    if job_id is not None:
        query = query.filter(Application.job_id == job_id)
    days = PERIODS.get(period, PERIODS["all"])[1]
    if days:
        query = query.filter(Application.created_at >= (now or utcnow()) - timedelta(days=days))
    return query.order_by(Application.created_at).all()


def _scored(applications):
    """(application, interview, scorecard) for completed interviews that have a final score."""
    rows = []
    for a in applications:
        iv = latest_interview(a)
        if iv is not None and iv.status == Interview.STATUS_COMPLETED and iv.scorecard is not None \
                and iv.scorecard.final_score is not None:
            rows.append((a, iv, iv.scorecard))
    return rows


# ---------- sections ----------

def funnel(applications):
    n = len(applications)
    latest = {a.id: latest_interview(a) for a in applications}
    stages = [
        ("applied", "Applied", n),
        ("screened", "Resume screened", sum(a.analysis_status == Application.ANALYSIS_DONE for a in applications)),
        ("shortlisted", "Shortlisted", sum(a.status == Application.STATUS_SHORTLISTED or latest[a.id] is not None
                                          for a in applications)),
        ("booked", "Interview booked", sum(latest[a.id] is not None for a in applications)),
        ("completed", "Interview completed", sum(1 for a in applications if latest[a.id] is not None
                                                 and latest[a.id].status == Interview.STATUS_COMPLETED)),
        ("recommended", "Recommended (strong or worth considering)",
         sum(1 for a in applications if latest[a.id] is not None and latest[a.id].scorecard is not None
             and latest[a.id].scorecard.recommendation in ("strong", "consider"))),
    ]
    side = {
        "withdrawn": sum(a.status == Application.STATUS_WITHDRAWN for a in applications),
        "rejected": sum(a.status == Application.STATUS_REJECTED for a in applications),
        "screening_failed": sum(a.analysis_status == Application.ANALYSIS_FAILED for a in applications),
        "no_show": sum(1 for a in applications if latest[a.id] is not None
                       and latest[a.id].status == Interview.STATUS_NO_SHOW),
        "cancelled": sum(1 for a in applications if latest[a.id] is not None
                         and latest[a.id].status == Interview.STATUS_CANCELLED),
        "upcoming": sum(1 for a in applications if latest[a.id] is not None
                        and latest[a.id].status in (Interview.STATUS_SCHEDULED, Interview.STATUS_IN_PROGRESS)),
    }
    return {"stages": [{"key": k, "label": label, "count": c, "pct": _pct(c, n)} for k, label, c in stages],
            "side": side}


def scores(applications, bands):
    rows = _scored(applications)
    finals = [sc.final_score for _, _, sc in rows]
    bins = [0] * SCORE_BINS
    for f in finals:
        bins[min(SCORE_BINS - 1, int(max(0.0, f) // (100 / SCORE_BINS)))] += 1
    recommendations = Counter(sc.recommendation for _, _, sc in rows)
    pairs = [(a.match_score, sc.final_score) for a, _, sc in rows if a.match_score is not None]
    return {"count": len(rows), "mean": _mean(finals), "median": _median(finals),
            "provisional": sum(sc.status != Scorecard.STATUS_COMPLETE for _, _, sc in rows),
            "histogram": [{"from": i * 10, "to": i * 10 + 9 if i < SCORE_BINS - 1 else 100, "count": c}
                          for i, c in enumerate(bins)],
            "bands": [{"min": m, "key": k, "label": label, "count": recommendations.get(k, 0)} for m, k, label in bands],
            "components": {"nlp": _mean(sc.nlp_score for _, _, sc in rows),
                           "emotion": _mean(sc.emotion_score for _, _, sc in rows),
                           "context": _mean(sc.context_score for _, _, sc in rows)},
            "resume_vs_interview": correlation(pairs)}


def correlation(pairs):
    """Pearson correlation between resume match and interview score: does the resume screen predict the interview?"""
    if len(pairs) < MIN_FOR_CORRELATION:
        return {"n": len(pairs), "r": None}
    xs, ys = zip(*pairs)
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"n": len(pairs), "r": None}
    return {"n": len(pairs), "r": round(statistics.correlation(xs, ys), 2)}


def jobs_table(applications):
    by_job = defaultdict(list)
    for a in applications:
        by_job[a.job].append(a)
    out = []
    for job, apps in sorted(by_job.items(), key=lambda kv: kv[0].title.lower()):
        rows = _scored(apps)
        out.append({"job": job, "applicants": len(apps), "interviewed": len(rows),
                    "final": _mean(sc.final_score for _, _, sc in rows),
                    "nlp": _mean(sc.nlp_score for _, _, sc in rows),
                    "emotion": _mean(sc.emotion_score for _, _, sc in rows),
                    "context": _mean(sc.context_score for _, _, sc in rows),
                    "recommended": sum(sc.recommendation in ("strong", "consider") for _, _, sc in rows)})
    return out


def ranking(applications, flags_by_interview):
    """Interviewed candidates, best final score first (for one job: a like-for-like comparison)."""
    rows = []
    for a, iv, sc in _scored(applications):
        rows.append({"application": a, "interview": iv, "scorecard": sc, "final": sc.final_score,
                     "flags": flags_by_interview.get(iv.id, 0)})
    rows.sort(key=lambda r: (-r["final"], r["application"].candidate.name.lower()))
    return rows


def questions(applications):
    """Per question: how candidates did, how often a follow-up was needed. Resume-based questions are personal, so
    they are summarised together per job instead of one row each."""
    groups = {}
    for a, iv, sc in _scored(applications):
        for item in (sc.details or {}).get("items", []):
            personal = item.get("category") == "resume"
            key = (a.job_id, "resume" if personal else item.get("question", ""))
            g = groups.setdefault(key, {"job": a.job, "question": "Resume-based questions (personal to each candidate)"
                                        if personal else item.get("question", ""), "category": item.get("category"),
                                        "source": None if personal else item.get("source"), "personal": personal,
                                        "items": []})
            g["items"].append(item)
    out = []
    for g in groups.values():
        items = g.pop("items")
        answered = [i for i in items if i.get("state") != "unanswered"]
        scored = [i for i in items if i.get("score") is not None]
        originals = [i["original"] for i in items if i.get("original", {}).get("state") == "scored"]
        avg = _mean(i["score"] for i in scored)
        flag = None
        if len(scored) >= MIN_FOR_STATS and avg is not None:
            flag = "hard" if avg < HARD_QUESTION else "easy" if avg >= EASY_QUESTION else None
        out.append(g | {"asked": len(items), "unanswered": len(items) - len(answered), "scored": len(scored),
                        "average": avg, "original_average": _mean(o["nlp"] for o in originals),
                        "follow_up_rate": _pct(sum(1 for i in items if i.get("follow_up")), len(items)),
                        "recovered": sum(1 for i in items if i.get("state") == "recovered"),
                        "dont_know_rate": _pct(sum(1 for o in originals if o.get("verdict") in ("dont_know", "off_topic")),
                                               len(originals)),
                        "flag": flag})
    out.sort(key=lambda q: (q["job"].title.lower(), q["average"] if q["average"] is not None else 101))
    return out


def skills(applications):
    """Average score per required skill, from the JD-based technical questions (weakest first)."""
    by_skill = defaultdict(list)
    for a, iv, sc in _scored(applications):
        for item in (sc.details or {}).get("items", []):
            m = JD_SOURCE.match(item.get("source") or "")
            if m and item.get("score") is not None:
                by_skill[m.group("skill").strip().lower()].append((m.group("skill").strip(), item["score"]))
    out = [{"skill": values[0][0], "answers": len(values), "average": _mean(v for _, v in values)}
           for values in by_skill.values()]
    out.sort(key=lambda s: (s["average"], s["skill"].lower()))
    return out


def integrity(interviews):
    ids = [iv.id for iv in interviews]
    counts, per_interview = Counter(), Counter()
    if ids:
        for interview_id, kind in (InterviewEvent.query.with_entities(InterviewEvent.interview_id, InterviewEvent.kind)
                                   .filter(InterviewEvent.interview_id.in_(ids),
                                           InterviewEvent.kind.in_(InterviewEvent.FLAGGED))):
            counts[kind] += 1
            per_interview[interview_id] += 1
    return {"interviews": len(ids), "flagged_interviews": len(per_interview),
            "kinds": [{"kind": k, "label": InterviewEvent.LABELS.get(k, k), "count": c} for k, c in counts.most_common()],
            "per_interview": dict(per_interview)}


def health(applications, interviews):
    """Processing problems HR may need to act on (re-score, re-analyse, re-check), with the interviews affected."""
    ids = [iv.id for iv in interviews]
    answers = InterviewAnswer.query.filter(InterviewAnswer.interview_id.in_(ids)).all() if ids else []
    problems = defaultdict(set)
    for ans in answers:
        if ans.transcript_status == InterviewAnswer.TRANSCRIPT_FAILED:
            problems["transcription_failed"].add(ans.interview_id)
        if ans.score_status == InterviewAnswer.SCORE_FAILED:
            problems["scoring_failed"].add(ans.interview_id)
        if ans.emotion_status == InterviewAnswer.EMOTION_FAILED:
            problems["emotion_failed"].add(ans.interview_id)
    for iv in interviews:
        if iv.scorecard is not None and iv.scorecard.consistency_status == Scorecard.CHECK_FAILED:
            problems["consistency_failed"].add(iv.id)
        if iv.status == Interview.STATUS_COMPLETED and iv.scorecard is not None \
                and iv.scorecard.status == Scorecard.STATUS_PROVISIONAL:
            problems["provisional"].add(iv.id)
    by_id = {iv.id: iv for iv in interviews}
    labels = {"transcription_failed": "Answers whose transcription failed",
              "scoring_failed": "Answers whose scoring failed",
              "emotion_failed": "Answers whose facial/voice analysis failed",
              "consistency_failed": "Consistency checks that failed",
              "provisional": "Finished interviews with a provisional scorecard"}
    return {"screening_failed": [a for a in applications if a.analysis_status == Application.ANALYSIS_FAILED],
            "problems": [{"key": k, "label": label, "interviews": sorted((by_id[i] for i in problems[k]),
                                                                          key=lambda iv: iv.id)}
                         for k, label in labels.items() if problems[k]],
            "answers": len(answers),
            "emotion_insufficient": sum(a.emotion_status == InterviewAnswer.EMOTION_INSUFFICIENT for a in answers),
            "emotion_low_reliability": sum(1 for a in answers if a.emotion_status == InterviewAnswer.EMOTION_DONE
                                           and (a.emotion or {}).get("reliability") == "low")}


def timing(applications):
    to_interview, length = [], []
    for a in applications:
        iv = latest_interview(a)
        if iv is None or iv.status != Interview.STATUS_COMPLETED:
            continue
        if iv.started_at:
            to_interview.append((as_utc(iv.started_at) - as_utc(a.created_at)).total_seconds() / 86400)
        if iv.started_at and iv.completed_at:
            length.append((as_utc(iv.completed_at) - as_utc(iv.started_at)).total_seconds() / 60)
    return {"days_to_interview": _median(to_interview), "interview_minutes": _median(length)}


def weekly(applications, now=None, weeks=12):
    """Applications and completed interviews per week, for the last `weeks` weeks (Monday-based, UTC)."""
    now = now or utcnow()
    start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0) \
        - timedelta(weeks=weeks - 1)
    buckets = [{"week": start + timedelta(weeks=i), "applications": 0, "interviews": 0} for i in range(weeks)]

    def put(moment, key):
        if moment is None:
            return
        index = int((as_utc(moment) - start).total_seconds() // (7 * 86400))
        if 0 <= index < weeks:
            buckets[index][key] += 1

    for a in applications:
        put(a.created_at, "applications")
        iv = latest_interview(a)
        if iv is not None and iv.status == Interview.STATUS_COMPLETED:
            put(iv.completed_at, "interviews")
    return buckets


def build(job_id=None, period="all", bands=(), now=None):
    """Everything the analytics page shows."""
    applications = cohort(job_id, period, now)
    interviews = [a.interview for a in applications if a.interview is not None]
    flags = integrity(interviews)
    return {"applications": len(applications), "funnel": funnel(applications), "scores": scores(applications, bands),
            "jobs": jobs_table(applications) if job_id is None else None,
            "ranking": ranking(applications, flags["per_interview"]),
            "questions": questions(applications), "skills": skills(applications), "integrity": flags,
            "health": health(applications, interviews), "timing": timing(applications),
            "weekly": weekly(applications, now)}


# ---------- export ----------

CSV_COLUMNS = ["Candidate", "Email", "Job", "Applied (UTC)", "Status", "Resume match", "Interview status",
               "Interview date (UTC)", "Final score", "NLP", "Emotion", "Context", "Recommendation",
               "Scorecard", "Integrity flags"]


def _cell(value):
    """A CSV cell that spreadsheet programs won't run as a formula (CSV injection)."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.1f}"
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def export_csv(job_id=None, period="all", recommendation_label=lambda k: k, now=None):
    applications = cohort(job_id, period, now)
    flags = integrity([a.interview for a in applications if a.interview is not None])["per_interview"]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(CSV_COLUMNS)
    for a in sorted(applications, key=lambda a: (a.job.title.lower(), a.candidate.name.lower())):
        iv = latest_interview(a)
        sc = iv.scorecard if iv is not None else None
        writer.writerow([_cell(v) for v in (
            a.candidate.name, a.candidate.email, a.job.title, as_utc(a.created_at).strftime("%Y-%m-%d %H:%M"),
            a.status_label, a.match_score, iv.status_label if iv else "",
            as_utc(iv.starts_at).strftime("%Y-%m-%d %H:%M") if iv else "",
            sc.final_score if sc else None, sc.nlp_score if sc else None, sc.emotion_score if sc else None,
            sc.context_score if sc else None,
            recommendation_label(sc.recommendation) if sc and sc.recommendation else "",
            sc.status_label if sc else "", flags.get(iv.id, 0) if iv else "")])
    return out.getvalue()


def job_choices():
    return Job.query.order_by(Job.title).all()
