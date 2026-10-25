"""PDF version of the interview scorecard (ReportLab; pure Python, no browser needed on the server)."""
import io
import os

from flask import current_app

from .timeutil import format_local

INK, MUTED, GRID, ACCENT = "#1f2430", "#6b7180", "#e3e5ea", "#2f5f8a"
_fonts_ready = False


def _fonts():
    """Bitstream Vera ships with ReportLab, so the PDF looks the same on every server."""
    global _fonts_ready
    if _fonts_ready:
        return
    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    folder = os.path.join(os.path.dirname(reportlab.__file__), "fonts")
    pdfmetrics.registerFont(TTFont("Vera", os.path.join(folder, "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("Vera-Bold", os.path.join(folder, "VeraBd.ttf")))
    pdfmetrics.registerFont(TTFont("Vera-Italic", os.path.join(folder, "VeraIt.ttf")))
    pdfmetrics.registerFontFamily("Vera", normal="Vera", bold="Vera-Bold", italic="Vera-Italic", boldItalic="Vera-Bold")
    _fonts_ready = True


def _fmt(v, digits=0):
    return "not assessed" if v is None else f"{v:.{digits}f}"


def build(interview, sc, recommendation_label):
    """The scorecard as PDF bytes."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    _fonts()
    d = sc.details or {}
    application = interview.application
    base = ParagraphStyle("base", fontName="Vera", fontSize=9, leading=12.5, textColor=colors.HexColor(INK),
                          alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=base, fontSize=7.8, leading=10.5, textColor=colors.HexColor(MUTED))
    h1 = ParagraphStyle("h1", parent=base, fontName="Vera-Bold", fontSize=17, leading=21)
    h2 = ParagraphStyle("h2", parent=base, fontName="Vera-Bold", fontSize=11, leading=15, spaceBefore=10, spaceAfter=4)
    big = ParagraphStyle("big", parent=base, fontName="Vera-Bold", fontSize=30, leading=34,
                         textColor=colors.HexColor(ACCENT))
    P = lambda text, style=base: Paragraph(text, style)  # noqa: E731
    E = lambda v: escape(str(v))  # noqa: E731

    def grid(rows, widths, header=True):
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        style = [("FONT", (0, 0), (-1, -1), "Vera", 8), ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
                 ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if header:
            style += [("FONT", (0, 0), (-1, 0), "Vera-Bold", 8), ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(MUTED))]
        t.setStyle(TableStyle(style))
        return t

    story = [P("Interview scorecard", h1),
             P(f"{E(application.candidate.name)} · {E(application.job.title)} · interviewed "
               f"{E(format_local(interview.started_at)) if interview.started_at else '—'}", small),
             Spacer(1, 6 * mm)]

    final = sc.final_score
    head = [[P(f"{final:.0f}<font size=12 color='{MUTED}'>/100</font>" if final is not None else "—", big),
             P(f"<b>{E(recommendation_label or 'Not available yet')}</b><br/>"
               f"<font color='{MUTED}' size=8>Status: {E(sc.status_label)}. AI recommendation only: the decision is "
               f"HR's.</font>", base)]]
    t = Table(head, colWidths=[45 * mm, 125 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [t, Spacer(1, 4 * mm)]

    story.append(P("Final score (paper §3.4, Eq. 1): 0.75 × NLP + 0.15 × Emotion + 0.10 × Context", h2))
    pw, uw = d.get("paper_weights", {}), d.get("weights", {})
    rows = [["Component", "Score", "Paper weight", "Weight used"]]
    for key, label in (("nlp", "NLP (answer content)"), ("emotion", "Emotion"), ("context", "Context")):
        rows.append([label, _fmt(d.get(key), 1), f"{pw.get(key, 0) * 100:.0f}%",
                     f"{uw[key] * 100:.1f}%" if key in uw else "—"])
    story.append(grid(rows, [70 * mm, 30 * mm, 35 * mm, 35 * mm]))
    for note in d.get("notes", []):
        story.append(P(E(note), small))

    story.append(P("Components", h2))
    speech, ei, cp = d.get("speech", {}), d.get("emotion_info", {}), d.get("context_parts", {})
    ind = ei.get("indicators") or {}
    rows = [["Speech", "", "Emotion indicators", "", "Context", ""],
            ["Clarity", _fmt(speech.get("clarity")), "Confidence", _fmt(ind.get("confidence")), "Consistency",
             f"No contradictions ({_fmt(cp.get('consistency'))})" if cp.get("consistency_findings") == 0
             else _fmt(cp.get("consistency"))],
            ["Fluency", _fmt(speech.get("fluency")), "Engagement", _fmt(ind.get("engagement")),
             "Adjustability", _fmt(cp.get("adjustability")) if cp.get("adjustability") is not None else "no follow-ups"],
            ["Pace (wpm)", _fmt(speech.get("words_per_minute")), "Stability", _fmt(ind.get("stability")), "", ""],
            ["Filler words", str(speech.get("fillers", 0)), "", "", "", ""]]
    story.append(grid(rows, [27 * mm, 25 * mm, 30 * mm, 25 * mm, 33 * mm, 30 * mm]))
    if ei.get("models"):
        story.append(P(f"Facial expressions: {E(', '.join(ei['models']))}; {ei['counts']['used']} answer(s) used"
                       + (f", {ei['counts']['low_reliability']} left out for low reliability"
                          if ei["counts"].get("low_reliability") else "") + ".", small))
    voice = ei.get("voice") or {}
    if voice.get("answers"):
        story.append(P(f"Voice: {E(', '.join(voice['models']))} included at {voice['weight'] * 100:.0f}% in "
                       f"{voice['answers']} of {ei['counts']['used']} answer(s); vocal nervousness "
                       f"{_fmt(voice.get('nervousness'))}.", small))

    c = sc.consistency if sc.consistency_status == "done" else None
    if c:
        story.append(P(f"Consistency findings ({c['score']:.0f}/10, AI certainty {E(c['confidence'])})", h2))
        story.append(P(E(c.get("summary", "")), base))
        for x in c.get("contradictions", []):
            story.append(P(f"<b>Contradiction:</b> “{E(x['first_quote'])}” vs “{E(x['second_quote'])}” — "
                           f"{E(x['explanation'])}", base))
        for x in c.get("resume_mismatches", []):
            story.append(P(f"<b>Differs from the resume</b> ({E(x['resume_claim'])}): “{E(x['quote'])}” — "
                           f"{E(x['explanation'])}", base))
        if not c.get("contradictions") and not c.get("resume_mismatches"):
            story.append(P("No contradictions found.", base))

    story.append(P("Questions", h2))
    story.append(P(f"One item per planned question; a better follow-up recovers "
                   f"{current_app.config['FOLLOW_UP_RECOVERY'] * 100:.0f}% of the gap and never lowers the item.", small))
    labels = {"technical": "Technical", "behavioral": "Behavioural", "situational": "Situational",
              "resume": "From resume"}
    rows = [["#", "Question", "Type", "Answer", "Follow-up", "Item"]]
    for it in d.get("items", []):
        o, f = it["original"], it.get("follow_up")
        rows.append([f"Q{it['position'] + 1}", P(E(it["question"]), small), labels.get(it["category"], it["category"]),
                     f"{o['nlp']:.0f}" if o["state"] == "scored" else o["state"].replace("_", " "),
                     (f"{f['nlp']:.0f}" if f["state"] == "scored" else f["state"]) if f else "—",
                     "waiting" if it["score"] is None else f"{it['score']:.0f}"])
    story.append(grid(rows, [10 * mm, 88 * mm, 24 * mm, 16 * mm, 17 * mm, 15 * mm]))

    flagged = [e for e in interview.events.all() if e.flagged]
    story.append(P("Integrity (not part of the score)", h2))
    if flagged:
        for e in flagged:
            extra = f" (about {e.details['seconds']:.0f} s)" if e.details and e.details.get("seconds") else ""
            pos = f" (Q{e.position + 1})" if e.position is not None else ""
            story.append(P(f"{E(format_local(e.created_at, '%I:%M:%S %p'))} · {E(e.label)}{pos}{extra}", base))
    else:
        story.append(P("No integrity flags.", base))

    story += [Spacer(1, 6 * mm),
              P("Generated by the AI Virtual Interviewer. Scores are AI estimates to support, not replace, the "
                "recruiter's judgement. Facial expression estimates are indicative only.", small)]

    buf = io.BytesIO()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Vera", 7)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(15 * mm, 10 * mm, f"Scorecard · {application.candidate.name} · {application.job.title}")
        canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm,
                            bottomMargin=18 * mm, title="Interview scorecard",
                            author="AI Virtual Interviewer", subject=f"{application.candidate.name} · {application.job.title}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
