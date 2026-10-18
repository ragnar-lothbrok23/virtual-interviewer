"""Resume text extraction and resume <-> job matching (paper Section 3.1)."""
import numpy as np
from docx import Document
from flask import current_app
from pypdf import PdfReader

from . import gemini

ALLOWED_EXTENSIONS = {"pdf", "docx"}

# Calibrated on the sample resumes: gemini-embedding-2 gives ~0.67-0.71 for a wrong-role resume
# and ~0.82 for a strong fit, so 0.60-0.85 is stretched linearly onto a 0-100 score.
SIMILARITY_FLOOR = 0.60
SIMILARITY_CEIL = 0.85


class ResumeError(ValueError):
    pass


def extract_text(path):
    ext = path.suffix.lower().lstrip(".")
    try:
        if ext == "pdf":
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        elif ext == "docx":
            doc = Document(path)
            parts = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))
            text = "\n".join(parts)
        else:
            raise ResumeError("Only PDF and DOCX resumes are supported.")
    except ResumeError:
        raise
    except Exception as e:
        raise ResumeError(f"Could not read the file: {e}") from e

    text = text.strip()
    if len(text) < 100:
        raise ResumeError("Very little text was found. If this is a scanned PDF, please upload a text-based PDF or DOCX.")
    return text


def profile_text(profile):
    """The parsed profile flattened into text for embedding (cleaner than raw resume text with addresses etc.)."""
    lines = [f"Summary: {profile['summary']}", f"Skills: {', '.join(profile['skills'])}"]
    for exp in profile["experience"]:
        lines.append(f"Experience: {exp['role']} at {exp['organisation']}. {' '.join(exp['highlights'])}")
    for edu in profile["education"]:
        lines.append(f"Education: {edu['degree']}, {edu['institution']}")
    if profile["projects"]:
        lines.append(f"Projects: {'; '.join(profile['projects'])}")
    if profile["certifications"]:
        lines.append(f"Certifications: {'; '.join(profile['certifications'])}")
    return "\n".join(lines)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def similarity_to_score(similarity):
    scaled = (similarity - SIMILARITY_FLOOR) / (SIMILARITY_CEIL - SIMILARITY_FLOOR)
    return round(float(np.clip(scaled, 0, 1)) * 100, 1)


def job_embedding(job):
    """Job embeddings are cached on the row so each job is embedded only once."""
    if job.embedding is None:
        job.embedding = gemini.embed(job.embedding_text).tobytes()
    return np.frombuffer(job.embedding, dtype=np.float32)


def analyse(resume_text, job):
    """Parse the resume with Gemini, then score it against the job. Returns (profile, similarity, score)."""
    profile = gemini.parse_resume(resume_text, job)
    similarity = cosine_similarity(gemini.embed(profile_text(profile)), job_embedding(job))
    return profile, similarity, similarity_to_score(similarity)


def initial_status(score):
    from ..models import Application

    return Application.STATUS_SHORTLISTED if score >= current_app.config["SHORTLIST_THRESHOLD"] else Application.STATUS_REVIEW
