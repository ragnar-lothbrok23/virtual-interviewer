"""Thin wrapper around the Gemini API (paper Section 3.1: resume parsing + semantic embeddings)."""
import time

import httpx
import numpy as np
from flask import current_app
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field


class GeminiError(RuntimeError):
    pass


class Experience(BaseModel):
    role: str
    organisation: str
    duration: str = Field(description="As written, e.g. 'Jun 2024 - Aug 2024' or '6 months'; empty if unknown")
    highlights: list[str] = Field(description="1-3 short bullet points of what they did")


class Education(BaseModel):
    degree: str
    institution: str
    year: str = Field(description="Graduation year or range; empty if unknown")
    score: str = Field(description="CGPA / percentage as written; empty if unknown")


class ResumeProfile(BaseModel):
    name: str
    email: str
    phone: str
    summary: str = Field(description="2-sentence neutral summary of the candidate's profile")
    skills: list[str] = Field(description="All technical and professional skills mentioned, normalised (e.g. 'JS' -> 'JavaScript')")
    experience: list[Experience]
    education: list[Education]
    projects: list[str] = Field(description="Project names with a few words on each")
    certifications: list[str]
    total_experience_years: float = Field(description="Total professional/internship experience in years; 0 if none")
    matched_skills: list[str] = Field(description="Required job skills the candidate clearly has (directly or via an equivalent)")
    missing_skills: list[str] = Field(description="Required job skills with no evidence in the resume")


def _client():
    client = current_app.extensions.get("gemini_client")
    if client is None:
        key = current_app.config["GEMINI_API_KEY"]
        if not key:
            current_app.logger.error("GEMINI_API_KEY is not set; AI features are unavailable")
            raise GeminiError("The AI service is not configured. Please contact the site administrator.")
        client = genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=30_000,  # ms; a normal parse takes ~10 s, so a stalled request is retried sooner
            # 429 is left out: on the free tier it usually means the daily quota is used up, so retrying only wastes time
            retry_options=types.HttpRetryOptions(attempts=3, http_status_codes=[500, 502, 503, 504]),
        ))
        current_app.extensions["gemini_client"] = client
    return client


def _friendly(e, action):
    if isinstance(e, GeminiError):
        return e
    if isinstance(e, genai_errors.ClientError) and e.code == 429:
        current_app.logger.error("Gemini quota exceeded: %s", e)
        return GeminiError("The AI service has reached its usage limit for now. Please try again later.")
    current_app.logger.error("Gemini %s failed: %s", action, e)
    return GeminiError(f"The AI service could not {action} right now. Please try again in a minute.")


def _with_retry(call, attempts=3):
    """Retry dropped connections ("Server disconnected"), which the SDK's status-code retry doesn't cover."""
    for attempt in range(attempts):
        try:
            return call()
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def parse_resume(resume_text, job):
    """Extract a structured profile from resume text and compare its skills with the job's required skills."""
    prompt = (
        "You are a resume parser for a recruitment system. Extract the candidate's details from the resume below. "
        "Only use information present in the resume; never invent details. Use empty strings or empty lists when missing.\n\n"
        f"Also compare the candidate against these REQUIRED JOB SKILLS: {', '.join(job.skill_list)}. "
        "Put each required skill into exactly one of matched_skills or missing_skills, using the job's spelling. "
        "Count a skill as matched if the resume shows it directly or through a clear equivalent "
        "(e.g. 'PostgreSQL' counts for 'SQL').\n\n"
        f"RESUME:\n{resume_text[:20000]}"
    )
    try:
        response = _with_retry(lambda: _client().models.generate_content(
            model=current_app.config["GEMINI_MODEL"],
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResumeProfile,
                # Extraction needs little reasoning; default thinking made this take ~50 s
                thinking_config=types.ThinkingConfig(thinking_level=current_app.config["GEMINI_PARSE_THINKING"]),
            ),
        ))
    except Exception as e:
        raise _friendly(e, "read your resume") from e
    profile = response.parsed
    if profile is None:
        raise GeminiError("Gemini returned an unreadable response. Please try again.")
    return profile.model_dump()


def embed(text):
    """Return the semantic embedding of `text` as a float32 numpy vector."""
    try:
        result = _with_retry(lambda: _client().models.embed_content(model=current_app.config["GEMINI_EMBED_MODEL"], contents=text))
    except Exception as e:
        raise _friendly(e, "analyse your resume") from e
    return np.asarray(result.embeddings[0].values, dtype=np.float32)
