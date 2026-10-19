"""Safe vector maths and an embedding cache shared by resume matching and answer scoring.

- safe_cosine never returns NaN: empty, zero-length, non-finite or mismatched vectors raise VectorError, so a broken
  embedding can't silently turn into a number inside a score.
- embed_cached stores embeddings keyed by (model, text). Editing a reference answer changes the text, and changing the
  embedding model changes the key, so stale embeddings are never reused.
"""
import hashlib

import numpy as np
from flask import current_app

from ..extensions import db


class VectorError(ValueError):
    """An embedding can't be used for similarity (empty, zero, non-finite, or wrong size)."""


def as_vector(values):
    vec = np.asarray(values, dtype=np.float32).ravel()
    if vec.size == 0:
        raise VectorError("empty embedding")
    if not np.all(np.isfinite(vec)):
        raise VectorError("embedding contains NaN or infinite values")
    return vec


def safe_cosine(a, b):
    a, b = as_vector(a), as_vector(b)
    if a.shape != b.shape:
        raise VectorError(f"embedding sizes differ ({a.size} vs {b.size})")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise VectorError("zero-length embedding")
    value = float(np.dot(a, b) / (na * nb))
    if not np.isfinite(value):
        raise VectorError("similarity is not a finite number")
    return max(-1.0, min(1.0, value))


def _cache_key(model, text):
    return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()


def embed_cached(text):
    """Embedding of `text`, reusing a stored copy when the same text was embedded with the same model before."""
    from ..models import EmbeddingCache
    from . import gemini

    model = current_app.config["GEMINI_EMBED_MODEL"]
    key = _cache_key(model, text)
    row = EmbeddingCache.query.filter_by(key=key).first()
    if row is not None:
        return np.frombuffer(row.vector, dtype=np.float32)
    vec = as_vector(gemini.embed(text))
    db.session.add(EmbeddingCache(key=key, model=model, dims=int(vec.size), vector=vec.tobytes()))
    try:
        db.session.commit()
    except Exception:  # a parallel job stored the same embedding first; that's fine
        db.session.rollback()
    return vec
