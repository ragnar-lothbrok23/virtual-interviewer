# Enhancement backlog

Agreed improvements and their status. Each item states whether it preserves the base
paper's methodology:

- **A — methodology-preserving**: reliability, validation, performance; valid scores don't change.
- **B — methodology-compatible**: adds information alongside the paper's method without altering it.
- **C — methodology-changing**: alters the paper's method; only as a documented experiment against the baseline.

Constraint for every item: **no additional AI calls per answer** — the wait between interview questions must not grow.

---

## Phase 7 hardening — answer scoring and adaptive follow-ups (agreed and implemented 19 Oct 2026)

**Status: all nine items done** (tests in `tests/test_scoring_hardening.py`). Where each lives:

| # | Implementation |
|---|---|
| 1 | `services/vectors.safe_cosine`, used by answer scoring and resume matching; unusable embeddings fail the answer with a reason (resume analysis retries and rebuilds the job embedding) |
| 2 | `services/vectors.embed_cached` + `EmbeddingCache` table, keyed by sha256(model + text) |
| 3 | `scoring.normalise_text`: whole-word, punctuation/case-insensitive quote matching; each quote stores its `criterion`; HR wording: quotes show what was said, not that it is correct |
| 4 | `AnswerScore.verdict` / `confidence` + fairness rules in the rubric prompt; `InterviewAnswer.verdict`, `confidence`, `needs_review` (low confidence, clarification request, or no assessable answer despite speech) |
| 5 | `scoring.follow_up_problem`; rejected follow-ups are logged as `follow_up_skipped` with the reason |
| 6 | Upcoming planned questions passed to `generate_follow_up`; `target_concept` stored as `InterviewAnswer.follow_up_target`, shown to HR as "Probes: ..." |
| 7 | `scoring.has_time_for_follow_up` (follow-up + all remaining questions, each prep + answer + grace) checked before any AI call |
| 8 | `scoring.rubric_value`: missing / non-numeric / NaN / out-of-range components are a retryable malformed response, never clamped or zeroed; zeros only for genuinely empty answers |
| 9 | `[paper]` / `[impl]` labels in `config.py` and `services/resume.py`; HR page no longer attributes the weights to the paper |

Real Gemini check (gemini-3.5-flash-lite): strong answer → correct, long keyword-stuffed wrong answer → incorrect (A1), "I don't know" → dont_know, "could you repeat the question?" → clarification_request; the generated follow-up avoided a pagination question that was still planned.

### Original backlog

| # | Enhancement | Category | Why | Latency |
|---|---|---|---|---|
| 1 | **Guard similarity against empty / zero / non-finite vectors.** `scoring._cosine` divides by the vector norms; a zero vector produces NaN, which would flow into `similarity_score` and `nlp_score`. Also reject mismatched dimensions. Treat as "unassessable", never as a silent number. | A | Real crash/corruption risk | None |
| 2 | **Cache the reference answer's embedding** (store it with the reference text and the embedding model name; invalidate when either changes). Today `_similarity` embeds the reference again for every answer. | A | Removes one embedding call per answer, so the follow-up decision is faster | **Lower** |
| 3 | **Evidence quotes tolerant of punctuation and case** (normalise both sides before matching, still rejecting invented text), and **tie each quote to the criterion it supports** (relevance / accuracy / fluency / clarity). A quote existing in the transcript does not prove the claim is correct — keep that distinction in the UI wording. | A | Real quotes are currently rejected when commas differ | None |
| 4 | **Answer verdict and assessment confidence in the existing rubric call**: `correct`, `partially_correct`, `incorrect`, `off_topic`, `dont_know`, `no_answer`, `clarification_request`, plus confidence (high/medium/low). Low confidence or unassessable answers are flagged **"needs human review"** for HR instead of being presented as certain. Covers most edge cases (alternative terminology, short-but-correct, long-but-wrong, keyword repetition, self-correction, contradictions within an answer, code/SQL in answers). Prompt must say: don't equate length with knowledge; don't penalise accent, grammar or concise answers; similarity to the reference is not correctness. | B | Explainability and fairness; formula unchanged | None (same call) |
| 5 | **Validate generated follow-up questions** before inserting them: non-empty, bounded length, phrased as a question, not identical/near-identical to the original question, does not contain the reference answer. On failure, skip the follow-up (interview moves on). | A | Prevents malformed or answer-revealing questions reaching candidates | None |
| 6 | **Avoid duplicate follow-ups and record why they were asked**: include the remaining planned questions in the follow-up prompt so it doesn't ask something coming later; return a `target_concept` (the missing concept/misconception probed) stored with the follow-up and shown to HR. | B | Relevance and traceability | None (same call) |
| 7 | **Remaining-time check before adding a follow-up**: skip it if `prep + answer time` would exceed the interview's hard deadline. | A | Prevents interviews timing out because of a late follow-up | None |
| 8 | **Explicit failure instead of silent zeros** for missing, null, NaN or infinite score components. Zero remains correct only for genuinely empty/unassessable answers (`no_answer`). Valid scores are never altered by these checks. | A | Data integrity | None |
| 9 | **Label paper values vs implementation decisions** in config and documentation (see table below); keep every implementation parameter configurable so a paper-baseline configuration stays identifiable for evaluation. | Docs | Research honesty | None |

Tests required for each item: unit tests for the edge cases (zero/NaN vectors, dimension mismatch, punctuation in
quotes, malformed follow-ups, insufficient time), fake-AI tests for each verdict, a cache-invalidation test when HR
edits a reference answer, and the existing suites on SQLite, PostgreSQL and the browser test.

### Paper requirement vs implementation decision

| Item | Base paper | Current implementation |
|---|---|---|
| Follow-up when a technical answer's **semantic similarity is low** | Specified (§3.3) | Implemented as specified |
| NLP components: content relevance, fluency, domain accuracy; speech clarity and speed | Specified (§3.4) | Implemented per answer |
| Weights inside the NLP score (30% relevance, 30% accuracy, 20% similarity, 20% fluency) | **Not specified** | Implementation decision (`NLP_WEIGHTS`) |
| Follow-up threshold (similarity score < 55) | **Not specified** ("low") | Implementation decision (`FOLLOW_UP_THRESHOLD`) |
| Maximum follow-ups (3), no chaining | **Not specified** | Implementation decision (`MAX_FOLLOW_UPS`) |
| Decision timeout (45 s) | **Not specified** | Implementation decision (`DECISION_TIMEOUT_SECONDS`) |
| Questions personalised from the candidate's **resume** | Motivated (§3.3 "Individualization", "personalized and role-conscious interviews"); tailoring to the candidate's background is named as **future work** | Implementation extension (Category B), `Job.resume_question_count` (0 = paper baseline) |
| Follow-ups for resume-based answers | Not specified (rule is stated for technical answers) | Same similarity rule applied (`FOLLOW_UP_CATEGORIES`) |
| Final score 0.75 NLP + 0.15 Emotion + 0.10 Context | Specified (§3.4, Eq. 1) | Phase 9 |
| Turn latency 2.3 s per candidate turn | Reported result (§4.4, GPU server) | Measured in Phase 12; currently ~5–7 s after technical answers, ~1–2 s otherwise |

### Interview format (verified against the paper)

The paper describes a **turn-based, recorded interview analysed in real time** (“the audio and video stream of the
interview is recorded”, “2.3 s per candidate turn”, “adapting interview questions” depending on performance) and
states the system is “not confined to a conversation flow”. It does not describe the AI replying, giving hints or
answering clarification questions. The implementation follows the same turn-based format.

---

## Resume-based questions (requested and implemented 20 Oct 2026) — Category B

Each candidate gets up to `Job.resume_question_count` questions (default 3, max 5) written from their parsed resume:
experience, projects and a required skill whose evidence is only "listed". Asked after the first job question;
weak answers get the paper's similarity-based follow-up.

| Concern | Handling |
|---|---|
| Latency | Generated in the background at booking (one Gemini call per candidate, ~2–3 s); nothing added during the interview |
| Comparability between candidates | The job's questions stay identical; resume questions are labelled "From resume" and scored on the same rubric |
| Paper-baseline evaluation | `resume_question_count = 0` per job |
| Prompt injection via the resume | Profile passed as untrusted data inside tags; output validated |
| Invented or unfair questions | Each must name an experience/project/skill that exists in the profile; personal/sensitive topics (age, family, health, religion, caste, nationality, salary, gaps) rejected; repeats of job questions rejected; partial sets reported to HR |
| Outdated sets (resume re-analysed, job or its questions edited, count changed) | Fingerprint of the inputs; housekeeping regenerates for interviews within 72 h; outdated sets are never used |
| Generation failure | Retried by the worker; then hourly by housekeeping; HR can "Prepare now"; the interview falls back to job questions and logs why |
| Old profiles (projects stored as strings) | Supported |

Real Gemini check: 6/6 questions across two real resumes were specific (named the internship, project or skill)
and passed validation.

## Deliberately not planned (or moved to a later phase)

| Idea | Decision | Reason |
|---|---|---|
| Hints / clarification dialogue during the live interview | Not planned | Turns the paper's turn-based interview into a conversational system, which the paper distinguishes itself from |
| Detecting contradictions with **earlier** answers in real time | Phase 9 | Needs cross-answer context (slower); belongs to the paper's **Context** score, which explicitly includes *consistency* |
| Full rubric/reference version history with an HR rubric editor | Not planned | Heavy; each answer already stores a snapshot of the reference answer it was scored against |
| Extra AI calls per answer (second opinions, ensembles, separate classifiers) | Not planned | Doubles latency and quota |
| Changing weights, threshold or the similarity method | Phase 12 experiment only (Category C) | Must be compared against the paper-baseline configuration |
| Large evaluation datasets, human-rater agreement | Phase 12 | Evaluation work |
| Handling embedding-model upgrades beyond storing the model name | Not planned | Model name storage (item 2) is sufficient |
| Streaming transcription during the answer to reach ~2.3 s per turn | Phase 12/13 consideration | Large architectural change; the paper attributes its latency to a GPU server |
