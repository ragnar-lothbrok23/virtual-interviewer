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
| **Technical questions generated from the job description** | Specified in spirit ("dynamic and domain specific question generation", §1/§3.3); method not specified | Implemented with Gemini, once per job (`Job.jd_question_count`, 0 = HR/template questions only) |
| Questions personalised from the candidate's **resume** | Motivated (§3.3 "Individualization", "personalized and role-conscious interviews"); tailoring to the candidate's background is named as **future work** | Implementation extension (Category B), `Job.resume_question_count` (0 = paper baseline) |
| Follow-up also when the rubric verdict is incorrect / partially correct / off topic / don't know, even with high similarity | **Not in the paper** (rule is similarity only) | **Deviation (Category C)**, chosen by the user on 20 Oct 2026 after a real interview: six wrong but on-topic answers scored similarity 68–93 (accuracy 10/100) and got no follow-up. `FOLLOW_UP_TRIGGER` (`similarity` = paper baseline). No extra AI calls (rubric reused for scoring); decision 9–12 s measured |
| Transcribing the audio track only | Not specified | Implementation decision: 3–5× faster than video (57 s answer: 7–12 s vs 24–67 s) |
| Follow-ups for resume-based answers | Not specified (rule is stated for technical answers) | Same similarity rule applied (`FOLLOW_UP_CATEGORIES`) |
| DeepFace facial analysis of the video for confidence, nervousness and attentiveness | Specified (§3.2) | **Replaced by HSEmotion (deviation, Category C, chosen by the user on 20 Oct 2026)** after a labelled test: 86% vs 57% of frames correct, macro-F1 0.74 vs 0.26. DeepFace stays selectable as the paper baseline (`EMOTION_MODEL=deepface`). Analysis per recorded answer, on a separate worker |
| Emotion from the candidate's **voice** (emotion2vec+, ONNX Runtime) combined with the face | Audio and video streams are analysed (§3.2); emotion is read from the face (DeepFace); a voice-emotion model is not specified | **Implementation extension (Category C: it changes the emotion score), requested by the user on 25 Oct 2026.** Chosen on a labelled recording (16 clips): emotion2vec+ 25%, wav2vec2 38% but calls calm speech "sad", Gemini 25%; face 81%. Weight 10% (`VOICE_EMOTION_WEIGHT`; 0 or `VOICE_EMOTION_ENABLED=0` = paper baseline) because more voice lowered accuracy (face alone 13/16, +10% voice 12/16, +30% 6/16). No extra AI calls: runs locally on the emotion worker (~1.6 s per 15 s of speech) |
| Face alignment, histogram normalisation, temporal smoothing | Specified (§3.2) | Implemented: CLAHE for face detection (it made emotion readings worse, so DeepFace reads the original frame), DeepFace alignment, exponential moving average |
| Mapping of DeepFace's 7 emotions to confidence / nervousness; attentiveness from face presence and head turn; emotion score = 50% confidence + 50% attentiveness; 2 frames/s; YuNet threshold 0.7 | **Not specified** | Implementation decisions (`EMOTION_*` settings) |
| "Insufficient" instead of a low score when the face is visible in < 50% of frames; reliability label | Not specified | Implementation decision (fairness: lighting and camera problems must not lower a score) |
| Final score 0.75 NLP + 0.15 Emotion + 0.10 Context | Specified (§3.4, Eq. 1) | Phase 9 |
| Turn latency 2.3 s per candidate turn | Reported result (§4.4, GPU server) | Measured in Phase 12; currently ~5–7 s after technical answers, ~1–2 s otherwise |

### Interview format (verified against the paper)

The paper describes a **turn-based, recorded interview analysed in real time** (“the audio and video stream of the
interview is recorded”, “2.3 s per candidate turn”, “adapting interview questions” depending on performance) and
states the system is “not confined to a conversation flow”. It does not describe the AI replying, giving hints or
answering clarification questions. The implementation follows the same turn-based format.

---

## JD-based technical questions (requested and implemented 20 Oct 2026) — Category B

Up to `Job.jd_question_count` technical questions (default 4, max 8) written by Gemini from the job description and
required skills, stored in the job's question bank (`InterviewQuestion.origin = "jd"`, with `skill` and `difficulty`).

| Concern | Handling |
|---|---|
| Comparability | Generated per job, not per candidate: everyone applying for the job gets the same technical questions |
| HR control | Shown in Job -> Questions marked "From JD"; editing one makes it HR's own (never replaced); count 0 removes the generated ones |
| Correctness of scoring | Each comes with a reference answer (key points, valid alternatives), so similarity and the follow-up rule work as in the paper |
| Staying current | Fingerprint of title, description, skills and count; editing the job regenerates untouched JD questions; upcoming interviews' resume questions are refreshed so they don't repeat them |
| Overlap with resume questions | They replace the generic per-skill "describe a project" templates (unless HR edited them) |
| Quality checks | Must test a required skill, be a real question of sensible length, have a substantial reference, not repeat other questions, no personal topics; partial sets reported |
| Limits | Never exceeds the 15-question limit (HR is told how many were added) |
| Failure | Worker retries, then hourly via housekeeping, HR "Write them now"; interviews fall back to the default set and log why |
| Latency / quota | One Gemini call per job (~3–4 s), never during an interview |

Real Gemini check on the three seeded jobs: 12/12 questions valid, one per required skill, mixed difficulty.

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

## Phase 9 (scorecard) — implemented 21 Oct 2026

The agreed rule below is implemented in `services/scorecard.py` (`FOLLOW_UP_RECOVERY = 0.5`: item =
original + 0.5 × (follow-up − original) when the follow-up is better, else the original). Other Phase 9 decisions:

| Item | Base paper | Implementation |
|---|---|---|
| FinalScore = 0.75 NLP + 0.15 Emotion + 0.10 Context | Specified (Eq. 1) | As specified (`FINAL_SCORE_WEIGHTS`) |
| Emotion indicators: stability, engagement, confidence | Specified (§3.4) | confidence = composure, engagement = attentiveness, stability = 100 − nervousness; per-answer emotion score = a third each (was 50/50 composure/attentiveness) |
| Contextual metrics: consistency, adjustability during follow-ups | Specified (§3.4), method not | Consistency: one Gemini check per interview (quotes verified against the transcripts; resume conflicts kept separately); adjustability = 50 + (follow-up − original), clamped 0-100; context = their average |
| Missing component | Not specified | Its weight is shared out over the others (never scored as 0); noted on the scorecard |
| Low-reliability facial analysis | Not specified | Left out of the emotion component |
| Recommendation bands (strong ≥ 75, consider ≥ 60) | “recommendation to recruiters”, no bands | Implementation decision (`RECOMMENDATION_BANDS`); HR decides |
| Integrity flags | Not in the formula | Shown on the scorecard, not scored |

### Original plan (kept for the record)

### A question and its follow-up count as one item

**Problem.** `interview_nlp_summary` averages every scored answer, so a follow-up counts as an extra question:

- Candidate A answers Q5 wrong and the follow-up wrong: Q5's topic is counted **twice** in their average.
- Candidate B answers Q5 equally wrong after the interview's `MAX_FOLLOW_UPS` were used: counted **once**.
- Interviews with follow-ups therefore have more items than those without, so candidates aren't scored on the same
  number of questions.

**Rule to implement.** Each planned question is one item. If it has a follow-up, the item's score combines both
answers: a good follow-up answer recovers part of the original score (the gap was phrasing, not knowledge), a weak
one confirms the low score **without adding a second penalty**. Exact combination to be fixed in Phase 9 (e.g.
item = max(original, weighted mix of original and follow-up)), configurable and documented as an implementation
decision; the per-answer scores stay visible to HR unchanged.

**Tests.** Same item count with and without follow-ups; wrong + wrong = one low item; wrong + right raises the item;
right answers unaffected; interviews without follow-ups score exactly as before.

**Context (current behaviour, unchanged).** Follow-ups never chain: an answer to a follow-up is scored normally
(verdict, accuracy, notes) and the interview moves straight to the next planned question; at most `MAX_FOLLOW_UPS`
(3) per interview. HR sees the original and its follow-up together.

## Planned for Phase 12 (evaluation) — agreed 20 Oct 2026

### Is DeepFace's emotion model good enough, or should a stronger one produce the emotion score?

**Decided 20 Oct 2026 (earlier than planned):** the user chose to replace DeepFace with **HSEmotion**
(`enet_b0_8_best_afew`, EfficientNet-B0 trained on AffectNet, ONNX Runtime) after the labelled test below was run:

| On the user's labelled clip (238 frames) | DeepFace (fixed pipeline) | HSEmotion |
|---|---|---|
| Frames read correctly | 57% | 86% |
| Macro-F1 (7 expressions) | 0.26 | 0.74 |
| Held expressions correct | 7 of 12 | 11 of 12 |
| Right group (composed / nervous / other) | 65% | 86% |
| Surprise / sad / angry / fear / disgust recognised | ~0% each | 100 / 70 / 50 / 53 / 42% |
| Model load / time per frame (incl. detection) | 18-40 s / ~76 ms | ~2 s / ~21 ms |

Caveats: the model and margin were chosen on the same clip they were measured on (one person, one room), so the
86% is optimistic for others; the gap to DeepFace is far larger than that effect. Phase 12 should repeat the test with
more people. Implementation: `services/emotion_models.py` (SHA-256-checked model files, `download-emotion-models`
command), faces aligned with a 10% margin, an eighth expression (contempt, counted as "other"), the model's name stored
with every analysis. DeepFace remains selectable (`EMOTION_MODEL=deepface`, `requirements-deepface.txt`).

**Original plan (kept for the record):**

**Evidence so far (Phase 8, real interviews).** Face detection and attentiveness are reliable, but DeepFace's
emotion classifier (FER-2013, 48x48 grayscale) is not: five near-identical neutral frames of one candidate were
labelled angry, fear, sad, neutral and happy at 93-100% certainty; with the best preprocessing a neutral face was
read as neutral only ~41% of the time and the top label changed between consecutive frames ~50-60% of the time.
The user agreed DeepFace is inaccurate and chose to keep it for now and decide at the end.

**Plan.** A labelled comparison on the user's own webcam: the user makes deliberate expressions (neutral, smile,
surprise, sad, frown) for a few seconds each in the live preview while the intended expression is recorded; DeepFace
and a modern in-the-wild expression model (e.g. one trained on AffectNet) are scored against those labels
(accuracy, frame-to-frame stability, speed on CPU).

**Options to decide between afterwards.**

| Option | Meaning |
|---|---|
| A. Keep DeepFace | Paper-faithful; current mitigations (smoothing, whole-answer averages, reliability label, "insufficient" instead of low scores, HR caveats, 15% weight) |
| B. Stronger model produces the score, DeepFace kept as baseline | Documented deviation from the paper (Category C); DeepFace still runs and is stored for comparison; detection/attentiveness unchanged |
| C. Emotion score from attentiveness and integrity only | Most reliable, but drops the paper's confidence/nervousness signals |

Switch only if the alternative wins clearly on the user's own footage, not on published numbers alone.

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
