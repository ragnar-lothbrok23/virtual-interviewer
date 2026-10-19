/* Interview room: consent -> device check -> question loop (read, record, upload) -> done.
 * All progress is kept on the server, so reloading the page (or reconnecting) resumes at the current question. */
(function () {
  "use strict";

  const root = document.getElementById("room");
  if (!root) return;
  const urls = {
    start: root.dataset.startUrl, state: root.dataset.stateUrl, ask: root.dataset.askUrl,
    answer: root.dataset.answerUrl, event: root.dataset.eventUrl, done: root.dataset.doneUrl,
  };
  const csrf = document.querySelector('meta[name="csrf-token"]').content;
  const $ = (id) => document.getElementById(id);
  let state = JSON.parse(root.dataset.state);
  let stream = null;
  let recorder = null;
  let chunks = [];
  let recordingStarted = 0;
  let timers = [];
  let pendingUpload = null;
  let mimeType = "";

  // ---------- helpers ----------

  function show(step) {
    ["step-consent", "step-check", "step-question", "step-done"].forEach((s) => $(s).classList.toggle("d-none", s !== step));
  }

  function fail(message) {
    const box = $("step-error");
    box.textContent = message;
    box.classList.remove("d-none");
  }

  function clearTimers() {
    timers.forEach(clearInterval);
    timers = [];
  }

  function fmt(seconds) {
    const s = Math.max(0, Math.ceil(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  async function api(url, options = {}) {
    const response = await fetch(url, {
      credentials: "same-origin",
      ...options,
      headers: { "X-CSRFToken": csrf, Accept: "application/json", ...(options.headers || {}) },
    });
    let data = {};
    try { data = await response.json(); } catch (e) { /* non-JSON error page */ }
    if (!response.ok) {
      const error = new Error(data.error || `Request failed (${response.status})`);
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function postJSON(url, body) {
    return api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  }

  function sendEvent(kind, details) {
    if (state.status !== "in_progress") return;
    fetch(urls.event, {
      method: "POST", keepalive: true, credentials: "same-origin",
      headers: { "X-CSRFToken": csrf, "Content-Type": "application/json" },
      body: JSON.stringify({ kind, position: state.position, details: details || {} }),
    }).catch(() => {});
  }

  function setProgress() {
    $("progress-label").textContent = state.total ? `Question ${Math.min(state.position + 1, state.total)} of ${state.total}` : "";
  }

  // ---------- integrity signals ----------

  document.addEventListener("visibilitychange", () => sendEvent(document.hidden ? "tab_hidden" : "tab_visible"));
  let blurTimer = null;
  window.addEventListener("blur", () => {
    blurTimer = setTimeout(() => { if (!document.hidden) sendEvent("window_blur"); }, 1500);
  });
  window.addEventListener("focus", () => clearTimeout(blurTimer));
  window.addEventListener("beforeunload", (e) => {
    if (recorder && recorder.state === "recording") { e.preventDefault(); e.returnValue = ""; }
  });

  // ---------- step 1: consent ----------

  $("consent-box").addEventListener("change", (e) => { $("consent-next").disabled = !e.target.checked; });
  $("consent-next").addEventListener("click", () => { show("step-check"); runChecks(); });

  // ---------- step 2: device check ----------

  function mark(check, ok, note) {
    const li = document.querySelector(`[data-check="${check}"]`);
    li.querySelector("i").className = ok ? "bi bi-check-circle-fill text-success" : "bi bi-x-circle-fill text-danger";
    if (note) {
      let span = li.querySelector(".note");
      if (!span) { span = document.createElement("div"); span.className = "note text-muted"; li.appendChild(span); }
      span.textContent = note;
    }
  }

  function help(text) {
    const box = $("check-help");
    box.textContent = text;
    box.classList.toggle("d-none", !text);
  }

  async function runChecks() {
    const results = { browser: false, camera: false, mic: false, network: false };
    const ready = () => {
      const ok = results.browser && results.camera && results.mic && results.network;
      $("start-btn").disabled = !ok;
      $("start-btn").textContent = state.status === "in_progress" ? "Continue interview" : "Start interview";
    };

    const candidates = ["video/webm;codecs=vp8,opus", "video/webm;codecs=vp9,opus", "video/webm", "video/mp4"];
    const supported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
    mimeType = supported ? (candidates.find((t) => MediaRecorder.isTypeSupported(t)) || "") : "";
    results.browser = supported && !!mimeType;
    mark("browser", results.browser, results.browser ? null : "Please use the latest Chrome or Edge on a computer.");
    if (!results.browser) { help("Your browser can't record the interview. Open this page in Chrome or Edge."); return; }

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15, max: 24 } },
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      $("preview").srcObject = stream;
      $("self-view").srcObject = stream;
      results.camera = stream.getVideoTracks().length > 0;
      mark("camera", results.camera);
    } catch (err) {
      mark("camera", false, err.name === "NotAllowedError" ? "Permission was denied." : "No camera or microphone found.");
      mark("mic", false);
      help("Allow camera and microphone access (click the camera icon in the address bar), then reload this page.");
      sendEvent("device_error", { error: err.name });
      return;
    }

    // Microphone: wait until the candidate actually makes some sound
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      ctx.createMediaStreamSource(stream).connect(analyser);
      const buf = new Uint8Array(analyser.fftSize);
      const tick = () => {
        analyser.getByteTimeDomainData(buf);
        let peak = 0;
        for (const v of buf) peak = Math.max(peak, Math.abs(v - 128));
        const level = Math.min(100, (peak / 128) * 250);
        $("mic-level").style.width = `${level}%`;
        if (level > 18 && !results.mic) { results.mic = true; mark("mic", true); ready(); }
        requestAnimationFrame(tick);
      };
      tick();
    } catch (e) {
      results.mic = stream.getAudioTracks().length > 0;  // can't measure the level; trust the track
      mark("mic", results.mic);
    }

    // Network: a quick round trip to the server
    try {
      const t0 = performance.now();
      const fresh = await api(urls.state);
      const ms = performance.now() - t0;
      state = fresh;
      results.network = true;
      mark("network", true, ms > 1500 ? "Slow connection: uploads may take a while." : null);
    } catch (e) {
      mark("network", false, "Can't reach the server. Check your internet connection.");
    }
    ready();
  }

  $("start-btn").addEventListener("click", async () => {
    $("start-btn").disabled = true;
    try {
      state = await postJSON(urls.start, { consent: true });
      next();
    } catch (e) {
      fail(e.message);
      $("start-btn").disabled = false;
    }
  });

  // ---------- step 3: question loop ----------

  async function next() {
    setProgress();
    if (state.waiting) { waitForNextQuestion(); return; }
    if (state.finished || !state.question) { finish(); return; }
    clearTimers();
    try {
      state = await postJSON(urls.ask, { position: state.question.position });
    } catch (e) {
      if (e.status === 425) { waitForNextQuestion(); return; }  // still being prepared
      fail(e.message); return;
    }
    const q = state.question;
    show("step-question");
    $("q-text").textContent = q.text;
    $("q-category").textContent = q.follow_up ? "Follow-up question"
      : ({ technical: "Technical", behavioral: "Behavioural", situational: "Situational" }[q.category] || q.category);
    $("phase-prep").classList.remove("d-none");
    $("phase-record").classList.add("d-none");
    $("phase-upload").classList.add("d-none");

    let remaining = q.prep_seconds;
    $("prep-timer").textContent = fmt(remaining);
    const prep = setInterval(() => {
      remaining -= 1;
      $("prep-timer").textContent = fmt(remaining);
      if (remaining <= 0) startRecording(q);
    }, 1000);
    timers.push(prep);
    $("answer-now").onclick = () => startRecording(q);
  }

  function startRecording(q) {
    if (recorder && recorder.state === "recording") return;
    clearTimers();
    $("phase-prep").classList.add("d-none");
    $("phase-record").classList.remove("d-none");
    chunks = [];
    recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 250000, audioBitsPerSecond: 64000 });
    recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    recorder.onstop = () => upload(q);
    recorder.start(1000);
    recordingStarted = performance.now();

    const limit = q.answer_seconds;
    $("submit-answer").disabled = true;
    setTimeout(() => { $("submit-answer").disabled = false; }, 3000);
    const timer = setInterval(() => {
      const elapsed = (performance.now() - recordingStarted) / 1000;
      $("answer-timer").textContent = fmt(limit - elapsed);
      $("answer-bar").style.width = `${Math.max(0, 100 - (elapsed / limit) * 100)}%`;
      $("answer-bar").classList.toggle("bg-danger", limit - elapsed < 15);
      if (elapsed >= limit) stopRecording();
    }, 250);
    timers.push(timer);
    $("submit-answer").onclick = stopRecording;
  }

  function stopRecording() {
    clearTimers();
    if (recorder && recorder.state === "recording") recorder.stop();
  }

  async function upload(q) {
    $("phase-record").classList.add("d-none");
    $("phase-upload").classList.remove("d-none");
    $("retry-upload").classList.add("d-none");
    const duration = (performance.now() - recordingStarted) / 1000;
    const blob = new Blob(chunks, { type: (mimeType || "video/webm").split(";")[0] });
    pendingUpload = { q, blob, duration };
    await sendUpload();
  }

  async function sendUpload() {
    const { q, blob, duration } = pendingUpload;
    for (let attempt = 1; attempt <= 4; attempt++) {
      $("upload-status").textContent = attempt === 1 ? "Saving your answer…" : `Connection problem, retrying (${attempt}/4)…`;
      const form = new FormData();
      form.append("position", String(q.position));
      form.append("duration", duration.toFixed(1));
      form.append("media", blob, `answer-${q.position}.${blob.type.includes("mp4") ? "mp4" : "webm"}`);
      try {
        state = await api(urls.answer, { method: "POST", body: form });
        pendingUpload = null;
        next();
        return;
      } catch (e) {
        if (e.status && e.status < 500 && e.status !== 429) { fail(e.message); return; }  // not worth retrying
        await new Promise((r) => setTimeout(r, 1500 * attempt));
      }
    }
    $("upload-status").textContent = "We couldn't save your answer. Check your connection and try again.";
    $("retry-upload").classList.remove("d-none");
  }

  // After a technical answer the server decides whether to ask a follow-up; wait for it (it never takes long:
  // the server moves on by itself if the decision is slow)
  function waitForNextQuestion() {
    show("step-question");
    $("phase-prep").classList.add("d-none");
    $("phase-record").classList.add("d-none");
    $("phase-upload").classList.remove("d-none");
    $("retry-upload").classList.add("d-none");
    $("upload-status").textContent = "Preparing your next question…";
    let tries = 0;
    const poll = async () => {
      try {
        state = await api(urls.state);
        if (!state.waiting) { next(); return; }
      } catch (e) { /* keep trying; temporary network problems are common */ }
      if (++tries < 120) setTimeout(poll, 1500); else fail("The next question couldn't be loaded. Please reload this page.");
    };
    setTimeout(poll, 1000);
  }

  $("retry-upload").addEventListener("click", () => { $("retry-upload").classList.add("d-none"); sendUpload(); });

  // ---------- step 4: done ----------

  function finish() {
    clearTimers();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    $("done-link").href = urls.done;
    $("progress-label").textContent = "";
    show("step-done");
  }

  // ---------- entry ----------

  if (state.finished) {
    finish();
  } else if (state.status === "in_progress") {
    show("step-check");  // rejoining: devices need permission again, consent was already given
    runChecks();
  } else {
    show("step-consent");
  }
})();
