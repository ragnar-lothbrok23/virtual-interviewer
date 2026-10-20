// Keeps a page up to date while background work runs (scoring, analysis, question generation) without reloading it:
// the page is fetched in the background and only the regions marked data-live="<key>" whose content changed are
// swapped in place. No scroll jumps, no flicker, and anything outside those regions (e.g. a video being watched)
// is left alone. The server says whether to keep going: <div data-live-busy="1" hidden> means work is still running.
// If the page's structure changed (a region appeared or disappeared), it falls back to one reload that keeps the
// scroll position.
(function () {
  "use strict";
  const INTERVAL = 6000;
  const SCROLL_KEY = "live-refresh-scroll:" + location.pathname;

  function isBusy(doc) {
    const marker = doc.querySelector("[data-live-busy]");
    return !!marker && marker.getAttribute("data-live-busy") === "1";
  }

  function keys(doc) {
    return Array.from(doc.querySelectorAll("[data-live]")).map((el) => el.getAttribute("data-live")).sort().join("|");
  }

  function restoreScroll() {
    try {
      const saved = sessionStorage.getItem(SCROLL_KEY);
      if (saved !== null) {
        sessionStorage.removeItem(SCROLL_KEY);
        window.scrollTo(0, parseInt(saved, 10) || 0);
      }
    } catch (e) { /* storage unavailable: nothing to restore */ }
  }

  function reloadKeepingScroll() {
    try { sessionStorage.setItem(SCROLL_KEY, String(window.scrollY)); } catch (e) { /* best effort */ }
    // Without the #anchor, so the reload doesn't jump back to it
    history.replaceState(null, "", location.pathname + location.search);
    location.reload();
  }

  async function tick() {
    let doc;
    try {
      const response = await fetch(location.pathname + location.search, {
        credentials: "same-origin", cache: "no-store", headers: { "X-Live-Refresh": "1" },
      });
      if (!response.ok || response.redirected) throw new Error("HTTP " + response.status);
      doc = new DOMParser().parseFromString(await response.text(), "text/html");
    } catch (e) {
      setTimeout(tick, INTERVAL * 2);  // network hiccup or signed out: try again later, never reload in a loop
      return;
    }
    if (keys(doc) !== keys(document)) {
      reloadKeepingScroll();
      return;
    }
    document.querySelectorAll("[data-live]").forEach((el) => {
      const key = el.getAttribute("data-live");
      const fresh = Array.from(doc.querySelectorAll("[data-live]")).find((f) => f.getAttribute("data-live") === key);
      // Don't swap a region the user is typing in
      if (fresh && fresh.innerHTML !== el.innerHTML && !el.contains(document.activeElement)) {
        el.innerHTML = fresh.innerHTML;
      }
    });
    if (isBusy(doc)) setTimeout(tick, INTERVAL);
  }

  restoreScroll();
  if (isBusy(document)) setTimeout(tick, INTERVAL);
})();
