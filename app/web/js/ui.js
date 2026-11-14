/*
 * ui.js: shared helpers for both views: icons, formatting, theme colours,
 * one animation loop, canvas sizing, screen wake lock, cursor hiding,
 * fullscreen and the raw-event inspector.
 */
(function (root) {
  'use strict';
  var doc = root.document;

  // ---- icons (24x24 line icons, stroke = currentColor) ----------------------
  var P = {
    cameraOff: '<path d="M4.5 6.5H7l1.6-2.2h6.8L17 6.5h2.5A1.5 1.5 0 0 1 21 8v9.5a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5V8a1.5 1.5 0 0 1 1.5-1.5z"/><circle cx="12" cy="12.6" r="3.4"/><path d="M3.5 3.5l17 17"/>',
    swipe_right: '<path d="M6 12h13"/><path d="M14 7l5 5-5 5"/><path d="M3 8h2.5M3 16h2.5"/>',
    swipe_left: '<path d="M18 12H5"/><path d="M10 7l-5 5 5 5"/><path d="M21 8h-2.5M21 16h-2.5"/>',
    push: '<path d="M12 3v10"/><path d="M8.2 9.4 12 13.2l3.8-3.8"/><path d="M6.5 20.5h11"/><path d="M8 20.5V17h8v3.5"/>',
    pull: '<path d="M12 13.5V3.5"/><path d="M8.2 7.3 12 3.5l3.8 3.8"/><path d="M6.5 20.5h11"/><path d="M8 20.5V17h8v3.5"/>',
    breath: '<path d="M3 8.5h10.5a3 3 0 1 0-3-3"/><path d="M3 12.5h15a3 3 0 1 1-3 3"/><path d="M3 16.5h6"/>',
    signal: '<path d="M5 19.5v-3M10 19.5v-6.5M15 19.5v-10M20 19.5V5"/>',
    alert: '<path d="M12 3.6 2.6 20h18.8z"/><path d="M12 10v4.4"/><path d="M12 17.3v.2"/>',
    check: '<path d="M5 12.6l4.4 4.4L19 7.4"/>',
    enter: '<path d="M14 4h4.5v16H14"/><path d="M3.5 12h10"/><path d="M10 8.5l3.5 3.5-3.5 3.5"/>',
    empty: '<path d="M4 20h16"/><path d="M6 20V9.5L12 4.5l6 5V20"/>',
    calibrate: '<circle cx="12" cy="12" r="6.5"/><path d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4"/><circle cx="12" cy="12" r="1.2"/>',
    link: '<path d="M9.5 14.5l5-5"/><path d="M11 7l1.6-1.6a3.8 3.8 0 0 1 5.4 5.4L16.4 12.4"/><path d="M13 17l-1.6 1.6a3.8 3.8 0 0 1-5.4-5.4L7.6 11.6"/>',
    lock: '<rect x="5" y="11" width="14" height="9.5" rx="2"/><path d="M8.5 11V8a3.5 3.5 0 0 1 7 0v3"/>',
    ping: '<circle cx="6.5" cy="12" r="1.8"/><path d="M10.5 8.4a5 5 0 0 1 0 7.2"/><path d="M14 5.4a9.4 9.4 0 0 1 0 13.2"/><path d="M17.5 2.6a13.4 13.4 0 0 1 0 18.8"/>',
    person: '<circle cx="12" cy="6.5" r="2.8"/><path d="M6.5 20.5v-4.5a5.5 5.5 0 0 1 11 0v4.5"/>'
  };
  function icon(name, extraClass) {
    var body = P[name] || P.signal;
    return '<svg class="icon' + (extraClass ? ' ' + extraClass : '') + '" viewBox="0 0 24 24" aria-hidden="true">' + body + '</svg>';
  }

  // ---- labels and formatting ----------------------------------------------
  var GESTURE_TEXT = { swipe_left: 'Swipe left', swipe_right: 'Swipe right', push: 'Push', pull: 'Pull / wave', unknown: 'Unknown' };
  var PRESENCE_TEXT = { empty: 'Room empty', entering: 'Someone entering', present: 'Person present', unknown: 'Waiting for sensor' };

  function pct(c) { return c == null ? '' : Math.round(c * 100) + '%'; }
  function duration(ms) {
    var s = Math.max(0, Math.floor(ms / 1000));
    if (s < 60) return s + ' s';
    var m = Math.floor(s / 60); s = s % 60;
    if (m < 60) return m + ' min ' + (s < 10 ? '0' : '') + s + ' s';
    var h = Math.floor(m / 60); m = m % 60;
    return h + ' h ' + m + ' min';
  }
  function ago(ms) { return ms < 1500 ? 'now' : duration(ms) + ' ago'; }
  function clock(ms) { return new Date(ms).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' }); }
  function khz(hz) { return (hz / 1000).toFixed(hz % 1000 === 0 ? 0 : 1); }
  // Everything the engine transmits (tones and/or a chirp band) as one span, e.g. "18–21.5 kHz".
  function signalSpan(s) {
    if (!s) return null;
    var f = [];
    if (s.carriersHz && s.carriersHz.length) f = f.concat(s.carriersHz); else if (s.carrierHz != null) f.push(s.carrierHz);
    if (s.bandHz) f = f.concat(s.bandHz);
    if (!f.length) return null;
    var lo = Math.min.apply(null, f), hi = Math.max.apply(null, f);
    var label = s.bandHz ? 'Band' : f.length > 1 ? f.length + ' tones' : 'Carrier';
    return { label: label, text: lo === hi ? khz(lo) : khz(lo) + '\u2013' + khz(hi) };
  }
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }

  // ---- theme -----------------------------------------------------------------
  var themeCache = null;
  function themeColor(name) {
    if (!themeCache) themeCache = {};
    if (!(name in themeCache)) themeCache[name] = getComputedStyle(doc.documentElement).getPropertyValue('--' + name).trim();
    return themeCache[name];
  }
  function setTheme(name) {
    var t = name === 'dark' || name === 'night' ? 'night' : 'daylight';
    if (t === 'night') doc.documentElement.setAttribute('data-theme', 'night');
    else doc.documentElement.removeAttribute('data-theme');
    themeCache = null;
    try { root.localStorage.setItem('sonar.theme', t === 'night' ? 'dark' : 'light'); } catch (e) { /* storage blocked: theme just won't persist */ }
    doc.dispatchEvent(new CustomEvent('sonar:theme', { detail: t }));
  }
  function toggleTheme() { setTheme(doc.documentElement.getAttribute('data-theme') === 'night' ? 'light' : 'dark'); }

  // Parse "#rrggbb" into [r, g, b].
  function rgb(hex) {
    var h = hex.replace('#', '');
    if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join('');
    var n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  // ---- one requestAnimationFrame loop for every canvas/orb --------------------
  var drawers = [];
  var running = false;
  function frame(ts) {
    for (var i = 0; i < drawers.length; i++) {
      var d = drawers[i];
      if (d.active && !d.active()) continue;
      try { d.draw(ts); } catch (err) { console.error('[sonar] draw failed:', err); }
    }
    if (running) root.requestAnimationFrame(frame);
  }
  function addDrawer(d) {
    drawers.push(d);
    if (!running) { running = true; root.requestAnimationFrame(frame); }
  }

  // Keep a canvas backing store matched to its CSS size and the screen's pixel density.
  function fitCanvas(canvas, onResize) {
    function fit() {
      var r = canvas.getBoundingClientRect();
      var dpr = Math.min(root.devicePixelRatio || 1, 2);
      var w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; if (onResize) onResize(w, h, dpr); }
    }
    if (root.ResizeObserver) new ResizeObserver(fit).observe(canvas);
    else root.addEventListener('resize', fit);
    fit();
    return fit;
  }

  // Is this element actually on screen (not inside a hidden slide or closed overlay)?
  function visible(el) {
    if (doc.hidden || !el.isConnected) return false;
    var r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    var cs = getComputedStyle(el.closest('[data-live-scope]') || el);
    return cs.visibility !== 'hidden' && cs.opacity !== '0';
  }

  function reducedMotion() { return root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches; }

  // ---- screen wake lock: keep the laptop from dimming mid-demo ----------------
  function keepAwake() {
    var lock = null;
    function request() {
      if (!('wakeLock' in navigator) || doc.hidden) return;
      navigator.wakeLock.request('screen').then(function (l) { lock = l; }).catch(function (err) {
        console.warn('[sonar] wake lock unavailable:', err && err.message);
      });
    }
    doc.addEventListener('visibilitychange', function () { if (!doc.hidden && (!lock || lock.released)) request(); });
    request();
  }

  // ---- hide the mouse pointer after 2.5 s without movement ---------------------
  function autoHideCursor() {
    var t = null;
    function wake() {
      doc.body.classList.remove('cursor-idle');
      clearTimeout(t);
      t = setTimeout(function () { doc.body.classList.add('cursor-idle'); }, 2500);
    }
    root.addEventListener('mousemove', wake, { passive: true });
    wake();
  }

  function toggleFullscreen() {
    if (doc.fullscreenElement) { doc.exitFullscreen().catch(function () {}); return; }
    doc.documentElement.requestFullscreen().catch(function (err) { console.warn('[sonar] fullscreen refused:', err && err.message); });
  }

  // ---- raw event inspector (press D) ------------------------------------------
  function debugPanel(store, startOpen) {
    var el = doc.createElement('pre');
    el.className = 'debug' + (startOpen ? ' is-on' : '');
    el.setAttribute('aria-live', 'off');
    doc.body.appendChild(el);
    var lines = [];
    var counts = {};
    store.on('raw', function (r) {
      for (var i = 0; i < r.events.length; i++) { var k = r.events[i].kind; counts[k] = (counts[k] || 0) + 1; }
      if (!el.classList.contains('is-on')) return;
      var parsed = r.events.map(function (e) {
        if (e.kind === 'unknown') return 'UNKNOWN (' + e.reason + ')';
        var copy = {};
        for (var k in e) {
          if (k === 'rows') copy.rows = e.rows.length + ' rows x ' + (e.rows[0] ? e.rows[0].length : 0);
          else if (k === 'waveform') copy.waveform = e.waveform ? e.waveform.length + ' samples' : null;
          else if (k !== 'raw') copy[k] = e[k];
        }
        return JSON.stringify(copy);
      }).join('\n    ');
      var text = r.text.length > 240 ? r.text.slice(0, 240) + '...' : r.text;
      lines.unshift('<b>raw</b> ' + escapeHtml(text) + '\n  -> ' + escapeHtml(parsed));
      if (lines.length > 14) lines.length = 14;
    });
    setInterval(function () {
      if (!el.classList.contains('is-on')) return;
      var c = store.state.conn;
      el.innerHTML = '<b>' + escapeHtml(c.url || '') + '</b>  socket=' + c.socket + ' health=' + c.health +
        '  adapter=' + root.SonarAdapter.VERSION + '  link delay=' + (store.linkLatencyMs() == null ? 'n/a' : Math.round(store.linkLatencyMs()) + ' ms') +
        '\ncounts ' + escapeHtml(JSON.stringify(counts)) + '\n\n' + lines.join('\n\n');
    }, 300);
    return { toggle: function () { el.classList.toggle('is-on'); } };
  }

  root.SonarUI = {
    icon: icon, GESTURE_TEXT: GESTURE_TEXT, PRESENCE_TEXT: PRESENCE_TEXT,
    pct: pct, duration: duration, ago: ago, clock: clock, khz: khz, signalSpan: signalSpan, escapeHtml: escapeHtml,
    themeColor: themeColor, setTheme: setTheme, toggleTheme: toggleTheme, rgb: rgb,
    addDrawer: addDrawer, fitCanvas: fitCanvas, visible: visible, reducedMotion: reducedMotion,
    keepAwake: keepAwake, autoHideCursor: autoHideCursor, toggleFullscreen: toggleFullscreen,
    debugPanel: debugPanel
  };
})(typeof self !== 'undefined' ? self : this);
