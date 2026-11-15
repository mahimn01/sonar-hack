/*
 * widgets.js: live components shared by the presenter and the Care Monitor.
 * Each one takes a container element and the store, and keeps itself updated.
 */
(function (root) {
  'use strict';
  var doc = root.document;
  var UI = root.SonarUI;

  function h(html) { var t = doc.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; }
  function every(ms, fn) { fn(); return setInterval(fn, ms); }

  // Shared helper: estimate how far behind "now" to draw the waveform, so the
  // line scrolls smoothly even though breathing messages arrive a few times a second.
  function lagTracker(store) {
    var arrivals = [], lag = 0.9;
    store.on('breathing', function () {
      arrivals.push(performance.now());
      if (arrivals.length > 8) arrivals.shift();
      if (arrivals.length >= 3) {
        var iv = (arrivals[arrivals.length - 1] - arrivals[0]) / (arrivals.length - 1) / 1000;
        lag = Math.max(0.25, Math.min(2.5, iv * 1.4 + 0.12));
      }
    });
    return function () { return lag; };
  }

  // Slow envelope so a breath-hold visibly flattens the line instead of zooming into noise.
  function envelope(halfLifeS) {
    var env = 0, last = 0;
    return function (maxAbs, nowMs) {
      var dt = last ? (nowMs - last) / 1000 : 0; last = nowMs;
      env = Math.max(env * Math.pow(0.5, dt / halfLifeS), maxAbs, 1e-6);
      return env;
    };
  }

  function windowStats(wave, start, end) {
    var i = Math.max(0, wave.indexAt(start)), s = 0, n = 0, maxAbs = 0;
    for (var k = i; k < wave.t.length && wave.t[k] <= end; k++) { var v = wave.v[k]; if (isFinite(v)) { s += v; n++; } }
    var mean = n ? s / n : 0;
    for (var j = i; j < wave.t.length && wave.t[j] <= end; j++) { var w = wave.v[j]; if (isFinite(w)) maxAbs = Math.max(maxAbs, Math.abs(w - mean)); }
    return { mean: mean, maxAbs: maxAbs, n: n };
  }

  // ---- breathing waveform ------------------------------------------------------
  function waveform(container, store, opts) {
    opts = opts || {};
    var seconds = opts.seconds || store.config.waveformSeconds;
    var canvas = doc.createElement('canvas');
    canvas.setAttribute('aria-hidden', 'true');
    container.classList.add('wave');
    container.appendChild(canvas);
    var ctx = canvas.getContext('2d');
    UI.fitCanvas(canvas);
    var getLag = lagTracker(store), env = envelope(12);

    UI.addDrawer({
      active: function () { return UI.visible(container); },
      draw: function () {
        var W = canvas.width, H = canvas.height, dpr = Math.min(root.devicePixelRatio || 1, 2);
        ctx.clearRect(0, 0, W, H);
        var wave = store.state.wave;
        var end = store.engineNow() - getLag(), start = end - seconds;
        var st = windowStats(wave, start - 0.2, end + 0.1);   // same span that gets drawn
        var e = env(st.maxAbs, performance.now());
        var y0 = H * 0.5, amp = H * 0.4;
        var alarm = !!store.state.alerts.apnea;
        var color = UI.themeColor(alarm ? 'alarm' : 'tide');

        // Time grid every 5 s.
        ctx.strokeStyle = UI.themeColor('line');
        ctx.lineWidth = 1 * dpr;
        ctx.beginPath();
        for (var g = Math.ceil(start / 5) * 5; g <= end; g += 5) { var gx = (g - start) / seconds * W; ctx.moveTo(gx, H * 0.08); ctx.lineTo(gx, H * 0.92); }
        ctx.moveTo(0, y0); ctx.lineTo(W, y0);
        ctx.stroke();

        if (opts.labels !== false) {
          ctx.fillStyle = UI.themeColor('ink-2');
          var u = Math.min(root.innerWidth / 100, root.innerHeight / 56.25);   // same stage unit as the CSS
          ctx.font = '500 ' + Math.round(Math.max(10, u * 0.9) * dpr) + 'px ' + getComputedStyle(doc.body).fontFamily;
          ctx.textBaseline = 'bottom';
          ctx.textAlign = 'left'; ctx.fillText('−' + seconds + ' s', 6 * dpr, H - 2 * dpr);
          ctx.textAlign = 'right'; ctx.fillText('now', W - 6 * dpr, H - 2 * dpr);
        }

        if (st.n < 2) {
          ctx.setLineDash([6 * dpr, 6 * dpr]);
          ctx.strokeStyle = UI.themeColor('ink-3');
          ctx.lineWidth = 2 * dpr;
          ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(W, y0); ctx.stroke();
          ctx.setLineDash([]);
          return;
        }

        // Build the line once, use it for the soft fill and the stroke.
        var i0 = Math.max(0, wave.indexAt(start) - 1);
        var path = new Path2D(), fill = new Path2D(), open = false, firstX = 0, lastX = 0, lastY = y0;
        for (var i = i0; i < wave.t.length && wave.t[i] <= end + 0.05; i++) {
          var v = wave.v[i];
          var x = (wave.t[i] - start) / seconds * W;
          if (!isFinite(v)) {
            if (open) { fill.lineTo(lastX, y0); fill.lineTo(firstX, y0); fill.closePath(); }
            open = false; continue;
          }
          var y = y0 - (v - st.mean) / e * amp;
          if (!open) { path.moveTo(x, y); fill.moveTo(x, y0); fill.lineTo(x, y); firstX = x; open = true; }
          else { path.lineTo(x, y); fill.lineTo(x, y); }
          lastX = x; lastY = y;
        }
        if (open) { fill.lineTo(lastX, y0); fill.lineTo(firstX, y0); fill.closePath(); }

        ctx.globalAlpha = 0.14; ctx.fillStyle = color; ctx.fill(fill);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = color; ctx.lineWidth = (opts.lineWidth || 3) * dpr; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        ctx.stroke(path);

        // Leading dot at "now".
        var vEnd = wave.valueAt(end);
        if (isFinite(vEnd) && lastX > W - 40 * dpr) {
          var yEnd = y0 - (vEnd - st.mean) / e * amp;
          ctx.globalAlpha = 0.18; ctx.beginPath(); ctx.arc(lastX, yEnd, 11 * dpr, 0, Math.PI * 2); ctx.fill();
          ctx.globalAlpha = 1; ctx.beginPath(); ctx.arc(lastX, yEnd, 4.5 * dpr, 0, Math.PI * 2); ctx.fill();
        }
      }
    });
  }

  // ---- live Doppler spectrogram strip ---------------------------------------------
  function spectrogram(container, store, opts) {
    opts = opts || {};
    var cols = opts.columns || 360;
    container.classList.add('spectro');
    var canvas = doc.createElement('canvas');
    canvas.setAttribute('aria-hidden', 'true');
    container.appendChild(canvas);
    var empty = h('<div class="spectro__empty">Waiting for the live Doppler stream</div>');
    container.appendChild(empty);
    if (opts.labels !== false) {
      container.appendChild(h('<span class="axis-note" style="left:0.6em;top:0.5em">toward</span>'));
      container.appendChild(h('<span class="axis-note" style="left:0.6em;bottom:0.5em">away</span>'));
      container.appendChild(h('<span class="axis-note" style="right:0.6em;bottom:0.5em">now</span>'));
    }
    var ctx = canvas.getContext('2d');
    UI.fitCanvas(canvas);
    var buf = doc.createElement('canvas'), bctx = buf.getContext('2d');
    var bins = 0, lut = null, pending = 0, dirty = true, col = null;

    function buildLut() {
      var stops = [0, 1, 2, 3, 4].map(function (i) { return UI.rgb(UI.themeColor('cm-' + i)); });
      lut = new Uint8ClampedArray(256 * 3);
      for (var i = 0; i < 256; i++) {
        var x = i / 255 * 4, k = Math.min(3, Math.floor(x)), f = x - k;
        for (var c = 0; c < 3; c++) lut[i * 3 + c] = stops[k][c] + (stops[k + 1][c] - stops[k][c]) * f;
      }
    }
    // Track the noise floor (running median of each row) and subtract it, so the
    // background stays quiet and real motion stands out whatever the engine's scaling.
    var floorEst = null;
    function trackFloor(row) {
      var s = row.slice().sort(function (a, b) { return a - b; });
      var med = s[s.length >> 1] || 0;
      floorEst = floorEst == null ? med : floorEst + (med - floorEst) * 0.05;
      return Math.min(0.9, Math.max(0, floorEst));
    }
    function drawColumn(x, row) {
      if (!col || col.height !== bins) col = bctx.createImageData(1, bins);
      var d = col.data, f = trackFloor(row), scale = 1 / (1 - f);
      for (var j = 0; j < bins; j++) {
        var v = (row[j] - f) * scale; if (!(v >= 0)) v = 0; if (v > 1) v = 1;
        var li = Math.round(Math.sqrt(v) * 255) * 3;            // sqrt lifts faint echoes
        var p = (bins - 1 - j) * 4;                              // bin 0 (most negative / away) at the bottom
        d[p] = lut[li]; d[p + 1] = lut[li + 1]; d[p + 2] = lut[li + 2]; d[p + 3] = 255;
      }
      bctx.putImageData(col, x, 0);
    }
    function redrawAll() {
      var R = store.state.spectro.rows;
      bins = R.length ? R[R.length - 1].length : 0;
      buf.width = cols; buf.height = Math.max(1, bins);
      bctx.fillStyle = UI.themeColor('cm-0'); bctx.fillRect(0, 0, cols, buf.height);
      var n = Math.min(cols, R.length);
      for (var k = 0; k < n; k++) { var row = R[R.length - n + k]; if (row.length === bins) drawColumn(cols - n + k, row); }
    }
    store.on('spectrogram', function (rows) { pending += rows.length; });
    doc.addEventListener('sonar:theme', function () { lut = null; dirty = true; });

    UI.addDrawer({
      active: function () { return UI.visible(container); },
      draw: function () {
        var R = store.state.spectro.rows;
        var stale = !R.length || Date.now() - store.state.spectro.lastAt > 4000;
        empty.style.display = stale ? '' : 'none';
        container.classList.toggle('is-empty', !R.length);
        canvas.style.opacity = R.length ? (stale ? '0.35' : '1') : '0';
        if (!R.length) return;
        if (!lut) buildLut();
        var nb = R[R.length - 1].length;
        if (dirty || nb !== bins) { redrawAll(); dirty = false; pending = 0; }
        else if (pending > 0) {
          var n = Math.min(pending, cols, R.length);
          bctx.globalCompositeOperation = 'copy'; bctx.drawImage(buf, -n, 0); bctx.globalCompositeOperation = 'source-over';
          for (var k = 0; k < n; k++) { var row = R[R.length - n + k]; if (row.length === bins) drawColumn(cols - n + k, row); }
          pending = 0;
        } else return;                                           // nothing new: keep the last frame
        ctx.imageSmoothingEnabled = true;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(buf, 0, 0, canvas.width, canvas.height);
      }
    });
    root.addEventListener('resize', function () { dirty = true; });
  }

  // ---- breathing orb with the rate inside ---------------------------------------
  function breathing(container, store, opts) {
    opts = opts || {};
    var el = h(
      '<div class="orb" data-state="idle">' +
      '<div class="orb__halo"></div><div class="orb__core"></div>' +
      '<div class="orb__text"><span class="orb__value num">—</span><span class="orb__unit">Waiting for sensor</span></div>' +
      '</div>');
    container.appendChild(el);
    var valueEl = el.querySelector('.orb__value'), unitEl = el.querySelector('.orb__unit');
    var getLag = lagTracker(store), env = envelope(12);
    var shown = null, target = null, lastFrame = 0, prevV = 0, rising = false, lastRipple = 0, state = 'idle';
    var still = UI.reducedMotion();

    function setText() {
      var S = store.state, B = S.breathing, now = Date.now();
      var presence = S.presence.state;
      var fresh = B.lastEventAt && now - B.lastEventAt < 8000;
      var next, unit;
      if (S.alerts.apnea) { next = 'apnea'; unit = 'without breathing'; }
      else if (presence === 'empty') { next = 'idle'; unit = 'No one in range'; }
      else if (B.bpm != null && B.lastValidAt && now - B.lastValidAt < 6000) { next = 'live'; unit = 'breaths / min'; }
      else if (presence === 'present' || presence === 'entering' || fresh) {
        next = 'measuring';
        var need = store.config.breathMinSeconds || 10;
        unit = presence === 'entering' ? 'Measuring once they settle'
          : B.windowS != null && B.windowS < need ? 'Hold still · ' + Math.floor(B.windowS) + ' of ' + need + ' s'
          : 'Measuring · hold still';
      }
      else { next = 'idle'; unit = S.conn.health === 'live' ? 'Waiting for a still person' : 'Waiting for sensor'; }
      if (next !== state) { state = next; el.setAttribute('data-state', state); }
      unitEl.textContent = unit;
      if (state === 'live') {
        target = B.bpm;
        if (shown == null) shown = target;
      } else if (state === 'apnea') {
        // A big seconds counter reads from across the room better than words.
        target = null; shown = null;
        var secs = Math.floor((now - S.alerts.apnea.since) / 1000);
        valueEl.innerHTML = secs < 100 ? secs + '<span class="orb__suffix">s</span>' : Math.floor(secs / 60) + '<span class="orb__suffix">min</span>';
      } else {
        target = null; shown = null;
        valueEl.textContent = '—';
      }
    }
    every(250, setText);
    store.on('alerts', setText);
    store.on('presence', setText);

    UI.addDrawer({
      active: function () { return UI.visible(container); },
      draw: function (ts) {
        var dt = lastFrame ? Math.min(0.1, (ts - lastFrame) / 1000) : 0; lastFrame = ts;
        if (target != null) {
          shown += (target - shown) * (1 - Math.exp(-dt / 0.35));
          var txt = String(Math.round(shown));
          if (valueEl.textContent !== txt) valueEl.textContent = txt;
        }
        if (still) return;
        var x = 0;
        if (state === 'live' || state === 'measuring') {
          var wave = store.state.wave, end = store.engineNow() - getLag();
          var st = windowStats(wave, end - 12, end);
          var e = env(st.maxAbs, performance.now());
          var v = wave.valueAt(end);
          if (isFinite(v) && st.n > 4) x = Math.max(-1, Math.min(1, (v - st.mean) / e));
          // A ripple leaves the orb at the top of each breath.
          if (x < prevV && rising && prevV > 0.4 && ts - lastRipple > 1600 && state === 'live') {
            lastRipple = ts;
            var r = doc.createElement('div'); r.className = 'orb__ripple';
            r.addEventListener('animationend', function () { r.remove(); });
            el.insertBefore(r, el.querySelector('.orb__text'));
          }
          rising = x > prevV; prevV = x;
        }
        el.style.setProperty('--breath', x.toFixed(3));
      }
    });
    return el;
  }

  // ---- where the rate sits against the typical resting range -------------------------
  function rateRange(container, store) {
    var el = h(
      '<div class="range">' +
      '<div class="range__track"><span class="range__band"></span><span class="range__mark"></span></div>' +
      '<div class="range__scale num"><span>6</span><span>12</span><span>20</span><span>30</span></div>' +
      '<div class="range__note">Typical adult resting range: 12–20 breaths/min</div>' +
      '</div>');
    container.appendChild(el);
    var mark = el.querySelector('.range__mark');
    function pos(v) { return Math.max(0, Math.min(1, (v - 6) / 24)) * 100; }
    el.querySelector('.range__band').style.left = pos(12) + '%';
    el.querySelector('.range__band').style.width = (pos(20) - pos(12)) + '%';
    every(500, function () {
      var B = store.state.breathing, ok = B.bpm != null && B.lastValidAt && Date.now() - B.lastValidAt < 6000 && !store.state.alerts.apnea;
      mark.style.opacity = ok ? '1' : '0';
      if (ok) {
        mark.style.left = pos(B.bpm) + '%';
        mark.setAttribute('data-zone', B.bpm < 10 || B.bpm > 24 ? 'out' : (B.bpm < 12 || B.bpm > 20 ? 'edge' : 'in'));
      }
    });
  }

  // ---- presence room ----------------------------------------------------------------
  var ROOM_SVG =
    '<svg class="presence__art" viewBox="0 0 160 96" aria-hidden="true">' +
    '<path class="wall" d="M12 88V16a6 6 0 0 1 6-6h124a6 6 0 0 1 6 6v72"/>' +
    '<path class="wall" d="M4 88h152"/>' +
    '<path class="thing" d="M18 66h32M24 66v22M44 66v22"/>' +
    '<path class="thing" d="M24 66h20M24 66l-3.5-15"/>' +
    '<path class="arc" d="M32 51a10 10 0 0 1 0 14"/>' +
    '<path class="arc" d="M39 46a17 17 0 0 1 0 24"/>' +
    '<path class="arc" d="M46 41a24 24 0 0 1 0 34"/>' +
    '<g class="person"><circle cx="90" cy="41" r="6.5"/><path d="M80.5 88V62.5a9.5 9.5 0 0 1 19 0V88z"/></g>' +
    '<path class="thing" d="M124 88V38h20v50"/>' +
    '<rect class="door-leaf" x="124" y="38" width="20" height="50"/>' +
    '</svg>';

  function presence(container, store, opts) {
    opts = opts || {};
    var el = h(
      '<div class="presence" data-state="unknown">' + (opts.art === false ? '' : ROOM_SVG) +
      '<div class="presence__state">Waiting for sensor</div>' +
      '<div class="presence__since"></div></div>');
    container.appendChild(el);
    var stateEl = el.querySelector('.presence__state'), sinceEl = el.querySelector('.presence__since');
    function render() {
      var P = store.state.presence;
      var shownState = store.state.conn.health === 'offline' && P.state === 'unknown' ? 'unknown' : P.state;
      el.setAttribute('data-state', shownState);
      stateEl.textContent = UI.PRESENCE_TEXT[shownState] || UI.PRESENCE_TEXT.unknown;
      if (shownState === 'unknown') { sinceEl.textContent = store.state.conn.health === 'live' ? 'No presence data yet' : ''; return; }
      var c = P.confidence != null ? ' · ' + UI.pct(P.confidence) + ' confidence' : '';
      sinceEl.textContent = 'for ' + UI.duration(Date.now() - P.since) + c;
    }
    store.on('presence', render);
    every(1000, render);
  }

  // ---- gesture tiles (live slide) -----------------------------------------------------
  function gestureTiles(container, store, actions) {
    var labels = ['swipe_left', 'swipe_right', 'push'];             // the gestures the trained model has
    var el = h('<div class="gtiles"></div>');
    labels.forEach(function (g) {
      el.appendChild(h(
        '<div class="gtile" data-g="' + g + '">' +
        '<span class="gtile__icon">' + UI.icon(g) + '</span>' +
        '<span class="gtile__name">' + UI.GESTURE_TEXT[g] + '</span>' +
        '<span class="gtile__does">' + (actions[g] || '') + '</span>' +
        '<span class="gtile__conf num"></span></div>'));
    });
    container.appendChild(el);
    var timers = {};
    store.on('gesture', function (g) {
      var tile = el.querySelector('[data-g="' + g.label + '"]');
      if (!tile) return;
      tile.classList.remove('is-hit', 'is-miss');
      void tile.offsetWidth;                                    // restart the highlight if the same tile fires twice
      tile.classList.add(g.accepted ? 'is-hit' : 'is-miss');
      tile.querySelector('.gtile__conf').textContent = g.accepted ? UI.pct(g.confidence) : (g.reason === 'low confidence' ? UI.pct(g.confidence) + ' low' : 'ignored');
      clearTimeout(timers[g.label]);
      timers[g.label] = setTimeout(function () { tile.classList.remove('is-hit', 'is-miss'); }, 1400);
    });
  }

  // ---- gesture feed (monitor) --------------------------------------------------------
  function gestureFeed(container, store, opts) {
    opts = opts || {};
    var max = opts.max || 5;
    var list = h('<ul class="feed" aria-label="Recent gestures"></ul>');
    container.appendChild(list);
    function render(isNew) {
      var G = store.state.gestures.slice(0, max);
      if (!G.length) { list.innerHTML = '<li class="feed__empty">No gestures yet. Swipe or push at the laptop.</li>'; return; }
      var now = Date.now();
      list.innerHTML = G.map(function (g, i) {
        var meta = g.accepted ? UI.pct(g.confidence) + ' · ' + UI.ago(now - g.at) : (g.reason === 'low confidence' ? UI.pct(g.confidence) + ' too unsure' : 'ignored: ' + g.reason);
        return '<li class="feed__item' + (g.accepted ? '' : ' is-rejected') + '" style="' + (i === 0 && isNew ? '' : 'animation:none') + '">' +
          UI.icon(g.label === 'unknown' ? 'signal' : g.label) + '<b>' + UI.GESTURE_TEXT[g.label] + '</b><span class="feed__meta">' + UI.escapeHtml(meta) + '</span></li>';
      }).join('');
    }
    store.on('gesture', function () { render(true); });
    every(1000, function () { render(false); });
  }

  // ---- signal health ------------------------------------------------------------------
  function signal(container, store) {
    var el = h(
      '<div class="signal">' +
      '<div class="signal__item"><span class="signal__label">Echo SNR</span><span class="signal__value" data-k="snr">—</span><div class="meter" data-k="meter"><i></i></div></div>' +
      '<div class="signal__item"><span class="signal__label" data-k="carrierLabel">Carrier</span><span class="signal__value" data-k="carrier">—</span></div>' +
      '<div class="signal__item"><span class="signal__label">Frame rate</span><span class="signal__value" data-k="fps">—</span></div>' +
      '<div class="signal__item"><span class="signal__label" data-k="latLabel">Events</span><span class="signal__value" data-k="lat">—</span></div>' +
      '</div>');
    container.appendChild(el);
    var q = function (k) { return el.querySelector('[data-k="' + k + '"]'); };
    every(500, function () {
      var s = store.state.status, live = store.state.conn.health === 'live';
      if (!s) return;
      if (s.snrDb != null) {
        q('snr').innerHTML = s.snrDb.toFixed(0) + '<small>dB</small>';
        var meter = q('meter');
        meter.firstElementChild.style.transform = 'scaleX(' + Math.max(0.04, Math.min(1, s.snrDb / 70)).toFixed(3) + ')';
        meter.setAttribute('data-level', s.snrDb >= 35 ? 'good' : s.snrDb >= 20 ? 'fair' : 'poor');
      }
      var span = UI.signalSpan(s);
      if (span) { q('carrierLabel').textContent = span.label; q('carrier').innerHTML = span.text + '<small>kHz</small>'; }
      if (s.fps != null) q('fps').innerHTML = Math.round(s.fps) + '<small>fps</small>';
      // Show the engine's own pipeline latency when it reports one; otherwise the live event rate.
      if (s.latencyMs != null) {
        q('latLabel').textContent = 'Latency';
        q('lat').innerHTML = live ? Math.round(s.latencyMs) + '<small>ms</small>' : '—';
      } else {
        q('latLabel').textContent = 'Events';
        q('lat').innerHTML = live && store.state.msgRate != null ? Math.round(store.state.msgRate) + '<small>per s</small>' : '—';
      }
    });
  }

  // ---- connection indicator ------------------------------------------------------------
  function conn(container, store) {
    var el = h('<span class="conn" data-health="connecting" role="status"><span class="conn__dot"></span><span class="conn__text">Connecting</span></span>');
    container.appendChild(el);
    var text = el.querySelector('.conn__text');
    every(500, function () {
      var C = store.state.conn, s = store.state.status;
      el.setAttribute('data-health', C.health);
      var t;
      if (C.health === 'live') {
        t = 'Sonar live';
        var span = UI.signalSpan(s);
        if (span) t += ' · ' + span.text + ' kHz';
      } else if (C.health === 'stalled') t = 'Connected, no data';
      else if (C.health === 'offline') t = 'Sensor offline · retrying';
      else t = 'Connecting';
      if (text.textContent !== t) text.textContent = t;
    });
    return el;
  }

  // ---- alert banner (highest-priority care alert) ------------------------------------------
  function alertBanner(container, store, opts) {
    opts = opts || {};
    var el = h('<div class="banner" data-level="info" role="alert"><span class="banner__icon"></span><div><span class="banner__title"></span><small class="banner__detail"></small></div></div>');
    container.appendChild(el);
    var iconEl = el.querySelector('.banner__icon'), titleEl = el.querySelector('.banner__title'), detailEl = el.querySelector('.banner__detail');
    var baseTitle = doc.title, lastKey = '';
    function pickAlert() {
      var A = store.state.alerts, now = Date.now(), s = store.state.status;
      if (A.apnea) return { key: 'apnea', level: 'alert', icon: 'alert', title: 'No breathing movement detected', detail: 'for ' + UI.duration(now - A.apnea.since) + ' · person present and still' };
      if (A.offline) return { key: 'offline', level: 'warn', icon: 'link', title: 'Sensor offline', detail: 'Reconnecting to ' + store.state.conn.url };
      if (A.calibrating) return { key: 'cal', level: 'info', icon: 'calibrate', title: 'Calibrating sonar', detail: 'Keep still for a few seconds' };
      if (A.lowSignal && s) return { key: 'low', level: 'warn', icon: 'signal', title: 'Weak echo signal', detail: 'SNR ' + s.snrDb.toFixed(0) + ' dB · raise the volume or check exclusive mode' };
      if (A.entered && opts.entered !== false) return { key: 'enter', level: 'info', icon: 'enter', title: 'Person entered', detail: UI.clock(A.entered.at) };
      return null;
    }
    function render() {
      var a = pickAlert();
      el.classList.toggle('is-on', !!a);
      if (!a) { if (lastKey) { doc.title = baseTitle; lastKey = ''; } return; }
      el.setAttribute('data-level', a.level);
      if (a.key !== lastKey) { iconEl.innerHTML = UI.icon(a.icon); lastKey = a.key; }
      titleEl.textContent = a.title;
      detailEl.textContent = a.detail;
      doc.title = (a.level === 'alert' ? 'ALERT: ' : '') + a.title + ' · ' + baseTitle;
    }
    store.on('alerts', render);
    every(250, render);
    return el;
  }

  root.SonarWidgets = {
    waveform: waveform, spectrogram: spectrogram, breathing: breathing, rateRange: rateRange,
    presence: presence, gestureTiles: gestureTiles, gestureFeed: gestureFeed,
    signal: signal, conn: conn, alertBanner: alertBanner, h: h
  };
})(typeof self !== 'undefined' ? self : this);
