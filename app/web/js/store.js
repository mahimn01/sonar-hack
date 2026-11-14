/*
 * store.js: live state shared by the presenter and the Care Monitor.
 *
 * Takes normalised events from adapter.js and keeps:
 *   - connection health (live / stalled / offline),
 *   - presence with "since" times,
 *   - a breathing waveform timeline (merges full windows or incremental chunks),
 *   - judged gestures (confidence, cooldown and return-stroke filters),
 *   - care alerts (no breathing while present, person entered, sensor offline, calibrating, weak signal),
 *   - an activity log.
 * Views subscribe with store.on(topic, fn). Topics: conn, hello, status, presence, breathing,
 * gesture, spectrogram, alerts, log, raw.
 */
(function (root) {
  'use strict';

  // ---- breathing waveform timeline -----------------------------------------

  function Timeline(maxSeconds) { this.t = []; this.v = []; this.max = maxSeconds; }

  // Add samples whose last sample is at tLast (or first at t0). Overlapping older
  // samples are replaced, so both "full recent window" and "new chunk" messages work.
  Timeline.prototype.merge = function (samples, fs, tLast, t0) {
    var n = samples.length;
    if (!n || !(fs > 0)) return;
    var dt = 1 / fs;
    var start = t0 != null ? t0 : tLast - (n - 1) * dt;
    var cut = this.t.length;
    while (cut > 0 && this.t[cut - 1] >= start - dt * 0.5) cut--;
    this.t.length = cut; this.v.length = cut;
    if (cut > 0 && start - this.t[cut - 1] > Math.max(1.5, 4 * dt)) {   // a gap: break the line
      this.t.push(this.t[cut - 1] + dt); this.v.push(NaN);
    }
    for (var i = 0; i < n; i++) { this.t.push(start + i * dt); this.v.push(samples[i]); }
    var limit = this.t[this.t.length - 1] - this.max, drop = 0;
    while (drop < this.t.length && this.t[drop] < limit) drop++;
    if (drop) { this.t.splice(0, drop); this.v.splice(0, drop); }
  };

  Timeline.prototype.lastTime = function () { return this.t.length ? this.t[this.t.length - 1] : null; };

  // Index of the last sample at or before time x (binary search), -1 if none.
  Timeline.prototype.indexAt = function (x) {
    var lo = 0, hi = this.t.length - 1, ans = -1;
    while (lo <= hi) { var mid = (lo + hi) >> 1; if (this.t[mid] <= x) { ans = mid; lo = mid + 1; } else hi = mid - 1; }
    return ans;
  };

  Timeline.prototype.valueAt = function (x) {
    var i = this.indexAt(x);
    if (i < 0) return NaN;
    if (i >= this.t.length - 1) return x - this.t[i] < 0.5 ? this.v[i] : NaN;
    var a = this.v[i], b = this.v[i + 1], f = (x - this.t[i]) / (this.t[i + 1] - this.t[i]);
    return a + (b - a) * f;
  };

  // RMS around the mean over [t0, t1]; NaN if fewer than 4 samples.
  Timeline.prototype.rms = function (t0, t1) {
    var s = 0, s2 = 0, n = 0, i = Math.max(0, this.indexAt(t0));
    for (; i < this.t.length && this.t[i] <= t1; i++) {
      var v = this.v[i];
      if (this.t[i] >= t0 && isFinite(v)) { s += v; s2 += v * v; n++; }
    }
    if (n < 4) return NaN;
    var m = s / n;
    return Math.sqrt(Math.max(0, s2 / n - m * m));
  };

  Timeline.prototype.clear = function () { this.t.length = 0; this.v.length = 0; };

  // ---- store ---------------------------------------------------------------

  var OPPOSITE = { swipe_left: 'swipe_right', swipe_right: 'swipe_left', push: 'pull', pull: 'push' };

  function median(arr) {
    if (!arr.length) return null;
    var s = arr.slice().sort(function (a, b) { return a - b; });
    return s[Math.floor(s.length / 2)];
  }

  function createStore(cfg) {
    var subs = {};
    var nowMs = function () { return Date.now(); };

    var S = {
      conn: { socket: 'connecting', health: 'connecting', since: nowMs(), lastMsgAt: 0, retryInMs: 0, url: null },
      simulated: false,
      hello: null,
      status: null,
      presence: { state: 'unknown', since: nowMs(), confidence: null, prev: null },
      breathing: {
        bpm: null, confidence: null, lastEventAt: 0, lastValidAt: 0, validStreakStart: 0,
        armed: false, baselineRms: null, flatSince: 0, rangeM: null, breathState: null, engineApnea: false, windowS: null
      },
      wave: new Timeline(60),
      gestures: [],                 // newest first: {label, confidence, at, accepted, reason}
      lastAccepted: null,
      lastMotionAt: 0,
      spectro: { rows: [], max: 900, binsHz: null, axisLabel: null, lastAt: 0 },
      alerts: { apnea: null, entered: null, offline: null, calibrating: false, lowSignal: false },
      log: [],                      // newest first: {at, level, icon, text}
      latency: [],                  // recent receive-minus-engine-time samples (ms)
      msgRate: null,                // messages per second from the engine
      clockOffset: null             // local seconds minus engine seconds (min-tracked)
    };

    function on(topic, fn) { (subs[topic] = subs[topic] || []).push(fn); return fn; }
    function emit(topic, payload) {
      var l = subs[topic]; if (!l) return;
      for (var i = 0; i < l.length; i++) {
        try { l[i](payload, S); } catch (err) { console.error('[sonar] subscriber for ' + topic + ' failed:', err); }
      }
    }

    function log(level, icon, text) {
      S.log.unshift({ at: nowMs(), level: level, icon: icon, text: text });
      if (S.log.length > 60) S.log.length = 60;
      emit('log', S.log[0]);
    }

    // Engine clock -> local clock. Engine and browser share a machine, but min-tracking
    // the offset keeps us correct even if they do not.
    function trackClock(evt) {
      if (evt.tLocal) return;
      var off = nowMs() / 1000 - evt.t;
      if (S.clockOffset == null || off < S.clockOffset) S.clockOffset = off;
      else S.clockOffset += (off - S.clockOffset) * 0.002;          // drift slowly upward
      S.latency.push(off * 1000);
      if (S.latency.length > 60) S.latency.shift();
    }
    function engineNow() { return nowMs() / 1000 - (S.clockOffset || 0); }
    // Engine-send to browser-receive delay. Only meaningful when both share a clock
    // (same laptop); anything outside a sane range means they do not, so hide it.
    function linkLatencyMs() {
      var m = median(S.latency);
      if (m == null || m < -50 || m > 5000) return null;
      return Math.max(0, m);
    }

    // ---- per-kind ingestion ------------------------------------------------

    function ingestGesture(g) {
      if (g.label === 'none') return;
      var now = nowMs();
      S.lastMotionAt = now;
      disarm();
      var c = g.confidence == null ? 1 : g.confidence;
      var reason = null;
      if (g.label === 'unknown') reason = 'unknown label "' + g.rawLabel + '"';
      else if (c < cfg.minConfidence) reason = 'low confidence';
      else if (S.lastAccepted && now - S.lastAccepted.at < cfg.cooldownMs) reason = 'too soon';
      else if (S.lastAccepted && OPPOSITE[g.label] === S.lastAccepted.label && now - S.lastAccepted.at < cfg.returnStrokeMs) reason = 'return stroke';
      var judged = { label: g.label, rawLabel: g.rawLabel, confidence: g.confidence, at: now, t: g.t, accepted: !reason, reason: reason };
      if (judged.accepted) S.lastAccepted = judged;
      S.gestures.unshift(judged);
      if (S.gestures.length > 30) S.gestures.length = 30;
      emit('gesture', judged);
    }

    function ingestPresence(p) {
      if (p.state === 'unknown') return;
      var now = nowMs(), prev = S.presence.state;
      S.presence.confidence = p.confidence;
      if (p.state !== prev) {
        S.presence.prev = prev;
        S.presence.state = p.state;
        S.presence.since = now;
        S.lastMotionAt = now;
        disarm();
        if (p.state === 'entering' && prev !== 'present') {
          S.alerts.entered = { at: now };
          log('info', 'enter', 'Person entered the room');
          emit('alerts', S.alerts);
        } else if (p.state === 'present' && prev === 'empty') {
          S.alerts.entered = { at: now };
          log('info', 'enter', 'Person detected in the room');
          emit('alerts', S.alerts);
        } else if (p.state === 'empty' && prev !== 'unknown') {
          log('info', 'empty', 'Room is empty');
          S.wave.clear();
        }
      }
      emit('presence', S.presence);
    }

    // The engine scales each window to its own peak, so a long breath-hold would be
    // stretched back to full height. When it also reports the real peak-to-peak chest
    // movement (amplitude_mm, from the 3rd to 97th percentile), undo the scaling so the
    // line and the flatline check work in millimetres.
    function toMillimetres(wave, amplitudeMm) {
      if (amplitudeMm == null || !(amplitudeMm >= 0) || wave.length < 5) return wave;
      var s = wave.filter(function (v) { return isFinite(v); }).sort(function (a, b) { return a - b; });
      if (s.length < 5) return wave;
      var pp = s[Math.round(0.97 * (s.length - 1))] - s[Math.round(0.03 * (s.length - 1))];
      if (!(pp > 1e-9)) return wave;
      var k = amplitudeMm / pp;
      return wave.map(function (v) { return v * k; });
    }

    function isValidBreath(b) {
      if (b.bpm == null || b.apnea === true) return false;
      if (b.breathState && b.breathState !== 'breathing') return false;
      return b.confidence == null || b.confidence >= cfg.breathMinConfidence;
    }

    function ingestBreathing(b) {
      var now = nowMs(), B = S.breathing;
      B.lastEventAt = now;
      B.confidence = b.confidence;
      B.breathState = b.breathState;
      B.engineApnea = b.apnea === true || b.breathState === 'apnea';
      B.windowS = b.windowS;
      if (b.rangeM != null) B.rangeM = b.rangeM;
      if (b.breathState === 'motion') { S.lastMotionAt = now; disarm(); }   // the engine saw the body move
      if (isValidBreath(b)) {
        if (!B.validStreakStart || now - B.lastValidAt > 3000) B.validStreakStart = now;
        B.lastValidAt = now;
        B.bpm = b.bpm;
      } else if (b.bpm != null && B.bpm == null) {
        B.bpm = b.bpm;                                           // low-confidence first estimate, shown as "measuring"
      }
      if (b.waveform && b.fs) {
        var tLast = b.t;
        var t0 = b.tLocal ? null : b.t0;                             // t0 only trusted with an engine clock
        if (b.tLocal) tLast = engineNow();
        S.wave.merge(toMillimetres(b.waveform, b.amplitude), b.fs, tLast, t0);
      }
      emit('breathing', B);
    }

    function ingestStatus(s) {
      S.status = s;
      if (s.source && /mock|sim|fake|replay/i.test(s.source)) S.simulated = true;
      var cal = s.calibrating === true;
      if (cal !== S.alerts.calibrating) {
        S.alerts.calibrating = cal;
        if (cal) { log('info', 'calibrate', 'Calibrating sonar'); disarm(); }
        else log('ok', 'check', 'Calibration complete');
        emit('alerts', S.alerts);
      }
      var low = s.snrDb != null && s.snrDb < cfg.lowSnrDb;
      if (low !== S.alerts.lowSignal) {
        S.alerts.lowSignal = low;
        if (low) log('warn', 'signal', 'Weak echo signal (' + s.snrDb.toFixed(0) + ' dB SNR)');
        emit('alerts', S.alerts);
      }
      emit('status', s);
    }

    function ingestSpectrogram(sp) {
      var R = S.spectro;
      for (var i = 0; i < sp.rows.length; i++) R.rows.push(sp.rows[i]);
      if (R.rows.length > R.max) R.rows.splice(0, R.rows.length - R.max);
      if (sp.binsHz) R.binsHz = sp.binsHz;
      if (sp.axisLabel) R.axisLabel = sp.axisLabel;
      R.lastAt = nowMs();
      emit('spectrogram', sp.rows);
    }

    function ingestHello(h) {
      S.hello = h;
      if (h.source && /mock|sim|fake|replay/i.test(h.source)) S.simulated = true;
      emit('hello', h);
    }

    function ingest(evt) {
      if (evt.kind !== 'unknown') trackClock(evt);
      switch (evt.kind) {
        case 'gesture': ingestGesture(evt); break;
        case 'presence': ingestPresence(evt); break;
        case 'breathing': ingestBreathing(evt); break;
        case 'status': ingestStatus(evt); break;
        case 'spectrogram': ingestSpectrogram(evt); break;
        case 'hello': ingestHello(evt); break;
        default: break;
      }
    }

    // Raw WebSocket text -> adapter -> ingest.
    var msgCount = 0, rateSince = nowMs();
    function ingestRaw(text) {
      S.conn.lastMsgAt = nowMs();
      msgCount++;
      var events = root.SonarAdapter.adapt(text);
      emit('raw', { text: text, events: events });
      for (var i = 0; i < events.length; i++) ingest(events[i]);
    }

    function setSocket(state, info) {
      S.conn.socket = state;
      if (info && info.retryInMs != null) S.conn.retryInMs = info.retryInMs;
      evaluate();
    }

    // ---- care alert logic (runs on a timer and after events) -------------------

    function disarm() {
      S.breathing.armed = false;
      S.breathing.validStreakStart = 0;
      S.breathing.flatSince = 0;
      S.breathing.baselineRms = null;
    }

    function evaluate() {
      var now = nowMs(), B = S.breathing, C = S.conn, A = S.alerts;
      if (now - rateSince >= 2000) { S.msgRate = msgCount * 1000 / (now - rateSince); msgCount = 0; rateSince = now; }

      // Connection health.
      var health;
      if (C.socket !== 'open') health = C.socket === 'connecting' && C.health === 'connecting' ? 'connecting' : 'offline';
      else if (!C.lastMsgAt || now - C.lastMsgAt > 2500) health = 'stalled';
      else health = 'live';
      if (health !== C.health) {
        var was = C.health;
        C.health = health; C.since = now;
        if (health === 'live' && (was === 'offline' || was === 'stalled')) log('ok', 'link', 'Sensor connected');
        emit('conn', C);
      }
      var offline = health === 'offline' || health === 'stalled';
      if (offline && now - C.since > 3000) {
        if (!A.offline) { A.offline = { since: C.since }; log('warn', 'link', 'Sensor offline'); disarm(); emit('alerts', A); }
      } else if (!offline && A.offline) { A.offline = null; emit('alerts', A); }

      // Breathing: arm once breathing has been steady for 5 s while someone is still.
      var still = S.presence.state === 'present' && now - S.lastMotionAt >= cfg.motionHoldSeconds * 1000 && !A.calibrating && !A.offline;
      if (!still) {
        if (B.armed) disarm();
      } else if (!B.armed && B.validStreakStart && now - B.validStreakStart >= 5000 && now - B.lastValidAt < 2500) {
        B.armed = true;
      }

      // Flatline detector on the waveform: amplitude collapses well below the person's own baseline.
      var tNow = S.wave.lastTime();
      if (B.armed && tNow != null) {
        var recent = S.wave.rms(tNow - 3, tNow);
        if (isFinite(recent)) {
          if (B.baselineRms == null) B.baselineRms = recent;
          if (!B.flatSince && recent > B.baselineRms * 0.5) B.baselineRms += (recent - B.baselineRms) * 0.05;
          if (recent < B.baselineRms * 0.3) { if (!B.flatSince) B.flatSince = now - 3000; }
          else if (recent > B.baselineRms * 0.5) B.flatSince = 0;
        }
      }

      // Apnea: the engine says so, or the waveform is flat, or valid breathing stopped arriving.
      var apneaSince = 0;
      if (B.armed) {
        var limit = cfg.apneaSeconds * 1000;
        if (B.engineApnea) apneaSince = Math.min(now, B.lastValidAt || now);
        if (B.flatSince && now - B.flatSince >= limit) apneaSince = apneaSince ? Math.min(apneaSince, B.flatSince) : B.flatSince;
        if (B.lastValidAt && now - B.lastValidAt >= limit && !B.flatSince) {
          // No valid estimate for a while. Only call it apnea if the waveform is not showing big motion.
          var motion = tNow != null && B.baselineRms != null && S.wave.rms(tNow - 3, tNow) > B.baselineRms * 2.5;
          if (!motion) apneaSince = apneaSince ? Math.min(apneaSince, B.lastValidAt) : B.lastValidAt;
        }
      }
      if (apneaSince && !A.apnea) {
        A.apnea = { since: apneaSince };
        log('alert', 'alert', 'No breathing movement for ' + Math.round((now - apneaSince) / 1000) + ' s');
        emit('alerts', A);
      } else if (!apneaSince && A.apnea) {
        var dur = Math.round((now - A.apnea.since) / 1000);
        A.apnea = null;
        log('ok', 'breath', 'Breathing resumed (paused ' + dur + ' s)');
        emit('alerts', A);
      }

      if (A.entered && now - A.entered.at > 6000) { A.entered = null; emit('alerts', A); }
    }

    // Activity log entry when breathing is first locked during a presence session.
    var lockedLogged = false;
    on('breathing', function (B) {
      if (B.lastValidAt && !lockedLogged && S.presence.state === 'present' && B.bpm != null) {
        lockedLogged = true;
        log('ok', 'breath', 'Breathing measured: ' + Math.round(B.bpm) + ' breaths/min');
      }
    });
    on('presence', function (P) { if (P.state !== 'present') lockedLogged = false; });

    setInterval(evaluate, 250);

    return {
      state: S,
      on: on,
      emit: emit,
      ingest: ingest,
      ingestRaw: ingestRaw,
      setSocket: setSocket,
      evaluate: evaluate,
      engineNow: engineNow,
      linkLatencyMs: linkLatencyMs,
      log: log,
      config: cfg
    };
  }

  // Wire a store to a live connection.
  function start(cfg) {
    var store = createStore(cfg);
    store.state.conn.url = cfg.wsUrl;
    var conn = root.SonarConnection.connect(cfg.wsUrl, {
      onMessage: store.ingestRaw,
      onState: store.setSocket
    });
    store.connection = conn;
    return store;
  }

  root.SonarStore = { createStore: createStore, start: start, Timeline: Timeline };
})(typeof self !== 'undefined' ? self : this);
