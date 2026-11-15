/*
 * adapter.js: the ONLY place in the web UI that knows the engine's wire format.
 *
 * The sonar engine publishes JSON over a WebSocket. This module turns whatever
 * it sends into a small set of normalised events that the rest of the UI uses:
 *
 *   {kind:'gesture',     label, rawLabel, confidence, durationS, peakDb, source, t, tLocal}
 *   {kind:'presence',    state, rawState, confidence, rangeM, moving, motionlessS, changed, t, tLocal}
 *   {kind:'breathing',   bpm, confidence, waveform, fs, t0, t, apnea, breathState, amplitude, rangeM, windowS,
 *                        held, stoppedS, motionlessS, tLocal}
 *   {kind:'status',      snrDb, snrList, carrierHz, carriersHz, bandHz, fps, calibrating, source, device, latencyMs,
 *                        locked, clipped, dropped, activityDb, noiseDb, classifier, lastDecision, t, tLocal}
 *   {kind:'spectrogram', rows, binsHz, axisLabel, autoscaled, frameS, t, tLocal}
 *   {kind:'fall',        stage, seq, burstT, burstDb, burstLevel, fastLevel, oneSided, direction, durationS, stillnessS,
 *                        needsS, rangeBeforeM, rangeAfterM, confidence, reason, experimental, t, tLocal}
 *   {kind:'hello',       source, version, schema, labels, classifier, t, tLocal}
 *   {kind:'unknown',     raw, reason}
 * and Sonar Care's own messages (see docs/SONAR_CARE.md):
 *   {kind:'careState', ...}  {kind:'careAlert', change, alert}  {kind:'careAction', gesture, result, ...}
 *   {kind:'careTimeline', entry}  {kind:'careSettings', rev, settings}  {kind:'careHello', ...}
 *
 * When INTERFACES.md changes, edit the alias tables and pickers below. Nothing
 * else in web/ reads raw engine fields. The Python twin is app/adapter.py, and
 * both are tested against app/tests/fixtures/engine_events.json.
 *
 * Loads as a classic browser script (window.SonarAdapter) and as a Node module.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.SonarAdapter = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var VERSION = 'provisional-2';

  // Message type names the engine might use, mapped to our kinds.
  var TYPE_ALIASES = {
    // Sonar Care's own messages (app/care/service.py), on the same stream as the engine's.
    care_hello: 'careHello', care_state: 'careState', care_alert: 'careAlert', care_action: 'careAction',
    care_timeline: 'careTimeline', care_settings: 'careSettings',
    fall: 'fall', care_fall: 'fall', possible_fall: 'fall',
    gesture: 'gesture', gestures: 'gesture',
    presence: 'presence', occupancy: 'presence', room: 'presence',
    breathing: 'breathing', breath: 'breathing', respiration: 'breathing', resp: 'breathing', vitals: 'breathing',
    status: 'status', health: 'status', signal: 'status', stats: 'status',
    spectrogram: 'spectrogram', doppler: 'spectrogram', spec: 'spectrogram',
    hello: 'hello', info: 'hello', meta: 'hello'
  };

  // Gesture class names, mapped to the four the UI acts on (plus 'none').
  var GESTURE_ALIASES = {
    swipe_left: 'swipe_left', left: 'swipe_left', swipeleft: 'swipe_left', left_swipe: 'swipe_left',
    swipe_right: 'swipe_right', right: 'swipe_right', swiperight: 'swipe_right', right_swipe: 'swipe_right',
    push: 'push', toward: 'push', towards: 'push', approach: 'push',
    pull: 'pull', wave: 'pull', pull_wave: 'pull', pullwave: 'pull', away: 'pull', retreat: 'pull',
    none: 'none', no_gesture: 'none', nogesture: 'none', idle: 'none', background: 'none', noise: 'none', other: 'none', null: 'none'
  };

  // Presence states, mapped to empty | present | entering.
  var PRESENCE_ALIASES = {
    empty: 'empty', empty_room: 'empty', absent: 'empty', none: 'empty', vacant: 'empty', nobody: 'empty', no_person: 'empty',
    present: 'present', person_present: 'present', person_present_still: 'present', present_still: 'present',
    still: 'present', occupied: 'present', sitting: 'present', standing: 'present', stationary: 'present',
    entering: 'entering', person_walking_in: 'entering', walking_in: 'entering', walking: 'entering',
    moving: 'entering', enter: 'entering', entered: 'entering', arrival: 'entering', motion: 'entering'
  };

  // Optional breathing-state strings, mapped to breathing | apnea | motion | absent | measuring.
  var BREATH_STATE_ALIASES = {
    breathing: 'breathing', normal: 'breathing', ok: 'breathing', tracking: 'breathing', locked: 'breathing',
    apnea: 'apnea', apnoea: 'apnea', breath_hold: 'apnea', no_breathing: 'apnea', stopped: 'apnea', flat: 'apnea',
    motion: 'motion', moving: 'motion', unstable: 'motion',
    absent: 'absent', empty: 'absent', no_person: 'absent',
    measuring: 'measuring', acquiring: 'measuring', searching: 'measuring', warming_up: 'measuring', calibrating: 'measuring',
    no_signal: 'measuring', nosignal: 'measuring', weak: 'measuring'
  };

  // ---- small helpers -------------------------------------------------------

  function isObj(v) { return v !== null && typeof v === 'object' && !Array.isArray(v); }

  function key(s) {
    return String(s == null ? '' : s).trim().toLowerCase().replace(/[\s\-\/]+/g, '_');
  }

  // First present (non-null) field among aliases.
  function pick(obj, names) {
    for (var i = 0; i < names.length; i++) {
      if (Object.prototype.hasOwnProperty.call(obj, names[i]) && obj[names[i]] != null) return obj[names[i]];
    }
    return undefined;
  }

  function num(v) {
    if (typeof v === 'number') return isFinite(v) ? v : null;
    if (typeof v === 'string' && v.trim() !== '') { var n = Number(v); return isFinite(n) ? n : null; }
    return null;
  }

  function bool(v) {
    if (typeof v === 'boolean') return v;
    if (typeof v === 'number') return v !== 0;
    if (typeof v === 'string') {
      var k = key(v);
      if (['true', 'yes', 'on', '1', 'running', 'active', 'in_progress'].indexOf(k) >= 0) return true;
      if (['false', 'no', 'off', '0', 'done', 'idle', 'complete', 'finished'].indexOf(k) >= 0) return false;
    }
    return null;
  }

  function boolOrNull(v) { return v == null ? null : bool(v); }
  function strOrNull(v) { return v == null ? null : String(v); }

  // Confidence as 0..1. Accepts 0..1 or 0..100 (percent).
  function conf(v) {
    var n = num(v);
    if (n == null) return null;
    if (n > 1 && n <= 100) n = n / 100;
    return Math.max(0, Math.min(1, n));
  }

  // Timestamp in unix seconds. Accepts seconds or milliseconds; falls back to local clock.
  function time(obj, nowS) {
    var t = num(pick(obj, ['t', 'ts', 'time', 'timestamp', 't_s', 'unix']));
    if (t == null) {
      var ms = num(pick(obj, ['t_ms', 'ts_ms', 'time_ms']));
      if (ms != null) t = ms / 1000;
    }
    if (t != null && t > 1e12) t = t / 1000;   // milliseconds sent in a seconds field
    if (t == null || t < 1e8) return { t: nowS, tLocal: true };
    return { t: t, tLocal: false };
  }

  function numArray(v) {
    if (!Array.isArray(v)) return null;
    var out = new Array(v.length);
    for (var i = 0; i < v.length; i++) { var n = num(v[i]); out[i] = n == null ? NaN : n; }
    return out;
  }

  // A single number from a scalar, an array (median) or a {freq: value} map (median).
  function scalarOf(v) {
    var n = num(v);
    if (n != null) return { value: n, list: null };
    var list = null;
    if (Array.isArray(v)) list = numArray(v).filter(function (x) { return isFinite(x); });
    else if (isObj(v)) list = Object.keys(v).map(function (k) { return num(v[k]); }).filter(function (x) { return x != null; });
    if (!list || !list.length) return { value: null, list: null };
    var s = list.slice().sort(function (a, b) { return a - b; });
    return { value: s[Math.floor(s.length / 2)], list: list };
  }

  // ---- per-kind parsers ----------------------------------------------------

  function parseGesture(m, ts) {
    var raw = pick(m, ['label', 'gesture', 'name', 'class', 'cls', 'prediction', 'pred', 'value']);
    var label = GESTURE_ALIASES[key(raw)] || (raw == null ? 'none' : 'unknown');
    return {
      kind: 'gesture', label: label, rawLabel: raw == null ? null : String(raw),
      confidence: conf(pick(m, ['confidence', 'conf', 'p', 'prob', 'probability', 'score'])),
      durationS: num(pick(m, ['duration_s', 'duration'])),
      peakDb: num(pick(m, ['peak_db', 'peak'])),
      source: strOrNull(pick(m, ['source', 'classifier'])),
      t: ts.t, tLocal: ts.tLocal
    };
  }

  function parsePresence(m, ts) {
    var raw = pick(m, ['state', 'presence', 'label', 'class', 'status', 'value']);
    var state = PRESENCE_ALIASES[key(raw)];
    if (!state && typeof raw === 'boolean') state = raw ? 'present' : 'empty';
    return {
      kind: 'presence', state: state || 'unknown', rawState: raw == null ? null : String(raw),
      confidence: conf(pick(m, ['confidence', 'conf', 'p', 'prob', 'probability', 'score'])),
      rangeM: num(pick(m, ['range_m', 'distance_m', 'range', 'distance'])),
      moving: boolOrNull(pick(m, ['moving', 'is_moving'])),
      motionlessS: num(pick(m, ['motionless_s', 'still_s', 'motionless'])),
      changed: boolOrNull(pick(m, ['changed'])),
      t: ts.t, tLocal: ts.tLocal
    };
  }

  function parseBreathing(m, ts) {
    var bpm = num(pick(m, ['bpm', 'rate_bpm', 'breaths_per_min', 'breathing_rate', 'rr', 'resp_rate', 'rate', 'value']));
    if (bpm != null && (bpm <= 0 || bpm > 120)) bpm = null;       // 0 / garbage means "no estimate"
    var wf = pick(m, ['waveform', 'wave', 'samples', 'displacement', 'signal', 'trace', 'phase']);
    var fs = num(pick(m, ['fs', 'waveform_fs', 'fs_hz', 'sample_rate_hz', 'sample_rate', 'srate']));
    var t0 = num(pick(m, ['t0', 'waveform_t0', 't_start', 'start_t']));
    if (isObj(wf)) {                                               // {"samples":[...], "fs":20}
      fs = fs != null ? fs : num(pick(wf, ['fs', 'sample_rate', 'fs_hz']));
      t0 = t0 != null ? t0 : num(pick(wf, ['t0', 't_start']));
      wf = pick(wf, ['samples', 'data', 'values', 'y']);
    }
    var waveform = numArray(wf);
    if (waveform && !waveform.length) waveform = null;
    if (t0 != null && t0 > 1e12) t0 = t0 / 1000;
    var stateRaw = pick(m, ['breath_state', 'breathing_state', 'state']);
    var apnea = bool(pick(m, ['apnea', 'apnoea', 'breath_hold', 'no_breathing']));
    var breathState = BREATH_STATE_ALIASES[key(stateRaw)] || null;
    if (apnea === true) breathState = 'apnea';
    return {
      kind: 'breathing', bpm: bpm,
      confidence: conf(pick(m, ['confidence', 'conf', 'quality', 'p', 'score'])),
      waveform: waveform, fs: fs != null && fs > 0 ? fs : null, t0: t0,
      apnea: apnea, breathState: breathState,
      amplitude: num(pick(m, ['amplitude', 'amplitude_mm', 'rms', 'displacement_mm'])),
      rangeM: num(pick(m, ['range_m', 'distance_m', 'range', 'distance'])),
      windowS: num(pick(m, ['window_s', 'window', 'window_sec'])),
      held: boolOrNull(pick(m, ['held'])),
      stoppedS: num(pick(m, ['stopped_s'])),
      motionlessS: num(pick(m, ['motionless_s', 'still_s'])),
      t: ts.t, tLocal: ts.tLocal
    };
  }

  // The classifier's verdict on the latest movement burst, including rejected ones ("none", "blocked:weak").
  function lastDecision(v) {
    if (!isObj(v) || v.label == null) return null;
    return { label: String(v.label), confidence: conf(v.confidence), peakDb: num(v.peak_db) };
  }

  function parseStatus(m, ts) {
    var snr = scalarOf(pick(m, ['snr_db', 'snr', 'snrDb', 'snr_dB']));
    // The engine sends its first tone (carrier_hz) and the full list (tones_hz): the list wins, so both tones show.
    var car = scalarOf(pick(m, ['tones_hz', 'carriers_hz', 'carrier_hz', 'carrier', 'f0', 'f0_hz', 'tone_hz', 'freq_hz']));
    var band = pick(m, ['band_hz', 'band', 'chirp_band_hz', 'chirp_hz', 'sweep_hz']);
    var bandHz = null;
    if (Array.isArray(band) && band.length >= 2) {
      var lo = num(band[0]), hi = num(band[1]);
      if (lo != null && hi != null) bandHz = [Math.min(lo, hi), Math.max(lo, hi)];
    } else {
      var flo = num(pick(m, ['f_lo', 'f_low', 'f_min', 'fmin', 'f_start'])), fhi = num(pick(m, ['f_hi', 'f_high', 'f_max', 'fmax', 'f_end']));
      if (flo != null && fhi != null) bandHz = [Math.min(flo, fhi), Math.max(flo, fhi)];
    }
    var cal = pick(m, ['calibrating', 'is_calibrating', 'calibration']);
    return {
      kind: 'status',
      snrDb: snr.value, snrList: snr.list,
      carrierHz: car.value, carriersHz: car.list,
      bandHz: bandHz,
      fps: num(pick(m, ['fps', 'frame_rate', 'frames_per_s', 'hop_hz', 'update_hz'])),
      calibrating: cal == null ? null : bool(cal),
      source: pick(m, ['source', 'mode', 'origin']) != null ? String(pick(m, ['source', 'mode', 'origin'])) : null,
      device: pick(m, ['device', 'mic', 'input_device']) != null ? String(pick(m, ['device', 'mic', 'input_device'])) : null,
      latencyMs: num(pick(m, ['latency_ms', 'lag_ms', 'pipeline_latency_ms'])),
      locked: boolOrNull(pick(m, ['locked', 'lock'])),
      clipped: boolOrNull(pick(m, ['clipped', 'clipping'])),
      dropped: num(pick(m, ['dropped', 'dropouts'])),
      activityDb: num(pick(m, ['activity_db', 'activity'])),
      noiseDb: num(pick(m, ['noise_db'])),
      classifier: strOrNull(pick(m, ['classifier'])),
      lastDecision: lastDecision(pick(m, ['last_decision'])),
      t: ts.t, tLocal: ts.tLocal
    };
  }

  function parseSpectrogram(m, ts) {
    var rows = pick(m, ['rows', 'frames', 'columns', 'data', 'spec', 'S', 'values']);
    if (rows == null) { var one = pick(m, ['row', 'frame', 'column']); if (one != null) rows = [one]; }
    if (!Array.isArray(rows) || !rows.length) return { kind: 'unknown', raw: m, reason: 'spectrogram without rows' };
    if (!Array.isArray(rows[0])) rows = [rows];                    // a single flat row
    var out = [], lo = Infinity, hi = -Infinity;
    for (var i = 0; i < rows.length; i++) {
      var r = numArray(rows[i]);
      if (!r || !r.length) continue;
      for (var j = 0; j < r.length; j++) { if (isFinite(r[j])) { if (r[j] < lo) lo = r[j]; if (r[j] > hi) hi = r[j]; } }
      out.push(r);
    }
    if (!out.length) return { kind: 'unknown', raw: m, reason: 'spectrogram rows empty' };
    // Expected 0..1. dB or raw magnitudes get a fixed or per-message rescale so the UI still shows something.
    var autoscaled = false;
    if (lo < -0.01 || hi > 1.5) {
      autoscaled = true;
      var a, b;
      if (hi <= 0.5 && lo < -1) { a = Math.max(lo, hi - 80); b = hi; }      // dB-like: show the top 80 dB
      else { a = lo; b = hi; }
      var span = b - a || 1;
      for (var k = 0; k < out.length; k++) for (var q = 0; q < out[k].length; q++) {
        out[k][q] = isFinite(out[k][q]) ? Math.max(0, Math.min(1, (out[k][q] - a) / span)) : 0;
      }
    } else {
      for (var k2 = 0; k2 < out.length; k2++) for (var q2 = 0; q2 < out[k2].length; q2++) {
        var v = out[k2][q2]; out[k2][q2] = isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
      }
    }
    return {
      kind: 'spectrogram', rows: out,
      binsHz: numArray(pick(m, ['bins_hz', 'freqs_hz', 'doppler_hz', 'f_hz', 'axis_hz'])),
      axisLabel: pick(m, ['axis', 'label', 'units']) != null ? String(pick(m, ['axis', 'label', 'units'])) : null,
      autoscaled: autoscaled,
      frameS: num(pick(m, ['frame_s', 'hop_s', 'dt'])),
      t: ts.t, tLocal: ts.tLocal
    };
  }

  // The experimental "possible fall" event. Today Sonar Care's own detector sends it on the relay, not the engine.
  function parseFall(m, ts) {
    var stage = pick(m, ['stage', 'state', 'phase']);
    stage = stage == null ? null : String(stage).toLowerCase();
    if (['candidate', 'confirmed', 'dismissed', 'moved'].indexOf(stage) < 0) {
      return { kind: 'unknown', raw: m, reason: 'fall event without a known stage' };
    }
    var direction = strOrNull(m.direction);
    return {
      kind: 'fall', stage: stage, seq: num(m.seq), burstT: num(m.burst_t),
      burstDb: num(m.burst_db), burstLevel: num(m.burst_level), fastLevel: num(m.fast_level),
      oneSided: num(m.one_sided), direction: direction === 'toward' || direction === 'away' ? direction : null,
      durationS: num(m.duration_s), stillnessS: num(m.stillness_s), needsS: num(m.needs_s),
      rangeBeforeM: num(m.range_before_m), rangeAfterM: num(m.range_after_m),
      confidence: conf(m.confidence), reason: strOrNull(m.reason),
      experimental: true, t: ts.t, tLocal: ts.tLocal
    };
  }

  function parseHello(m, ts) {
    var src = pick(m, ['source', 'mode', 'origin', 'engine']);
    var ver = pick(m, ['version', 'engine_version']);
    var sch = pick(m, ['schema', 'schema_version', 'protocol']);
    var labels = pick(m, ['labels', 'gestures', 'classes']);
    return {
      kind: 'hello', source: src == null ? null : String(src), version: ver == null ? null : String(ver),
      schema: sch == null ? null : String(sch),
      labels: Array.isArray(labels) ? labels.map(String) : null,
      classifier: strOrNull(pick(m, ['classifier'])),
      t: ts.t, tLocal: ts.tLocal
    };
  }

  // ---- Sonar Care's own messages ---------------------------------------------
  // Field names on the wire are snake_case (see docs/SONAR_CARE.md); the app uses camelCase like the rest.

  function d(v) { return isObj(v) ? v : {}; }
  function yes(v) { return bool(v) === true; }

  function careAlert(a) {
    if (!isObj(a)) return null;
    return {
      id: strOrNull(a.id), kind: strOrNull(a.kind), level: strOrNull(a.level), state: strOrNull(a.state),
      title: strOrNull(a.title), detail: strOrNull(a.detail),
      since: num(a.since), raisedAt: num(a.raised_at), activeAt: num(a.active_at), escalateAt: num(a.escalate_at),
      ackedAt: num(a.acked_at), resolvedAt: num(a.resolved_at), resolution: strOrNull(a.resolution),
      ackS: num(a.ack_s), count: num(a.count), thresholdS: num(a.threshold_s), breathingDetected: boolOrNull(a.breathing_detected),
      experimental: yes(a.experimental), direction: strOrNull(a.direction), confidence: conf(a.confidence), movedAt: num(a.moved_at)
    };
  }

  function fallBurst(b) {
    if (!isObj(b)) return null;
    return {
      t: num(b.t), durationS: num(b.duration_s), level: num(b.level), fastLevel: num(b.fast_level), oneSided: num(b.one_sided),
      direction: strOrNull(b.direction), counted: yes(b.counted), why: strOrNull(b.why), outcome: strOrNull(b.outcome)
    };
  }

  // The experimental fall alert's live read-out inside care_state (app/care/fall.py snapshot()).
  function fallState(f) {
    f = d(f);
    var w = f.watching, th = d(f.thresholds);
    return {
      enabled: yes(f.enabled), sensitivity: strOrNull(f.sensitivity),
      level: num(f.level), slowLevel: num(f.slow_level), hasRows: yes(f.has_rows), error: strOrNull(f.error),
      watching: isObj(w) ? { since: num(w.since), stillS: num(w.still_s), needsS: num(w.needs_s), direction: strOrNull(w.direction) } : null,
      lastBurst: fallBurst(f.last_burst),
      thresholds: {
        burstLevel: num(th.burst_level), fastLevel: num(th.fast_level), oneSided: num(th.one_sided),
        minS: num(th.min_s), maxS: num(th.max_s), stillnessS: num(th.stillness_s)
      }
    };
  }

  function careEntry(e) {
    if (!isObj(e)) return null;
    return {
      id: num(e.id), t: num(e.t), kind: strOrNull(e.kind), level: strOrNull(e.level), icon: strOrNull(e.icon),
      text: strOrNull(e.text), detail: strOrNull(e.detail), alertId: strOrNull(e.alert_id)
    };
  }

  function careGesture(g) {
    if (!isObj(g) || g.gesture == null) return null;
    return {
      gesture: String(g.gesture), confidence: conf(g.confidence), result: strOrNull(g.result), reason: strOrNull(g.reason),
      action: strOrNull(g.action), actionLabel: strOrNull(g.action_label), t: num(g.t)
    };
  }

  function parseCareHello(m, ts) {
    return {
      kind: 'careHello', version: strOrNull(m.version), dryRun: boolOrNull(m.dry_run), mode: strOrNull(m.mode),
      gestures: Array.isArray(m.gestures) ? m.gestures.map(String) : [],
      timeline: Array.isArray(m.timeline) ? m.timeline.map(careEntry).filter(Boolean) : [],
      t: ts.t, tLocal: ts.tLocal
    };
  }

  // The settings document belongs to the local API (GET/POST /api/settings) and is passed through as it is.
  function parseCareSettings(m, ts) {
    return { kind: 'careSettings', rev: num(m.rev), settings: d(m.settings), t: ts.t, tLocal: ts.tLocal };
  }

  function parseCareAction(m, ts) {
    return {
      kind: 'careAction', gesture: strOrNull(m.gesture), confidence: conf(m.confidence), result: strOrNull(m.result),
      reason: strOrNull(m.reason), action: strOrNull(m.action), actionLabel: strOrNull(m.action_label), detail: strOrNull(m.detail),
      delayedS: num(m.delayed_s), helpUntil: num(m.help_until), doublePush: boolOrNull(m.double_push), gapS: num(m.gap_s),
      t: ts.t, tLocal: ts.tLocal
    };
  }

  function parseCareAlert(m, ts) {
    return { kind: 'careAlert', change: strOrNull(m.change), alert: careAlert(m.alert), t: ts.t, tLocal: ts.tLocal };
  }

  function parseCareTimeline(m, ts) {
    return { kind: 'careTimeline', entry: careEntry(m.entry), t: ts.t, tLocal: ts.tLocal };
  }

  function parseCareState(m, ts) {
    var eng = d(m.engine), link = d(m.link), sonar = d(m.sonar), pres = d(m.presence), br = d(m.breathing);
    var still = d(m.stillness), hlp = d(m.help), ctl = d(m.control), mon = d(m.monitoring);
    var keys = d(ctl.keys), mh = d(mon.help), mm = d(mon.no_movement), mb = d(mon.no_breathing), mf = d(mon.fall);
    var mapping = {}, rawMap = d(ctl.mapping);
    for (var g in rawMap) mapping[g] = { action: strOrNull(d(rawMap[g]).action), label: strOrNull(d(rawMap[g]).label) };
    return {
      kind: 'careState', version: strOrNull(m.version), starting: yes(m.starting),
      dryRun: boolOrNull(m.dry_run), quiet: boolOrNull(m.quiet), mode: strOrNull(m.mode),
      engine: {
        managed: boolOrNull(eng.managed), state: strOrNull(eng.state), mode: strOrNull(eng.mode), port: num(eng.port),
        url: strOrNull(eng.url), pid: num(eng.pid), model: strOrNull(eng.model), restarts: num(eng.restarts),
        lastError: strOrNull(eng.last_error), uptimeS: num(eng.uptime_s), retryInS: num(eng.retry_in_s)
      },
      link: { state: strOrNull(link.state), health: strOrNull(link.health), ageS: num(link.age_s) },
      sonar: {
        snrDb: num(sonar.snr_db), signal: strOrNull(sonar.signal), calibrating: yes(sonar.calibrating),
        locked: boolOrNull(sonar.locked), source: strOrNull(sonar.source), noiseDb: num(sonar.noise_db), fps: num(sonar.fps),
        clipped: boolOrNull(sonar.clipped), device: strOrNull(sonar.device)
      },
      presence: {
        state: strOrNull(pres.state), since: num(pres.since), confidence: conf(pres.confidence), rangeM: num(pres.range_m),
        moving: yes(pres.moving), confirmed: yes(pres.confirmed), motionlessS: num(pres.motionless_s)
      },
      breathing: { bpm: num(br.bpm), state: strOrNull(br.state), confidence: conf(br.confidence), armed: yes(br.armed) },
      stillness: { stillS: num(still.still_s), limitS: num(still.limit_s), lastMovement: strOrNull(still.last_movement) },
      help: { pending: yes(hlp.pending), since: num(hlp.since), until: num(hlp.until), heldAction: strOrNull(hlp.held_action) },
      practice: yes(m.practice),
      fall: fallState(m.fall),
      lastGesture: careGesture(m.last_gesture),
      alerts: Array.isArray(m.alerts) ? m.alerts.map(careAlert).filter(Boolean) : [],
      control: {
        enabled: yes(ctl.enabled), preset: strOrNull(ctl.preset), presetLabel: strOrNull(ctl.preset_label), mapping: mapping,
        keys: {
          mode: strOrNull(keys.mode), trusted: boolOrNull(keys.trusted), backend: strOrNull(keys.backend), canPress: yes(keys.can_press),
          media: strOrNull(keys.media), note: strOrNull(keys.note), settingsPath: strOrNull(keys.settings_path)
        }
      },
      monitoring: {
        enabled: yes(mon.enabled),
        help: { enabled: yes(mh.enabled), windowS: num(mh.window_s), minGapS: num(mh.min_gap_s), confirmS: num(mh.confirm_s), pushTiming: strOrNull(mh.push_timing) },
        noMovement: { enabled: yes(mm.enabled), seconds: num(mm.seconds), breathingCounts: yes(mm.breathing_counts) },
        noBreathing: { enabled: yes(mb.enabled), seconds: num(mb.seconds) },
        fall: {
          enabled: yes(mf.enabled), sensitivity: strOrNull(mf.sensitivity), stillnessS: num(mf.stillness_s),
          burstLevel: num(mf.burst_level), fastLevel: num(mf.fast_level), oneSided: num(mf.one_sided),
          minS: num(mf.min_s), maxS: num(mf.max_s)
        }
      },
      room: strOrNull(m.room), onboarded: yes(m.onboarded), settingsRev: num(m.settings_rev), pages: num(m.pages),
      t: ts.t, tLocal: ts.tLocal
    };
  }

  var PARSERS = {
    gesture: parseGesture, presence: parsePresence, breathing: parseBreathing,
    status: parseStatus, spectrogram: parseSpectrogram, hello: parseHello, fall: parseFall,
    careHello: parseCareHello, careState: parseCareState, careAlert: parseCareAlert, careAction: parseCareAction,
    careTimeline: parseCareTimeline, careSettings: parseCareSettings
  };

  // ---- envelope handling ---------------------------------------------------

  function adaptObject(m, nowS, out, depth) {
    if (depth > 3) { out.push({ kind: 'unknown', raw: m, reason: 'nested too deep' }); return; }
    if (Array.isArray(m)) { for (var i = 0; i < m.length; i++) adaptObject(m[i], nowS, out, depth + 1); return; }
    if (!isObj(m)) { out.push({ kind: 'unknown', raw: m, reason: 'not an object' }); return; }

    if (Array.isArray(m.events)) { adaptObject(m.events, nowS, out, depth + 1); return; }   // {"events":[...]} batch

    var typeRaw = pick(m, ['type', 'event', 'kind', 'topic', 'msg_type']);
    var body = m;
    var inner = pick(m, ['data', 'payload', 'body']);
    if (typeRaw != null && isObj(inner) && key(typeRaw).indexOf('care_') !== 0) {   // {"type":"gesture","data":{...}}
      body = {};
      for (var k in inner) body[k] = inner[k];
      for (var k2 in m) if (!(k2 in body) && k2 !== 'data' && k2 !== 'payload' && k2 !== 'body') body[k2] = m[k2];
    }

    if (typeRaw == null) {
      // Snapshot form: {"gesture":{...}, "breathing":{...}, ...} without a type.
      var found = false;
      for (var name in m) {
        var kindName = TYPE_ALIASES[key(name)];
        if (kindName && isObj(m[name])) {
          var sub = {}; for (var s in m[name]) sub[s] = m[name][s];
          if (sub.t == null && m.t != null) sub.t = m.t;
          sub.type = kindName;
          adaptObject(sub, nowS, out, depth + 1);
          found = true;
        }
      }
      if (!found) out.push({ kind: 'unknown', raw: m, reason: 'no type field' });
      return;
    }

    var kind = TYPE_ALIASES[key(typeRaw)];
    if (!kind) { out.push({ kind: 'unknown', raw: m, reason: 'unknown type "' + typeRaw + '"' }); return; }
    out.push(PARSERS[kind](body, time(body, nowS)));
  }

  /**
   * adapt(raw[, nowSeconds]) -> Array of normalised events.
   * `raw` may be a JSON string, a parsed object, an array of messages or a batch envelope.
   */
  function adapt(raw, nowS) {
    if (nowS == null) nowS = Date.now() / 1000;
    var msg = raw;
    if (typeof raw === 'string') {
      try { msg = JSON.parse(raw); } catch (e) { return [{ kind: 'unknown', raw: raw, reason: 'invalid JSON' }]; }
    }
    var out = [];
    adaptObject(msg, nowS, out, 0);
    return out;
  }

  return {
    VERSION: VERSION,
    adapt: adapt,
    GESTURES: ['swipe_left', 'swipe_right', 'push', 'pull'],
    PRESENCE_STATES: ['empty', 'entering', 'present'],
    _aliases: { TYPE_ALIASES: TYPE_ALIASES, GESTURE_ALIASES: GESTURE_ALIASES, PRESENCE_ALIASES: PRESENCE_ALIASES }
  };
});
