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
 *   {kind:'spectrogram', rows, binsHz, axisLabel, autoscaled, t, tLocal}
 *   {kind:'hello',       source, version, schema, labels, classifier, t, tLocal}
 *   {kind:'unknown',     raw, reason}
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
    var car = scalarOf(pick(m, ['carrier_hz', 'carrier', 'f0', 'f0_hz', 'tone_hz', 'tones_hz', 'freq_hz', 'carriers_hz']));
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
      t: ts.t, tLocal: ts.tLocal
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

  var PARSERS = {
    gesture: parseGesture, presence: parsePresence, breathing: parseBreathing,
    status: parseStatus, spectrogram: parseSpectrogram, hello: parseHello
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
    if (typeRaw != null && isObj(inner)) {                         // {"type":"gesture","data":{...}}
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
