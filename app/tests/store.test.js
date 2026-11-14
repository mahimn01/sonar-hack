// Care-alert and gesture-filter logic in web/js/store.js, driven with a fake clock.
// Run: node --test app/tests/        (read-only)
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { loadWeb } = require('./helpers.js');

const CFG = {
  minConfidence: 0.6, cooldownMs: 700, returnStrokeMs: 1100,
  apneaSeconds: 10, breathMinConfidence: 0.4, motionHoldSeconds: 6, lowSnrDb: 15, waveformSeconds: 20
};

// A fake engine: feeds breathing every 0.5 s with 20 Hz waveform chunks.
function rig() {
  const clock = { now: 1794600000000 };
  const sb = loadWeb(['adapter.js', 'store.js'], { clock });
  const store = sb.SonarStore.createStore(CFG);
  let phase = 0, amp = 1;
  const send = (msg) => store.ingestRaw(JSON.stringify({ t: clock.now / 1000, ...msg }));
  store.setSocket('open', {});
  function step(seconds, { bpm = 14, conf = 0.9 } = {}) {
    for (let k = 0; k < seconds * 2; k++) {
      const chunk = [];
      for (let i = 0; i < 10; i++) { phase += 2 * Math.PI * (14 / 60) / 20; chunk.push(amp * Math.sin(phase) + 0.01 * Math.cos(7 * phase)); }
      clock.now += 500;
      send({ type: 'breathing', bpm, confidence: conf, waveform: chunk, fs: 20 });   // estimator keeps saying 14, like a real lagging one
      send({ type: 'status', snr_db: 55, carrier_hz: 19000, fps: 30, calibrating: false });
      store.evaluate();
    }
  }
  return { clock, store, send, step, setAmp: (a) => { amp = a; } };
}

test('gesture filter: confidence, cooldown and return stroke', () => {
  const { clock, store, send } = rig();
  const judged = [];
  store.on('gesture', (g) => judged.push([g.label, g.accepted, g.reason]));
  send({ type: 'gesture', label: 'swipe_right', confidence: 0.4 });
  clock.now += 100; send({ type: 'gesture', label: 'swipe_right', confidence: 0.9 });
  clock.now += 300; send({ type: 'gesture', label: 'swipe_right', confidence: 0.9 });
  clock.now += 500; send({ type: 'gesture', label: 'swipe_left', confidence: 0.9 });
  clock.now += 400; send({ type: 'gesture', label: 'swipe_left', confidence: 0.9 });
  clock.now += 100; send({ type: 'gesture', label: 'none', confidence: 0.99 });
  assert.deepStrictEqual(judged, [
    ['swipe_right', false, 'low confidence'],
    ['swipe_right', true, null],
    ['swipe_right', false, 'too soon'],
    ['swipe_left', false, 'return stroke'],
    ['swipe_left', true, null]
  ]);
});

test('no-breathing alert: arms on steady breathing, fires after 10 s flat, clears on recovery', () => {
  const { store, send, step, setAmp } = rig();
  send({ type: 'presence', state: 'present' });
  step(14);
  assert.strictEqual(store.state.breathing.armed, true, 'should arm after steady breathing while still');
  assert.strictEqual(store.state.alerts.apnea, null);

  setAmp(0.02);                       // breath hold: chest stops moving, estimator still says 14
  step(6);
  assert.strictEqual(store.state.alerts.apnea, null, 'must not fire before 10 s');
  step(6);
  assert.ok(store.state.alerts.apnea, 'should fire by ~10-12 s of no breathing');
  assert.ok(store.state.log.some((l) => /No breathing movement/.test(l.text)));

  setAmp(1);
  step(4);
  assert.strictEqual(store.state.alerts.apnea, null, 'should clear when breathing resumes');
  assert.ok(store.state.log.some((l) => /Breathing resumed/.test(l.text)));
});

test('no-breathing alert stays quiet while someone is gesturing', () => {
  const { store, send, step, setAmp } = rig();
  send({ type: 'presence', state: 'present' });
  step(14);
  setAmp(0.02);
  for (let i = 0; i < 8; i++) { send({ type: 'gesture', label: 'push', confidence: 0.9 }); step(2); }
  assert.strictEqual(store.state.alerts.apnea, null, 'gestures mean motion, not apnea');
});

test('no alert for a person who never showed steady breathing', () => {
  const { store, send, step, setAmp } = rig();
  setAmp(0.02);
  send({ type: 'presence', state: 'present' });
  step(20, { conf: 0.2 });
  assert.strictEqual(store.state.alerts.apnea, null);
});

test('presence changes are logged and raise the entered notice', () => {
  const { store, send } = rig();
  send({ type: 'presence', state: 'empty' });
  send({ type: 'presence', state: 'person_walking_in' });
  assert.ok(store.state.alerts.entered);
  assert.strictEqual(store.state.presence.state, 'entering');
  assert.ok(store.state.log.some((l) => l.text === 'Person entered the room'));
});

test('calibration and weak signal surface as alerts; mock source is flagged', () => {
  const { store, send } = rig();
  send({ type: 'status', snr_db: 9, carrier_hz: 19000, calibrating: true, source: 'mock' });
  assert.strictEqual(store.state.alerts.calibrating, true);
  assert.strictEqual(store.state.alerts.lowSignal, true);
  assert.strictEqual(store.state.simulated, true);
});

test('waveform timeline merges full windows and chunks without duplicates', () => {
  const sb = loadWeb(['store.js']);
  const tl = new sb.SonarStore.Timeline(60);
  const fs = 20, t = 1000;
  tl.merge([1, 2, 3, 4], fs, t + 0.15);                    // samples at t, t+.05, t+.10, t+.15
  tl.merge([3, 4, 5, 6], fs, t + 0.25);                    // overlapping window
  assert.deepStrictEqual(Array.from(tl.v), [1, 2, 3, 4, 5, 6]);
  tl.merge([7], fs, t + 0.30);                              // incremental chunk
  assert.strictEqual(tl.v.length, 7);
  tl.merge([9], fs, t + 5);                                 // gap -> NaN break
  assert.ok(Number.isNaN(tl.v[tl.v.length - 2]));
  assert.ok(Math.abs(tl.valueAt(t + 0.075) - 2.5) < 1e-9);
});

test('no-breathing alert with engine-shaped data (1 Hz, whole window scaled to its peak)', () => {
  // Mirrors sonar/breathing.py: every second, up to 30 s of 10 Hz displacement divided by the window's peak.
  const clock = { now: 1794600000000 };
  const sb = loadWeb(['adapter.js', 'store.js'], { clock });
  const store = sb.SonarStore.createStore(CFG);
  store.setSocket('open', {});
  const fs = 10, hist = [];
  let phase = 0, amp = 1, bpm = 14, state = 'tracking';
  const send = (msg) => store.ingestRaw(JSON.stringify({ t: clock.now / 1000, ...msg }));
  function second() {
    for (let i = 0; i < fs; i++) { phase += 2 * Math.PI * (14 / 60) / fs; hist.push(amp * Math.sin(phase) + 0.004 * Math.sin(17 * phase)); }
    while (hist.length > 30 * fs) hist.shift();
    const peak = Math.max(...hist.map(Math.abs));
    clock.now += 1000;
    send({ type: 'breathing', bpm, confidence: bpm ? 0.8 : 0.2, state, fs, waveform: hist.map((v) => v / peak) });
    for (let k = 0; k < 4; k++) store.evaluate();
  }
  send({ type: 'presence', state: 'present' });
  for (let s = 0; s < 20; s++) second();
  assert.strictEqual(store.state.breathing.armed, true);

  amp = 0.01;                                   // breath hold; the estimator keeps reporting 14 for a while
  let firedAt = null;
  for (let s = 1; s <= 20 && firedAt == null; s++) {
    if (s === 9) { bpm = null; state = 'acquiring'; }
    second();
    if (store.state.alerts.apnea) firedAt = s;
  }
  assert.ok(firedAt != null && firedAt >= 9 && firedAt <= 14, 'fired after ' + firedAt + ' s');

  amp = 1; bpm = 14; state = 'tracking';
  for (let s = 0; s < 5; s++) second();
  assert.strictEqual(store.state.alerts.apnea, null, 'clears once breathing resumes');
});

test('engine-reported body motion suppresses the alert', () => {
  const clock = { now: 1794600000000 };
  const sb = loadWeb(['adapter.js', 'store.js'], { clock });
  const store = sb.SonarStore.createStore(CFG);
  store.setSocket('open', {});
  const send = (msg) => store.ingestRaw(JSON.stringify({ t: clock.now / 1000, ...msg }));
  send({ type: 'presence', state: 'present' });
  for (let s = 0; s < 20; s++) {
    clock.now += 1000;
    send({ type: 'breathing', bpm: 14, confidence: 0.8, state: 'tracking', fs: 10, waveform: Array.from({ length: 100 }, (_, i) => Math.sin(i / 3)) });
    store.evaluate();
  }
  assert.strictEqual(store.state.breathing.armed, true);
  for (let s = 0; s < 15; s++) {
    clock.now += 1000;
    send({ type: 'breathing', bpm: null, confidence: 0, state: 'moving', fs: 10 });
    store.evaluate();
  }
  assert.strictEqual(store.state.alerts.apnea, null);
});

test('a breath-hold longer than the engine window stays flagged and flat (amplitude_mm rescale)', () => {
  const clock = { now: 1794600000000 };
  const sb = loadWeb(['adapter.js', 'store.js'], { clock });
  const store = sb.SonarStore.createStore(CFG);
  store.setSocket('open', {});
  const fs = 10, hist = [];
  let phase = 0, amp = 1, bpm = 14, state = 'tracking', seed = 1;
  const noise = () => { seed = (seed * 16807) % 2147483647; return (seed / 2147483647 - 0.5) * 0.02; };
  const send = (msg) => store.ingestRaw(JSON.stringify({ t: clock.now / 1000, ...msg }));
  function second() {
    for (let i = 0; i < fs; i++) { phase += 2 * Math.PI * (14 / 60) / fs; hist.push(amp * Math.sin(phase) + noise()); }
    while (hist.length > 30 * fs) hist.shift();
    const s = hist.slice().sort((a, b) => a - b);
    const ppMm = 4 * (s[Math.round(0.97 * (s.length - 1))] - s[Math.round(0.03 * (s.length - 1))]);
    const peak = Math.max(...hist.map(Math.abs));
    clock.now += 1000;
    send({ type: 'breathing', bpm, confidence: bpm ? 0.8 : 0.1, state, fs, amplitude_mm: ppMm, waveform: hist.map((v) => v / peak) });
    for (let k = 0; k < 4; k++) store.evaluate();
  }
  send({ type: 'presence', state: 'present' });
  for (let s = 0; s < 25; s++) second();
  amp = 0;
  for (let s = 1; s <= 45; s++) { if (s === 9) { bpm = null; state = 'acquiring'; } second(); }
  // 45 s in: the engine's window holds nothing but noise, scaled to full height. Still an alarm, still flat.
  assert.ok(store.state.alerts.apnea, 'alert must hold through a long breath-hold');
  const W = store.state.wave, tEnd = W.t[W.t.length - 1];
  assert.ok(W.rms(tEnd - 5, tEnd) < 0.2, 'rescaled line stays flat, rms=' + W.rms(tEnd - 5, tEnd));
});
