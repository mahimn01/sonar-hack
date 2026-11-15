// Live check of the browser client code (connection.js + adapter.js + store.js) against the mock engine.
// Starts app/mock_engine.py on a free port, connects exactly as the pages do, then stops it.
// Run: node --test app/tests/     (needs Node 22+ for the built-in WebSocket, and websockets in the Python env)
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const net = require('node:net');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { loadWeb } = require('./helpers.js');

const APP = path.join(__dirname, '..');

function python() {
  if (process.env.PYTHON) return process.env.PYTHON;
  // The venv is in app/ on the old Windows setup and at the repo root on the Mac.
  for (const root of [APP, path.join(APP, '..')]) {
    const venv = process.platform === 'win32' ? path.join(root, '.venv', 'Scripts', 'python.exe') : path.join(root, '.venv', 'bin', 'python');
    if (fs.existsSync(venv)) return venv;
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

function freePort() {
  return new Promise((resolve) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => resolve(p)); }); });
}

async function waitForPort(port, ms = 10000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    const ok = await new Promise((resolve) => {
      const c = net.connect(port, '127.0.0.1', () => { c.end(); resolve(true); });
      c.on('error', () => resolve(false));
    });
    if (ok) return;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('mock engine did not start on port ' + port);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

test('browser client parses the live mock stream', { skip: typeof WebSocket === 'undefined' && 'needs Node 22+ (global WebSocket)', timeout: 30000 }, async () => {
  const port = await freePort();
  const mock = spawn(python(), [path.join(APP, 'mock_engine.py'), '--port', String(port), '--scenario', 'quiet', '--quiet', '--seed', '5'], { stdio: 'ignore' });
  try {
    await waitForPort(port);
    const sb = loadWeb(['adapter.js', 'connection.js', 'store.js'], { extra: { WebSocket } });
    const store = sb.SonarStore.start({
      wsUrl: 'ws://127.0.0.1:' + port, minConfidence: 0.6, cooldownMs: 700, returnStrokeMs: 1100,
      apneaSeconds: 10, breathMinConfidence: 0.4, motionHoldSeconds: 6, lowSnrDb: 15, waveformSeconds: 20
    });
    const unknown = [];
    store.on('raw', (r) => r.events.forEach((e) => { if (e.kind === 'unknown') unknown.push(e); }));

    await sleep(5500);                                      // the mock needs ~4 s of "breaths" before it reports a rate
    const S = store.state;
    assert.strictEqual(S.conn.health, 'live');
    assert.strictEqual(S.simulated, true, 'mock data must be flagged as simulated');
    assert.strictEqual(S.presence.state, 'present');
    assert.ok(Math.abs(S.breathing.bpm - 14) < 1.5, 'bpm ' + S.breathing.bpm);
    assert.ok(S.wave.t.length > 100, 'waveform samples ' + S.wave.t.length);   // a settled person: ~25 s at 10 Hz
    assert.ok(S.spectro.rows.length > 60 && S.spectro.rows[0].length === 64);
    assert.ok(Math.abs(S.status.snrDb - 55) < 5);
    assert.deepStrictEqual(unknown, []);

    store.connection.send({ cmd: 'gesture', label: 'swipe_right', confidence: 0.91 });
    await sleep(1200);
    assert.strictEqual(S.gestures[0].label, 'swipe_right');
    assert.strictEqual(S.gestures[0].accepted, true);
    store.connection.close();
  } finally {
    mock.kill();
  }
});
