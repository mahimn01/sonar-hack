// Shared helpers for the Node tests: load the browser scripts into a sandbox with a controllable clock.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const WEB = path.join(__dirname, '..', 'web', 'js');

// Load web scripts (classic browser scripts) into one sandbox, the way the page does.
// `clock` lets a test move time forward by hand; pass null to use the real clock.
function loadWeb(files, { clock = null, extra = {} } = {}) {
  const RealDate = Date;
  const FakeDate = clock
    ? class extends RealDate { constructor(...a) { super(...(a.length ? a : [clock.now])); } static now() { return clock.now; } }
    : RealDate;
  // Real timers are unref'd so the page's forever-running intervals don't keep the test process alive.
  const unref = (t) => { if (t && t.unref) t.unref(); return t; };
  const sandbox = {
    console, Date: FakeDate, Math, JSON, URLSearchParams,
    setTimeout: (fn, ms) => unref(setTimeout(fn, ms)), clearTimeout,
    setInterval: clock ? () => 0 : (fn, ms) => unref(setInterval(fn, ms)),   // fake clock: tests call evaluate() themselves
    clearInterval,
    ...extra,
  };
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  for (const f of files) vm.runInContext(fs.readFileSync(path.join(WEB, f), 'utf8'), sandbox, { filename: f });
  return sandbox;
}

module.exports = { loadWeb, WEB };
