// Checks web/js/adapter.js against the same fixtures as test_adapter.py.
// Run: node --test app/tests/        (Node 18+; read-only)
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const adapter = require('../web/js/adapter.js');

const FIX = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'engine_events.json'), 'utf8'));

function close(a, b) { return Math.abs(a - b) <= Math.max(1e-6, Math.abs(b) * 1e-9); }

function check(evt, expect) {
  const problems = [];
  for (const [key, want] of Object.entries(expect)) {
    let got;
    if (key.endsWith('_len')) got = (evt[key.slice(0, -4)] || []).length;
    else if (key === 'rows_shape') got = [evt.rows.length, evt.rows[0].length];
    else if (key === 'rows_in_01') got = evt.rows.every((r) => r.every((v) => v >= 0 && v <= 1));
    else got = key in evt ? evt[key] : '<missing>';
    const exact = (want !== null && typeof want === 'object' && !Array.isArray(want)) || (Array.isArray(want) && want.some((w) => typeof w === 'string'));
    const ok = exact
      ? JSON.stringify(got) === JSON.stringify(want)        // a nested object or a list of names: compared exactly
      : Array.isArray(want)
        ? Array.isArray(got) && got.length === want.length && got.every((g, i) => close(g, want[i]))
        : typeof want === 'number' && typeof got === 'number' ? close(got, want) : got === want;
    if (!ok) problems.push(`${key}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
  }
  return problems;
}

for (const c of FIX.cases) {
  test(`adapter.js: ${c.name}`, () => {
    const raw = 'raw_text' in c ? c.raw_text : c.raw;
    const events = adapter.adapt(raw, FIX.now);
    assert.strictEqual(events.length, c.expect.length, JSON.stringify(events));
    events.forEach((e, i) => assert.deepStrictEqual(check(e, c.expect[i]), [], JSON.stringify(e)));
  });
}
