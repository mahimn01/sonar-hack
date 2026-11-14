/*
 * config.js: runtime settings, read once from the page URL.
 *
 * Every value can be overridden with a query parameter, e.g.
 *   presenter.html?ws=ws://127.0.0.1:8765&minConf=0.7&cooldown=900&theme=dark
 */
(function (root) {
  'use strict';
  var q = new URLSearchParams(root.location ? root.location.search : '');

  function str(name, dflt) { var v = q.get(name); return v == null || v === '' ? dflt : v; }
  function num(name, dflt) { var v = Number(q.get(name)); return q.get(name) == null || !isFinite(v) ? dflt : v; }
  function flag(name, dflt) {
    var v = q.get(name);
    if (v == null) return dflt;
    return !(v === '0' || v === 'false' || v === 'off' || v === 'no');
  }

  // Default engine address: same host the page was served from (so a phone on the
  // LAN can view the monitor), port 8765. Pages opened from disk use 127.0.0.1.
  var host = root.location && root.location.hostname ? root.location.hostname : '127.0.0.1';
  if (host.indexOf(':') >= 0) host = '[' + host + ']';             // IPv6 literal

  var storedTheme = null;
  try { storedTheme = root.localStorage.getItem('sonar.theme'); } catch (e) { storedTheme = null; }

  root.SonarConfig = {
    wsUrl: str('ws', 'ws://' + host + ':8765'),

    // Gesture acceptance (applied on top of whatever the engine already does).
    minConfidence: num('minConf', 0.6),      // ignore gestures below this confidence
    cooldownMs: num('cooldown', 700),        // ignore any gesture this soon after an accepted one
    returnStrokeMs: num('returnMs', 1100),   // ignore the opposite swipe (hand coming back) for this long

    // Care alerts.
    apneaSeconds: num('apneaSec', 10),       // AASM apnea definition: breathing stops for >= 10 s
    breathMinConfidence: num('breathConf', 0.4),
    motionHoldSeconds: num('motionHold', 6), // no apnea alert this soon after a gesture or presence change
    breathMinSeconds: num('breathMin', 10),  // engine needs this much stillness for a first rate (sonar breath_min_s)
    lowSnrDb: num('lowSnr', 15),

    // Display.
    waveformSeconds: num('waveSec', 20),
    theme: str('theme', storedTheme || 'light'),
    room: str('room', 'Room 1 · Bedroom'),
    hud: flag('hud', true),
    debug: flag('debug', false),
    startSlide: num('slide', 1)
  };
})(typeof self !== 'undefined' ? self : this);
