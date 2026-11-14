/*
 * monitor.js: the Care Monitor dashboard.
 * Keys: T theme, F fullscreen, D raw event inspector.
 */
(function (root) {
  'use strict';
  var doc = root.document, UI = root.SonarUI, W = root.SonarWidgets, cfg = root.SonarConfig;

  UI.setTheme(cfg.theme);
  var store = root.SonarStore.start(cfg);
  root.sonarStore = store;

  doc.querySelectorAll('[data-icon]').forEach(function (el) { el.innerHTML = UI.icon(el.getAttribute('data-icon')); });
  doc.getElementById('room').textContent = cfg.room;
  doc.getElementById('apneaRule').textContent = cfg.apneaSeconds + ' s';

  W.conn(doc.getElementById('conn'), store);
  W.alertBanner(doc.getElementById('banner'), store);
  W.breathing(doc.getElementById('orb'), store);
  W.rateRange(doc.getElementById('range'), store);
  W.waveform(doc.getElementById('wave'), store, { seconds: cfg.waveformSeconds });
  W.presence(doc.getElementById('presence'), store);
  W.gestureFeed(doc.getElementById('feed'), store, { max: 3 });
  W.spectrogram(doc.getElementById('spectro'), store, { columns: 520 });
  W.signal(doc.getElementById('signal'), store);

  // Breathing status line, quality and alarm styling.
  var panel = doc.getElementById('breathPanel'), statusEl = doc.getElementById('breathStatus');
  var qualityEl = doc.getElementById('quality'), distanceEl = doc.getElementById('distance'), distanceFact = doc.getElementById('distanceFact');
  function breathText() {
    var S = store.state, B = S.breathing, now = Date.now(), p = S.presence.state;
    var valid = B.bpm != null && B.lastValidAt && now - B.lastValidAt < 6000;
    var zone = '';
    var text;
    if (S.alerts.apnea) text = 'No breathing movement';
    else if (S.alerts.offline) text = 'Sensor offline';
    else if (p === 'empty') text = 'No one in range';
    else if (valid) {
      var r = B.bpm;
      if (r > 24) { text = 'Very fast breathing'; zone = 'out'; }
      else if (r > 20) { text = 'Faster than typical'; zone = 'edge'; }
      else if (r < 10) { text = 'Very slow breathing'; zone = 'out'; }
      else if (r < 12) { text = 'Slower than typical'; zone = 'edge'; }
      else text = 'Breathing steadily';
    } else if (p === 'present' || p === 'entering') text = 'Measuring. Stay still for a few breaths';
    else text = S.conn.health === 'live' ? 'Waiting for a still person' : 'Waiting for sensor';
    if (statusEl.textContent !== text) statusEl.textContent = text;
    panel.classList.toggle('is-alarm', !!S.alerts.apnea);
    panel.setAttribute('data-zone', zone);
    qualityEl.textContent = B.confidence != null && (valid || S.alerts.apnea) ? UI.pct(B.confidence) : '—';
    if (B.rangeM != null) { distanceFact.hidden = false; distanceEl.textContent = B.rangeM.toFixed(1) + ' m'; }
  }
  setInterval(breathText, 300);
  store.on('alerts', breathText);
  breathText();

  // Activity log.
  var logEl = doc.getElementById('log');
  var ICON_FOR = { enter: 'enter', empty: 'empty', breath: 'breath', alert: 'alert', link: 'link', calibrate: 'calibrate', check: 'check', signal: 'signal' };
  function renderLog() {
    var L = store.state.log.slice(0, 6);
    if (!L.length) { logEl.innerHTML = '<li class="log__empty">Nothing yet. Events appear here as they happen.</li>'; return; }
    logEl.innerHTML = L.map(function (e) {
      return '<li data-level="' + e.level + '"><time>' + UI.clock(e.at) + '</time>' + UI.icon(ICON_FOR[e.icon] || 'check') + '<span>' + UI.escapeHtml(e.text) + '</span></li>';
    }).join('');
  }
  store.on('log', renderLog);
  renderLog();

  // Clock.
  var clockEl = doc.getElementById('clock');
  function tick() { clockEl.textContent = new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }); }
  tick(); setInterval(tick, 5000);

  function markSimulated() { doc.body.classList.toggle('is-simulated', store.state.simulated); }
  store.on('hello', markSimulated);
  store.on('status', markSimulated);

  var debug = UI.debugPanel(store, cfg.debug);
  doc.addEventListener('keydown', function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var k = e.key.toLowerCase();
    if (k === 't') UI.toggleTheme();
    else if (k === 'f') UI.toggleFullscreen();
    else if (k === 'd') debug.toggle();
    else return;
    e.preventDefault();
  });

  UI.keepAwake();
  UI.autoHideCursor();
})(typeof self !== 'undefined' ? self : this);
