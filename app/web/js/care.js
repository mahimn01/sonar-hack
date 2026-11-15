/*
 * care.js: the Sonar Care page shell.
 *
 * Starts the live store on the service's relay (engine events + care events on one stream), and owns
 * what every view shares: the top bar, the three views and their hash routes (#/, #/settings, #/setup),
 * the theme, and the alert layer (banners for "No movement" / "No breathing", the full-screen help
 * request with its cancel countdown and Acknowledge).
 *
 * Alerts shown here are the SERVICE's decisions (store.state.care.alerts). This page never decides one.
 * Keys: T theme, F full screen, D raw event inspector. With --mock: arrows, P, H, M, E rehearse events.
 */
(function (root) {
  'use strict';
  var doc = root.document, UI = root.SonarUI, W = root.SonarWidgets, cfg = root.SonarConfig, Api = root.SonarCareApi;
  var boot = root.SonarCareBoot || null;
  function $(id) { return doc.getElementById(id); }

  doc.querySelectorAll('[data-icon]').forEach(function (el) { el.innerHTML = UI.icon(el.getAttribute('data-icon')); });

  if (!boot) {                                             // opened without the service behind it
    $('noService').hidden = false;
    root.SonarCare = { offline: true };
    return;
  }

  // ---- theme: ?theme=, else the saved choice, else follow the system ----------------------------
  var query = new URLSearchParams(root.location.search);
  var themeParam = query.get('theme'), storedTheme = null;
  try { storedTheme = root.localStorage.getItem('sonar.theme'); } catch (e) { storedTheme = null; }
  var system = root.matchMedia ? root.matchMedia('(prefers-color-scheme: dark)') : null;
  var themeBtn = $('themeBtn');
  function applyTheme() {
    var t = themeParam || storedTheme || (system && system.matches ? 'dark' : 'light');
    UI.setTheme(t, { persist: false });
    themeBtn.innerHTML = UI.icon(UI.isDark() ? 'sun' : 'moon');
    themeBtn.setAttribute('aria-label', UI.isDark() ? 'Switch to the light theme' : 'Switch to the dark theme');
  }
  function toggleTheme() {
    themeParam = null;
    storedTheme = UI.isDark() ? 'light' : 'dark';
    try { root.localStorage.setItem('sonar.theme', storedTheme); } catch (e) { /* storage blocked: lasts until reload */ }
    applyTheme();
  }
  if (system && system.addEventListener) system.addEventListener('change', function () { if (!storedTheme && !themeParam) applyTheme(); });
  themeBtn.addEventListener('click', toggleTheme);
  applyTheme();

  // ---- live data ----------------------------------------------------------------------------------
  cfg.careAlerts = true;                                   // the service decides alerts; this page shows them
  cfg.wsUrl = query.get('ws') || boot.relay;
  var store = root.SonarStore.start(cfg);
  root.sonarStore = store;

  function careState() { return store.state.care.state; }
  function since(t) { return Math.max(0, Date.now() - store.careTimeMs(t)); }

  // ---- top bar --------------------------------------------------------------------------------------
  var connEl = W.h('<span class="conn" data-health="connecting" role="status"><span class="conn__dot"></span><span class="conn__text">Connecting</span></span>');
  $('conn').appendChild(connEl);
  var CONN_TEXT = { live: 'Sonar live', stalled: 'No data', offline: 'Sensor offline', connecting: 'Connecting' };
  function renderConn() {
    var h = store.state.conn.health, text = store.state.alerts.calibrating && h === 'live' ? 'Calibrating' : CONN_TEXT[h] || 'Connecting';
    connEl.setAttribute('data-health', h);
    if (connEl.lastChild.textContent !== text) connEl.lastChild.textContent = text;
  }
  store.on('conn', renderConn); store.on('alerts', renderConn); setInterval(renderConn, 1000);
  var clockEl = $('clock');
  function tickClock() { clockEl.textContent = new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }); }
  tickClock(); setInterval(tickClock, 5000);
  function renderBar() {
    var st = careState();
    doc.body.classList.toggle('is-simulated', !!store.state.simulated);
    if (!st) return;
    if ($('room').textContent !== (st.room || '')) $('room').textContent = st.room || '';
    $('dryTag').hidden = !st.dryRun;
  }
  store.on('care', renderBar); store.on('hello', renderBar); store.on('status', renderBar);

  // ---- views and routes -----------------------------------------------------------------------------
  var views = { dashboard: $('view-dashboard'), settings: $('view-settings'), setup: $('view-setup') };
  var titles = { dashboard: 'Sonar Care', settings: 'Settings · Sonar Care', setup: 'Setup · Sonar Care' };
  var hooks = {}, current = null, baseTitle = 'Sonar Care';
  function registerView(name, hook) { hooks[name] = hook; if (current === name && hook.show) hook.show(); }
  function route() {
    var name = (root.location.hash || '#/').replace(/^#\/?/, '').split(/[/?]/)[0] || 'dashboard';
    if (!views[name]) name = 'dashboard';
    Object.keys(views).forEach(function (k) { views[k].hidden = k !== name; });
    doc.querySelectorAll('.tabs a').forEach(function (a) {
      if (a.getAttribute('data-view') === name) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    baseTitle = titles[name];
    if (name !== current) {
      if (current && hooks[current] && hooks[current].hide) hooks[current].hide();
      current = name;
      if (hooks[name] && hooks[name].show) hooks[name].show();
    }
    renderTitle();
  }
  root.addEventListener('hashchange', route);

  // ---- toast ------------------------------------------------------------------------------------------
  var toastEl = $('toast'), toastTimer = null;
  function toast(text, tone) {
    toastEl.textContent = text;
    toastEl.setAttribute('data-tone', tone || '');
    toastEl.classList.add('is-on');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastEl.classList.remove('is-on'); }, tone === 'alert' ? 6000 : 3200);
  }
  function failed(what) { return function (err) { toast(what + ': ' + (err && err.message ? err.message : 'no answer'), 'alert'); }; }

  // ---- alert banners (everything except a help request that is counting down or waiting) ----------------
  var ALERT_COPY = {
    no_movement: function (a) { return 'Someone is present but nothing has moved. ' + (a.breathingDetected ? 'Breathing is still detected.' : 'Breathing is not confirmed.'); },
    no_breathing: function () { return 'Breathing was steady, then the breathing movement stopped. The person is present and still.'; },
    help: function () { return 'Two pushes in a row at the laptop.'; },
    fall: function (a) { return a.detail || 'A sudden movement, then no movement. Experimental: it can be wrong.'; }
  };
  var ALERT_ICON = { no_movement: 'still', no_breathing: 'breath', help: 'help', fall: 'fall' };
  var ALERT_TAG = { fall: 'Experimental fall alert' };       // a label next to the title: this one is a prototype rule
  var bannersEl = $('alerts'), banners = {};

  function bannerModel(a) {
    var lasted = UI.duration((a.resolvedAt != null ? a.resolvedAt - a.since : 0) * 1000);
    if (a.state === 'active') {
      return { tone: a.level === 'alert' ? 'alert' : 'attention', size: 'big', title: a.title, live: true,
        detail: (ALERT_COPY[a.kind] || function () { return a.detail || ''; })(a), button: 'Acknowledge' };
    }
    if (a.kind === 'help' || a.kind === 'fall') {           // acknowledged: stays a few seconds as a receipt
      return { tone: 'ok', size: 'small', title: (a.kind === 'help' ? 'Help request' : 'Possible fall') + ' acknowledged', live: false,
        detail: 'Answered ' + UI.duration((a.ackS || 0) * 1000) + ' after it went out.', button: null };
    }
    if (a.state === 'acknowledged') {
      return { tone: 'quiet', size: 'small', title: a.title, live: true,
        detail: 'Acknowledged. This clears by itself when ' + (a.kind === 'no_breathing' ? 'breathing' : 'movement') + ' returns.', button: null };
    }
    return { tone: 'ok', size: 'small', title: a.title + ': ' + (a.resolution || 'over'), live: false,
      detail: 'It lasted ' + lasted + ' and was not acknowledged.', button: 'Dismiss' };
  }

  function renderBanners() {
    var seen = {};
    store.state.care.alerts.forEach(function (a) {
      if (a.kind === 'help' && (a.state === 'pending' || a.state === 'active')) return;     // the help screen has it
      seen[a.id] = true;
      var m = bannerModel(a), b = banners[a.id];
      var key = m.tone + '|' + m.size + '|' + (m.button || '');
      if (!b || b.key !== key) {                            // (re)build only when the kind of banner changes
        if (b) b.el.remove();
        var el = W.h('<div class="abanner" role="' + (m.size === 'big' ? 'alert' : 'status') + '"><span class="abanner__icon">' + UI.icon(ALERT_ICON[a.kind] || 'alert') +
          '</span><div><div class="abanner__title"></div><div class="abanner__detail"></div></div></div>');
        if (m.button) {
          var btn = W.h('<button class="btn' + (m.button === 'Dismiss' ? ' btn--quiet' : '') + '" type="button"></button>');
          btn.textContent = m.button;
          btn.addEventListener('click', function () { btn.disabled = true; Api.post('/api/ack', { id: a.id }).catch(failed('Could not acknowledge')).then(function () { btn.disabled = false; }); });
          el.appendChild(btn);
        }
        b = banners[a.id] = { el: el, key: key, title: el.querySelector('.abanner__title'), detail: el.querySelector('.abanner__detail') };
        bannersEl.appendChild(el);
      }
      b.el.setAttribute('data-tone', m.tone);
      b.el.setAttribute('data-size', m.size);
      b.el.setAttribute('data-level', a.level || '');
      var title = UI.escapeHtml(m.title) + (m.live ? ' <span>· ' + UI.duration(since(a.since)) + '</span>' : '') +
        (ALERT_TAG[a.kind] && m.size === 'big' ? ' <em class="abanner__tag">' + ALERT_TAG[a.kind] + '</em>' : '');
      if (b.title.innerHTML !== title) b.title.innerHTML = title;
      if (b.detail.textContent !== m.detail) b.detail.textContent = m.detail;
    });
    Object.keys(banners).forEach(function (id) { if (!seen[id]) { banners[id].el.remove(); delete banners[id]; } });
  }

  // ---- the help request: the whole screen ---------------------------------------------------------------
  var helpEl = $('helpScreen'), helpKey = null, helpFocusBack = null;
  var RING = 2 * Math.PI * 46;
  function helpAlert() {
    var list = store.state.care.alerts;
    for (var i = 0; i < list.length; i++) if (list[i].kind === 'help' && (list[i].state === 'pending' || list[i].state === 'active')) return list[i];
    return null;
  }
  function helpButton(label, cls, onClick) {
    var b = W.h('<button class="btn btn--big ' + cls + '" type="button"></button>');
    b.textContent = label;
    b.addEventListener('click', function () { b.disabled = true; onClick().then(function () { b.disabled = false; }); });
    return b;
  }
  function renderHelp() {
    var a = helpAlert();
    if (!a) {
      if (helpKey) {
        helpKey = null;
        helpEl.hidden = true;
        if (helpFocusBack && helpFocusBack.focus) helpFocusBack.focus();
        helpFocusBack = null;
      }
      return;
    }
    var key = a.id + ':' + a.state, st = careState();
    if (key !== helpKey) {
      if (!helpKey) helpFocusBack = doc.activeElement;
      helpKey = key;
      helpEl.setAttribute('data-state', a.state);
      var buttons = $('helpButtons');
      buttons.innerHTML = '';
      if (a.state === 'pending') {
        $('helpMark').innerHTML = '<svg class="ring" viewBox="0 0 100 100"><circle class="ring__track" cx="50" cy="50" r="46"/><circle class="ring__left" id="helpRing" cx="50" cy="50" r="46" stroke-dasharray="' + RING.toFixed(1) + '"/></svg><span class="helpscreen__count" id="helpCount"></span>';
        $('helpTitle').textContent = 'Help gesture seen';
        $('helpDetail').textContent = 'Two pushes in a row at the laptop. The alert goes out in a moment.';
        buttons.appendChild(helpButton('Cancel', 'btn--quiet', function () { return Api.post('/api/help/cancel').catch(failed('Could not cancel')); }));
        buttons.appendChild(helpButton('Send now', '', function () { return Api.post('/api/help/send').catch(failed('Could not send')); }));
        $('helpFoot').textContent = 'Pushed by accident? Cancel here. A hand gesture can never cancel a help request, and another push sends it at once.';
        helpEl.hidden = false;
        helpEl.focus();                                      // on purpose not on a button: a stray Enter must not cancel a real call
      } else {
        $('helpMark').innerHTML = UI.icon('help');
        $('helpTitle').textContent = 'Help requested';
        var ack = helpButton('Acknowledge', '', function () { return Api.post('/api/ack', { id: a.id }).catch(failed('Could not acknowledge')); });
        buttons.appendChild(ack);
        $('helpFoot').textContent = 'Acknowledging stops the reminders and records how long it took.';
        helpEl.hidden = false;
        ack.focus();
      }
    }
    if (a.state === 'pending') {
      var left = Math.max(0, store.careTimeMs(a.escalateAt) - Date.now()), total = Math.max(1, (a.escalateAt - a.raisedAt) * 1000);
      var ring = $('helpRing'), count = $('helpCount');
      if (ring) ring.setAttribute('stroke-dashoffset', (RING * (1 - Math.min(1, left / total))).toFixed(1));
      if (count) count.textContent = String(Math.ceil(left / 1000));
      $('helpTime').textContent = 'Alert goes out in ' + Math.ceil(left / 1000) + ' s';
    } else {
      var room = st && st.room ? st.room + ' · ' : '';
      $('helpDetail').textContent = room + (a.count > 1 ? 'asked ' + a.count + ' times' : 'two pushes at the laptop');
      $('helpTime').textContent = 'Waiting ' + UI.duration(since(a.activeAt != null ? a.activeAt : a.raisedAt)) + ' · since ' + UI.clock(store.careTimeMs(a.since));
    }
  }
  helpEl.setAttribute('tabindex', '-1');
  helpEl.addEventListener('keydown', function (e) {          // keep Tab inside the dialog while it is open
    if (e.key !== 'Tab') return;
    var f = helpEl.querySelectorAll('button:not([disabled])');
    if (!f.length) { e.preventDefault(); return; }
    var first = f[0], last = f[f.length - 1];
    if (e.shiftKey && (doc.activeElement === first || doc.activeElement === helpEl)) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && doc.activeElement === last) { e.preventDefault(); first.focus(); }
  });

  function renderTitle() {
    var loud = store.state.care.alerts.filter(function (a) { return a.state === 'active' || a.state === 'pending'; })[0];
    var t = loud ? (loud.level === 'alert' ? '▲ ' : '⚠ ') + loud.title + ' · Sonar Care' : baseTitle;
    if (doc.title !== t) doc.title = t;
  }
  function renderAlerts() { renderBanners(); renderHelp(); renderTitle(); }
  store.on('care', renderAlerts);
  store.on('careAlert', renderAlerts);
  setInterval(renderAlerts, 250);

  // ---- keys -----------------------------------------------------------------------------------------------
  var debug = UI.debugPanel(store, cfg.debug);
  var REHEARSE = {                                          // only with --mock: fire scripted events from the keyboard
    ArrowRight: { cmd: 'gesture', label: 'swipe_right' }, ArrowLeft: { cmd: 'gesture', label: 'swipe_left' },
    p: { cmd: 'gesture', label: 'push' }, h: { cmd: 'double_push' }, m: { cmd: 'moving', seconds: 2 },
    g: { cmd: 'fall', direction: 'toward', still_s: 30 }    // the experimental fall alert (switch it on in Settings first)
  };
  doc.addEventListener('keydown', function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || tag === 'BUTTON' || (e.target && e.target.isContentEditable)) return;
    var k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (k === 't') toggleTheme();
    else if (k === 'f') UI.toggleFullscreen();
    else if (k === 'd') debug.toggle();
    else if (boot.mode === 'mock' && current === 'dashboard' && REHEARSE[k]) store.connection.send(REHEARSE[k]);
    else if (boot.mode === 'mock' && current === 'dashboard' && k === 'e') {
      store.connection.send({ cmd: 'presence', state: store.state.presence.state === 'empty' ? 'entering' : 'empty' });
    } else return;
    e.preventDefault();
  });

  UI.keepAwake();

  root.SonarCare = {
    store: store, api: Api, boot: boot, toast: toast, failed: failed, registerView: registerView,
    state: careState, since: since, go: function (hash) { root.location.hash = hash; }
  };
  route();
})(typeof self !== 'undefined' ? self : this);
