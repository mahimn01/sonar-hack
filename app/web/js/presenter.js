/*
 * presenter.js: slide deck driven by sonar gestures, with keyboard fallback.
 *
 *   swipe_right -> next slide       swipe_left -> previous slide
 *   push        -> live panel on/off
 *
 * Gestures pass the store's filters first (confidence, cooldown, return stroke).
 * If Sonar Care is also pressing keys for the same gestures, the key that follows a gesture this
 * page already acted on is dropped, so one movement never moves two slides.
 */
(function (root) {
  'use strict';
  var doc = root.document, UI = root.SonarUI, W = root.SonarWidgets, cfg = root.SonarConfig;

  var ACTIONS = { swipe_right: 'next', swipe_left: 'prev', push: 'overlay' };
  var TILE_TEXT = { swipe_left: 'Previous slide', swipe_right: 'Next slide', push: 'Live panel on or off' };

  UI.setTheme(cfg.theme);
  var store = root.SonarStore.start(cfg);
  root.sonarStore = store;                                       // handy in DevTools and for tests

  // Fill <span data-icon="name"> placeholders with inline SVG.
  doc.querySelectorAll('[data-icon]').forEach(function (el) { el.innerHTML = UI.icon(el.getAttribute('data-icon')); });

  var slides = Array.prototype.slice.call(doc.querySelectorAll('.slide'));
  var overlay = doc.getElementById('overlay');
  var hud = doc.getElementById('hud');
  var counter = doc.getElementById('counter');
  var progress = doc.getElementById('progress');
  var announce = doc.getElementById('announce');
  var help = doc.getElementById('help');
  var current = 0, navLocked = false, lastGestureNav = { action: null, at: 0 }, lastPushAt = 0;

  // ---- live widgets ---------------------------------------------------------------
  W.conn(doc.getElementById('hudConn'), store);
  W.conn(doc.getElementById('overlayConn'), store);
  W.breathing(doc.getElementById('liveOrb'), store);
  W.presence(doc.getElementById('livePresence'), store);
  W.rateRange(doc.getElementById('liveRange'), store);
  W.waveform(doc.getElementById('liveWave'), store, { seconds: 16 });
  W.gestureTiles(doc.getElementById('liveTiles'), store, TILE_TEXT);
  W.spectrogram(doc.getElementById('liveSpectro'), store);
  W.spectrogram(doc.getElementById('howSpectro'), store, { columns: 480 });
  W.signal(doc.getElementById('techSignal'), store);
  W.breathing(doc.getElementById('overlayOrb'), store);
  W.presence(doc.getElementById('overlayPresence'), store, { art: false });
  W.waveform(doc.getElementById('overlayWave'), store, { seconds: 12, labels: false, lineWidth: 2.5 });
  W.spectrogram(doc.getElementById('overlaySpectro'), store, { columns: 240, labels: false });

  // ---- navigation ------------------------------------------------------------------
  function go(index, dir) {
    index = Math.max(0, Math.min(slides.length - 1, index));
    if (index === current && slides[index].classList.contains('is-active')) return false;
    var from = slides[current], to = slides[index];
    if (dir == null) dir = index > current ? 1 : -1;
    if (from !== to) {
      from.classList.remove('is-active', 'from-left', 'from-right');
      from.classList.add(dir > 0 ? 'to-right' : 'to-left');
    }
    // Place the incoming slide on the side the hand came from, then animate it in.
    to.classList.add('no-anim');
    to.classList.remove('to-left', 'to-right', 'from-left', 'from-right');
    to.classList.add(dir > 0 ? 'from-left' : 'from-right');
    void to.offsetWidth;
    to.classList.remove('no-anim');
    to.classList.add('is-active');
    to.classList.remove('from-left', 'from-right');
    current = index;
    counter.textContent = (index + 1) + ' / ' + slides.length;
    progress.style.transform = 'scaleX(' + ((index + 1) / slides.length).toFixed(4) + ')';
    announce.textContent = 'Slide ' + (index + 1) + ': ' + (to.getAttribute('data-title') || '');
    try { history.replaceState(null, '', '#' + (index + 1)); } catch (e) { /* file:// may refuse */ }
    return true;
  }
  function next() { return go(current + 1, 1); }
  function prev() { return go(current - 1, -1); }
  var deck = doc.getElementById('deck');
  function setOverlay(open) {
    var on = open == null ? !overlay.classList.contains('is-open') : open;
    overlay.classList.toggle('is-open', on);
    deck.classList.toggle('overlay-open', on);
  }

  function run(action) {
    if (action === 'next') return next();
    if (action === 'prev') return prev();
    if (action === 'overlay') { setOverlay(); return true; }
    return false;
  }

  // ---- gestures -> HUD + actions ------------------------------------------------------
  var hudIcon = hud.querySelector('.hud__icon'), hudLabel = hud.querySelector('.hud__label'), hudConf = hud.querySelector('.hud__conf');
  var hudTimer = null;
  store.on('gesture', function (g) {
    hudIcon.innerHTML = UI.icon(g.label === 'unknown' ? 'signal' : g.label);
    var ic = hudIcon.firstElementChild;
    var shift = g.label === 'swipe_left' ? -0.5 : g.label === 'swipe_right' ? 0.5 : 0;
    if (shift && !UI.reducedMotion()) {
      ic.style.transform = 'translateX(' + (-shift) + 'em)';
      requestAnimationFrame(function () { requestAnimationFrame(function () { ic.style.transform = 'none'; }); });
    }
    hudLabel.textContent = UI.GESTURE_TEXT[g.label] || g.label;
    var action = ACTIONS[g.label];
    if (g.accepted) {
      hudConf.textContent = UI.pct(g.confidence);
      if (action && navLocked) hudConf.textContent += ' · paused';
    } else {
      hudConf.textContent = g.reason === 'low confidence' ? UI.pct(g.confidence) + ' · too unsure' : 'ignored';
    }
    hud.classList.remove('is-flash', 'is-flash-miss');
    void hud.offsetWidth;
    hud.classList.add(g.accepted ? 'is-flash' : 'is-flash-miss');
    clearTimeout(hudTimer);
    hud.classList.remove('is-stale');
    hudTimer = setTimeout(function () {
      hud.classList.remove('is-flash', 'is-flash-miss');
      hudTimer = setTimeout(function () { hud.classList.add('is-stale'); }, 2000);
    }, 1600);

    if (g.label === 'push') lastPushAt = Date.now();
    if (!g.accepted || !action || navLocked) return;
    if (run(action)) lastGestureNav = { action: action, at: Date.now() };
  });

  // ---- keyboard fallback ---------------------------------------------------------------
  var KEYS = {
    ArrowRight: 'next', ArrowDown: 'next', PageDown: 'next', ' ': 'next', Enter: 'next', n: 'next',
    ArrowLeft: 'prev', ArrowUp: 'prev', PageUp: 'prev', Backspace: 'prev', p: 'prev'
  };
  var debug = UI.debugPanel(store, cfg.debug);
  doc.addEventListener('keydown', function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (k === 'Escape') { help.classList.remove('is-open'); return; }
    var action = KEYS[e.key] || KEYS[k];
    if (action) {
      e.preventDefault();
      // If Sonar Care (or keys_bridge.py) is also running, the same gesture arrives a second time as a key press. Drop it.
      if (lastGestureNav.action === action && Date.now() - lastGestureNav.at < 450) return;
      // Sonar Care presses Space for a push about 2.5 s later (it waits in case a second push follows).
      if (e.key === ' ' && Date.now() - lastPushAt < 4500) return;
      run(action);
      return;
    }
    if (/^[0-9]$/.test(k)) { go(k === '0' ? 9 : Number(k) - 1); return; }
    switch (k) {
      case 'o': setOverlay(); break;
      case 'f': UI.toggleFullscreen(); break;
      case 't': UI.toggleTheme(); break;
      case 'h': hud.classList.toggle('is-hidden'); break;
      case 'l': navLocked = !navLocked; doc.body.classList.toggle('nav-locked', navLocked); break;
      case 'd': debug.toggle(); break;
      case 'm': root.open('monitor.html' + location.search, '_blank'); break;
      case 'c': root.open('care.html', '_blank'); break;
      case '?': case '/': help.classList.toggle('is-open'); break;
      default: return;
    }
    e.preventDefault();
  });

  store.on('hello', function () { doc.body.classList.toggle('is-simulated', store.state.simulated); });
  store.on('status', function () { doc.body.classList.toggle('is-simulated', store.state.simulated); });
  if (!cfg.hud) hud.classList.add('is-hidden');

  UI.keepAwake();
  UI.autoHideCursor();

  var start = parseInt((location.hash || '').replace('#', ''), 10) || cfg.startSlide || 1;
  current = Math.max(0, Math.min(slides.length - 1, start - 1));
  go(current, 1);

  // Exposed for the end-to-end check and for rehearsal from DevTools.
  root.sonarPresenter = {
    get index() { return current; }, get overlayOpen() { return overlay.classList.contains('is-open'); },
    go: go, next: next, prev: prev, setOverlay: setOverlay
  };
})(typeof self !== 'undefined' ? self : this);
