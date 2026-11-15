/*
 * care-settings.js: the Settings view of Sonar Care.
 *
 * Every control saves the moment it changes (POST /api/settings with just that value). The service
 * checks it, stores it and pushes the new settings to every open page, so the menu bar, this page
 * and a second browser window always agree. A refused value comes back with the reason and the
 * control returns to what the service has.
 *
 * Sections: touch-free control (preset, key per gesture, permission), room monitor (no-movement and
 * no-breathing alerts), the help gesture, how alerts are announced, and this device (room name, engine).
 */
(function (root) {
  'use strict';
  var Care = root.SonarCare;
  if (!Care || Care.offline) return;
  var doc = root.document, UI = root.SonarUI, W = root.SonarWidgets, store = Care.store, Api = Care.api;
  var view = doc.getElementById('view-settings');
  var GESTURES = [['swipe_left', 'Swipe left'], ['swipe_right', 'Swipe right'], ['push', 'Push']];
  var meta = null, built = false;

  function h(html) { return W.h(html); }
  function esc(s) { return UI.escapeHtml(s == null ? '' : String(s)); }
  function settings() { return store.state.care.settings; }
  function get(obj, path) { return path.split('.').reduce(function (o, k) { return o == null ? undefined : o[k]; }, obj); }
  function patchOf(path, value) {
    var out = {}, cur = out, keys = path.split('.');
    keys.forEach(function (k, i) { cur[k] = i === keys.length - 1 ? value : {}; cur = cur[k]; });
    return out;
  }
  function save(patch, okText) {
    return Api.post('/api/settings', patch).then(function () { if (okText) Care.toast(okText); }, function (err) {
      Care.toast(err.message || 'Could not save', 'alert');
      sync();                                               // put the control back to what the service has
    });
  }

  // ---- small builders ------------------------------------------------------------------------------
  function toggle(path, label, hint) {
    return '<label class="switch"><input type="checkbox" data-path="' + path + '"><span class="switch__track" aria-hidden="true"></span>' +
      '<span class="switch__text"><b>' + esc(label) + '</b>' + (hint ? '<small>' + esc(hint) + '</small>' : '') + '</span></label>';
  }
  function number(path, label, unit, hint, step) {
    var lim = (meta.limits[path] || [0, 9999]);
    return '<label class="field"><span class="field__label"><b>' + esc(label) + '</b>' + (hint ? '<small>' + esc(hint) + '</small>' : '') + '</span>' +
      '<span class="field__input"><input type="number" inputmode="decimal" data-path="' + path + '" min="' + lim[0] + '" max="' + lim[1] + '" step="' + (step || 1) + '">' +
      '<span class="field__unit">' + esc(unit) + '</span></span></label>';
  }
  function slider(path, label, hint, step, percent) {
    var lim = meta.limits[path];
    return '<label class="field field--slider"><span class="field__label"><b>' + esc(label) + '</b>' + (hint ? '<small>' + esc(hint) + '</small>' : '') + '</span>' +
      '<span class="field__input"><input type="range" data-path="' + path + '" min="' + lim[0] + '" max="' + lim[1] + '" step="' + step + '"' + (percent ? ' data-percent="1"' : '') + '>' +
      '<output class="field__unit num"></output></span></label>';
  }
  function select(path, label, hint, options) {
    return '<label class="field"><span class="field__label"><b>' + esc(label) + '</b>' + (hint ? '<small>' + esc(hint) + '</small>' : '') + '</span>' +
      '<span class="field__input"><select data-path="' + path + '">' + options.map(function (o) { return '<option value="' + o[0] + '">' + esc(o[1]) + '</option>'; }).join('') + '</select></span></label>';
  }
  function section(id, icon, title, lede, body) {
    return '<section class="sect" id="set-' + id + '" aria-labelledby="set-' + id + '-t"><header class="sect__head"><span class="sect__icon">' + UI.icon(icon) + '</span>' +
      '<div><h2 id="set-' + id + '-t">' + esc(title) + '</h2><p>' + lede + '</p></div></header><div class="sect__body">' + body + '</div></section>';
  }

  // ---- build once ------------------------------------------------------------------------------------
  function build() {
    var actionOptions = function () {
      var groups = { keys: 'Keys', media: 'Media', none: '' }, html = '';
      ['none', 'keys', 'media'].forEach(function (g) {
        var items = meta.actions.filter(function (a) { return a.group === g; });
        var opts = items.map(function (a) { return '<option value="' + a.id + '">' + esc(a.label) + '</option>'; }).join('');
        html += g === 'none' ? opts : '<optgroup label="' + groups[g] + '">' + opts + '</optgroup>';
      });
      return html;
    };
    var presets = meta.presets.map(function (p) {
      return '<label class="choice"><input type="radio" name="preset" value="' + p.id + '"><span class="choice__box"><b>' + esc(p.label) + '</b><small>' + esc(p.hint) + '</small></span></label>';
    }).join('');
    var mapRows = GESTURES.map(function (g) {
      return '<div class="maprow" data-g="' + g[0] + '"><span class="maprow__icon">' + UI.icon(g[0]) + '</span><b>' + g[1] + '</b>' +
        '<span class="maprow__arrow" aria-hidden="true">' + UI.icon('arrow') + '</span>' +
        '<select aria-label="' + g[1] + ' does" data-map="' + g[0] + '">' + actionOptions() + '</select>' +
        '<button class="btn btn--quiet btn--small" type="button" data-test="' + g[0] + '" title="Presses this key 3 seconds from now, so you can click into another app first">Try in 3 s</button></div>';
    }).join('');
    var stillPresets = meta.no_movement_presets.map(function (p) {
      return '<button class="btn btn--quiet btn--small" type="button" data-still="' + p.id + '">' + esc(p.label) + ' · ' + (p.seconds < 60 ? p.seconds + ' s' : p.seconds / 60 + ' min') + '</button>';
    }).join('');

    view.innerHTML = '<div class="page__inner">' +
      '<header class="page__head"><h1 class="page__title">Settings</h1><p class="page__lede">Changes are saved and take effect at once.</p></header>' +
      '<div class="sects">' +

      section('control', 'keyboard', 'Touch-free control', 'A gesture presses a real key in whatever app is in front.',
        toggle('control.enabled', 'Gestures press keys', 'Off: gestures are still recognised and shown, but nothing is pressed.') +
        '<fieldset class="choices"><legend>Gestures control</legend>' + presets + '</fieldset>' +
        '<div class="maprows" role="group" aria-label="Key for each gesture">' + mapRows + '</div>' +
        '<div class="note" id="keysBox"></div>' +
        '<details class="more"><summary>Fine tuning</summary>' +
        slider('control.min_confidence', 'How sure the model must be', 'Higher: fewer accidental presses, more missed gestures.', 0.01, true) +
        number('control.cooldown_s', 'Pause after a gesture', 's', 'Nothing else acts for this long.', 0.1) +
        number('control.repeat_s', 'Same gesture again', 's', 'Sooner than this is the same movement seen twice.', 0.1) +
        number('control.return_stroke_s', 'Hand coming back', 's', 'The opposite swipe is ignored for this long.', 0.1) +
        '</details>') +

      section('monitor', 'person', 'Room monitor', 'Caregiver alerts when someone is present.',
        toggle('monitoring.enabled', 'Monitoring', 'Off: no alerts and no help gesture.') +
        '<div class="group">' +
        toggle('monitoring.no_movement.enabled', '“No movement” alert', 'Someone is present and nothing has moved for the time below.') +
        number('monitoring.no_movement.seconds', 'Alert after', 's', 'How long without movement.', 1) +
        '<div class="quick" role="group" aria-label="Quick choices">' + stillPresets + '</div>' +
        toggle('monitoring.no_movement.breathing_counts', 'Detected breathing counts as movement',
          'On: a still person whose breathing is being tracked raises no alert. Off: any stillness does. Use Off to show the alert in a demo, because breathing detection is experimental.') +
        '</div><div class="group">' +
        toggle('monitoring.no_breathing.enabled', '“No breathing movement” alert', 'A steady breathing rate was seen, then the movement stopped. Experimental on one microphone.') +
        number('monitoring.no_breathing.seconds', 'Alert after', 's', '10 s is the clinical length of an apnea.', 1) +
        '</div>') +

      section('fall', 'fall', 'Fall alert (experimental)', 'One sudden one-way movement, then stillness, raises “Possible fall”. A prototype rule: it can be wrong.',
        toggle('monitoring.fall.enabled', 'Experimental fall alert', 'Off unless you switch it on. Needs Monitoring to be on.') +
        select('monitoring.fall.sensitivity', 'Sensitivity', 'How big the sudden movement has to be.', [['low', 'Low: only big movements'], ['medium', 'Medium'], ['high', 'High: smaller movements count']]) +
        number('monitoring.fall.stillness_s', 'Stillness before the alert', 's', 'Nothing may move for this long after the sudden movement. Shorter is quicker but fires more often when someone just steps away.', 1) +
        '<div class="note" id="fallBox" aria-live="polite"></div>' +
        '<details class="more"><summary>Thresholds for tuning</summary>' +
        number('monitoring.fall.burst_level', 'Size needed', '', '0 to 1: how much of the Doppler strip the movement lights up. Sensitivity scales this and the next one.', 0.01) +
        number('monitoring.fall.fast_level', 'Fast part needed', '', '0 to 1: how much of it is faster than about 0.9 m/s.', 0.01) +
        number('monitoring.fall.one_sided', 'One-way score needed', '', '0 = as much toward as away, 1 = all one way. Gestures, sitting down and walking past go both ways.', 0.05) +
        number('monitoring.fall.min_s', 'Shortest movement', 's', 'A dropped pillow is quick: lower this if the box says “too short”.', 0.1) +
        number('monitoring.fall.max_s', 'Longest movement', 's', 'Longer than this is walking or tidying, not a fall.', 0.1) +
        '<p class="fine">Try it with something safe: drop a pillow or a bag about a metre in front of the laptop, step back and keep still. The box above says what the last big movement looked like and why it did or did not count. The guide is in docs/SONAR_CARE.md.</p>' +
        '</details>') +

      section('help', 'help', 'Help gesture', 'Two pushes in a row raise “Help requested”.',
        toggle('monitoring.help.enabled', 'Two pushes call for help', 'Needs Monitoring to be on.') +
        number('monitoring.help.window_s', 'Second push within', 's', 'Counted from the first push.', 0.1) +
        number('monitoring.help.min_gap_s', 'But not sooner than', 's', 'The hand coming back after one push often reads as a second push 0.4 to 1.0 s later.', 0.05) +
        number('monitoring.help.confirm_s', 'Time to cancel on screen', 's', 'The request shows with a Cancel button for this long before the alert goes out. 0 sends it at once.', 0.5) +
        '<fieldset class="choices choices--two"><legend>A single push</legend>' +
        '<label class="choice"><input type="radio" name="push_timing" value="wait"><span class="choice__box"><b>Waits for the window</b><small>Its key is pressed only once no second push came. One movement never does both.</small></span></label>' +
        '<label class="choice"><input type="radio" name="push_timing" value="instant"><span class="choice__box"><b>Acts at once</b><small>Its key is pressed immediately. A double push then presses once and also calls for help.</small></span></label>' +
        '</fieldset>') +

      section('alerts', 'alert', 'How alerts are announced', 'Besides the dashboard, on this laptop.',
        toggle('alerts.notification', 'macOS notification', 'A banner at the top right of the screen.') +
        toggle('alerts.sound', 'Sound', 'A chime through the laptop speakers.') +
        toggle('alerts.speak', 'Say it aloud', 'After the chime, the alert is spoken: “Help requested”.') +
        slider('alerts.volume', 'Alert volume', 'Relative to the system volume. The system volume itself must stay where the sonar was set up.', 0.05, true) +
        number('alerts.repeat_s', 'Repeat the sound every', 's', 'While a help or no-breathing alert is not acknowledged. 0 = never.', 5) +
        '<div class="quick"><button class="btn btn--quiet btn--small" type="button" id="testAlert">Play a test alert</button></div>') +

      section('device', 'engine', 'This device', 'Where the sensor is and how it runs.',
        '<label class="field"><span class="field__label"><b>Room name</b><small>Shown on the dashboard and in a help alert.</small></span>' +
        '<span class="field__input"><input type="text" id="roomInput" maxlength="60" autocomplete="off"></span></label>' +
        '<div class="note" id="engineBox"></div>' +
        '<div class="quick" id="engineButtons"></div>' +
        '<p class="fine" id="filePath"></p>' +
        '<div class="quick"><a class="btn btn--quiet btn--small" href="#/setup">Run the setup again</a></div>') +

      '</div></div>';

    // ---- wiring ----
    view.addEventListener('change', onChange);
    view.addEventListener('input', function (e) {             // live value next to a slider
      if (e.target.type === 'range') showRange(e.target);
    });
    view.addEventListener('click', onClick);
    built = true;
  }

  function showRange(input) {
    var out = input.parentNode.querySelector('output');
    if (out) out.textContent = input.getAttribute('data-percent') ? Math.round(input.value * 100) + ' %' : input.value;
  }

  function onChange(e) {
    var t = e.target, s = settings();
    if (!s) return;
    if (t.name === 'preset') { save({ control: { preset: t.value } }); return; }
    if (t.name === 'push_timing') { save({ monitoring: { help: { push_timing: t.value } } }); return; }
    if (t.getAttribute('data-map')) {                         // editing one key turns the current mapping into "Custom"
      var custom = {}, map = currentMap(s);
      GESTURES.forEach(function (g) { custom[g[0]] = map[g[0]]; });
      custom[t.getAttribute('data-map')] = t.value;
      save({ control: { preset: 'custom', custom: custom } });
      return;
    }
    if (t.id === 'roomInput') { save({ room: t.value }); return; }
    var path = t.getAttribute('data-path');
    if (!path) return;
    if (t.type === 'checkbox') save(patchOf(path, t.checked));
    else if (t.tagName === 'SELECT') save(patchOf(path, t.value));
    else if (t.value !== '' && isFinite(Number(t.value))) save(patchOf(path, Number(t.value)));
    else sync();
  }

  function onClick(e) {
    var t = e.target.closest ? e.target.closest('button') : null;
    if (!t) return;
    if (t.getAttribute('data-test')) {
      var action = currentMap(settings())[t.getAttribute('data-test')];
      if (!action || action === 'off') { Care.toast('This gesture is set to Off'); return; }
      t.disabled = true;
      Care.toast('Pressing it in 3 seconds: click into the app you want to control');
      Api.post('/api/test-action', { action: action, delay_s: 3 }).then(function (r) {
        Care.toast(r.result === 'sent' ? 'Done: ' + r.detail : r.result === 'dry_run' ? 'Dry run: ' + r.detail : r.detail, r.result === 'sent' || r.result === 'dry_run' ? '' : 'alert');
      }, Care.failed('Could not press the key')).then(function () { t.disabled = false; });
    } else if (t.getAttribute('data-still')) {
      var p = meta.no_movement_presets.filter(function (x) { return x.id === t.getAttribute('data-still'); })[0];
      save({ monitoring: { no_movement: { enabled: true, seconds: p.seconds, breathing_counts: p.breathing_counts } } }, 'No-movement alert: ' + p.label.toLowerCase());
    } else if (t.id === 'testAlert') {
      Api.post('/api/test-alert').then(function (r) { Care.toast(r.quiet ? 'Quiet mode: nothing is played (started with --quiet)' : 'A test alert was sent to this laptop'); }, Care.failed('Could not send the test alert'));
    } else if (t.getAttribute('data-engine')) {
      Api.post('/api/engine', { action: t.getAttribute('data-engine') }).catch(Care.failed('Could not ' + t.getAttribute('data-engine') + ' the engine'));
    } else if (t.id === 'openAccess') {
      Api.post('/api/accessibility').then(function () { Care.toast('System Settings is opening on the Accessibility page'); }, Care.failed('Could not open System Settings'));
    }
  }

  function currentMap(s) {
    var preset = meta.presets.filter(function (p) { return p.id === s.control.preset; })[0];
    return s.control.preset === 'custom' || !preset ? s.control.custom : preset.map;
  }

  // ---- show the service's values (skipping the control being edited right now) ---------------------------
  function sync() {
    var s = settings();
    if (!built || !s) return;
    var active = doc.activeElement;
    view.querySelectorAll('[data-path]').forEach(function (el) {
      var v = get(s, el.getAttribute('data-path'));
      if (el.type === 'checkbox') el.checked = !!v;
      else if (el !== active) { el.value = v; if (el.type === 'range') showRange(el); }
    });
    view.querySelectorAll('input[name="preset"]').forEach(function (el) { el.checked = el.value === s.control.preset; });
    view.querySelectorAll('input[name="push_timing"]').forEach(function (el) { el.checked = el.value === s.monitoring.help.push_timing; });
    var map = currentMap(s);
    view.querySelectorAll('[data-map]').forEach(function (el) { if (el !== active) el.value = map[el.getAttribute('data-map')] || 'off'; });
    var room = doc.getElementById('roomInput');
    if (room !== active) room.value = s.room;
    // Things that depend on a switch above them look switched off too.
    view.querySelector('#set-control .sect__body').setAttribute('data-off', s.control.enabled ? '0' : '1');
    view.querySelector('#set-monitor .sect__body').setAttribute('data-off', s.monitoring.enabled ? '0' : '1');
    view.querySelector('#set-help .sect__body').setAttribute('data-off', s.monitoring.enabled && s.monitoring.help.enabled ? '0' : '1');
    view.querySelector('#set-fall .sect__body').setAttribute('data-off', s.monitoring.enabled ? '0' : '1');
    view.querySelectorAll('[data-still]').forEach(function (b) {
      var p = meta.no_movement_presets.filter(function (x) { return x.id === b.getAttribute('data-still'); })[0];
      var on = s.monitoring.no_movement.seconds === p.seconds && s.monitoring.no_movement.breathing_counts === p.breathing_counts;
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    syncStatus();
  }

  // ---- live status boxes: key permission and the engine -----------------------------------------------------
  function syncStatus() {
    var st = Care.state();
    if (!built || !st) return;
    var k = st.control.keys, html;
    if (k.mode === 'dry_run') {
      html = '<span class="note__icon" data-tone="caution">' + UI.icon('info') + '</span><div><b>Dry run</b><p>Gestures are logged with the key they would press. Nothing is pressed. Start Sonar Care without <code>--dry-run</code> to press keys.</p></div>';
    } else if (k.mode === 'blocked') {
      html = '<span class="note__icon" data-tone="alert">' + UI.icon('alert') + '</span><div><b>macOS is blocking key presses</b>' +
        '<p>Open <b>' + esc(k.settingsPath || 'System Settings > Privacy & Security > Accessibility') + '</b> and switch on the app you started Sonar Care from (Terminal, iTerm, …). This page notices by itself when it is allowed.</p>' +
        '<p><button class="btn btn--small" type="button" id="openAccess">Open Accessibility settings</button></p></div>';
    } else if (k.mode === 'live') {
      html = '<span class="note__icon" data-tone="ok">' + UI.icon('check') + '</span><div><b>Key presses are allowed</b><p>Keys go to the app in front. Media keys: ' + esc(k.media || 'unknown') + '.</p></div>';
    } else {
      html = '<span class="note__icon" data-tone="caution">' + UI.icon('info') + '</span><div><b>Key presses are not available on this system</b><p>' + esc(k.note || '') + '</p></div>';
    }
    var box = doc.getElementById('keysBox');
    if (box.getAttribute('data-key') !== k.mode + '|' + k.media) { box.innerHTML = html; box.setAttribute('data-key', k.mode + '|' + k.media); }

    // The experimental fall alert: what the detector sees right now, and what it made of the last big movement.
    var f = st.fall, fbox = doc.getElementById('fallBox'), fhtml, two = function (v) { return v == null ? '–' : v.toFixed(2); };
    if (f && f.error) {
      fhtml = '<span class="note__icon" data-tone="alert">' + UI.icon('alert') + '</span><div><b>Stopped after an internal error</b><p>The fall alert switched itself off so the rest of Sonar Care keeps working. ' +
        'Switch it off and on again to restart it.</p><p class="note__error">' + esc(f.error) + '</p></div>';
    } else if (!f || !f.enabled) {
      fhtml = '<span class="note__icon" data-tone="caution">' + UI.icon('info') + '</span><div><b>Switched off</b><p>Nothing is watched for falls.' +
        (st.mode === 'mock' ? ' To rehearse: switch it on, open the dashboard and press G.' : '') + '</p></div>';
    } else if (!f.hasRows) {
      fhtml = '<span class="note__icon" data-tone="alert">' + UI.icon('alert') + '</span><div><b>No movement picture from the engine</b><p>The fall alert reads the engine’s Doppler strip. It is not arriving (engine stopped, or started with --no-spectrogram).</p></div>';
    } else {
      var th = f.thresholds, b = f.lastBurst, w = f.watching;
      fhtml = '<span class="note__icon" data-tone="' + (w ? 'caution' : 'ok') + '">' + UI.icon(w ? 'still' : 'check') + '</span><div><b>' +
        (w ? 'Sudden movement seen: still for ' + Math.round(w.stillS) + ' of ' + Math.round(w.needsS) + ' s' : 'Watching') + '</b>' +
        '<p>Movement level now <span class="num">' + two(f.level) + '</span> (slow movement <span class="num">' + two(f.slowLevel) + '</span>). ' +
        'A movement counts from size <span class="num">' + two(th.burstLevel) + '</span>, fast part <span class="num">' + two(th.fastLevel) + '</span>, one-way score <span class="num">' + two(th.oneSided) +
        '</span>, lasting ' + th.minS + ' to ' + th.maxS + ' s.</p>' +
        (b ? '<p>Last big movement: size <span class="num">' + two(b.level) + '</span>, fast part <span class="num">' + two(b.fastLevel) + '</span>, one-way score <span class="num">' + two(b.oneSided) + '</span> ' +
          esc(b.direction === 'toward' ? 'toward the laptop' : 'away from the laptop') + ', ' + (b.durationS == null ? '–' : b.durationS.toFixed(1)) + ' s. ' + esc(b.why || '') +
          (b.outcome && b.outcome !== 'ignored' ? ' → ' + esc(b.outcome) : '') + '.</p>' : '<p>No movement big enough to judge yet.</p>') + '</div>';
    }
    if (fbox && fbox.innerHTML !== fhtml) fbox.innerHTML = fhtml;

    var e = st.engine, word = { running: 'running', starting: 'starting…', restarting: 'restarting…', stopping: 'stopping…', stopped: 'stopped', failed: 'could not start',
      attached: 'running (started elsewhere)', external: 'started elsewhere' }[e.state] || e.state;
    var tone = e.state === 'failed' ? 'alert' : e.state === 'running' || e.state === 'attached' || e.state === 'external' ? 'ok' : 'caution';
    var how = { live: 'this laptop’s speakers and one microphone', sim: 'the simulated visitor (no sound)', mock: 'the mock engine (scripted data)', attach: 'an engine at ' + esc(e.url || '') }[st.mode] || '';
    var ehtml = '<span class="note__icon" data-tone="' + tone + '">' + UI.icon('engine') + '</span><div><b>Engine ' + esc(word) + '</b><p>' + how +
      (e.model && st.mode !== 'mock' && e.managed ? ' · model ' + esc(e.model) : '') + (e.restarts ? ' · restarted ' + e.restarts + '×' : '') + '</p>' +
      (e.lastError ? '<p class="note__error">' + esc(e.lastError) + '</p>' : '') + '</div>';
    var ebox = doc.getElementById('engineBox'), ekey = [e.state, e.restarts, e.lastError, e.model].join('|');
    if (ebox.getAttribute('data-key') !== ekey) { ebox.innerHTML = ehtml; ebox.setAttribute('data-key', ekey); }
    var buttons = doc.getElementById('engineButtons'), bkey = (e.managed ? 'm' : 'x') + e.state;
    if (buttons.getAttribute('data-key') !== bkey) {
      buttons.setAttribute('data-key', bkey);
      if (!e.managed) buttons.innerHTML = '<span class="fine">Sonar Care did not start this engine, so it does not stop it.</span>';
      else if (e.state === 'stopped' || e.state === 'failed') buttons.innerHTML = '<button class="btn btn--small" type="button" data-engine="start">Start the engine</button>';
      else buttons.innerHTML = '<button class="btn btn--quiet btn--small" type="button" data-engine="restart">Restart the engine</button><button class="btn btn--quiet btn--small" type="button" data-engine="stop">Stop the engine</button>';
    }
  }

  function show() {
    if (built) { sync(); return; }
    view.innerHTML = '<div class="page__inner"><h1 class="page__title">Settings</h1><p class="page__lede">Loading…</p></div>';
    Api.get('/api/settings').then(function (r) {
      meta = r.meta;
      if (!settings()) store.state.care.settings = r.settings;
      build();
      doc.getElementById('filePath').textContent = 'Saved in ' + r.path;
      sync();
    }, function (err) {
      view.innerHTML = '<div class="page__inner"><h1 class="page__title">Settings</h1><p class="page__lede">Could not load the settings: ' + esc(err.message) + '</p></div>';
    });
  }

  store.on('careSettings', sync);
  store.on('care', syncStatus);
  Care.registerView('settings', { show: show });
})(typeof self !== 'undefined' ? self : this);
