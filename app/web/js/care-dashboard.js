/*
 * care-dashboard.js: the Dashboard view of Sonar Care.
 *
 *   Room       is someone there, how far, how long still (with the no-movement countdown)
 *   Breathing  the early breathing-rate estimate and the chest waveform
 *   Control    what each gesture does right now, lighting up as gestures arrive
 *   Timeline   everything that happened, newest first
 *   Sensor     signal strength, the live Doppler strip, the engine's state
 *
 * Live readings come from the engine's events (store.state.presence / breathing / status). Everything
 * that is a DECISION (alerts, which key a gesture pressed, how long someone has been still) comes from
 * the Sonar Care service (store.state.care).
 */
(function (root) {
  'use strict';
  var Care = root.SonarCare;
  if (!Care || Care.offline) return;
  var doc = root.document, UI = root.SonarUI, W = root.SonarWidgets, store = Care.store, Api = Care.api;
  function $(id) { return doc.getElementById(id); }

  var GESTURE_NAME = { swipe_left: 'Swipe left', swipe_right: 'Swipe right', push: 'Push' };
  var KEY_GLYPH = { left: '←', right: '→', up: '↑', down: '↓', space: '␣', return: '⏎', escape: 'esc', tab: '⇥',
    page_up: '⇞', page_down: '⇟', home: '↖', end: '↘', media_play_pause: '⏯', media_next: '⏭', media_previous: '⏮' };

  W.breathing($('orb'), store);
  W.waveform($('wave'), store, { seconds: 20 });
  W.spectrogram($('spectro'), store, { columns: 300, labels: false });

  var dash = $('view-dashboard');

  // ---- Room -------------------------------------------------------------------------------------------
  var roomCard = $('roomCard'), roomState = $('roomState'), roomSub = $('roomSub'), ruler = $('ruler'), person = $('rulerPerson');
  var stillEl = $('still'), stillFill = $('stillFill'), stillText = $('stillText'), stillLimit = $('stillLimit'), monChip = $('monChip');

  function setText(el, text) { if (el.textContent !== text) el.textContent = text; }
  // 45 -> "45 s", 60 -> "1 min", 75 -> "1 min 15 s"
  function dur(seconds) {
    var s = Math.max(0, Math.round(seconds));
    if (s < 60) return s + ' s';
    var m = Math.floor(s / 60); s = s % 60;
    return m + ' min' + (s ? ' ' + s + ' s' : '');
  }
  function range(m) { return m == null ? '' : 'about ' + m.toFixed(1) + ' m away'; }
  function openAlert(kind) {
    var list = store.state.care.alerts;
    for (var i = 0; i < list.length; i++) if (list[i].kind === kind && (list[i].state === 'active' || list[i].state === 'pending' || list[i].state === 'acknowledged') && list[i].resolvedAt == null) return list[i];
    return null;
  }
  // How long the person has been still, counted on from the service's last number so the bar moves smoothly.
  function stillNow(st) {
    if (!st || st.stillness.stillS == null) return null;
    return st.stillness.stillS + Math.max(0, Math.min(3, store.engineNow() - st.t));
  }

  function renderRoom() {
    var S = store.state, st = S.care.state, P = S.presence, health = S.conn.health;
    var eng = st ? st.engine : null;
    var head, sub = '', tone = '', rs = 'unknown', moving = false, live = health === 'live';
    dash.setAttribute('data-live', live ? '1' : '0');

    if (!live) {
      tone = 'muted';
      if (eng && eng.state === 'failed') { head = 'The sensor could not start'; sub = eng.lastError || 'Look at the engine log for the reason.'; tone = 'alert'; }
      else if (eng && eng.state === 'starting') { head = 'Starting the sensor'; sub = 'This takes a few seconds.'; }
      else if (eng && eng.state === 'restarting') { head = 'Restarting the sensor'; sub = eng.lastError || 'It stopped unexpectedly and is coming back.'; }
      else if (eng && eng.state === 'stopped') { head = 'The sensor is stopped'; sub = 'Start the engine from the Sonar Care menu.'; }
      else if (health === 'connecting') { head = 'Connecting to the sensor'; sub = 'One moment.'; }
      else { head = 'Sensor offline'; sub = 'Reconnecting by itself.'; }
    } else if (S.alerts.calibrating) {
      head = 'Calibrating'; sub = 'Keep still for a few seconds.'; tone = 'muted';
    } else if (P.state === 'empty') {
      head = 'Room is empty'; sub = 'Nobody within about 1.3 m of the laptop.'; rs = 'empty';
    } else if (P.state === 'entering') {
      head = 'Someone is arriving'; sub = range(P.rangeM); rs = 'entering'; moving = true;
    } else if (P.state === 'present') {
      if (st && !st.presence.confirmed) {
        head = 'Checking…'; sub = 'A faint sign of someone. Waiting for a clearer signal.'; rs = 'faint'; tone = 'muted';
      } else {
        head = 'Someone is here'; rs = 'present'; moving = !!P.moving;
        var still = stillNow(st), parts = [];
        if (moving) parts.push('Moving');
        else if (still != null && still >= 5) parts.push('Still for ' + dur(still));
        else parts.push('Settled');
        if (P.rangeM != null) parts.push(range(P.rangeM));
        sub = parts.join(' · ');
        var help = openAlert('help'), fell = openAlert('fall');
        if ((help && help.state !== 'acknowledged') || (fell && fell.state !== 'acknowledged') || openAlert('no_breathing')) tone = 'alert';
        else if (openAlert('no_movement')) tone = 'attention';
      }
    } else { head = 'Waiting for the sensor'; sub = 'No presence reading yet.'; tone = 'muted'; }

    setText(roomState, head);
    setText(roomSub, sub || ' ');
    roomCard.setAttribute('data-tone', tone);
    ruler.setAttribute('data-state', live ? rs : 'unknown');
    ruler.setAttribute('data-moving', moving && live ? '1' : '0');
    if (P.rangeM != null && rs !== 'empty') {
      var x = 96 + Math.max(0.2, Math.min(1.5, P.rangeM)) * 260;
      person.style.transform = 'translateX(' + x.toFixed(1) + 'px)';
    }

    // Monitoring chip and the stillness countdown.
    var mon = st ? st.monitoring : null;
    var fall = st && st.fall && st.fall.enabled ? st.fall : null;
    if (mon) { setText(monChip, mon.enabled ? 'Monitoring on' + (fall ? ' · fall alert (experimental)' : '') : 'Monitoring off'); monChip.setAttribute('data-tone', mon.enabled ? 'ok' : ''); }
    var stillS = stillNow(st), show = live && mon && mon.enabled && mon.noMovement.enabled && stillS != null && rs === 'present';
    var watch = live && fall && fall.watching && !openAlert('fall') ? fall.watching : null;
    stillEl.hidden = !show && !watch;
    if (watch) {                                            // the experimental fall alert saw a sudden movement and is counting the stillness
      var seen = Math.min(watch.needsS, watch.stillS + Math.max(0, Math.min(3, store.engineNow() - st.t)));
      stillFill.style.transform = 'scaleX(' + Math.min(1, seen / watch.needsS).toFixed(3) + ')';
      stillEl.setAttribute('data-tone', 'attention');
      setText(stillText, 'Sudden movement · still ' + dur(seen));
      setText(stillLimit, 'possible fall at ' + dur(watch.needsS));
    } else if (show) {
      var limit = st.stillness.limitS || mon.noMovement.seconds, frac = Math.min(1, stillS / limit), alert = openAlert('no_movement');
      stillFill.style.transform = 'scaleX(' + frac.toFixed(3) + ')';
      stillEl.setAttribute('data-tone', frac >= 0.6 || alert ? 'attention' : '');
      setText(stillText, stillS < 5 && !alert ? 'Moved just now' : 'Still for ' + dur(stillS));
      setText(stillLimit, alert ? 'no-movement alert raised' : 'alert after ' + dur(limit) + ' still');
    }
  }

  // ---- Breathing ------------------------------------------------------------------------------------------
  var breathCard = $('breathCard'), breathStatus = $('breathStatus'), waveCap = $('waveCap');
  function renderBreath() {
    var S = store.state, B = S.breathing, p = S.presence.state, now = Date.now();
    var valid = B.bpm != null && B.lastValidAt && now - B.lastValidAt < 6000, text;
    if (S.alerts.apnea) text = 'No breathing movement';
    else if (S.conn.health !== 'live') text = ' ';
    else if (p === 'empty') text = 'Nobody to measure';
    else if (valid) text = B.bpm > 24 ? 'Very fast breathing' : B.bpm > 20 ? 'Faster than typical' : B.bpm < 10 ? 'Very slow breathing' : B.bpm < 12 ? 'Slower than typical' : 'Breathing steadily';
    else if (p === 'present' || p === 'entering') text = 'No confident reading yet';
    else text = ' ';
    setText(breathStatus, text);
    breathCard.setAttribute('data-tone', S.alerts.apnea ? 'alert' : '');
    // The line is the raw chest movement the sensor sees. It can look clean before the engine trusts a rate.
    var tEnd = S.wave.lastTime(), hasWave = tEnd != null && store.engineNow() - tEnd < 6 && S.conn.health === 'live' && p !== 'empty';
    setText(waveCap, !hasWave ? '\u00a0' : valid ? 'Chest movement, last 20 s' : 'Raw chest movement. A rate shows once the engine is confident.');
  }

  // ---- Touch-free control -----------------------------------------------------------------------------------
  var grows = $('grows'), ctlChip = $('ctlChip'), presetName = $('presetName'), ctlLast = $('ctlLast');
  var rowsKey = '', rows = {}, flashTimers = {}, waitAnim = null;

  function keyHtml(m) {
    if (!m || !m.action || m.action === 'off') return 'Off';
    var glyph = KEY_GLYPH[m.action];
    return (glyph ? '<span aria-hidden="true">' + glyph + '</span>' : '') + UI.escapeHtml(m.label || m.action);
  }
  function restingNote(g, st) {
    var help = st.monitoring.enabled && st.monitoring.help.enabled, m = st.control.mapping[g];
    if (g === 'help') return help ? 'second push within ' + st.monitoring.help.windowS + ' s of the first' : (st.monitoring.enabled ? 'help gesture is off' : 'monitoring is off');
    if (!st.control.enabled) return 'gesture control is off';
    if (!m || m.action === 'off') return 'does nothing';
    if (g === 'push' && help && st.monitoring.help.pushTiming === 'wait') return 'acts after ' + st.monitoring.help.windowS + ' s, unless you push again';
    return g === 'swipe_left' ? 'hand moves to your left' : g === 'swipe_right' ? 'hand moves to your right' : 'palm toward the screen and back';
  }
  function buildRows(st) {
    var key = JSON.stringify([st.control.mapping, st.control.enabled, st.monitoring.enabled, st.monitoring.help]);
    if (key === rowsKey) return;
    rowsKey = key;
    grows.innerHTML = '';
    rows = {};
    ['swipe_left', 'swipe_right', 'push', 'help'].forEach(function (g) {
      var isHelp = g === 'help', m = st.control.mapping[g];
      var helpOn = st.monitoring.enabled && st.monitoring.help.enabled;
      var off = isHelp ? !helpOn : (!st.control.enabled || !m || m.action === 'off');
      var li = W.h('<li class="grow" data-g="' + g + '"' + (off ? ' data-off="1"' : '') + '>' +
        '<span class="grow__icon">' + UI.icon(isHelp ? 'double_push' : g) + '</span>' +
        '<span class="grow__name">' + (isHelp ? 'Push twice' : GESTURE_NAME[g]) + '</span>' +
        '<span class="grow__key">' + (isHelp ? (helpOn ? UI.icon('help') + 'Help' : 'Off') : keyHtml(m)) + '</span>' +
        '<span class="grow__note"></span>' +
        '<i class="grow__wait"></i></li>');
      rows[g] = { el: li, note: li.querySelector('.grow__note'), wait: li.querySelector('.grow__wait'), rest: restingNote(g, st) };
      rows[g].note.textContent = rows[g].rest;
      grows.appendChild(li);
    });
  }
  // Step the card down until its four rows fit: full, one line per gesture, then no "Last:" line.
  var controlCard = $('controlCard');
  function fitControl() {
    controlCard.removeAttribute('data-compact');
    if (grows.scrollHeight <= grows.clientHeight + 1) return;
    controlCard.setAttribute('data-compact', '1');
    if (grows.scrollHeight > grows.clientHeight + 1) controlCard.setAttribute('data-compact', '2');
  }
  if (root.ResizeObserver) new ResizeObserver(fitControl).observe(controlCard);

  function renderControl() {
    var st = store.state.care.state;
    if (!st) return;
    var before = rowsKey;
    buildRows(st);
    if (rowsKey !== before) fitControl();
    setText(presetName, st.control.presetLabel || '');
    var tone = 'ok', text = 'On';
    if (!st.control.enabled) { tone = ''; text = 'Off'; }
    else if (st.practice) { tone = 'caution'; text = 'Practice'; }
    else if (openAlert('help') && openAlert('help').state !== 'acknowledged') { tone = 'caution'; text = 'Paused'; }
    else if (st.dryRun) { tone = 'caution'; text = 'Dry run'; }
    else if (st.control.keys.mode === 'blocked') { tone = 'alert'; text = 'Keys blocked'; }
    setText(ctlChip, text); ctlChip.setAttribute('data-tone', tone);
    var last = st.lastGesture;
    if (last && last.gesture) {
      var what = (GESTURE_NAME[last.gesture] || last.gesture) + (last.confidence != null ? ' ' + UI.pct(last.confidence) : '');
      setText(ctlLast, 'Last: ' + what + ' · ' + outcome(last) + ' · ' + UI.ago(Care.since(last.t)));
    } else setText(ctlLast, 'No gesture yet. Swipe or push a hand-length from the screen.');
  }
  function outcome(a) {
    switch (a.result) {
      case 'sent': return 'pressed ' + (a.actionLabel || a.action);
      case 'dry_run': return 'would press ' + (a.actionLabel || a.action);
      case 'blocked': return 'blocked by macOS';
      case 'failed': return 'could not press';
      case 'held': return 'waiting for a second push';
      case 'armed': return 'first push';
      case 'help': return 'help';
      case 'practice': return 'practice';
      case 'paused': return 'paused';
      case 'off': return 'control is off';
      case 'unmapped': return 'no action';
      default: return a.reason ? 'ignored: ' + a.reason : 'ignored';
    }
  }
  function flash(g, cls, note, ms) {
    var r = rows[g];
    if (!r) return;
    r.el.classList.remove('is-hit', 'is-miss', 'is-help');
    void r.el.offsetWidth;                                 // restart the highlight if the same row fires twice
    r.el.classList.add(cls);
    r.note.textContent = note;
    clearTimeout(flashTimers[g]);
    flashTimers[g] = setTimeout(function () { r.el.classList.remove('is-hit', 'is-miss', 'is-help'); r.note.textContent = r.rest; }, ms || 1800);
  }
  store.on('careAction', function (a) {
    renderControl();
    var push = rows.push;
    if (push && waitAnim && a.gesture === 'push') { waitAnim.cancel(); waitAnim = null; }
    if (a.result === 'help') { flash('push', 'is-help', 'second push'); flash('help', 'is-help', 'help requested', 3200); return; }
    if (a.result === 'held' || a.result === 'armed') {
      var ms = a.helpUntil != null ? Math.max(300, store.careTimeMs(a.helpUntil) - Date.now()) : 2500;
      flash('push', 'is-hit', a.result === 'held' ? 'waiting for a second push…' : 'first push: one more calls for help', ms);
      flash('help', 'is-hit', 'push again now for help', ms);
      if (push && push.wait.animate && !UI.reducedMotion()) {
        waitAnim = push.wait.animate([{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }], { duration: ms, easing: 'linear' });
      }
      return;
    }
    var hit = a.result === 'sent' || a.result === 'dry_run' || a.result === 'practice';
    flash(a.gesture, hit ? 'is-hit' : 'is-miss', outcome(a));
  });

  // ---- Sensor ------------------------------------------------------------------------------------------------
  var snrValue = $('snrValue'), snrWord = $('snrWord'), snrMeter = $('snrMeter'), engineLine = $('engineLine'), calBtn = $('calBtn');
  var SIGNAL_WORD = { good: 'good signal', fair: 'fair signal', weak: 'weak signal', unknown: 'signal' };
  function renderSensor() {
    var S = store.state, s = S.status, st = S.care.state, live = S.conn.health === 'live';
    var level = st ? st.sonar.signal : 'unknown';
    if (s && s.snrDb != null && live) {
      snrValue.innerHTML = s.snrDb.toFixed(0) + '<small>dB</small>';
      snrMeter.firstElementChild.style.transform = 'scaleX(' + Math.max(0.04, Math.min(1, s.snrDb / 60)).toFixed(3) + ')';
      snrMeter.setAttribute('data-level', level === 'weak' ? 'poor' : level);
    } else { snrValue.textContent = '—'; snrMeter.firstElementChild.style.transform = 'scaleX(0)'; level = 'unknown'; }
    setText(snrWord, S.alerts.calibrating ? 'calibrating' : SIGNAL_WORD[level] || 'signal');
    snrWord.parentNode.setAttribute('data-level', level);
    calBtn.disabled = !live || S.alerts.calibrating;

    var html = '';
    if (st) {
      var e = st.engine, span = UI.signalSpan(s);
      var what = st.mode === 'mock' ? '<b>Mock engine</b> · scripted data for rehearsal'
        : st.mode === 'sim' ? '<b>Simulated visitor</b> · the real engine, no sound'
        : st.mode === 'attach' ? '<b>Attached</b> to an engine started elsewhere'
        : '<b>This laptop</b> · its own speakers and one microphone';
      if (e.state === 'failed') what = '<b>Engine could not start</b>';
      else if (e.state === 'restarting') what = '<b>Engine restarting</b>';
      else if (e.state === 'starting') what = '<b>Engine starting</b>';
      else if (e.state === 'stopped' && e.managed) what = '<b>Engine stopped</b>';
      html = what + (span && live ? ' · ' + span.text + ' kHz' : '') + (e.restarts ? ' · restarted ' + e.restarts + '×' : '');
    }
    if (engineLine.innerHTML !== html) engineLine.innerHTML = html || ' ';
  }
  calBtn.addEventListener('click', function () {
    calBtn.disabled = true;
    Api.post('/api/calibrate').then(function (r) { Care.toast(r.sent ? 'Calibrating: keep still for 5 seconds' : 'The sensor is not connected', r.sent ? '' : 'alert'); }, Care.failed('Could not calibrate'));
  });

  // ---- Timeline ------------------------------------------------------------------------------------------------
  var tlEl = $('timeline');
  var TL_ICON = { ping: 'ping', breath: 'breath', enter: 'enter', empty: 'empty', link: 'link', calibrate: 'calibrate', check: 'check', settings: 'settings',
    engine: 'engine', help: 'help', still: 'still', swipe_left: 'swipe_left', swipe_right: 'swipe_right', push: 'push', test: 'test',
    no_movement: 'still', no_breathing: 'breath' };
  function renderTimeline(fresh) {
    var T = store.state.care.timeline.slice(0, 60);
    if (!T.length) { tlEl.innerHTML = '<li class="tl__empty">Nothing yet. Arrivals, gestures and alerts appear here as they happen.</li>'; return; }
    tlEl.innerHTML = T.map(function (e, i) {
      var time = new Date(store.careTimeMs(e.t)).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' });
      return '<li data-level="' + UI.escapeHtml(e.level || 'info') + '"' + (i === 0 && fresh ? ' class="is-new"' : '') + '><time>' + time + '</time>' +
        UI.icon(TL_ICON[e.icon] || 'info') + '<span><b>' + UI.escapeHtml(e.text || '') + '</b>' + (e.detail ? ' <span>' + UI.escapeHtml(e.detail) + '</span>' : '') + '</span></li>';
    }).join('');
  }
  store.on('careTimeline', function (entry) { renderTimeline(!!entry); });

  // ---- keep it all fresh -------------------------------------------------------------------------------------------
  function renderAll() { renderRoom(); renderBreath(); renderControl(); renderSensor(); }
  ['care', 'presence', 'alerts', 'conn', 'status', 'careAlert'].forEach(function (topic) { store.on(topic, renderAll); });
  setInterval(renderAll, 500);
  renderAll();
  renderTimeline(false);
})(typeof self !== 'undefined' ? self : this);
