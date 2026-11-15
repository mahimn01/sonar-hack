/*
 * care-setup.js: the first-run Setup view of Sonar Care (re-runnable from the menu and from Settings).
 *
 *   1 Welcome      what it does; no camera, no audio leaves the laptop
 *   2 Sensor       is the engine running, do speakers + microphone work, live signal strength
 *   3 Calibrate    "keep still for 5 s"
 *   4 Permission   macOS Accessibility, needed to press keys
 *   5 Practice     swipe left, swipe right, push, double push: an animated picture, live feedback, hints
 *   6 Done         a summary and two quick choices
 *
 * While the Practice step is open the service is in practice mode: gestures are recognised and
 * reported here, but they press no key and raise no alert. The page says "still here" every 30 s;
 * if it disappears, the service leaves practice mode by itself after 90 s.
 */
(function (root) {
  'use strict';
  var Care = root.SonarCare;
  if (!Care || Care.offline) return;
  var doc = root.document, UI = root.SonarUI, W = root.SonarWidgets, store = Care.store, Api = Care.api;
  var view = doc.getElementById('view-setup');

  var STEPS = [['welcome', 'Welcome'], ['sensor', 'Sensor'], ['calibrate', 'Calibrate'], ['access', 'Permission'], ['practice', 'Practice'], ['done', 'Done']];
  var LESSONS = [
    { id: 'swipe_left', name: 'Swipe left', need: 2,
      how: 'Hold an open hand a hand-length (20 to 25 cm) in front of the screen. Sweep it to your <b>left</b> in about half a second, then let it drop to your lap.' },
    { id: 'swipe_right', name: 'Swipe right', need: 2,
      how: 'The same, the other way: sweep the open hand to your <b>right</b>, then let it drop. Do not swipe back.' },
    { id: 'push', name: 'Push', need: 2,
      how: 'Move your palm quickly <b>toward the screen and straight back</b>, in one motion. Stop about 10 cm from the glass.' },
    { id: 'help', name: 'Double push for help', need: 1,
      how: 'Push, bring the hand back, then <b>push again about a second later</b>. Two pushes in a row call for help.' }
  ];
  var GESTURE_NAME = { swipe_left: 'a swipe left', swipe_right: 'a swipe right', push: 'a push' };

  var step = 0, shown = false, heartbeat = null, practiceOn = false;
  var lesson = 0, hits = 0, results = {}, feedback = null, advanceTimer = null, lastEventAt = 0, lastDecisionKey = '';
  var calibration = { asked: false, seen: false, done: false };

  function esc(s) { return UI.escapeHtml(s == null ? '' : String(s)); }
  function $(sel) { return view.querySelector(sel); }
  function st() { return Care.state(); }
  function live() { return store.state.conn.health === 'live'; }

  // ---- pictures: a laptop seen from the front, a hand in front of it -----------------------------------
  function art(kind) {
    return '<svg class="gart gart--' + kind + '" viewBox="0 0 320 232" aria-hidden="true">' +
      '<rect class="gart__screen" x="56" y="14" width="208" height="140" rx="12"/>' +
      '<rect class="gart__glass" x="68" y="26" width="184" height="116" rx="6"/>' +
      '<path class="gart__base" d="M32 160h256l-16 20H48z"/>' +
      '<circle class="gart__ripple" cx="160" cy="84" r="26"/>' +
      '<circle class="gart__ripple gart__ripple--2" cx="160" cy="84" r="26"/>' +
      '<g class="gart__hand"><g transform="translate(130 46)">' +
      '<rect x="14" y="30" width="34" height="36" rx="13"/><rect x="14" y="9" width="7.5" height="32" rx="3.75"/>' +
      '<rect x="23" y="2" width="7.5" height="38" rx="3.75"/><rect x="32" y="4" width="7.5" height="36" rx="3.75"/>' +
      '<rect x="40.5" y="11" width="7" height="30" rx="3.5"/><rect x="1" y="37" width="18" height="8" rx="4" transform="rotate(-38 10 41)"/>' +
      '</g></g>' +
      (kind === 'swipe_left' ? '<path class="gart__arrow" d="M218 208H104m14-11-14 11 14 11"/>'
        : kind === 'swipe_right' ? '<path class="gart__arrow" d="M102 208h114m-14-11 14 11-14 11"/>'
        : '<path class="gart__arrow" d="M160 226v-26m-11 12 11-12 11 12"/>') +
      (kind === 'help' ? '<text class="gart__label" x="182" y="222">× 2</text>' : '') +
      '</svg>';
  }

  // ---- frame ----------------------------------------------------------------------------------------------
  function render() {
    var id = STEPS[step][0];
    var dots = STEPS.map(function (s, i) {
      return '<li' + (i === step ? ' aria-current="step"' : '') + ' data-done="' + (i < step ? '1' : '0') + '"><span>' + (i < step ? UI.icon('check') : i + 1) + '</span>' + s[1] + '</li>';
    }).join('');
    view.innerHTML = '<div class="page__inner setup" data-step="' + id + '"><ol class="steps" aria-label="Setup steps">' + dots + '</ol>' +
      '<div class="setup__body">' + BODY[id]() + '</div></div>';
    view.scrollTop = 0;
    setPractice(id === 'practice');
    update();
    var focus = view.querySelector('[data-autofocus]');
    if (focus && shown) focus.focus({ preventScroll: true });
  }
  function nav(back, nextLabel, nextAttrs) {
    return '<div class="setup__nav">' + (back ? '<button class="btn btn--quiet" type="button" data-go="back">Back</button>' : '') +
      '<button class="btn btn--tide" type="button" data-go="next" data-autofocus ' + (nextAttrs || '') + '>' + (nextLabel || 'Next') + '</button></div>';
  }
  function check(key, title) {
    return '<li class="check" data-check="' + key + '" data-state="wait"><span class="check__icon"></span><div><b>' + title + '</b><p></p></div></li>';
  }

  var BODY = {
    welcome: function () {
      return '<div class="setup__split"><div class="setup__copy">' +
        '<h1 class="page__title">This laptop is now a sensor</h1>' +
        '<p class="page__lede">Its speakers play sound too high to hear. Its microphone listens to the echoes. Sonar Care turns them into three things.</p>' +
        '<ul class="pillars">' +
        '<li>' + UI.icon('keyboard') + '<div><b>Touch-free control</b><span>Swipe or push in the air to press a key: next slide, pause the music.</span></div></li>' +
        '<li>' + UI.icon('person') + '<div><b>Room monitor</b><span>Is someone there, are they still, and an early breathing estimate, for a caregiver.</span></div></li>' +
        '<li>' + UI.icon('help') + '<div><b>Help gesture</b><span>Two pushes in a row call for help.</span></div></li></ul>' +
        '<p class="setup__privacy">' + UI.icon('cameraOff') + '<span><b>No camera.</b> No sound is recorded, and nothing leaves this laptop.</span></p>' +
        '<div class="setup__nav"><button class="btn btn--tide btn--wide" type="button" data-go="next" data-autofocus>Start the setup</button>' +
        '<button class="btn btn--quiet" type="button" data-go="skip">Skip for now</button></div>' +
        '<p class="fine">About two minutes. You can run it again from the menu bar or from Settings.</p></div>' +
        '<div class="setup__art">' + art('swipe_right') + '</div></div>';
    },
    sensor: function () {
      return '<h1 class="page__title">Is the sensor working?</h1><p class="page__lede">Three quick checks. They update live.</p>' +
        '<ul class="checks">' + check('engine', 'The sensor is running') + check('audio', 'Speakers and microphone') + check('signal', 'Signal strength') + '</ul>' +
        '<div class="snr"><div class="snr__scale"><i id="snrFill"></i><b style="left:41.67%"></b><b style="left:58.33%"></b></div>' +
        '<div class="snr__labels num"><span>weak</span><span style="left:41.67%">25 dB</span><span style="left:58.33%">35 dB</span><span>good</span></div></div>' +
        '<div class="note" id="sensorTip"></div>' + nav(true, 'Next');
    },
    calibrate: function () {
      return '<div class="setup__split"><div class="setup__copy"><h1 class="page__title">Keep still for five seconds</h1>' +
        '<p class="page__lede">The sensor learns what the empty space in front of the laptop sounds like. Hands away from the laptop, nobody walking past.</p>' +
        '<ul class="checks">' + check('cal', 'Calibration') + '</ul>' +
        '<p class="fine">Calibrate again whenever you move the laptop, change the screen angle or change the volume.</p>' +
        '<div class="setup__nav"><button class="btn btn--quiet" type="button" data-go="back">Back</button>' +
        '<button class="btn" type="button" id="calNow">Calibrate now</button>' +
        '<button class="btn btn--tide" type="button" data-go="next" data-autofocus>Next</button></div></div>' +
        '<div class="setup__art"><div class="calring" id="calRing"><svg viewBox="0 0 100 100"><circle class="calring__track" cx="50" cy="50" r="44"/><circle class="calring__go" cx="50" cy="50" r="44"/></svg><span id="calWord">Ready</span></div></div></div>';
    },
    access: function () {
      return '<h1 class="page__title">Let gestures press keys</h1>' +
        '<p class="page__lede">macOS asks before any app may press keys in other apps. This is only needed for touch-free control; the room monitor and the help gesture work without it.</p>' +
        '<ul class="checks">' + check('keys', 'Permission to press keys') + '</ul>' +
        '<ol class="howto" id="accessHow"><li>Open <b>System Settings &gt; Privacy &amp; Security &gt; Accessibility</b>.</li>' +
        '<li>Switch on the app you started Sonar Care from (Terminal, iTerm, …).</li><li>Come back here. This page notices by itself.</li></ol>' +
        '<div class="setup__nav"><button class="btn btn--quiet" type="button" data-go="back">Back</button>' +
        '<button class="btn" type="button" id="openAccess2">Open Accessibility settings</button>' +
        '<button class="btn btn--tide" type="button" data-go="next" data-autofocus>Next</button></div>';
    },
    practice: function () {
      var L = LESSONS[lesson];
      var tabs = LESSONS.map(function (x, i) {
        var r = results[x.id];
        return '<li' + (i === lesson ? ' aria-current="step"' : '') + ' data-result="' + (r ? r.result : '') + '">' + (r && r.result === 'ok' ? UI.icon('check') : '') + esc(x.name) + '</li>';
      }).join('');
      return '<div class="lesson"><ol class="lesson__tabs" aria-label="Gestures to practise">' + tabs + '</ol>' +
        '<div class="setup__split"><div class="setup__art">' + art(L.id) + '</div>' +
        '<div class="setup__copy"><h1 class="page__title">' + esc(L.name) + '</h1><p class="page__lede">' + L.how + '</p>' +
        '<div class="fb" id="fb" data-tone="wait" aria-live="polite"><div class="fb__dots" id="fbDots"></div><b id="fbTitle"></b><p id="fbText"></p>' +
        (L.id === 'help' ? '<div class="beat" id="beat"><i id="beatFill"></i><span class="beat__zone"></span><span class="beat__a">too soon</span><span class="beat__b">second push now</span></div>' : '') + '</div>' +
        '<p class="fine">Practice only: nothing is pressed and no alert is raised on this page.</p>' +
        '<div class="setup__nav"><button class="btn btn--quiet" type="button" data-go="lessonBack">Back</button>' +
        '<button class="btn btn--quiet" type="button" data-go="lessonSkip">Skip this one</button></div></div></div></div>';
    },
    done: function () {
      var s = store.state.care.settings || {}, c = s.control || {}, m = (s.monitoring || {}).no_movement || {};
      var rows = LESSONS.map(function (x) {
        var r = results[x.id];
        var ok = r && r.result === 'ok';
        return '<li class="check" data-state="' + (ok ? 'ok' : 'skip') + '"><span class="check__icon">' + UI.icon(ok ? 'check' : 'arrow') + '</span><div><b>' + esc(x.name) + '</b><p>' +
          (ok ? 'Recognised' + (r.confidence != null ? ', ' + UI.pct(r.confidence) + ' sure' : '') + (r.gap != null ? ', pushes ' + r.gap.toFixed(1) + ' s apart' : '') : 'Skipped: practise it any time from Setup') + '</p></div></li>';
      }).join('');
      var preset = function (id, label, hint) {
        return '<label class="choice"><input type="radio" name="donePreset" value="' + id + '"' + (c.preset === id ? ' checked' : '') + '><span class="choice__box"><b>' + label + '</b><small>' + hint + '</small></span></label>';
      };
      var demo = m.seconds === 15 && m.breathing_counts === false;
      return '<h1 class="page__title">You are set</h1><p class="page__lede">Two choices, then the dashboard. Everything can be changed later in Settings.</p>' +
        '<div class="done"><div><ul class="checks">' + rows + '</ul></div><div class="done__choices">' +
        '<fieldset class="choices choices--one"><legend>Gestures control</legend>' +
        preset('slides', 'Slides / PDF', 'swipe = next / previous, push = space') + preset('media', 'Media', 'swipe = next / previous track, push = play / pause') +
        preset('scroll', 'Scroll', 'swipe = page down / up') + '</fieldset>' +
        '<fieldset class="choices choices--two"><legend>“No movement” alert after</legend>' +
        '<label class="choice"><input type="radio" name="doneStill" value="standard"' + (demo ? '' : ' checked') + '><span class="choice__box"><b>1 minute</b><small>for real use</small></span></label>' +
        '<label class="choice"><input type="radio" name="doneStill" value="demo"' + (demo ? ' checked' : '') + '><span class="choice__box"><b>15 seconds</b><small>to show it in a demo</small></span></label>' +
        '</fieldset></div></div>' +
        '<div class="setup__nav"><button class="btn btn--quiet" type="button" data-go="back">Back</button>' +
        '<button class="btn btn--tide btn--wide" type="button" data-go="finish" data-autofocus>Open the dashboard</button></div>';
    }
  };

  // ---- live parts of the current step -------------------------------------------------------------------------
  function setCheck(key, state, text) {
    var li = view.querySelector('[data-check="' + key + '"]');
    if (!li) return;
    if (li.getAttribute('data-state') !== state) {
      li.setAttribute('data-state', state);
      li.querySelector('.check__icon').innerHTML = state === 'ok' ? UI.icon('check') : state === 'bad' ? UI.icon('alert') : state === 'info' ? UI.icon('info') : '<i class="spin"></i>';
    }
    var p = li.querySelector('p');
    if (p.innerHTML !== text) p.innerHTML = text;
  }

  function update() {
    if (!shown) return;
    var id = STEPS[step][0], S = store.state, s = S.status, c = st();
    if (id === 'sensor') {
      var e = c ? c.engine : null, tip = '';
      if (live()) setCheck('engine', 'ok', c && c.mode === 'mock' ? 'A mock engine with scripted data: good for rehearsing, not a real sensor.' : c && c.mode === 'sim' ? 'The real engine with its simulated visitor (no sound).' : 'Running and sending data.');
      else if (e && e.state === 'failed') { setCheck('engine', 'bad', esc(e.lastError || 'It could not start.')); tip = 'Another app may be holding the speakers or the microphone. Close Zoom, Teams, FaceTime and browser tabs that play sound, then choose Start engine in the Sonar Care menu.'; }
      else setCheck('engine', 'wait', e && e.state === 'restarting' ? 'It stopped and is restarting…' : 'Starting… this takes a few seconds.');
      if (live() && s) {
        if (s.locked === false) setCheck('audio', 'bad', 'The microphone does not hear the speakers. Is the sound muted, or are headphones connected?');
        else setCheck('audio', 'ok', 'The microphone hears the speakers' + (s.device && !/mock|sim/i.test(s.device) ? ' (' + esc(s.device) + ')' : '') + '.');
        var snr = s.snrDb, q = c ? c.sonar.signal : 'unknown';
        if (snr != null) {
          setCheck('signal', q === 'weak' ? 'bad' : q === 'fair' ? 'info' : 'ok', '<span class="num">' + snr.toFixed(0) + ' dB</span> · ' +
            (q === 'weak' ? 'weak: gestures will be unreliable.' : q === 'fair' ? 'usable. A little more volume helps.' : 'good.'));
          var fill = doc.getElementById('snrFill');
          if (fill) { fill.style.transform = 'scaleX(' + Math.max(0.02, Math.min(1, snr / 60)).toFixed(3) + ')'; fill.setAttribute('data-q', q); }
          if (q === 'weak') tip = 'Raise the volume to about 70 %, unplug headphones, turn off any sound enhancer, then calibrate in the next step. The sound stays inaudible.';
          else if (s.clipped) tip = 'The microphone is overloaded. Lower the volume a little.';
        } else setCheck('signal', 'wait', 'Measuring…');
      } else { setCheck('audio', 'wait', 'Waiting for the sensor.'); setCheck('signal', 'wait', 'Waiting for the sensor.'); }
      var tipEl = doc.getElementById('sensorTip');
      if (tipEl) {
        var html = tip ? '<span class="note__icon" data-tone="caution">' + UI.icon('info') + '</span><div><b>What to try</b><p>' + tip + '</p></div>' : '';
        if (tipEl.innerHTML !== html) tipEl.innerHTML = html;
      }
      var next = view.querySelector('[data-go="next"]');
      if (next) next.textContent = live() ? 'Next' : 'Continue anyway';
    } else if (id === 'calibrate') {
      var cal = S.alerts.calibrating, ring = doc.getElementById('calRing'), word = doc.getElementById('calWord');
      if (cal) calibration.seen = true;
      if (calibration.seen && !cal) calibration.done = true;
      if (ring) ring.setAttribute('data-state', cal ? 'run' : calibration.done ? 'done' : 'idle');
      if (word) word.textContent = cal ? 'Keep still' : calibration.done ? 'Done' : 'Ready';
      if (!live()) setCheck('cal', 'wait', 'Waiting for the sensor.');
      else if (cal) setCheck('cal', 'wait', 'Learning the empty space… keep still.');
      else if (calibration.done) setCheck('cal', 'ok', 'Calibrated just now.');
      else setCheck('cal', 'info', 'The sensor calibrated itself when it started. Do it again here if anything has moved since.');
      var btn = doc.getElementById('calNow');
      if (btn) btn.disabled = !live() || cal;
    } else if (id === 'access') {
      var k = c ? c.control.keys : null, how = doc.getElementById('accessHow'), open = doc.getElementById('openAccess2');
      if (!k) setCheck('keys', 'wait', 'Checking…');
      else if (k.mode === 'dry_run') setCheck('keys', 'info', 'Dry run: gestures are only logged, so no permission is needed right now.');
      else if (k.mode === 'live') setCheck('keys', 'ok', 'Allowed. Gestures can press keys in the app in front.');
      else if (k.mode === 'blocked') setCheck('keys', 'bad', 'Not allowed yet. Follow the three steps below.');
      else setCheck('keys', 'info', esc(k.note || 'Key presses are not available on this system.'));
      var need = !!k && k.mode === 'blocked';
      if (how) how.hidden = !need;
      if (open) open.hidden = !need;
    } else if (id === 'practice') {
      drawFeedback();
    }
  }

  // ---- practice ---------------------------------------------------------------------------------------------------
  function setPractice(on) {
    if (on === practiceOn) return;
    practiceOn = on;
    clearInterval(heartbeat);
    Api.post('/api/practice', { active: on }).catch(function () { /* the service ends practice mode by itself after 90 s */ });
    if (on) heartbeat = setInterval(function () { Api.post('/api/practice', { active: true }).catch(function () {}); }, 30000);
  }
  function say(tone, title, text) { feedback = { tone: tone, title: title, text: text || '' }; drawFeedback(); }
  function waitingText() {
    var L = LESSONS[lesson];
    if (!live()) return { tone: 'wait', title: 'Waiting for the sensor', text: 'It is not sending data right now.' };
    return { tone: 'wait', title: L.id === 'help' ? 'Waiting for your first push…' : 'Waiting for ' + (GESTURE_NAME[L.id] || 'the gesture') + '…',
      text: hits ? 'Once more, the same way.' : 'Try it now.' };
  }
  function drawFeedback() {
    var box = doc.getElementById('fb');
    if (!box) return;
    var f = feedback || waitingText(), L = LESSONS[lesson];
    box.setAttribute('data-tone', f.tone);
    doc.getElementById('fbTitle').textContent = f.title;
    doc.getElementById('fbText').textContent = f.text;
    var dots = '';
    for (var i = 0; i < L.need; i++) dots += '<i data-on="' + (i < hits ? '1' : '0') + '">' + (i < hits ? UI.icon('check') : '') + '</i>';
    var d = doc.getElementById('fbDots');
    if (d.innerHTML !== dots) d.innerHTML = dots;
  }
  function gotIt(conf, gap) {
    var L = LESSONS[lesson];
    hits += 1;
    var best = results[L.id] && results[L.id].confidence != null ? Math.max(results[L.id].confidence, conf || 0) : conf;
    if (hits >= L.need) {
      results[L.id] = { result: 'ok', confidence: best, gap: gap };
      say('ok', L.id === 'help' ? 'That is the help gesture' : 'Recognised: ' + L.name.toLowerCase(),
        (gap != null ? 'Your pushes were ' + gap.toFixed(1) + ' s apart. ' : '') + (conf != null ? UI.pct(conf) + ' sure. ' : '') + 'Moving on…');
      clearTimeout(advanceTimer);
      advanceTimer = setTimeout(nextLesson, 1500);
    } else {
      results[L.id] = { result: 'part', confidence: best };
      say('ok', 'Recognised' + (conf != null ? ' · ' + UI.pct(conf) + ' sure' : ''), 'Once more, the same way.');
    }
  }
  function nextLesson() {
    clearTimeout(advanceTimer);
    feedback = null; hits = 0;
    if (lesson < LESSONS.length - 1) { lesson += 1; render(); } else go(1);
  }
  function startBeat(a) {
    var fill = doc.getElementById('beatFill'), beat = doc.getElementById('beat'), s = st();
    if (!fill || !beat || !s) return;
    var windowS = s.monitoring.help.windowS || 2.5, gapS = s.monitoring.help.minGapS || 1;
    beat.style.setProperty('--gap', (gapS / windowS * 100).toFixed(1) + '%');
    beat.setAttribute('data-on', '1');
    if (fill.getAnimations) fill.getAnimations().forEach(function (x) { x.cancel(); });
    var ms = a.helpUntil != null ? Math.max(300, store.careTimeMs(a.helpUntil) - Date.now()) : windowS * 1000;
    if (fill.animate) fill.animate([{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }], { duration: ms, easing: 'linear', fill: 'forwards' });
    setTimeout(function () {
      if (STEPS[step][0] !== 'practice' || LESSONS[lesson].id !== 'help' || results.help && results.help.result === 'ok') return;
      if (feedback && feedback.tone === 'first') say('hint', 'Only one push was seen', 'Push, bring the hand back, then push again about a second later. Not faster.');
      beat.setAttribute('data-on', '0');
    }, ms + 250);
  }

  store.on('careAction', function (a) {
    if (!shown || STEPS[step][0] !== 'practice') return;
    var L = LESSONS[lesson];
    if (results[L.id] && results[L.id].result === 'ok') return;       // already passed: waiting to move on
    lastEventAt = Date.now();
    if (a.result === 'ignored') {
      if (a.reason === 'low confidence') say('hint', 'Seen, but not sure enough', (a.confidence != null ? UI.pct(a.confidence) + ' sure. ' : '') + 'Make the movement a little bigger and faster.');
      else if (a.reason === 'same push' && L.id === 'help') say('first', 'One push so far', 'That second movement came too soon and counts as the same push. Leave about a second between the two.');
      return;
    }
    if (a.result !== 'practice') return;
    if (L.id === 'help') {
      if (a.gesture !== 'push') { say('hint', 'That looked like ' + GESTURE_NAME[a.gesture], 'For help: push toward the screen, bring the hand back, push again.'); return; }
      if (a.doublePush) gotIt(a.confidence, a.gapS);
      else { say('first', 'First push seen', 'Now bring the hand back and push again.'); startBeat(a); }
      return;
    }
    if (a.gesture === L.id) gotIt(a.confidence, null);
    else if (L.id === 'push') say('hint', 'That looked like ' + GESTURE_NAME[a.gesture], 'For a push, move the palm straight toward the screen and straight back, not sideways.');
    else if (a.gesture === 'push') say('hint', 'That looked like a push', 'For a swipe, sweep sideways across the screen, staying the same distance from it.');
    else say('hint', 'That looked like ' + GESTURE_NAME[a.gesture], 'It was read as the other direction. Sweep the other way, and let the hand drop afterwards instead of swiping back.');
  });

  // A movement the engine saw but did not call a gesture: say why, so "nothing happened" is never a mystery.
  store.on('status', function (s) {
    if (!shown || STEPS[step][0] !== 'practice' || !s.lastDecision) return;
    var d = s.lastDecision, key = d.label + '|' + d.peakDb + '|' + d.confidence;
    if (key === lastDecisionKey) return;
    lastDecisionKey = key;
    if (Date.now() - lastEventAt < 1500 || (feedback && feedback.tone === 'ok')) return;
    if (d.label === 'blocked:weak') say('hint', 'A movement was seen, but too faint', 'Bring your hand closer: a hand-length from the screen.');
    else if (d.label === 'none') say('hint', 'A movement was seen, but not a gesture', 'Make it one clear, quick stroke.');
    else if (d.label === 'blocked:entering') say('hint', 'Wait a moment', 'The sensor thinks someone is still walking up. Sit still for two seconds, then try.');
  });

  // ---- moving between steps -------------------------------------------------------------------------------------------
  function go(delta) {
    step = Math.max(0, Math.min(STEPS.length - 1, step + delta));
    feedback = null; hits = 0;
    if (STEPS[step][0] === 'practice' && delta > 0) lesson = 0;
    render();
  }
  function finish(alsoSave) {
    var patch = { onboarded: true };
    if (alsoSave) {
      var p = view.querySelector('input[name="donePreset"]:checked'), s = view.querySelector('input[name="doneStill"]:checked');
      if (p) patch.control = { preset: p.value };
      if (s) patch.monitoring = { no_movement: s.value === 'demo' ? { enabled: true, seconds: 15, breathing_counts: false } : { enabled: true, seconds: 60, breathing_counts: true } };
    }
    setPractice(false);
    Api.post('/api/settings', patch).catch(Care.failed('Could not save'));
    step = 0; lesson = 0; results = {}; calibration = { asked: false, seen: false, done: false };
    Care.go('#/');
  }
  view.addEventListener('click', function (e) {
    var t = e.target.closest ? e.target.closest('button') : null;
    if (!t) return;
    var g = t.getAttribute('data-go');
    if (g === 'next') go(1);
    else if (g === 'back') go(-1);
    else if (g === 'skip') finish(false);
    else if (g === 'finish') finish(true);
    else if (g === 'lessonSkip') { if (!results[LESSONS[lesson].id] || results[LESSONS[lesson].id].result !== 'ok') results[LESSONS[lesson].id] = { result: 'skip' }; nextLesson(); }
    else if (g === 'lessonBack') { clearTimeout(advanceTimer); feedback = null; hits = 0; if (lesson > 0) { lesson -= 1; render(); } else go(-1); }
    else if (t.id === 'calNow') {
      calibration.asked = true; calibration.done = false; calibration.seen = false;
      Api.post('/api/calibrate').then(function (r) { if (!r.sent) Care.toast('The sensor is not connected', 'alert'); }, Care.failed('Could not calibrate'));
    } else if (t.id === 'openAccess2') {
      Api.post('/api/accessibility').then(function () { Care.toast('System Settings is opening on the Accessibility page'); }, Care.failed('Could not open System Settings'));
    }
  });

  ['care', 'status', 'conn', 'alerts'].forEach(function (topic) { store.on(topic, update); });
  setInterval(update, 500);
  root.addEventListener('pagehide', function () {
    if (practiceOn) root.fetch('/api/practice', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"active":false}', keepalive: true }).catch(function () {});
  });

  Care.registerView('setup', {
    show: function () { shown = true; render(); },
    hide: function () { shown = false; clearTimeout(advanceTimer); setPractice(false); }
  });
})(typeof self !== 'undefined' ? self : this);
