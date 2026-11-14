/*
 * connection.js: one WebSocket to the sonar engine, reconnecting forever.
 *
 * Reconnects quickly (250 ms doubling to a 2 s cap) so an engine restart during
 * the demo heals itself. Raw message strings go to onMessage; socket state
 * ('connecting' | 'open' | 'closed') goes to onState.
 */
(function (root) {
  'use strict';

  function connect(url, handlers) {
    var ws = null, attempt = 0, timer = null, stopped = false;
    var onMessage = handlers.onMessage || function () {};
    var onState = handlers.onState || function () {};

    function schedule() {
      if (stopped) return;
      var delay = Math.min(2000, 250 * Math.pow(2, attempt++));
      onState('closed', { retryInMs: delay });
      clearTimeout(timer);
      timer = setTimeout(open, delay);
    }

    function open() {
      if (stopped) return;
      onState('connecting', {});
      try {
        ws = new WebSocket(url);
      } catch (err) {
        // Bad URL or blocked port: report and keep retrying (the URL may be fixed by restarting the engine).
        console.warn('[sonar] WebSocket constructor failed:', err && err.message);
        schedule();
        return;
      }
      ws.onopen = function () { attempt = 0; onState('open', {}); };
      ws.onmessage = function (ev) {
        if (typeof ev.data === 'string') onMessage(ev.data);
        // Binary frames are not part of the schema; ignore them.
      };
      ws.onclose = function () { ws = null; schedule(); };
      ws.onerror = function () { /* onclose always follows and handles the retry */ };
    }

    open();

    return {
      url: url,
      send: function (obj) {
        if (ws && ws.readyState === 1) { ws.send(JSON.stringify(obj)); return true; }
        return false;
      },
      close: function () { stopped = true; clearTimeout(timer); if (ws) ws.close(); }
    };
  }

  root.SonarConnection = { connect: connect };
})(typeof self !== 'undefined' ? self : this);
