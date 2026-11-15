/*
 * care-api.js: the page's calls to the local Sonar Care API (app/care/api.py).
 *
 *   SonarCareApi.get('/api/settings')            -> Promise of the JSON reply
 *   SonarCareApi.post('/api/ack', {id: 'help-3'}) -> Promise of the JSON reply
 *
 * A refused request rejects with an Error whose message is the service's own reason
 * ("control.min_confidence must be between 0.3 and 0.99"), ready to show to a person.
 */
(function (root) {
  'use strict';

  function call(method, path, body) {
    var opts = { method: method, headers: {}, cache: 'no-store' };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    return root.fetch(path, opts).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (!res.ok) {
          var err = new Error(data && data.error ? data.error : 'The request failed (' + res.status + ')');
          err.status = res.status;
          throw err;
        }
        return data;
      });
    }, function () {
      throw new Error('Sonar Care is not answering. Is it still running?');
    });
  }

  root.SonarCareApi = {
    get: function (path) { return call('GET', path); },
    post: function (path, body) { return call('POST', path, body === undefined ? {} : body); }
  };
})(typeof self !== 'undefined' ? self : this);
