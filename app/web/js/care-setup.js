/*
 * care-setup.js: the first-run Setup view of Sonar Care (built in the next step).
 */
(function (root) {
  'use strict';
  var Care = root.SonarCare;
  if (!Care || Care.offline) return;
  var el = root.document.getElementById('view-setup');
  el.innerHTML = '<div class="page__inner"><h1 class="page__title">Setup</h1><p class="page__lede">The guided setup and gesture practice are coming in the next build step.</p><p><a class="btn" href="#/">Open the dashboard</a></p></div>';
})(typeof self !== 'undefined' ? self : this);
