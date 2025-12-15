// static/js/case_liens.js
(function() {
  const container = document.querySelector('[data-liens-root]');
  if (!container) return;

  const caseId = parseInt(container.dataset.caseId, 10);
  const arv = parseFloat(container.dataset.arv || '0');
  const rehab = parseFloat(container.dataset.rehab || '0');
  const closing = parseFloat(container.dataset.closing || '0');

  // then move your existing liens JS here,
  // but use `container.querySelector(...)` instead of document.getElementById(...)
  // and compute JSN offer from arv/rehab/closing vars above
})();
