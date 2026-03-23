/* ============================================================
   PneumoScan AI — Main App Controller
   File: static/js/app.js
   ============================================================
   This is the top-level controller. It handles:
     - Page navigation (showPage)
     - Toast notifications (showToast)
     - Shared badge helper (makeBadge)
     - App initialization on window.load
   All page-specific logic lives in its own file:
     dashboard.js → Dashboard
     scan.js      → New Scan
     history.js   → History
     about.js     → About
   ============================================================ */


// ── Page Navigation ───────────────────────────────────────────
/**
 * Shows the selected page, hides all others.
 * Updates active state on sidebar nav buttons.
 * Triggers the load function for that page.
 *
 * @param {string} name - Page name: 'dashboard' | 'scan' | 'history' | 'about'
 */
function showPage(name) {
  // Hide all pages
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  // Deactivate all nav items
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  // Show selected page + activate nav item
  document.getElementById('page-' + name).classList.add('active');
  document.getElementById('nav-' + name).classList.add('active');

  // Trigger page-specific data load
  if (name === 'dashboard') loadDashboard();
  if (name === 'history')   loadHistory();
  if (name === 'about')     loadAbout();
  // 'scan' has no async load — it's ready on DOM load
}


// ── Toast Notification ────────────────────────────────────────
/**
 * Shows a brief notification message at the bottom-right of the screen.
 *
 * @param {string} msg - Message to display
 * @param {number} ms  - Duration in milliseconds (default: 3000)
 */
function showToast(msg, ms = 3000) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), ms);
}


// ── Shared Badge Helper ───────────────────────────────────────
/**
 * Returns an HTML badge string for a scan record.
 * Used in both the Dashboard recent table and the History table.
 *
 * @param {Object} r - Scan record with is_pneumonia and prediction fields
 * @returns {string} HTML string for the badge element
 */
function makeBadge(r) {
  if (!r.is_pneumonia)
    return `<span class="badge b-ok">● Normal</span>`;
  if (r.prediction === 'Bacterial Pneumonia')
    return `<span class="badge b-ba">● Bacterial</span>`;
  return `<span class="badge b-vi">● Viral</span>`;
}


// ── App Initialization ────────────────────────────────────────
/**
 * Runs after the full page is loaded (DOM + layout settled).
 * Using window.load (not DOMContentLoaded) ensures canvas elements
 * have correct offsetWidth/offsetHeight for chart drawing.
 */
window.addEventListener('load', () => {
  loadDashboard();
});
