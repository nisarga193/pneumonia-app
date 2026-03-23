/* ============================================================
   PneumoScan AI — Dashboard Page Logic
   File: static/js/dashboard.js
   ============================================================
   Handles everything on the Dashboard page:
     - Stat cards (Total, Normal, Bacterial, Viral)
     - Class Distribution donut chart  → uses drawDonut() from charts.js
     - Recent Confidence bar chart     → uses drawBars()  from charts.js
     - Recent Scans quick-view table
   Called by: showPage('dashboard') in app.js
   ============================================================ */


/**
 * Main entry point for the Dashboard page.
 * Fetches stats + history in parallel, then renders everything.
 */
async function loadDashboard() {
  try {
    const [stats, hist] = await Promise.all([
      apiStats(),
      apiHistory(),
    ]);

    // 1. Fill stat cards
    document.getElementById('s-total').textContent     = stats.total;
    document.getElementById('s-normal').textContent    = stats.normal;
    document.getElementById('s-bacterial').textContent = stats.bacterial;
    document.getElementById('s-viral').textContent     = stats.viral;

    // 2. Draw charts (canvas drawing functions from charts.js)
    drawDonut(stats);                          // Class distribution
    drawBars(hist.slice(0, 20).reverse());     // Last 20 scans, oldest→newest

    // 3. Recent scans table (last 5)
    drawRecentTable(hist.slice(0, 5));

  } catch (e) {
    console.error('Dashboard load error:', e);
  }
}


/**
 * Renders the "Recent Scans — Quick View" table on the dashboard.
 * Shows last 5 scans with thumbnail, patient name, prediction, confidence, time.
 *
 * @param {Array} rows - Array of scan record objects
 */
function drawRecentTable(rows) {
  const el = document.getElementById('dash-recent');

  if (!rows.length) {
    el.innerHTML = '<div class="no-data">No scans yet. Go to New Scan to get started.</div>';
    return;
  }

  el.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Image</th>
          <th>Patient</th>
          <th>Prediction</th>
          <th>Confidence</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(r => `
          <tr>
            <td><img src="${r.heatmap_img}" class="thumb" alt="heatmap"/></td>
            <td>${r.patient_name}</td>
            <td>${makeBadge(r)}</td>
            <td style="font-family:'DM Mono',monospace">${(r.confidence * 100).toFixed(1)}%</td>
            <td style="color:var(--muted);font-size:0.76rem">${new Date(r.timestamp).toLocaleString()}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>`;
}
