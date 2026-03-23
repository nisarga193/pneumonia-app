/* ============================================================
   PneumoScan AI — History Page Logic
   File: static/js/history.js
   ============================================================
   Handles everything on the History page:
     - Loads all scan records from /api/history
     - Search bar: filter by patient name or patient ID
     - Filter buttons: All / Pneumonia / Normal
     - Renders table with thumbnail, name, ID, badge, confidence, time
     - Delete button per row (calls /api/history/:id DELETE)
   Called by: showPage('history') in app.js
   ============================================================ */


// Module-level state
let histData     = [];   // Full list of all scan records
let activeFilter = 'all'; // 'all' | 'pn' | 'ok'


/**
 * Main entry point for the History page.
 * Fetches all records and triggers a render.
 */
async function loadHistory() {
  histData = await apiHistory();
  renderHist();
}


/**
 * Called by the filter buttons (All / Pneumonia / Normal).
 * Updates activeFilter, toggles button styles, re-renders table.
 *
 * @param {string} f - Filter key: 'all', 'pn', or 'ok'
 */
function setFilter(f) {
  activeFilter = f;
  ['all', 'pn', 'ok'].forEach(k =>
    document.getElementById('f-' + k).classList.toggle('active', k === f)
  );
  renderHist();
}


/**
 * Called on every keystroke in the search input (oninput).
 * Re-renders the table with current search + filter applied.
 */
function renderHist() {
  const query = (document.getElementById('search-inp').value || '').toLowerCase();

  const rows = histData.filter(r => {
    // Text search: patient name or ID
    const matchSearch =
      r.patient_name.toLowerCase().includes(query) ||
      r.patient_id.toLowerCase().includes(query);

    // Filter tab
    const matchFilter =
      activeFilter === 'all' ||
      (activeFilter === 'pn' && r.is_pneumonia) ||
      (activeFilter === 'ok' && !r.is_pneumonia);

    return matchSearch && matchFilter;
  });

  const el = document.getElementById('hist-body');

  if (!rows.length) {
    el.innerHTML = '<div class="no-data">No records found.</div>';
    return;
  }

  el.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Image</th>
          <th>Patient</th>
          <th>ID</th>
          <th>Prediction</th>
          <th>Confidence</th>
          <th>Date &amp; Time</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(r => `
          <tr>
            <td><img src="${r.heatmap_img}" class="thumb" alt="heatmap"/></td>
            <td>${r.patient_name}</td>
            <td style="font-family:'DM Mono',monospace;font-size:0.78rem">${r.patient_id}</td>
            <td>${makeBadge(r)}</td>
            <td style="font-family:'DM Mono',monospace">${(r.confidence * 100).toFixed(1)}%</td>
            <td style="color:var(--muted);font-size:0.76rem">${new Date(r.timestamp).toLocaleString()}</td>
            <td><button class="del-btn" onclick="deleteRecord('${r.id}')">Delete</button></td>
          </tr>
        `).join('')}
      </tbody>
    </table>`;
}


/**
 * Deletes a scan record by ID, shows toast, refreshes the table.
 *
 * @param {string} id - UUID of the scan record to delete
 */
async function deleteRecord(id) {
  await apiDeleteRecord(id);
  showToast('Record deleted');
  loadHistory();
}
