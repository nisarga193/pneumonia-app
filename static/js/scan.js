/* ============================================================
   PneumoScan AI — New Scan Page Logic
   File: static/js/scan.js
   ============================================================
   Handles everything on the New Scan page:
     - File drag-and-drop and preview
     - Sending image to /api/predict via apiPredict() from api.js
     - Displaying the result panel with:
         • Original X-ray vs Grad-CAM heatmap side by side
         • Class probability bars (Normal / Bacterial / Viral)
         • Verdict dot, label, confidence score
         • Patient info metadata
         • Flag box (red = pneumonia, green = normal)
   ============================================================ */


// ── File drag-and-drop + preview ──────────────────────────────
const fileInp = document.getElementById('file-input');
const prevImg = document.getElementById('prev-img');
const dropZ   = document.getElementById('drop-zone');

fileInp.addEventListener('change', () => {
  const f = fileInp.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = e => {
    prevImg.src = e.target.result;
    prevImg.style.display = 'block';
  };
  rd.readAsDataURL(f);
});

dropZ.addEventListener('dragover', e => {
  e.preventDefault();
  dropZ.classList.add('dg');
});
dropZ.addEventListener('dragleave', () => {
  dropZ.classList.remove('dg');
});
dropZ.addEventListener('drop', e => {
  e.preventDefault();
  dropZ.classList.remove('dg');
  if (e.dataTransfer.files.length) {
    fileInp.files = e.dataTransfer.files;
    fileInp.dispatchEvent(new Event('change'));
  }
});


// ── Run analysis ──────────────────────────────────────────────
/**
 * Called when doctor clicks "Analyze X-Ray" button.
 * Sends image + patient info to backend, then renders result.
 */
async function runScan() {
  const f = fileInp.files[0];
  if (!f) {
    showToast('Please upload an X-ray first.');
    return;
  }

  // Show loading state
  const btn = document.getElementById('run-btn');
  document.getElementById('btn-txt').textContent  = 'Analyzing...';
  document.getElementById('btn-spin').style.display = 'block';
  btn.disabled = true;

  try {
    const data = await apiPredict(
      f,
      document.getElementById('p-name').value,
      document.getElementById('p-id').value
    );
    renderResult(data);
    showToast('Analysis complete ✓');
  } catch (e) {
    showToast('Error: ' + e.message);
  } finally {
    document.getElementById('btn-txt').textContent  = 'Analyze X-Ray';
    document.getElementById('btn-spin').style.display = 'none';
    btn.disabled = false;
  }
}


// ── Render result panel ───────────────────────────────────────
/**
 * Fills the result panel with prediction output.
 * Shows verdict, confidence, images, probability bars, metadata.
 *
 * @param {Object} d - Prediction record returned by /api/predict
 */
function renderResult(d) {
  const panel = document.getElementById('result-panel');
  panel.classList.add('show');

  const isp = d.is_pneumonia;

  // Verdict dot + label
  document.getElementById('v-dot').className  = 'v-dot ' + (isp ? 'pn' : 'ok');
  document.getElementById('v-lbl').textContent = d.prediction;
  document.getElementById('v-sub').textContent = isp
    ? '⚠️ Flagged for immediate doctor review'
    : '✓ No signs of pneumonia detected';
  document.getElementById('v-conf').textContent = (d.confidence * 100).toFixed(1) + '%';

  // X-ray images
  document.getElementById('orig-img').src = d.original_img;
  document.getElementById('heat-img').src = d.heatmap_img;

  // Probability bars
  const pr = d.probabilities;
  const pn = pr['Normal']               || 0;
  const pb = pr['Bacterial Pneumonia']  || 0;
  const pv = pr['Viral Pneumonia']      || 0;

  document.getElementById('pp-n').textContent = (pn * 100).toFixed(1) + '%';
  document.getElementById('pp-b').textContent = (pb * 100).toFixed(1) + '%';
  document.getElementById('pp-v').textContent = (pv * 100).toFixed(1) + '%';

  // Animate bar widths after a tiny delay (CSS transition needs display:block first)
  setTimeout(() => {
    document.getElementById('bf-n').style.width = (pn * 100) + '%';
    document.getElementById('bf-b').style.width = (pb * 100) + '%';
    document.getElementById('bf-v').style.width = (pv * 100) + '%';
  }, 60);

  // Flag box
  const fb = document.getElementById('flag-box');
  fb.className  = isp ? 'flag-box' : 'flag-box ok-box';
  fb.textContent = isp
    ? '🚨 Pneumonia detected. Red/yellow heatmap shows infected region. Verify with a radiologist.'
    : '✓ Lungs appear clear. No significant infection pattern detected.';

  // Patient metadata
  document.getElementById('m-patient').textContent = d.patient_name;
  document.getElementById('m-pid').textContent     = d.patient_id;
  document.getElementById('m-time').textContent    = new Date(d.timestamp).toLocaleString();
  document.getElementById('m-ms').textContent      = d.elapsed_ms + ' ms';

  // Scroll result into view
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
