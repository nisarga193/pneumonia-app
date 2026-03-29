/* ============================================================
   PneumoScan AI — New Scan Page (v3 clean rewrite)
   File: static/js/scan.js

   3 completely separate display states:
     showInvalid(d)   → non-X-ray: warning only, NO prediction, NO heatmap
     showNormal(d)    → valid X-ray, Normal: original image only, NO heatmap
     showPneumonia(d) → valid X-ray, Bacterial/Viral: original + Grad-CAM
   ============================================================ */

// ── File input + drag-and-drop ────────────────────────────────
const fileInp = document.getElementById('file-input');
const prevImg = document.getElementById('prev-img');
const dropZ = document.getElementById('drop-zone');

fileInp.addEventListener('change', () => {
  const f = fileInp.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = e => { prevImg.src = e.target.result; prevImg.style.display = 'block'; };
  rd.readAsDataURL(f);
});
dropZ.addEventListener('dragover', e => { e.preventDefault(); dropZ.classList.add('dg'); });
dropZ.addEventListener('dragleave', () => dropZ.classList.remove('dg'));
dropZ.addEventListener('drop', e => {
  e.preventDefault(); dropZ.classList.remove('dg');
  if (e.dataTransfer.files.length) {
    fileInp.files = e.dataTransfer.files;
    fileInp.dispatchEvent(new Event('change'));
  }
});

// ── Run scan ──────────────────────────────────────────────────
async function runScan() {
  const f = fileInp.files[0];
  if (!f) { showToast('Please upload a chest X-ray first.'); return; }

  const btn = document.getElementById('run-btn');
  document.getElementById('btn-txt').textContent = 'Analyzing...';
  document.getElementById('btn-spin').style.display = 'block';
  btn.disabled = true;

  resetResultPanel();   // always wipe previous state first

  try {
    const data = await apiPredict(
      f,
      document.getElementById('p-name').value,
      document.getElementById('p-id').value
    );

    if (data.is_invalid) showInvalid(data);
    else if (!data.is_pneumonia) showNormal(data);
    else showPneumonia(data);

  } catch (e) {
    showToast('Server error: ' + e.message);
  } finally {
    document.getElementById('btn-txt').textContent = 'Analyze X-Ray';
    document.getElementById('btn-spin').style.display = 'none';
    btn.disabled = false;
  }
}

// ── Reset everything before each new run ─────────────────────
function resetResultPanel() {
  const panel = document.getElementById('result-panel');
  panel.className = 'result-panel';   // strips show + state classes

  document.getElementById('v-dot').className = 'v-dot';
  document.getElementById('v-lbl').textContent = '—';
  document.getElementById('v-sub').textContent = '—';
  document.getElementById('v-conf').textContent = '—';
  document.getElementById('orig-img').src = '';
  document.getElementById('heat-img').src = '';

  ['bf-n', 'bf-b', 'bf-v'].forEach(id => document.getElementById(id).style.width = '0%');
  ['pp-n', 'pp-b', 'pp-v'].forEach(id => document.getElementById(id).textContent = '—');

  document.getElementById('flag-box').className = 'flag-box';
  document.getElementById('flag-box').innerHTML = '';

  // Hide all conditional sections
  setDisplay('sec-images', false);
  setDisplay('sec-probs', false);
  setDisplay('sec-meta', false);
  setDisplay('heat-img-wrap', false);

  // Reset image grid to 2 columns (default)
  document.getElementById('img-pair-wrap').style.gridTemplateColumns = '1fr 1fr';
}

// ════════════════════════════════════════════════════════════
//  STATE 1 — INVALID IMAGE
//  Only shows: warning message + uploaded image
//  Hides: prediction label, confidence, probability bars, Grad-CAM
// ════════════════════════════════════════════════════════════
function showInvalid(d) {
  const panel = document.getElementById('result-panel');
  panel.classList.add('show');

  document.getElementById('v-dot').className = 'v-dot dot-warn';
  document.getElementById('v-lbl').textContent = 'Invalid Image';
  document.getElementById('v-sub').textContent = '⚠️ ' + d.invalid_reason;
  document.getElementById('v-conf').textContent = '—';

  // Single image (no heatmap slot)
  setDisplay('sec-images', true);
  setDisplay('heat-img-wrap', false);
  document.getElementById('img-pair-wrap').style.gridTemplateColumns = '1fr';
  document.getElementById('orig-img').src = d.original_img;

  // Warning flag
  const fb = document.getElementById('flag-box');
  fb.className = 'flag-box warn-box';
  fb.innerHTML = `<strong>🚫 Not a valid chest X-ray</strong><br/>
    ${d.invalid_reason}<br/>
    <span style="font-size:0.72rem;opacity:0.75">Upload a proper grayscale chest X-ray (JPEG/PNG).</span>`;

  // NO probability bars, show meta only
  setDisplay('sec-probs', false);
  setDisplay('sec-meta', true);
  fillMeta(d);

  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  showToast('⚠️ Not a valid X-ray image');
}

// ════════════════════════════════════════════════════════════
//  STATE 2 — NORMAL (valid X-ray, no pneumonia found)
//  Shows: original image only, probability bars, green verdict
//  No Grad-CAM (not generated on backend either)
// ════════════════════════════════════════════════════════════
function showNormal(d) {
  const panel = document.getElementById('result-panel');
  panel.classList.add('show');

  document.getElementById('v-dot').className = 'v-dot dot-ok';
  document.getElementById('v-lbl').textContent = 'Normal';
  document.getElementById('v-sub').textContent = '✓ No signs of pneumonia detected';
  document.getElementById('v-conf').textContent = (d.confidence * 100).toFixed(1) + '%';

  // Original image only — hide heatmap slot
  setDisplay('sec-images', true);
  setDisplay('heat-img-wrap', false);
  document.getElementById('img-pair-wrap').style.gridTemplateColumns = '1fr';
  document.getElementById('orig-img').src = d.original_img;

  // Probability bars
  setDisplay('sec-probs', true);
  fillBars(d);

  // Info flag
  const fb = document.getElementById('flag-box');
  fb.className = 'flag-box ok-box';
  fb.textContent = '✓ Lungs appear clear. Grad-CAM is not generated for Normal predictions — it is only shown when pneumonia is detected.';

  setDisplay('sec-meta', true);
  fillMeta(d);

  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  showToast('✓ Normal — no pneumonia detected');
}

// ════════════════════════════════════════════════════════════
//  STATE 3 — PNEUMONIA detected (Bacterial or Viral)
//  Shows: original + Grad-CAM side by side, red verdict, bars
//  Grad-CAM was generated on backend and is in d.heatmap_img
// ════════════════════════════════════════════════════════════
function showPneumonia(d) {
  const panel = document.getElementById('result-panel');
  panel.classList.add('show');

  document.getElementById('v-dot').className = 'v-dot dot-pn';
  document.getElementById('v-lbl').textContent = d.prediction;
  document.getElementById('v-sub').textContent = '⚠️ Flagged for immediate doctor review';
  document.getElementById('v-conf').textContent = (d.confidence * 100).toFixed(1) + '%';

  // Both images side by side
  setDisplay('sec-images', true);
  setDisplay('heat-img-wrap', true);
  document.getElementById('img-pair-wrap').style.gridTemplateColumns = '1fr 1fr';
  document.getElementById('orig-img').src = d.original_img;
  document.getElementById('heat-img').src = d.heatmap_img;

  // Probability bars
  setDisplay('sec-probs', true);
  fillBars(d);

  // Pneumonia flag
  const fb = document.getElementById('flag-box');
  fb.className = 'flag-box';
  fb.textContent = '🚨 Pneumonia detected. Grad-CAM heatmap highlights the infected lung region in red/yellow. Please verify with a radiologist before any clinical decision.';

  setDisplay('sec-meta', true);
  fillMeta(d);

  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  showToast('🚨 Pneumonia detected — review required');
}

// ── Helpers ───────────────────────────────────────────────────
function setDisplay(id, show) {
  const el = document.getElementById(id);
  if (el) el.style.display = show ? 'block' : 'none';
}

function fillBars(d) {
  const pr = d.probabilities || {};
  const pn = pr['Normal'] || 0;
  const pb = pr['Bacterial Pneumonia'] || 0;
  const pv = pr['Viral Pneumonia'] || 0;
  document.getElementById('pp-n').textContent = (pn * 100).toFixed(1) + '%';
  document.getElementById('pp-b').textContent = (pb * 100).toFixed(1) + '%';
  document.getElementById('pp-v').textContent = (pv * 100).toFixed(1) + '%';
  setTimeout(() => {
    document.getElementById('bf-n').style.width = (pn * 100) + '%';
    document.getElementById('bf-b').style.width = (pb * 100) + '%';
    document.getElementById('bf-v').style.width = (pv * 100) + '%';
  }, 60);
}

function fillMeta(d) {
  document.getElementById('m-patient').textContent = d.patient_name || '—';
  document.getElementById('m-pid').textContent = d.patient_id || '—';
  document.getElementById('m-time').textContent = new Date(d.timestamp).toLocaleString();
  document.getElementById('m-ms').textContent = d.elapsed_ms + ' ms';
}