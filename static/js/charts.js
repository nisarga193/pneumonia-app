/* ============================================================
   PneumoScan AI — Chart Engine (no external library needed)
   File: static/js/charts.js
   ============================================================
   Contains two pure-canvas chart functions:
     - drawDonut(stats)   → Class distribution pie/donut chart
     - drawBars(history)  → Recent scan confidence bar chart
   Both draw directly onto <canvas> elements using the
   browser's built-in Canvas 2D API. Zero CDN dependencies.
   ============================================================ */

/**
 * Draws the donut (doughnut) chart on #chart-donut canvas.
 * Shows Normal / Bacterial / Viral scan distribution.
 * @param {Object} stats - { normal, bacterial, viral } counts from /api/stats
 */
function drawDonut(stats) {
  const canvas = document.getElementById('chart-donut');
  const box    = document.getElementById('donut-box');
  if (!canvas || !box) return;

  const W   = box.offsetWidth  || 400;
  const H   = box.offsetHeight || 200;
  const dpr = window.devicePixelRatio || 1;

  canvas.width  = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const vals   = [stats.normal || 0, stats.bacterial || 0, stats.viral || 0];
  const labels = ['Normal', 'Bacterial', 'Viral'];
  const colors = ['#22a67a', '#e07a30', '#7060d0'];
  const total  = vals.reduce((a, b) => a + b, 0);

  // No data state
  if (total === 0) {
    ctx.fillStyle = '#5a7a75';
    ctx.font = '13px Plus Jakarta Sans, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No scans yet — run your first scan!', W / 2, H / 2);
    return;
  }

  // Draw pie slices
  const cx     = W * 0.35;
  const cy     = H / 2;
  const outerR = Math.min(cx - 10, cy - 10);
  const innerR = outerR * 0.58;
  let   angle  = -Math.PI / 2;

  vals.forEach((v, i) => {
    if (v === 0) return;
    const sweep = (v / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, outerR, angle, angle + sweep);
    ctx.closePath();
    ctx.fillStyle = colors[i];
    ctx.fill();
    angle += sweep;
  });

  // White donut hole
  ctx.beginPath();
  ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();

  // Centre total label
  ctx.textAlign = 'center';
  ctx.fillStyle = '#1a2e2b';
  ctx.font = 'bold 22px Plus Jakarta Sans, sans-serif';
  ctx.fillText(total, cx, cy + 6);
  ctx.fillStyle = '#5a7a75';
  ctx.font = '10px Plus Jakarta Sans, sans-serif';
  ctx.fillText('total', cx, cy + 19);

  // Legend on the right side
  const lx = W * 0.65;
  let ly = cy - (labels.length * 28) / 2 + 10;
  labels.forEach((lbl, i) => {
    // Colour square
    ctx.fillStyle = colors[i];
    ctx.beginPath();
    ctx.roundRect(lx, ly - 7, 10, 10, 2);
    ctx.fill();
    // Label text
    ctx.fillStyle = '#1a2e2b';
    ctx.font = '12px Plus Jakarta Sans, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(lbl, lx + 15, ly + 2);
    // Count + percentage
    ctx.fillStyle = '#5a7a75';
    ctx.font = '10px Plus Jakarta Sans, sans-serif';
    const pct = ((vals[i] / total) * 100).toFixed(0);
    ctx.fillText(vals[i] + '  (' + pct + '%)', lx + 15, ly + 14);
    ly += 30;
  });
}


/**
 * Draws the confidence bar chart on #chart-conf canvas.
 * Green bars = Normal, Red bars = Pneumonia.
 * @param {Array} hist - array of scan history records (most recent last)
 */
function drawBars(hist) {
  const canvas = document.getElementById('chart-conf');
  const box    = document.getElementById('bar-box');
  if (!canvas || !box) return;

  const W   = box.offsetWidth  || 400;
  const H   = box.offsetHeight || 200;
  const dpr = window.devicePixelRatio || 1;

  canvas.width  = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  // No data state
  if (!hist || hist.length === 0) {
    ctx.fillStyle = '#5a7a75';
    ctx.font = '13px Plus Jakarta Sans, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No scans yet — run your first scan!', W / 2, H / 2);
    return;
  }

  const padL = 38, padR = 12, padT = 22, padB = 34;
  const cW = W - padL - padR;
  const cH = H - padT - padB;

  // Y-axis grid lines (0%, 25%, 50%, 75%, 100%)
  [0, 25, 50, 75, 100].forEach(pct => {
    const y = padT + cH * (1 - pct / 100);
    ctx.strokeStyle = 'rgba(26,74,66,0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + cW, y);
    ctx.stroke();
    ctx.fillStyle = '#5a7a75';
    ctx.font = '9px Plus Jakarta Sans, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(pct + '%', padL - 5, y + 3);
  });

  // Draw bars
  const n    = hist.length;
  const slot = cW / n;
  const barW = Math.max(4, Math.min(24, slot * 0.6));

  hist.forEach((r, i) => {
    const v    = Math.round((r.confidence || 0) * 100);
    const barH = (v / 100) * cH;
    const x    = padL + i * slot + (slot - barW) / 2;
    const y    = padT + cH - barH;
    const rad  = Math.min(3, barW / 2);

    ctx.fillStyle = r.is_pneumonia
      ? 'rgba(224,80,80,0.75)'
      : 'rgba(34,166,122,0.75)';

    // Rounded-top rectangle
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.lineTo(x + barW - rad, y);
    ctx.quadraticCurveTo(x + barW, y, x + barW, y + rad);
    ctx.lineTo(x + barW, y + barH);
    ctx.lineTo(x, y + barH);
    ctx.lineTo(x, y + rad);
    ctx.quadraticCurveTo(x, y, x + rad, y);
    ctx.closePath();
    ctx.fill();

    // X-axis scan label
    ctx.fillStyle = '#5a7a75';
    ctx.font = '8px Plus Jakarta Sans, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('#' + (i + 1), x + barW / 2, H - padB + 13);
  });

  // Top-right legend: ■ Normal  ■ Pneumonia
  ctx.fillStyle = 'rgba(34,166,122,0.75)';
  ctx.fillRect(padL, padT - 14, 10, 8);
  ctx.fillStyle = '#5a7a75';
  ctx.font = '9px Plus Jakarta Sans, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Normal', padL + 14, padT - 6);

  ctx.fillStyle = 'rgba(224,80,80,0.75)';
  ctx.fillRect(padL + 72, padT - 14, 10, 8);
  ctx.fillStyle = '#5a7a75';
  ctx.fillText('Pneumonia', padL + 86, padT - 6);
}
