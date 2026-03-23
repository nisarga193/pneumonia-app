/* ============================================================
   PneumoScan AI — About Page Logic
   File: static/js/about.js
   ============================================================
   Handles the About page:
     - Fetches model metadata from /api/metadata
     - Fills the Performance Metrics grid:
         • Test Accuracy
         • Macro Recall (sensitivity across all 3 classes)
         • Bacterial Pneumonia Recall
         • Viral Pneumonia Recall
   The pipeline steps and architecture text are static HTML
   (in pages/about.html). Only the perf numbers are dynamic.
   Called by: showPage('about') in app.js
   ============================================================ */


/**
 * Main entry point for the About page.
 * Loads model performance metrics from the backend metadata endpoint.
 */
async function loadAbout() {
  try {
    const m  = await apiMetadata();
    const rc = m.test_recall_per_class || {};

    document.getElementById('pa-acc').textContent  =
      ((m.test_accuracy      || 0) * 100).toFixed(1) + '%';

    document.getElementById('pa-rec').textContent  =
      ((m.test_macro_recall  || 0) * 100).toFixed(1) + '%';

    document.getElementById('pa-brec').textContent =
      ((rc['Bacterial Pneumonia'] || 0) * 100).toFixed(1) + '%';

    document.getElementById('pa-vrec').textContent =
      ((rc['Viral Pneumonia']     || 0) * 100).toFixed(1) + '%';

  } catch (e) {
    // Metadata endpoint may not exist in all deployments — fail silently
    console.warn('Could not load model metadata:', e.message);
  }
}
