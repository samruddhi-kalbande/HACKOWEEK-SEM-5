// ─── Tab Switching ─────────────────────────────────────────────────────────
const navButtons = document.querySelectorAll('.nav-btn');
const tabPanes = document.querySelectorAll('.tab-content');

navButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.getAttribute('data-tab');
    navButtons.forEach(b => b.classList.remove('active'));
    tabPanes.forEach(p => p.classList.remove('active'));

    btn.classList.add('active');
    const pane = document.getElementById(`pane-${target}`);
    if (pane) pane.classList.add('active');
  });
});

// ─── 1. Pipeline Evaluation & Metrics ──────────────────────────────────────
const elTestMetrics = document.getElementById('pipeline-test-metrics');
const elCvMetrics = document.getElementById('pipeline-cv-metrics');
const elConfusionMatrix = document.getElementById('confusion-matrix-box');
const canvasRoc = document.getElementById('canvas-roc');
const ctxRoc = canvasRoc.getContext('2d');
const elRocAucBadge = document.getElementById('roc-auc-badge');

async function loadEvaluation() {
  try {
    const res = await fetch('/api/pipeline/evaluate');
    const data = await res.json();
    if (data.success) {
      const ev = data.evaluation;
      renderEvaluation(ev);
    }
  } catch (err) {
    console.error('Failed to load evaluation:', err);
  }
}

function renderEvaluation(ev) {
  const tm = ev.test_metrics;
  const cv = ev.cross_validation;
  const cm = ev.confusion_matrix;

  // Test metrics
  elTestMetrics.innerHTML = `
    <div style="background: rgba(255, 255, 255, 0.03); padding: 0.85rem; border-radius: 8px;">
      <span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Accuracy</span>
      <div style="font-size: 1.5rem; font-weight: 700; color: #5eead4;">${tm.accuracy}%</div>
    </div>
    <div style="background: rgba(255, 255, 255, 0.03); padding: 0.85rem; border-radius: 8px;">
      <span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Precision</span>
      <div style="font-size: 1.5rem; font-weight: 700; color: #38bdf8;">${tm.precision}%</div>
    </div>
    <div style="background: rgba(255, 255, 255, 0.03); padding: 0.85rem; border-radius: 8px;">
      <span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Recall (Sensitivity)</span>
      <div style="font-size: 1.5rem; font-weight: 700; color: #a855f7;">${tm.recall}%</div>
    </div>
    <div style="background: rgba(255, 255, 255, 0.03); padding: 0.85rem; border-radius: 8px;">
      <span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">F1-Score</span>
      <div style="font-size: 1.5rem; font-weight: 700; color: #fbbf24;">${tm.f1_score}</div>
    </div>
  `;

  // CV metrics
  elCvMetrics.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
      <span style="font-size: 0.85rem; color: var(--text-secondary);">Mean Stratified CV Accuracy:</span>
      <strong style="font-size: 1.25rem; color: #34d399;">${cv.mean_accuracy}% (&plusmn;${cv.std_dev}%)</strong>
    </div>
    <div style="font-size: 0.82rem; color: var(--text-muted);">
      Fold Scores: ${cv.folds.map((f, i) => `Fold ${i+1}: <strong>${f}%</strong>`).join(' &bull; ')}
    </div>
  `;

  // Confusion Matrix
  elConfusionMatrix.innerHTML = `
    <table style="width: 100%; text-align: center; border-collapse: collapse; font-size: 0.88rem;">
      <tr>
        <th></th>
        <th style="padding: 8px; color: #5eead4;">Predicted: Standard</th>
        <th style="padding: 8px; color: #5eead4;">Predicted: Distinction</th>
      </tr>
      <tr>
        <td style="font-weight: 600; text-align: left; padding: 8px; color: #5eead4;">Actual: Standard</td>
        <td style="background: rgba(16, 185, 129, 0.15); color: #34d399; font-weight: 700; padding: 12px; border: 1px solid rgba(255, 255, 255, 0.05);">TN: ${cm.true_negatives}</td>
        <td style="background: rgba(244, 63, 94, 0.15); color: #fb7185; font-weight: 700; padding: 12px; border: 1px solid rgba(255, 255, 255, 0.05);">FP: ${cm.false_positives}</td>
      </tr>
      <tr>
        <td style="font-weight: 600; text-align: left; padding: 8px; color: #5eead4;">Actual: Distinction</td>
        <td style="background: rgba(244, 63, 94, 0.15); color: #fb7185; font-weight: 700; padding: 12px; border: 1px solid rgba(255, 255, 255, 0.05);">FN: ${cm.false_negatives}</td>
        <td style="background: rgba(16, 185, 129, 0.15); color: #34d399; font-weight: 700; padding: 12px; border: 1px solid rgba(255, 255, 255, 0.05);">TP: ${cm.true_positives}</td>
      </tr>
    </table>
  `;

  // Render ROC Curve
  elRocAucBadge.textContent = `AUC = ${ev.roc_curve.auc_score.toFixed(3)}`;
  drawRocCurve(ev.roc_curve);
}

function drawRocCurve(roc) {
  const w = canvasRoc.width;
  const h = canvasRoc.height;
  ctxRoc.clearRect(0, 0, w, h);

  const pad = 35;

  // Grid
  ctxRoc.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  ctxRoc.lineWidth = 1;
  ctxRoc.beginPath();
  ctxRoc.moveTo(pad, pad);
  ctxRoc.lineTo(pad, h - pad);
  ctxRoc.lineTo(w - pad, h - pad);
  ctxRoc.stroke();

  // Diagonal reference
  ctxRoc.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctxRoc.setLineDash([4, 4]);
  ctxRoc.beginPath();
  ctxRoc.moveTo(pad, h - pad);
  ctxRoc.lineTo(w - pad, pad);
  ctxRoc.stroke();
  ctxRoc.setLineDash([]);

  // ROC Curve
  ctxRoc.strokeStyle = '#14b8a6';
  ctxRoc.lineWidth = 2.5;
  ctxRoc.beginPath();

  const fpr = roc.fpr;
  const tpr = roc.tpr;

  fpr.forEach((fp, i) => {
    const tp = tpr[i];
    const x = pad + fp * (w - 2 * pad);
    const y = (h - pad) - tp * (h - 2 * pad);

    if (i === 0) ctxRoc.moveTo(x, y);
    else ctxRoc.lineTo(x, y);
  });
  ctxRoc.stroke();
}

// ─── 2. Clustering Suite ───────────────────────────────────────────────────
const elSelectAlgo = document.getElementById('select-algo');
const elGroupK = document.getElementById('group-k');
const elGroupDbscan = document.getElementById('group-dbscan');
const elInputK = document.getElementById('input-k');
const elInputEps = document.getElementById('input-eps');
const elInputMinSamples = document.getElementById('input-min-samples');
const elBtnRunClustering = document.getElementById('btn-run-clustering');
const canvasClusters = document.getElementById('canvas-clusters');
const ctxClusters = canvasClusters.getContext('2d');
const elClusterLegend = document.getElementById('cluster-legend');
const elSilhouetteBadge = document.getElementById('silhouette-badge');
const elClusterProfiles = document.getElementById('cluster-profiles-container');

const CLUSTER_COLORS = ['#14b8a6', '#6366f1', '#f59e0b', '#ec4899', '#3b82f6', '#10b981'];

elSelectAlgo.addEventListener('change', () => {
  if (elSelectAlgo.value === 'dbscan') {
    elGroupK.style.display = 'none';
    elGroupDbscan.style.display = 'block';
  } else {
    elGroupK.style.display = 'block';
    elGroupDbscan.style.display = 'none';
  }
});

async function executeClustering() {
  const payload = {
    algorithm: elSelectAlgo.value,
    k: parseInt(elInputK.value) || 3,
    eps: parseFloat(elInputEps.value) || 1.8,
    min_samples: parseInt(elInputMinSamples.value) || 2
  };

  elBtnRunClustering.textContent = 'Clustering...';

  try {
    const res = await fetch('/api/clustering/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    elBtnRunClustering.textContent = 'Run Clustering';

    if (data.success) {
      renderClustering(data.clustering);
    }
  } catch (err) {
    elBtnRunClustering.textContent = 'Run Clustering';
    console.error('Clustering error:', err);
  }
}

function renderClustering(cl) {
  elSilhouetteBadge.textContent = cl.silhouette_score > 0 ? `Silhouette: ${cl.silhouette_score}` : 'Density Partition';

  // 1. Render Scatter Plot on Canvas
  drawClusterScatter(cl.points);

  // 2. Render Legend
  const uniqueClusters = Array.from(new Set(cl.points.map(p => p.cluster))).sort();
  elClusterLegend.innerHTML = uniqueClusters.map(cid => {
    const color = cid === -1 ? '#94a3b8' : CLUSTER_COLORS[cid % CLUSTER_COLORS.length];
    const name = cid === -1 ? 'Noise / Outlier' : `Cluster ${cid}`;
    return `<span style="display: flex; align-items: center; gap: 4px;"><span style="width: 10px; height: 10px; border-radius: 50%; background: ${color};"></span>${name}</span>`;
  }).join('');

  // 3. Render Profiles
  elClusterProfiles.innerHTML = Object.entries(cl.cluster_profiles).map(([cid, prof]) => {
    const color = cid === "-1" ? '#94a3b8' : CLUSTER_COLORS[parseInt(cid) % CLUSTER_COLORS.length];
    return `
      <div class="cluster-card" style="border-left: 3px solid ${color};">
        <div class="cluster-card-title" style="color: ${color};">${prof.name} (${prof.count} students)</div>
        <div style="font-size: 0.84rem; color: var(--text-secondary); display: flex; flex-direction: column; gap: 0.25rem;">
          <div>Mean CGPA: <strong style="color: #f8fafc;">${prof.avg_cgpa}</strong></div>
          <div>Mean Attendance: <strong style="color: #f8fafc;">${prof.avg_attendance}%</strong></div>
          <div>Study Hours: <strong style="color: #f8fafc;">${prof.avg_study_hours} hrs/wk</strong></div>
        </div>
      </div>
    `;
  }).join('');
}

function drawClusterScatter(points) {
  const w = canvasClusters.width;
  const h = canvasClusters.height;
  ctxClusters.clearRect(0, 0, w, h);

  if (!points || points.length === 0) return;

  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const pad = 30;

  // Draw axes
  ctxClusters.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctxClusters.lineWidth = 1;
  ctxClusters.beginPath();
  ctxClusters.moveTo(pad, h / 2);
  ctxClusters.lineTo(w - pad, h / 2);
  ctxClusters.moveTo(w / 2, pad);
  ctxClusters.lineTo(w / 2, h - pad);
  ctxClusters.stroke();

  // Draw Points
  points.forEach(p => {
    const x = pad + ((p.x - minX) / (maxX - minX || 1)) * (w - 2 * pad);
    const y = (h - pad) - ((p.y - minY) / (maxY - minY || 1)) * (h - 2 * pad);
    const color = p.cluster === -1 ? '#94a3b8' : CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length];

    ctxClusters.fillStyle = color;
    ctxClusters.beginPath();
    ctxClusters.arc(x, y, 6, 0, 2 * Math.PI);
    ctxClusters.fill();

    ctxClusters.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctxClusters.lineWidth = 1;
    ctxClusters.stroke();
  });
}

elBtnRunClustering.addEventListener('click', executeClustering);

// ─── Initialization ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadEvaluation();
  executeClustering();
});
