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

// ─── Color Palettes ───────────────────────────────────────────────────────
const COLORS = {
  emerald: '#10b981',
  sky: '#38bdf8',
  violet: '#8b5cf6',
  amber: '#f59e0b',
  rose: '#f43f5e',
  orange: '#f97316',
  teal: '#14b8a6',
  pink: '#ec4899'
};
const MODEL_COLORS = [COLORS.emerald, COLORS.violet, COLORS.sky, COLORS.amber, COLORS.rose, COLORS.orange];

// ═══════════════════════════════════════════════════════════════════════════════
// 1. ENSEMBLE METHODS
// ═══════════════════════════════════════════════════════════════════════════════

const elModelsGrid = document.getElementById('ensemble-models-grid');
const canvasFeatureImp = document.getElementById('canvas-feature-importance');
const ctxFeatureImp = canvasFeatureImp.getContext('2d');

let ensembleData = null;

async function loadEnsemble() {
  try {
    const res = await fetch('/api/ensemble/compare');
    const json = await res.json();
    if (json.success) {
      ensembleData = json.data;
      renderEnsembleCards(json.data);
      renderFeatureImportance(json.data);
    }
  } catch (err) {
    elModelsGrid.innerHTML = `<div style="color: var(--accent-rose); padding: 1rem;">Error loading ensemble data: ${err.message}</div>`;
  }
}

function renderEnsembleCards(data) {
  const models = data.models;
  const keys = Object.keys(models);

  elModelsGrid.innerHTML = keys.map((key, idx) => {
    const m = models[key];
    const color = MODEL_COLORS[idx % MODEL_COLORS.length];
    return `
      <div class="model-card type-${m.type}">
        <div class="model-card-name" style="color: ${color};">${m.name}</div>
        <span class="model-card-type ${m.type}">${m.type}</span>
        <div class="metric-row"><span class="metric-label">Accuracy</span><span class="metric-value" style="color: #6ee7b7;">${m.accuracy}%</span></div>
        <div class="metric-row"><span class="metric-label">Precision</span><span class="metric-value" style="color: #38bdf8;">${m.precision}%</span></div>
        <div class="metric-row"><span class="metric-label">Recall</span><span class="metric-value" style="color: #c4b5fd;">${m.recall}%</span></div>
        <div class="metric-row"><span class="metric-label">F1 Score</span><span class="metric-value" style="color: #fbbf24;">${m.f1_score}</span></div>
        <div class="metric-row"><span class="metric-label">ROC AUC</span><span class="metric-value" style="color: #fb7185;">${m.roc_auc}</span></div>
        <div class="metric-row"><span class="metric-label">CV Mean</span><span class="metric-value" style="color: #6ee7b7;">${m.cv_mean}% ±${m.cv_std}%</span></div>
        <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 0.5rem;">
          Folds: ${m.cv_folds.map((f, i) => `F${i+1}: ${f}%`).join(' · ')}
        </div>
      </div>
    `;
  }).join('');
}

function renderFeatureImportance(data) {
  // Pick the model with the highest accuracy that has feature importance
  const models = data.models;
  let bestKey = null;
  let bestAcc = -1;

  for (const [key, m] of Object.entries(models)) {
    if (m.feature_importance && m.accuracy > bestAcc) {
      bestAcc = m.accuracy;
      bestKey = key;
    }
  }

  if (!bestKey) {
    ctxFeatureImp.fillStyle = '#64748b';
    ctxFeatureImp.font = '13px Inter';
    ctxFeatureImp.fillText('No feature importance data available.', 20, 140);
    return;
  }

  const imp = models[bestKey].feature_importance;
  const maxVal = Math.max(...imp);
  const w = canvasFeatureImp.width;
  const h = canvasFeatureImp.height;
  const pad = { top: 30, right: 20, bottom: 30, left: 40 };
  const barH = Math.min(20, (h - pad.top - pad.bottom) / imp.length - 4);
  const chartW = w - pad.left - pad.right;

  ctxFeatureImp.clearRect(0, 0, w, h);

  // Title
  ctxFeatureImp.fillStyle = '#6ee7b7';
  ctxFeatureImp.font = 'bold 12px Outfit';
  ctxFeatureImp.fillText(`Feature Importance — ${models[bestKey].name}`, pad.left, 18);

  imp.forEach((val, i) => {
    const y = pad.top + i * (barH + 6);
    const barW = (val / maxVal) * (chartW - 60);

    // Bar
    const grad = ctxFeatureImp.createLinearGradient(pad.left, y, pad.left + barW, y);
    grad.addColorStop(0, 'rgba(16, 185, 129, 0.7)');
    grad.addColorStop(1, 'rgba(139, 92, 246, 0.7)');
    ctxFeatureImp.fillStyle = grad;
    ctxFeatureImp.beginPath();
    ctxFeatureImp.roundRect(pad.left, y, barW, barH, 4);
    ctxFeatureImp.fill();

    // Index Label
    ctxFeatureImp.fillStyle = '#64748b';
    ctxFeatureImp.font = '11px JetBrains Mono';
    ctxFeatureImp.textAlign = 'right';
    ctxFeatureImp.fillText(`F${i}`, pad.left - 6, y + barH / 2 + 4);

    // Value
    ctxFeatureImp.fillStyle = '#94a3b8';
    ctxFeatureImp.textAlign = 'left';
    ctxFeatureImp.fillText(val.toFixed(4), pad.left + barW + 6, y + barH / 2 + 4);
  });
  ctxFeatureImp.textAlign = 'left';
}


// ═══════════════════════════════════════════════════════════════════════════════
// 2. BIAS-VARIANCE TRADE-OFF
// ═══════════════════════════════════════════════════════════════════════════════

const canvasBV = document.getElementById('canvas-bias-variance');
const ctxBV = canvasBV.getContext('2d');
const elBvOptimal = document.getElementById('bv-optimal-badge');
const elBvInterp = document.getElementById('bv-interpretation');
const elBvTableBody = document.getElementById('bv-table-body');

async function loadBiasVariance() {
  try {
    const res = await fetch('/api/bias-variance');
    const json = await res.json();
    if (json.success) renderBiasVariance(json.data);
  } catch (err) {
    console.error('Bias-variance error:', err);
  }
}

function renderBiasVariance(data) {
  const pts = data.depth_analysis;
  const opt = data.optimal_depth;

  // Draw chart
  drawBiasVarianceChart(pts, opt);

  // Optimal badge
  elBvOptimal.innerHTML = `
    <div class="stat-box" style="text-align: center; border: 1px solid rgba(16, 185, 129, 0.3);">
      <div class="stat-label">Optimal Tree Depth</div>
      <div class="stat-value" style="color: #6ee7b7; font-size: 2rem;">max_depth = ${opt}</div>
    </div>
  `;

  // Interpretation cards
  const interp = data.interpretation;
  const icons = { low_depth: '🟡', optimal_depth: '🟢', high_depth: '🔴' };
  const colors = { low_depth: '#fbbf24', optimal_depth: '#6ee7b7', high_depth: '#fb7185' };

  elBvInterp.innerHTML = Object.entries(interp).map(([k, v]) => `
    <div class="info-card" style="border-left: 3px solid ${colors[k]};">
      <div class="info-card-title" style="color: ${colors[k]};">${icons[k]} ${k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</div>
      <p>${v}</p>
    </div>
  `).join('');

  // Table
  elBvTableBody.innerHTML = pts.map(p => {
    const isOpt = p.depth === opt;
    const rowBg = isOpt ? 'rgba(16, 185, 129, 0.08)' : 'transparent';
    return `
      <tr style="border-bottom: 1px solid rgba(255,255,255,0.04); background: ${rowBg};">
        <td style="padding: 8px; font-weight: 600; color: ${isOpt ? '#6ee7b7' : '#f1f5f9'};">${p.depth}${isOpt ? ' ⭐' : ''}</td>
        <td style="padding: 8px; text-align: center; color: #38bdf8; font-family: 'JetBrains Mono'; font-size: 0.82rem;">${p.train_accuracy}%</td>
        <td style="padding: 8px; text-align: center; color: #f59e0b; font-family: 'JetBrains Mono'; font-size: 0.82rem;">${p.test_accuracy}%</td>
        <td style="padding: 8px; text-align: center; color: ${p.gap > 15 ? '#f43f5e' : p.gap > 5 ? '#fbbf24' : '#6ee7b7'}; font-family: 'JetBrains Mono'; font-size: 0.82rem;">${p.gap}%</td>
        <td style="padding: 8px; text-align: center; color: var(--text-muted); font-family: 'JetBrains Mono'; font-size: 0.82rem;">±${p.test_std}%</td>
      </tr>
    `;
  }).join('');
}

function drawBiasVarianceChart(pts, optDepth) {
  const w = canvasBV.width;
  const h = canvasBV.height;
  ctxBV.clearRect(0, 0, w, h);

  const pad = { top: 25, right: 25, bottom: 40, left: 50 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;

  const depths = pts.map(p => p.depth);
  const maxDepth = Math.max(...depths);
  const minDepth = Math.min(...depths);

  const allAcc = pts.flatMap(p => [p.train_accuracy, p.test_accuracy]);
  const minAcc = Math.floor(Math.min(...allAcc) / 5) * 5;
  const maxAcc = Math.ceil(Math.max(...allAcc) / 5) * 5;

  const xScale = d => pad.left + ((d - minDepth) / (maxDepth - minDepth)) * chartW;
  const yScale = a => pad.top + ((maxAcc - a) / (maxAcc - minAcc)) * chartH;

  // Grid
  ctxBV.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctxBV.lineWidth = 1;
  for (let a = minAcc; a <= maxAcc; a += 5) {
    const y = yScale(a);
    ctxBV.beginPath();
    ctxBV.moveTo(pad.left, y);
    ctxBV.lineTo(w - pad.right, y);
    ctxBV.stroke();

    ctxBV.fillStyle = '#64748b';
    ctxBV.font = '10px JetBrains Mono';
    ctxBV.textAlign = 'right';
    ctxBV.fillText(`${a}%`, pad.left - 6, y + 4);
  }

  // X labels
  ctxBV.textAlign = 'center';
  pts.forEach(p => {
    ctxBV.fillStyle = p.depth === optDepth ? '#6ee7b7' : '#64748b';
    ctxBV.font = p.depth === optDepth ? 'bold 11px JetBrains Mono' : '10px JetBrains Mono';
    ctxBV.fillText(p.depth, xScale(p.depth), h - pad.bottom + 18);
  });
  ctxBV.fillStyle = '#64748b';
  ctxBV.font = '11px Inter';
  ctxBV.fillText('Tree Depth (max_depth)', w / 2, h - 4);

  // Optimal vertical line
  ctxBV.strokeStyle = 'rgba(16, 185, 129, 0.3)';
  ctxBV.setLineDash([4, 4]);
  ctxBV.beginPath();
  ctxBV.moveTo(xScale(optDepth), pad.top);
  ctxBV.lineTo(xScale(optDepth), h - pad.bottom);
  ctxBV.stroke();
  ctxBV.setLineDash([]);

  // Train accuracy line
  drawLine(ctxBV, pts.map(p => [xScale(p.depth), yScale(p.train_accuracy)]), '#38bdf8', 2.5);
  // Test accuracy line
  drawLine(ctxBV, pts.map(p => [xScale(p.depth), yScale(p.test_accuracy)]), '#f59e0b', 2.5);

  // Points
  pts.forEach(p => {
    drawDot(ctxBV, xScale(p.depth), yScale(p.train_accuracy), '#38bdf8', p.depth === optDepth ? 5 : 3.5);
    drawDot(ctxBV, xScale(p.depth), yScale(p.test_accuracy), '#f59e0b', p.depth === optDepth ? 5 : 3.5);
  });

  // Legend
  ctxBV.font = '11px Inter';
  const lx = pad.left + 10;
  const ly = pad.top + 10;
  ctxBV.fillStyle = '#38bdf8';
  ctxBV.fillRect(lx, ly, 14, 3);
  ctxBV.fillText('Train', lx + 20, ly + 5);
  ctxBV.fillStyle = '#f59e0b';
  ctxBV.fillRect(lx + 70, ly, 14, 3);
  ctxBV.fillText('Test', lx + 90, ly + 5);
}


// ═══════════════════════════════════════════════════════════════════════════════
// 3. OVERFITTING & UNDERFITTING
// ═══════════════════════════════════════════════════════════════════════════════

const elLCGrid = document.getElementById('learning-curves-grid');
const elOfCauses = document.getElementById('of-causes');
const elOfPrevention = document.getElementById('of-prevention');

async function loadOverfitting() {
  try {
    const res = await fetch('/api/overfitting');
    const json = await res.json();
    if (json.success) renderOverfitting(json.data);
  } catch (err) {
    console.error('Overfitting error:', err);
  }
}

function renderOverfitting(data) {
  const scenarios = data.scenarios;
  const scColors = { underfit: '#fbbf24', good_fit: '#10b981', overfit: '#f43f5e' };
  const scIcons = { underfit: '🟡', good_fit: '🟢', overfit: '🔴' };

  // Build canvas for each scenario
  elLCGrid.innerHTML = Object.entries(scenarios).map(([key, sc]) => `
    <div class="canvas-wrapper" style="position: relative;">
      <div class="chart-title" style="color: ${scColors[key]};">${scIcons[key]} ${sc.label}</div>
      <canvas id="canvas-lc-${key}" width="400" height="230"></canvas>
      <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
        Final train: <strong style="color: #38bdf8;">${sc.train_mean[sc.train_mean.length-1]}%</strong> &nbsp;|&nbsp;
        Final val: <strong style="color: #f59e0b;">${sc.val_mean[sc.val_mean.length-1]}%</strong>
      </div>
    </div>
  `).join('');

  // Draw each learning curve
  Object.entries(scenarios).forEach(([key, sc]) => {
    const canvas = document.getElementById(`canvas-lc-${key}`);
    drawLearningCurve(canvas, sc, scColors[key]);
  });

  // Causes
  elOfCauses.innerHTML = data.causes.map(c => {
    const isUnder = c.startsWith('Underfitting');
    const isOver = c.startsWith('Overfitting');
    const color = isUnder ? '#fbbf24' : isOver ? '#fb7185' : '#6ee7b7';
    return `<div class="info-card" style="border-left: 3px solid ${color};"><p>${c}</p></div>`;
  }).join('');

  // Prevention
  elOfPrevention.innerHTML = data.prevention.map(p => `
    <div class="info-card" style="border-left: 3px solid var(--accent-sky);"><p>${p}</p></div>
  `).join('');
}

function drawLearningCurve(canvas, sc, accentColor) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const pad = { top: 15, right: 15, bottom: 35, left: 45 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;

  const sizes = sc.train_sizes;
  const maxSize = Math.max(...sizes);
  const minSize = Math.min(...sizes);

  const allAcc = [...sc.train_mean, ...sc.val_mean];
  const minAcc = Math.max(0, Math.floor(Math.min(...allAcc) / 5) * 5 - 5);
  const maxAcc = Math.min(100, Math.ceil(Math.max(...allAcc) / 5) * 5 + 5);

  const xScale = s => pad.left + ((s - minSize) / (maxSize - minSize || 1)) * chartW;
  const yScale = a => pad.top + ((maxAcc - a) / (maxAcc - minAcc || 1)) * chartH;

  // Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let a = minAcc; a <= maxAcc; a += 10) {
    const y = yScale(a);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = '#64748b';
    ctx.font = '9px JetBrains Mono';
    ctx.textAlign = 'right';
    ctx.fillText(`${a}%`, pad.left - 5, y + 3);
  }

  // X label
  ctx.fillStyle = '#64748b';
  ctx.font = '10px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Training Set Size', w / 2, h - 4);

  // Train scores
  const trainPts = sizes.map((s, i) => [xScale(s), yScale(sc.train_mean[i])]);
  const valPts = sizes.map((s, i) => [xScale(s), yScale(sc.val_mean[i])]);

  // Confidence band for validation
  ctx.fillStyle = 'rgba(249, 158, 11, 0.08)';
  ctx.beginPath();
  sizes.forEach((s, i) => {
    const x = xScale(s);
    const y1 = yScale(sc.val_mean[i] + sc.val_std[i]);
    if (i === 0) ctx.moveTo(x, y1); else ctx.lineTo(x, y1);
  });
  for (let i = sizes.length - 1; i >= 0; i--) {
    const x = xScale(sizes[i]);
    const y2 = yScale(sc.val_mean[i] - sc.val_std[i]);
    ctx.lineTo(x, y2);
  }
  ctx.closePath();
  ctx.fill();

  drawLine(ctx, trainPts, '#38bdf8', 2);
  drawLine(ctx, valPts, '#f59e0b', 2);

  trainPts.forEach(([x, y]) => drawDot(ctx, x, y, '#38bdf8', 3));
  valPts.forEach(([x, y]) => drawDot(ctx, x, y, '#f59e0b', 3));

  // Legend
  ctx.font = '10px Inter';
  ctx.textAlign = 'left';
  const lx = pad.left + 8;
  ctx.fillStyle = '#38bdf8';
  ctx.fillRect(lx, pad.top + 2, 12, 2.5);
  ctx.fillText('Train', lx + 16, pad.top + 7);
  ctx.fillStyle = '#f59e0b';
  ctx.fillRect(lx + 60, pad.top + 2, 12, 2.5);
  ctx.fillText('Val', lx + 76, pad.top + 7);
}


// ═══════════════════════════════════════════════════════════════════════════════
// 4. REGULARIZATION (L1/L2)
// ═══════════════════════════════════════════════════════════════════════════════

const canvasCSweep = document.getElementById('canvas-c-sweep');
const ctxCSweep = canvasCSweep.getContext('2d');
const elCoeffContainer = document.getElementById('coeff-comparison-container');
const elRegTableBody = document.getElementById('reg-table-body');

async function loadRegularization() {
  try {
    const res = await fetch('/api/regularization');
    const json = await res.json();
    if (json.success) renderRegularization(json.data);
  } catch (err) {
    console.error('Regularization error:', err);
  }
}

function renderRegularization(data) {
  drawCSweepChart(data);
  renderCoefficientBars(data);
  renderRegTable(data);
}

function drawCSweepChart(data) {
  const w = canvasCSweep.width;
  const h = canvasCSweep.height;
  ctxCSweep.clearRect(0, 0, w, h);

  const pad = { top: 25, right: 25, bottom: 40, left: 50 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;

  const cVals = data.c_values;
  const logC = cVals.map(c => Math.log10(c));
  const minLogC = Math.min(...logC);
  const maxLogC = Math.max(...logC);

  const l1Acc = data.c_sweep.l1.map(d => d.test_accuracy);
  const l2Acc = data.c_sweep.l2.map(d => d.test_accuracy);
  const allAcc = [...l1Acc, ...l2Acc];
  const minAcc = Math.floor(Math.min(...allAcc) / 5) * 5;
  const maxAcc = Math.ceil(Math.max(...allAcc) / 5) * 5 + 5;

  const xScale = lc => pad.left + ((lc - minLogC) / (maxLogC - minLogC || 1)) * chartW;
  const yScale = a => pad.top + ((maxAcc - a) / (maxAcc - minAcc || 1)) * chartH;

  // Grid
  ctxCSweep.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctxCSweep.lineWidth = 1;
  for (let a = minAcc; a <= maxAcc; a += 5) {
    const y = yScale(a);
    ctxCSweep.beginPath();
    ctxCSweep.moveTo(pad.left, y);
    ctxCSweep.lineTo(w - pad.right, y);
    ctxCSweep.stroke();
    ctxCSweep.fillStyle = '#64748b';
    ctxCSweep.font = '10px JetBrains Mono';
    ctxCSweep.textAlign = 'right';
    ctxCSweep.fillText(`${a}%`, pad.left - 6, y + 4);
  }

  // X labels
  ctxCSweep.textAlign = 'center';
  cVals.forEach(c => {
    ctxCSweep.fillStyle = '#64748b';
    ctxCSweep.font = '10px JetBrains Mono';
    ctxCSweep.fillText(c, xScale(Math.log10(c)), h - pad.bottom + 16);
  });
  ctxCSweep.fillStyle = '#64748b';
  ctxCSweep.font = '11px Inter';
  ctxCSweep.fillText('C (Inverse Regularization Strength)', w / 2, h - 2);

  // L1 line
  const l1Pts = logC.map((lc, i) => [xScale(lc), yScale(l1Acc[i])]);
  drawLine(ctxCSweep, l1Pts, '#fbbf24', 2.5);
  l1Pts.forEach(([x, y]) => drawDot(ctxCSweep, x, y, '#fbbf24', 4));

  // L2 line
  const l2Pts = logC.map((lc, i) => [xScale(lc), yScale(l2Acc[i])]);
  drawLine(ctxCSweep, l2Pts, '#38bdf8', 2.5);
  l2Pts.forEach(([x, y]) => drawDot(ctxCSweep, x, y, '#38bdf8', 4));

  // Legend
  ctxCSweep.textAlign = 'left';
  ctxCSweep.font = '11px Inter';
  const lx = pad.left + 10;
  ctxCSweep.fillStyle = '#fbbf24';
  ctxCSweep.fillRect(lx, pad.top + 8, 14, 3);
  ctxCSweep.fillText('L1 (Lasso)', lx + 20, pad.top + 13);
  ctxCSweep.fillStyle = '#38bdf8';
  ctxCSweep.fillRect(lx + 100, pad.top + 8, 14, 3);
  ctxCSweep.fillText('L2 (Ridge)', lx + 120, pad.top + 13);
}

function renderCoefficientBars(data) {
  const coef = data.coefficient_comparison;
  const noneCoef = coef.none || [];
  const l1Coef = coef.l1 || [];
  const l2Coef = coef.l2 || [];

  const maxAbs = Math.max(
    ...noneCoef.map(Math.abs),
    ...l1Coef.map(Math.abs),
    ...l2Coef.map(Math.abs),
    0.001
  );

  let html = '';
  const nFeatures = Math.max(noneCoef.length, l1Coef.length, l2Coef.length);

  for (let i = 0; i < nFeatures; i++) {
    const none = noneCoef[i] || 0;
    const l1 = l1Coef[i] || 0;
    const l2 = l2Coef[i] || 0;

    html += buildCoeffBarRow(`F${i}`, [
      { val: none, color: '#94a3b8', label: 'None' },
      { val: l1, color: '#fbbf24', label: 'L1' },
      { val: l2, color: '#38bdf8', label: 'L2' }
    ], maxAbs);
  }

  // Legend at top
  elCoeffContainer.innerHTML = `
    <div style="display: flex; gap: 1rem; font-size: 0.75rem; margin-bottom: 0.5rem;">
      <span style="display: flex; align-items: center; gap: 4px;"><span style="width:10px;height:4px;background:#94a3b8;border-radius:2px;"></span>No Reg</span>
      <span style="display: flex; align-items: center; gap: 4px;"><span style="width:10px;height:4px;background:#fbbf24;border-radius:2px;"></span>L1</span>
      <span style="display: flex; align-items: center; gap: 4px;"><span style="width:10px;height:4px;background:#38bdf8;border-radius:2px;"></span>L2</span>
    </div>
    ${html}
  `;
}

function buildCoeffBarRow(label, entries, maxAbs) {
  const rows = entries.map(e => {
    const pct = Math.min(100, (Math.abs(e.val) / maxAbs) * 100);
    const isZero = Math.abs(e.val) < 0.0001;
    return `
      <div class="coeff-bar-row">
        <span class="coeff-bar-label" style="font-size: 0.65rem; color: ${e.color};">${e.label}</span>
        <div class="coeff-bar-track">
          <div class="coeff-bar-fill" style="width: ${pct}%; background: ${e.color}; opacity: ${isZero ? 0.15 : 0.65};"></div>
        </div>
        <span class="coeff-bar-value" style="color: ${isZero ? '#f43f5e' : '#94a3b8'}; font-size: 0.7rem;">${isZero ? '0 ✕' : e.val.toFixed(3)}</span>
      </div>
    `;
  }).join('');

  return `
    <div style="margin-bottom: 0.6rem;">
      <div style="font-size: 0.72rem; font-weight: 600; color: #6ee7b7; margin-bottom: 0.15rem; font-family: 'JetBrains Mono';">${label}</div>
      ${rows}
    </div>
  `;
}

function renderRegTable(data) {
  const l1 = data.c_sweep.l1;
  const l2 = data.c_sweep.l2;

  elRegTableBody.innerHTML = l1.map((l1d, i) => {
    const l2d = l2[i];
    return `
      <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
        <td style="padding: 8px; font-weight: 600; color: #fbbf24; font-family: 'JetBrains Mono'; font-size: 0.82rem;">${l1d.C}</td>
        <td style="padding: 8px; text-align: center; color: #6ee7b7; font-family: 'JetBrains Mono'; font-size: 0.82rem;">${l1d.train_accuracy}%</td>
        <td style="padding: 8px; text-align: center; color: #38bdf8; font-family: 'JetBrains Mono'; font-size: 0.82rem;">${l1d.test_accuracy}%</td>
        <td style="padding: 8px; text-align: center; font-family: 'JetBrains Mono'; font-size: 0.82rem;">
          <span style="color: ${l1d.n_features_active < l1d.total_features ? '#fbbf24' : '#6ee7b7'};">
            ${l1d.n_features_active} / ${l1d.total_features}
          </span>
        </td>
        <td style="padding: 8px; text-align: center; color: #6ee7b7; font-family: 'JetBrains Mono'; font-size: 0.82rem;">${l2d.train_accuracy}%</td>
        <td style="padding: 8px; text-align: center; color: #38bdf8; font-family: 'JetBrains Mono'; font-size: 0.82rem;">${l2d.test_accuracy}%</td>
      </tr>
    `;
  }).join('');
}


// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY DRAWING FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

function drawLine(ctx, points, color, width) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  points.forEach(([x, y], i) => {
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawDot(ctx, x, y, color, r) {
  // Glow
  ctx.fillStyle = color + '30';
  ctx.beginPath();
  ctx.arc(x, y, r + 3, 0, Math.PI * 2);
  ctx.fill();
  // Dot
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  // Border
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
  ctx.lineWidth = 1;
  ctx.stroke();
}


// ═══════════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  loadEnsemble();
  loadBiasVariance();
  loadOverfitting();
  loadRegularization();
});
