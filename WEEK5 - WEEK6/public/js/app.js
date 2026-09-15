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

// ─── 1. Vector Operations ──────────────────────────────────────────────────
const elVecU1 = document.getElementById('vec-u1');
const elVecU2 = document.getElementById('vec-u2');
const elVecV1 = document.getElementById('vec-v1');
const elVecV2 = document.getElementById('vec-v2');
const elBtnCalcVectors = document.getElementById('btn-calc-vectors');
const elVectorResultsBox = document.getElementById('vector-results-box');

async function calculateVectors() {
  const u = [parseFloat(elVecU1.value) || 0, parseFloat(elVecU2.value) || 0];
  const v = [parseFloat(elVecV1.value) || 0, parseFloat(elVecV2.value) || 0];

  try {
    const res = await fetch('/api/linalg/vector-ops', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ u, v })
    });
    const data = await res.json();
    if (data.success) {
      const r = data.result;
      elVectorResultsBox.innerHTML = `
        <div class="metric-pill">
          <span class="form-label">Dot Product (u &bull; v)</span>
          <span class="metric-pill-val">${r.dot_product}</span>
        </div>
        <div class="metric-pill">
          <span class="form-label">Cosine Similarity</span>
          <span class="metric-pill-val" style="color: #a855f7;">${r.cosine_similarity}</span>
        </div>
        <div class="metric-pill">
          <span class="form-label">Angle Between Vectors</span>
          <span class="metric-pill-val" style="color: #34d399;">${r.angle_degrees}&deg;</span>
        </div>
        <div class="metric-pill">
          <span class="form-label">Norm ||u|| & ||v||</span>
          <span class="metric-pill-val" style="font-size: 1rem; color: #fbbf24;">${r.norm_u} &bull; ${r.norm_v}</span>
        </div>
      `;
    }
  } catch (err) {
    console.error('Vector calc error:', err);
  }
}

elBtnCalcVectors.addEventListener('click', calculateVectors);

// ─── 2. 2D Matrix Transformation Canvas ────────────────────────────────────
const canvasMat = document.getElementById('canvas-matrix');
const ctxMat = canvasMat.getContext('2d');
const elMatA = document.getElementById('mat-a');
const elMatB = document.getElementById('mat-b');
const elMatC = document.getElementById('mat-c');
const elMatD = document.getElementById('mat-d');
const elMatrixMetricsBox = document.getElementById('matrix-metrics-box');

function renderMatrixCanvas() {
  const a = parseFloat(elMatA.value) || 0;
  const b = parseFloat(elMatB.value) || 0;
  const c = parseFloat(elMatC.value) || 0;
  const d = parseFloat(elMatD.value) || 0;

  const det = (a * d - b * c);
  elMatrixMetricsBox.innerHTML = `
    Determinant det(A) = <strong>${det.toFixed(2)}</strong> &bull; Area Scale Factor: <strong>${Math.abs(det).toFixed(2)}x</strong> &bull; ${Math.abs(det) > 1e-4 ? '<span style="color: #34d399;">Invertible</span>' : '<span style="color: #fb7185;">Singular (Collapse)</span>'}
  `;

  // Draw on Canvas
  const width = canvasMat.width;
  const height = canvasMat.height;
  const originX = width / 2;
  const originY = height / 2;
  const scale = 40; // Pixels per unit

  ctxMat.clearRect(0, 0, width, height);

  // Axes
  ctxMat.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctxMat.lineWidth = 1;
  ctxMat.beginPath();
  ctxMat.moveTo(0, originY);
  ctxMat.lineTo(width, originY);
  ctxMat.moveTo(originX, 0);
  ctxMat.lineTo(originX, height);
  ctxMat.stroke();

  // Draw Original Unit Square [0,0]->[1,0]->[1,1]->[0,1]
  ctxMat.strokeStyle = 'rgba(6, 182, 212, 0.4)';
  ctxMat.setLineDash([4, 4]);
  ctxMat.beginPath();
  ctxMat.moveTo(originX, originY);
  ctxMat.lineTo(originX + 1 * scale, originY);
  ctxMat.lineTo(originX + 1 * scale, originY - 1 * scale);
  ctxMat.lineTo(originX, originY - 1 * scale);
  ctxMat.closePath();
  ctxMat.stroke();
  ctxMat.setLineDash([]);

  // Transformed Unit Square
  // [0, 0] -> [0, 0]
  // [1, 0] -> [a, c]
  // [1, 1] -> [a + b, c + d]
  // [0, 1] -> [b, d]
  const p0 = [originX, originY];
  const p1 = [originX + a * scale, originY - c * scale];
  const p2 = [originX + (a + b) * scale, originY - (c + d) * scale];
  const p3 = [originX + b * scale, originY - d * scale];

  ctxMat.fillStyle = 'rgba(168, 85, 247, 0.25)';
  ctxMat.strokeStyle = '#a855f7';
  ctxMat.lineWidth = 2;
  ctxMat.beginPath();
  ctxMat.moveTo(p0[0], p0[1]);
  ctxMat.lineTo(p1[0], p1[1]);
  ctxMat.lineTo(p2[0], p2[1]);
  ctxMat.lineTo(p3[0], p3[1]);
  ctxMat.closePath();
  ctxMat.fill();
  ctxMat.stroke();
}

[elMatA, elMatB, elMatC, elMatD].forEach(input => {
  input.addEventListener('input', renderMatrixCanvas);
});

// ─── 3. Covariance Matrix & Eigen-Decomposition ────────────────────────────
async function loadEigenAnalysis() {
  try {
    const res = await fetch('/api/linalg/eigen-analysis');
    const data = await res.json();
    if (data.success) {
      const r = data.result;
      document.getElementById('cov-matrix-display').innerHTML = `
        <table style="width: 100%; text-align: right; border-collapse: collapse;">
          <tr><th></th><th style="padding: 4px 8px; color: #a5b4fc;">Math</th><th style="padding: 4px 8px; color: #a5b4fc;">Reading</th><th style="padding: 4px 8px; color: #a5b4fc;">Writing</th></tr>
          <tr><td style="font-weight: 600; color: #a5b4fc;">Math</td>${r.covariance_matrix[0].map(v => `<td style="padding: 4px 8px;">${v}</td>`).join('')}</tr>
          <tr><td style="font-weight: 600; color: #a5b4fc;">Reading</td>${r.covariance_matrix[1].map(v => `<td style="padding: 4px 8px;">${v}</td>`).join('')}</tr>
          <tr><td style="font-weight: 600; color: #a5b4fc;">Writing</td>${r.covariance_matrix[2].map(v => `<td style="padding: 4px 8px;">${v}</td>`).join('')}</tr>
        </table>
      `;

      document.getElementById('eigen-metrics-display').innerHTML = `
        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
          <div class="metric-pill">
            <span class="form-label">&lambda;1 (Principal Component 1)</span>
            <span class="metric-pill-val" style="color: #38bdf8;">${r.eigenvalues[0]} &bull; ${r.explained_variance_ratio[0]}% Variance</span>
          </div>
          <div class="metric-pill">
            <span class="form-label">&lambda;2 (Principal Component 2)</span>
            <span class="metric-pill-val" style="color: #a855f7; font-size: 1.1rem;">${r.eigenvalues[1]} &bull; ${r.explained_variance_ratio[1]}% Variance</span>
          </div>
          <div class="metric-pill">
            <span class="form-label">&lambda;3 (Principal Component 3)</span>
            <span class="metric-pill-val" style="color: var(--text-muted); font-size: 1rem;">${r.eigenvalues[2]} &bull; ${r.explained_variance_ratio[2]}% Variance</span>
          </div>
        </div>
      `;
    }
  } catch (err) {
    console.error('Eigen analysis error:', err);
  }
}

// ─── 4. Gradient Descent Optimization Simulator ────────────────────────────
const elGdLr = document.getElementById('gd-lr');
const elGdEpochs = document.getElementById('gd-epochs');
const elBtnRunGd = document.getElementById('btn-run-gd');
const elGdEquation = document.getElementById('gd-equation');
const elGdLoss = document.getElementById('gd-loss');
const canvasLoss = document.getElementById('canvas-loss');
const ctxLoss = canvasLoss.getContext('2d');

async function runGradientDescent() {
  elBtnRunGd.textContent = 'Optimizing...';
  const payload = {
    learning_rate: parseFloat(elGdLr.value) || 0.05,
    epochs: parseInt(elGdEpochs.value) || 60
  };

  try {
    const res = await fetch('/api/calculus/gradient-descent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    elBtnRunGd.textContent = 'Run Gradient Descent';

    if (data.success) {
      const r = data.result;
      elGdEquation.textContent = r.fitted_equation;
      elGdLoss.textContent = `${r.final_loss} (MSE)`;

      // Render Loss Curve on Canvas
      drawLossCurve(r.history);
    }
  } catch (err) {
    elBtnRunGd.textContent = 'Run Gradient Descent';
    console.error('Gradient descent error:', err);
  }
}

function drawLossCurve(history) {
  const w = canvasLoss.width;
  const h = canvasLoss.height;
  ctxLoss.clearRect(0, 0, w, h);

  if (!history || history.length === 0) return;

  const maxLoss = Math.max(...history.map(pt => pt.loss));
  const minLoss = Math.min(...history.map(pt => pt.loss));
  const pad = 30;

  // Grid
  ctxLoss.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  ctxLoss.lineWidth = 1;
  ctxLoss.beginPath();
  ctxLoss.moveTo(pad, pad);
  ctxLoss.lineTo(pad, h - pad);
  ctxLoss.lineTo(w - pad, h - pad);
  ctxLoss.stroke();

  // Curve
  ctxLoss.strokeStyle = '#a855f7';
  ctxLoss.lineWidth = 2.5;
  ctxLoss.beginPath();

  history.forEach((pt, i) => {
    const x = pad + (i / (history.length - 1)) * (w - 2 * pad);
    const lossRange = (maxLoss - minLoss) || 1;
    const y = (h - pad) - ((pt.loss - minLoss) / lossRange) * (h - 2 * pad);

    if (i === 0) ctxLoss.moveTo(x, y);
    else ctxLoss.lineTo(x, y);
  });
  ctxLoss.stroke();
}

elBtnRunGd.addEventListener('click', runGradientDescent);

// ─── 5. Chain Rule & Backpropagation Trace ─────────────────────────────────
const elCrX = document.getElementById('cr-x');
const elCrTarget = document.getElementById('cr-target');
const elCrW1 = document.getElementById('cr-w1');
const elCrW2 = document.getElementById('cr-w2');
const elBtnCalcChainRule = document.getElementById('btn-calc-chainrule');
const elChainRuleResults = document.getElementById('chainrule-results');

async function calculateChainRule() {
  const payload = {
    x: parseFloat(elCrX.value) || 2.5,
    target: parseFloat(elCrTarget.value) || 1.0,
    w1: parseFloat(elCrW1.value) || 0.6,
    w2: parseFloat(elCrW2.value) || 0.9
  };

  try {
    const res = await fetch('/api/calculus/chain-rule', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      const f = data.result.forward;
      const g = data.result.chain_rule_gradients;

      elChainRuleResults.innerHTML = `
        <div class="metric-pill">
          <span class="form-label">Forward Prediction (&ycirc;) & Loss</span>
          <span class="metric-pill-val" style="color: #38bdf8;">${f.y_pred} (Error Loss: ${f.loss})</span>
        </div>
        <div class="metric-pill">
          <span class="form-label">Gradient &part;L / &part;W2 (Layer 2)</span>
          <span class="metric-pill-val" style="color: #a855f7;">${g.dL_dw2}</span>
        </div>
        <div class="metric-pill">
          <span class="form-label">Gradient &part;L / &part;W1 (Layer 1 via Chain Rule)</span>
          <span class="metric-pill-val" style="color: #34d399;">${g.dL_dw1}</span>
        </div>
      `;
    }
  } catch (err) {
    console.error('Chain rule error:', err);
  }
}

elBtnCalcChainRule.addEventListener('click', calculateChainRule);

// ─── Initialization ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  calculateVectors();
  renderMatrixCanvas();
  loadEigenAnalysis();
  runGradientDescent();
  calculateChainRule();
});
