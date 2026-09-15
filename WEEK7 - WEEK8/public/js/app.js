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

// ─── 1. Benchmarks Loader ──────────────────────────────────────────────────
const elRegressionCards = document.getElementById('regression-cards-container');
const elClassificationCards = document.getElementById('classification-cards-container');

async function loadBenchmarks(ridgeAlpha = 1.0, lassoAlpha = 0.05, knnK = 3) {
  try {
    const res = await fetch(`/api/models/benchmarks?ridge_alpha=${ridgeAlpha}&lasso_alpha=${lassoAlpha}&knn_k=${knnK}`);
    const data = await res.json();
    if (data.success) {
      renderBenchmarks(data.benchmarks);
    }
  } catch (err) {
    console.error('Failed to load benchmarks:', err);
  }
}

function renderBenchmarks(bm) {
  const reg = bm.regression;
  const clf = bm.classification;

  // Regression Cards
  elRegressionCards.innerHTML = `
    <div class="model-card">
      <div>
        <div class="model-name">${reg.linear.name}</div>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.75rem;">Standard Ordinary Least Squares regression fitting feature hyper-plane.</p>
      </div>
      <div>
        <div class="metric-row"><span>R² Score:</span> <span class="metric-val-highlight">${reg.linear.r2}</span></div>
        <div class="metric-row"><span>Mean Squared Error:</span> <span>${reg.linear.mse}</span></div>
        <div class="metric-row"><span>Top Feature Weight:</span> <span style="color: #38bdf8;">StudyHours (${reg.linear.weights.StudyHoursPerWeek})</span></div>
      </div>
    </div>

    <div class="model-card">
      <div>
        <div class="model-name">${reg.polynomial.name}</div>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.75rem;">Non-linear polynomial expansion with interaction terms.</p>
      </div>
      <div>
        <div class="metric-row"><span>R² Score:</span> <span class="metric-val-highlight">${reg.polynomial.r2}</span></div>
        <div class="metric-row"><span>Mean Squared Error:</span> <span>${reg.polynomial.mse}</span></div>
        <div class="metric-row"><span>Expanded Terms:</span> <span>${reg.polynomial.features_count} terms</span></div>
      </div>
    </div>

    <div class="model-card">
      <div>
        <div class="model-name">${reg.ridge.name}</div>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.75rem;">L2 regularized regression preventing weight explosion.</p>
      </div>
      <div>
        <div class="metric-row"><span>R² Score:</span> <span class="metric-val-highlight">${reg.ridge.r2}</span></div>
        <div class="metric-row"><span>Mean Squared Error:</span> <span>${reg.ridge.mse}</span></div>
        <div class="metric-row"><span>Alpha (Penalty):</span> <span>${reg.ridge.alpha}</span></div>
      </div>
    </div>

    <div class="model-card">
      <div>
        <div class="model-name">${reg.lasso.name}</div>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.75rem;">L1 regularized regression enforcing coefficient sparsity.</p>
      </div>
      <div>
        <div class="metric-row"><span>R² Score:</span> <span class="metric-val-highlight">${reg.lasso.r2}</span></div>
        <div class="metric-row"><span>Mean Squared Error:</span> <span>${reg.lasso.mse}</span></div>
        <div class="metric-row"><span>Zeroed Weights:</span> <span style="color: #fbbf24;">${reg.lasso.zeroed_features.length > 0 ? reg.lasso.zeroed_features.join(', ') : 'None'}</span></div>
      </div>
    </div>
  `;

  // Classification Cards
  elClassificationCards.innerHTML = `
    <div class="model-card">
      <div>
        <div class="model-name">${clf.logistic.name}</div>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.75rem;">Probabilistic classification estimating P(Distinction | Features).</p>
      </div>
      <div>
        <div class="metric-row"><span>Accuracy:</span> <span class="metric-val-highlight">${clf.logistic.accuracy}%</span></div>
        <div class="metric-row"><span>F1-Score:</span> <span>${clf.logistic.f1_score}</span></div>
        <div class="metric-row"><span>Primary Predictor:</span> <span style="color: #38bdf8;">StudyHours (${clf.logistic.coefficients.StudyHoursPerWeek})</span></div>
      </div>
    </div>

    <div class="model-card">
      <div>
        <div class="model-name">${clf.knn.name}</div>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.75rem;">Instance-based voting from closest student neighbors in scaled space.</p>
      </div>
      <div>
        <div class="metric-row"><span>Accuracy:</span> <span class="metric-val-highlight">${clf.knn.accuracy}%</span></div>
        <div class="metric-row"><span>F1-Score:</span> <span>${clf.knn.f1_score}</span></div>
        <div class="metric-row"><span>Neighbor Count (k):</span> <span>${clf.knn.k}</span></div>
      </div>
    </div>
  `;
}

// ─── 2. Live Student Predictor ─────────────────────────────────────────────
const elInputStudy = document.getElementById('input-study');
const elValStudy = document.getElementById('val-study');
const elInputAttendance = document.getElementById('input-attendance');
const elValAttendance = document.getElementById('val-attendance');
const elInputMath = document.getElementById('input-math');
const elValMath = document.getElementById('val-math');
const elInputReading = document.getElementById('input-reading');
const elValReading = document.getElementById('val-reading');
const elInputWriting = document.getElementById('input-writing');
const elValWriting = document.getElementById('val-writing');
const elBtnPredict = document.getElementById('btn-predict');

// Prediction Output DOM
const elPredConsensus = document.getElementById('pred-consensus-cgpa');
const elPredLinear = document.getElementById('pred-linear');
const elPredPoly = document.getElementById('pred-poly');
const elPredRidge = document.getElementById('pred-ridge');
const elPredLasso = document.getElementById('pred-lasso');
const elPredProbVal = document.getElementById('pred-prob-val');
const elPredProbBar = document.getElementById('pred-prob-bar');
const elPredLogisticVerdict = document.getElementById('pred-logistic-verdict');
const elPredKnnVerdict = document.getElementById('pred-knn-verdict');

elInputStudy.addEventListener('input', () => { elValStudy.textContent = `${elInputStudy.value} hrs/wk`; });
elInputAttendance.addEventListener('input', () => { elValAttendance.textContent = `${elInputAttendance.value}%`; });
elInputMath.addEventListener('input', () => { elValMath.textContent = `${elInputMath.value} / 100`; });
elInputReading.addEventListener('input', () => { elValReading.textContent = `${elInputReading.value} / 100`; });
elInputWriting.addEventListener('input', () => { elValWriting.textContent = `${elInputWriting.value} / 100`; });

async function executePrediction() {
  const payload = {
    study_hours: parseFloat(elInputStudy.value),
    attendance: parseFloat(elInputAttendance.value),
    math: parseFloat(elInputMath.value),
    reading: parseFloat(elInputReading.value),
    writing: parseFloat(elInputWriting.value),
    ridge_alpha: parseFloat(elInputRidgeAlpha.value) || 1.0,
    lasso_alpha: parseFloat(elInputLassoAlpha.value) || 0.05,
    knn_k: parseInt(elInputKnnK.value) || 3
  };

  elBtnPredict.textContent = 'Generating Forecasts...';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    elBtnPredict.textContent = 'Generate ML Forecasts';

    if (data.success) {
      const reg = data.prediction.regression_predictions;
      const clf = data.prediction.classification_prediction;

      elPredConsensus.textContent = reg.consensus_average;
      elPredLinear.textContent = reg.linear_cgpa;
      elPredPoly.textContent = reg.polynomial_cgpa;
      elPredRidge.textContent = reg.ridge_cgpa;
      elPredLasso.textContent = reg.lasso_cgpa;

      elPredProbVal.textContent = `${clf.high_performer_probability}%`;
      elPredProbBar.style.width = `${clf.high_performer_probability}%`;
      elPredLogisticVerdict.textContent = clf.logistic_verdict;
      elPredKnnVerdict.textContent = clf.knn_verdict;
    }
  } catch (err) {
    elBtnPredict.textContent = 'Generate ML Forecasts';
    console.error('Prediction error:', err);
  }
}

elBtnPredict.addEventListener('click', executePrediction);

// ─── 3. Hyperparameter Tuning ──────────────────────────────────────────────
const elInputRidgeAlpha = document.getElementById('input-ridge-alpha');
const elValRidgeAlpha = document.getElementById('val-ridge-alpha');
const elInputLassoAlpha = document.getElementById('input-lasso-alpha');
const elValLassoAlpha = document.getElementById('val-lasso-alpha');
const elInputKnnK = document.getElementById('input-knn-k');
const elValKnnK = document.getElementById('val-knn-k');
const elTuningFeedback = document.getElementById('tuning-feedback-box');

async function onTuningChange() {
  const rAlpha = parseFloat(elInputRidgeAlpha.value);
  const lAlpha = parseFloat(elInputLassoAlpha.value);
  const k = parseInt(elInputKnnK.value);

  elValRidgeAlpha.textContent = rAlpha;
  elValLassoAlpha.textContent = lAlpha;
  elValKnnK.textContent = k;

  await loadBenchmarks(rAlpha, lAlpha, k);

  elTuningFeedback.innerHTML = `
// Model hyperparameters updated:
Ridge alpha (L2): ${rAlpha}  --> Higher values shrink weights towards 0.
Lasso alpha (L1): ${lAlpha} --> Higher values enforce feature sparsity.
KNN neighbors (k): ${k}    --> Larger k creates smoother, less noisy decision boundaries.
  `;
}

elInputRidgeAlpha.addEventListener('input', onTuningChange);
elInputLassoAlpha.addEventListener('input', onTuningChange);
elInputKnnK.addEventListener('input', onTuningChange);

// ─── Initialization ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadBenchmarks();
  executePrediction();
});
