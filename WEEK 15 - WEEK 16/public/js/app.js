// Global chart instances
let benchmarkChart = null;
let activationChart = null;
let optimizerTrajChart = null;
let optimizerLossChart = null;
let lossCurveChart = null;
let accCurveChart = null;
let decisionBoundaryChart = null;

let activationsDataCache = null;

document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  loadDeploymentBenchmark();
  loadActivationCurves();
  loadOptimizerSimulation();
  initNeuralNetSimulator();
  loadDecisionBoundary();
  setupInferenceForm();
});

// ─────────────────────────────────────────────────────────────────────────────
// 1. Navigation
// ─────────────────────────────────────────────────────────────────────────────
function setupNavigation() {
  const navBtns = document.querySelectorAll('.nav-btn');
  const tabs = document.querySelectorAll('.tab-content');

  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;

      navBtns.forEach(b => b.classList.remove('active'));
      tabs.forEach(t => t.classList.remove('active'));

      btn.classList.add('active');
      const activeTab = document.getElementById(`${target}-tab`);
      if (activeTab) {
        activeTab.classList.add('active');
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Deployment Benchmark (Pickle vs Joblib)
// ─────────────────────────────────────────────────────────────────────────────
async function loadDeploymentBenchmark() {
  try {
    const res = await fetch('/api/deployment/benchmark');
    const json = await res.json();
    if (!json.success) return;

    const data = json.data;

    // Update stats
    document.getElementById('pickle-size').textContent = `${data.pickle.file_size_kb} KB`;
    document.getElementById('pickle-save').textContent = `${data.pickle.save_time_ms} ms`;
    document.getElementById('pickle-load').textContent = `${data.pickle.load_time_ms} ms`;

    document.getElementById('joblib-size').textContent = `${data.joblib.file_size_kb} KB`;
    document.getElementById('joblib-save').textContent = `${data.joblib.save_time_ms} ms`;
    document.getElementById('joblib-load').textContent = `${data.joblib.load_time_ms} ms`;

    // Render comparison chart
    const ctx = document.getElementById('deploymentBenchmarkChart').getContext('2d');
    if (benchmarkChart) benchmarkChart.destroy();

    benchmarkChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['File Size (KB)', 'Save Latency (ms)', 'Load Latency (ms)'],
        datasets: [
          {
            label: 'Python Pickle (.pkl)',
            data: [data.pickle.file_size_kb, data.pickle.save_time_ms, data.pickle.load_time_ms],
            backgroundColor: 'rgba(59, 130, 246, 0.65)',
            borderColor: '#3b82f6',
            borderWidth: 1.5,
            borderRadius: 6
          },
          {
            label: 'Scikit-Learn Joblib (.joblib)',
            data: [data.joblib.file_size_kb, data.joblib.save_time_ms, data.joblib.load_time_ms],
            backgroundColor: 'rgba(139, 92, 246, 0.75)',
            borderColor: '#8b5cf6',
            borderWidth: 1.5,
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#cbd5e1', font: { family: 'Inter' } } },
          tooltip: {
            backgroundColor: '#090d16',
            titleColor: '#fff',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1
          }
        },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Value (Log / Linear)', color: '#64748b' } }
        }
      }
    });
  } catch (err) {
    console.error('Failed to load deployment benchmark:', err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Activation Functions & Derivatives
// ─────────────────────────────────────────────────────────────────────────────
async function loadActivationCurves() {
  try {
    const res = await fetch('/api/activations');
    const json = await res.json();
    if (!json.success) return;

    activationsDataCache = json.data;

    // Wire chips
    const chips = document.querySelectorAll('.chip-btn');
    chips.forEach(chip => {
      chip.addEventListener('click', () => {
        chips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        renderActivation(chip.dataset.act);
      });
    });

    renderActivation('relu');
  } catch (err) {
    console.error('Failed to load activation functions:', err);
  }
}

function renderActivation(actKey) {
  if (!activationsDataCache) return;

  const fnInfo = activationsDataCache.functions[actKey];
  if (!fnInfo) return;

  document.getElementById('act-chart-title').textContent = `${fnInfo.name} & Derivative`;
  document.getElementById('act-range-tag').textContent = `Range: ${fnInfo.range}`;
  document.getElementById('act-name-display').textContent = fnInfo.name;
  document.getElementById('act-formula').textContent = fnInfo.formula;
  document.getElementById('act-deriv').textContent = fnInfo.derivative_formula;
  document.getElementById('act-pros').textContent = fnInfo.pros;
  document.getElementById('act-cons').textContent = fnInfo.cons;

  const ctx = document.getElementById('activationCurveChart').getContext('2d');
  if (activationChart) activationChart.destroy();

  activationChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: activationsDataCache.z_values,
      datasets: [
        {
          label: `${fnInfo.name} f(z)`,
          data: fnInfo.output,
          borderColor: '#8b5cf6',
          backgroundColor: 'rgba(139, 92, 246, 0.1)',
          borderWidth: 2.5,
          tension: 0.2,
          pointRadius: 0
        },
        {
          label: `Derivative f'(z)`,
          data: fnInfo.derivative,
          borderColor: '#06b6d4',
          borderDash: [5, 5],
          borderWidth: 2,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#cbd5e1' } },
        tooltip: {
          backgroundColor: '#090d16',
          titleColor: '#fff',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1
        }
      },
      scales: {
        x: { ticks: { color: '#94a3b8', maxTicksLimit: 11 }, grid: { color: 'rgba(255,255,255,0.03)' }, title: { display: true, text: 'Input z', color: '#64748b' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Output', color: '#64748b' } }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Loss & Optimizers Simulation
// ─────────────────────────────────────────────────────────────────────────────
async function loadOptimizerSimulation() {
  try {
    const res = await fetch('/api/optimizers/simulate');
    const json = await res.json();
    if (!json.success) return;

    const data = json.data;

    // 1. Loss trajectory line chart
    const ctxLoss = document.getElementById('optimizerLossChart').getContext('2d');
    if (optimizerLossChart) optimizerLossChart.destroy();

    optimizerLossChart = new Chart(ctxLoss, {
      type: 'line',
      data: {
        labels: data.steps,
        datasets: [
          {
            label: 'Vanilla SGD',
            data: data.loss_histories.SGD,
            borderColor: '#3b82f6',
            borderWidth: 2,
            pointRadius: 1,
            tension: 0.1
          },
          {
            label: 'SGD with Momentum (β=0.9)',
            data: data.loss_histories.Momentum,
            borderColor: '#10b981',
            borderWidth: 2,
            pointRadius: 1,
            tension: 0.1
          },
          {
            label: 'Adam Optimizer',
            data: data.loss_histories.Adam,
            borderColor: '#8b5cf6',
            borderWidth: 2.5,
            pointRadius: 1,
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#cbd5e1' } },
          tooltip: {
            backgroundColor: '#090d16',
            titleColor: '#fff',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1
          }
        },
        scales: {
          x: { title: { display: true, text: 'Step / Iteration', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.03)' } },
          y: { title: { display: true, text: 'Loss', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });

    // 2. 2D Path chart
    const ctxTraj = document.getElementById('optimizerTrajectoryChart').getContext('2d');
    if (optimizerTrajChart) optimizerTrajChart.destroy();

    optimizerTrajChart = new Chart(ctxTraj, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'SGD Trajectory',
            data: data.trajectories_2d.SGD,
            showLine: true,
            borderColor: '#3b82f6',
            backgroundColor: '#3b82f6',
            borderWidth: 2,
            pointRadius: 3
          },
          {
            label: 'Momentum Trajectory',
            data: data.trajectories_2d.Momentum,
            showLine: true,
            borderColor: '#10b981',
            backgroundColor: '#10b981',
            borderWidth: 2,
            pointRadius: 3
          },
          {
            label: 'Adam Trajectory',
            data: data.trajectories_2d.Adam,
            showLine: true,
            borderColor: '#c084fc',
            backgroundColor: '#c084fc',
            borderWidth: 2.5,
            pointRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#cbd5e1' } }
        },
        scales: {
          x: { title: { display: true, text: 'Parameter X', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { title: { display: true, text: 'Parameter Y', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });

  } catch (err) {
    console.error('Failed to load optimizer simulation:', err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. NumPy Neural Network Simulator
// ─────────────────────────────────────────────────────────────────────────────
function initNeuralNetSimulator() {
  const epochsInput = document.getElementById('nn-epochs');
  const epochsVal = document.getElementById('nn-epochs-val');
  const lrInput = document.getElementById('nn-lr');
  const lrVal = document.getElementById('nn-lr-val');
  const btnTrain = document.getElementById('btn-train-nn');

  epochsInput.addEventListener('input', () => epochsVal.textContent = epochsInput.value);
  lrInput.addEventListener('input', () => lrVal.textContent = lrInput.value);

  btnTrain.addEventListener('click', runNnTraining);

  // Auto-run baseline training
  runNnTraining();
}

async function runNnTraining() {
  const btnTrain = document.getElementById('btn-train-nn');
  const epochs = document.getElementById('nn-epochs').value;
  const lr = document.getElementById('nn-lr').value;
  const activation = document.getElementById('nn-act').value;
  const optimizer = document.getElementById('nn-opt').value;
  const hiddenUnits = document.getElementById('nn-hidden').value;

  btnTrain.disabled = true;
  btnTrain.innerHTML = '<span class="btn-icon">⏳</span> Training NumPy Net...';

  try {
    const res = await fetch('/api/neural-net/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        epochs: parseInt(epochs),
        learning_rate: parseFloat(lr),
        activation: activation,
        optimizer: optimizer,
        hidden_units: parseInt(hiddenUnits)
      })
    });

    const json = await res.json();
    if (!json.success) throw new Error(json.error);

    const data = json.data;
    const m = data.metrics;
    const hist = data.history;

    // Update scoreboard
    document.getElementById('nn-acc').textContent = `${m.accuracy}%`;
    document.getElementById('nn-f1').textContent = `${m.f1_score}%`;
    document.getElementById('nn-prec').textContent = `${m.precision}%`;
    document.getElementById('nn-rec').textContent = `${m.recall}%`;
    document.getElementById('nn-auc').textContent = `${m.roc_auc}%`;
    document.getElementById('nn-time').textContent = `${m.train_duration_ms} ms`;

    // Render Loss Chart
    const ctxLoss = document.getElementById('lossCurveChart').getContext('2d');
    if (lossCurveChart) lossCurveChart.destroy();

    lossCurveChart = new Chart(ctxLoss, {
      type: 'line',
      data: {
        labels: hist.epochs,
        datasets: [
          {
            label: 'Train Loss (BCE)',
            data: hist.train_loss,
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            borderWidth: 2,
            tension: 0.2
          },
          {
            label: 'Val Loss (BCE)',
            data: hist.val_loss,
            borderColor: '#f43f5e',
            borderDash: [4, 4],
            borderWidth: 2,
            tension: 0.2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#cbd5e1' } } },
        scales: {
          x: { title: { display: true, text: 'Epoch', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.03)' } },
          y: { title: { display: true, text: 'BCE Loss', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });

    // Render Accuracy Chart
    const ctxAcc = document.getElementById('accCurveChart').getContext('2d');
    if (accCurveChart) accCurveChart.destroy();

    accCurveChart = new Chart(ctxAcc, {
      type: 'line',
      data: {
        labels: hist.epochs,
        datasets: [
          {
            label: 'Train Accuracy (%)',
            data: hist.train_acc,
            borderColor: '#38bdf8',
            borderWidth: 2,
            tension: 0.2
          },
          {
            label: 'Val Accuracy (%)',
            data: hist.val_acc,
            borderColor: '#34d399',
            borderDash: [4, 4],
            borderWidth: 2,
            tension: 0.2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#cbd5e1' } } },
        scales: {
          x: { title: { display: true, text: 'Epoch', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.03)' } },
          y: { title: { display: true, text: 'Accuracy (%)', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });

  } catch (err) {
    console.error('Training error:', err);
    alert('Training failed: ' + err.message);
  } finally {
    btnTrain.disabled = false;
    btnTrain.innerHTML = '<span class="btn-icon">⚡</span> Run NumPy Training';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Non-linear Decision Boundary
// ─────────────────────────────────────────────────────────────────────────────
async function loadDecisionBoundary() {
  try {
    const res = await fetch('/api/neural-net/decision-boundary');
    const json = await res.json();
    if (!json.success) return;

    const data = json.data;
    const class0 = data.scatter_points.filter(p => p.label === 0);
    const class1 = data.scatter_points.filter(p => p.label === 1);

    const ctx = document.getElementById('decisionBoundaryChart').getContext('2d');
    if (decisionBoundaryChart) decisionBoundaryChart.destroy();

    decisionBoundaryChart = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Class 0 (Inner Circle)',
            data: class0,
            backgroundColor: 'rgba(56, 189, 248, 0.85)',
            borderColor: '#0284c7',
            pointRadius: 5
          },
          {
            label: 'Class 1 (Outer Ring)',
            data: class1,
            backgroundColor: 'rgba(192, 132, 252, 0.85)',
            borderColor: '#9333ea',
            pointRadius: 5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#cbd5e1' } },
          tooltip: {
            backgroundColor: '#090d16',
            titleColor: '#fff',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1
          }
        },
        scales: {
          x: { title: { display: true, text: 'Feature X₁', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { title: { display: true, text: 'Feature X₂', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });

  } catch (err) {
    console.error('Failed to load decision boundary:', err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Interactive Inference Form
// ─────────────────────────────────────────────────────────────────────────────
function setupInferenceForm() {
  const form = document.getElementById('inference-form');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const features = {
      Age: parseFloat(document.getElementById('inf-age').value),
      Gender: document.getElementById('inf-gender').value,
      Department: document.getElementById('inf-dept').value,
      StudyHoursPerWeek: parseFloat(document.getElementById('inf-hours').value),
      AttendanceRate: parseFloat(document.getElementById('inf-attendance').value),
      MathScore: parseFloat(document.getElementById('inf-math').value),
      ReadingScore: parseFloat(document.getElementById('inf-reading').value),
      WritingScore: parseFloat(document.getElementById('inf-writing').value)
    };

    const serializer = document.getElementById('inf-serializer').value;

    try {
      const res = await fetch('/api/deployment/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ serializer, features })
      });
      const json = await res.json();
      if (!json.success) throw new Error(json.error);

      const d = json.data;

      document.getElementById('result-placeholder').classList.add('d-none');
      document.getElementById('result-content').classList.remove('d-none');

      const badge = document.getElementById('pred-badge');
      badge.textContent = d.prediction_label;
      if (d.prediction === 1) {
        badge.className = 'prediction-badge badge-distinction';
      } else {
        badge.className = 'prediction-badge badge-standard';
      }

      document.getElementById('pred-prob').textContent = `${d.probability_distinction}%`;
      document.getElementById('pred-progress').style.width = `${d.probability_distinction}%`;

      document.getElementById('lat-load').textContent = `${d.load_latency_ms} ms`;
      document.getElementById('lat-inf').textContent = `${d.inference_latency_ms} ms`;
      document.getElementById('lat-ser').textContent = d.serializer_used.toUpperCase();

    } catch (err) {
      alert('Inference call failed: ' + err.message);
    }
  });
}
