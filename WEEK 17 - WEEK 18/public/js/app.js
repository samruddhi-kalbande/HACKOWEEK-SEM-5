let pytorchChart = null;
let sequenceLossChart = null;
let sequenceParamsChart = null;
let regularizationChart = null;
let cnnFiltersCache = null;

document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  loadAutogradDemo();
  setupPyTorchTraining();
  loadCnnArchitecture();
  loadSequenceComparison();
  loadRegularizationComparison();
  setupTransformersForm();
  setupLlmWorkbench();
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
// 2. PyTorch & Autograd
// ─────────────────────────────────────────────────────────────────────────────
async function loadAutogradDemo() {
  try {
    const res = await fetch('/api/pytorch/tensors-autograd');
    const json = await res.json();
    if (!json.success) return;

    const d = json.data;
    if (d.has_torch) {
      document.getElementById('pytorch-version').textContent = `PyTorch ${d.torch_version}`;
    }
    document.getElementById('autograd-x').textContent = d.autograd_demo.x_value.toFixed(2);
    document.getElementById('autograd-y').textContent = d.autograd_demo.y_value.toFixed(3);
    document.getElementById('autograd-grad').textContent = d.autograd_demo.computed_gradient.toFixed(2);
  } catch (err) {
    console.error('Failed to load autograd:', err);
  }
}

function setupPyTorchTraining() {
  const epochsInput = document.getElementById('pt-epochs');
  const epochsVal = document.getElementById('pt-epochs-val');
  const lrInput = document.getElementById('pt-lr');
  const lrVal = document.getElementById('pt-lr-val');
  const btnTrain = document.getElementById('btn-train-pytorch');

  epochsInput.addEventListener('input', () => epochsVal.textContent = epochsInput.value);
  lrInput.addEventListener('input', () => lrVal.textContent = lrInput.value);

  btnTrain.addEventListener('click', runPyTorchLoop);
  runPyTorchLoop();
}

async function runPyTorchLoop() {
  const btnTrain = document.getElementById('btn-train-pytorch');
  const epochs = document.getElementById('pt-epochs').value;
  const lr = document.getElementById('pt-lr').value;

  btnTrain.disabled = true;
  btnTrain.innerHTML = '<span>⏳</span> Training...';

  try {
    const res = await fetch('/api/pytorch/training-loop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs: parseInt(epochs), learning_rate: parseFloat(lr) })
    });
    const json = await res.json();
    if (!json.success) throw new Error(json.error);

    const data = json.data;
    document.getElementById('pt-final-loss').textContent = data.final_loss;
    document.getElementById('pt-final-acc').textContent = `${data.final_acc}%`;

    const labels = data.history.map(h => `Ep ${h.epoch}`);
    const lossData = data.history.map(h => h.loss);
    const accData = data.history.map(h => h.accuracy);

    const ctx = document.getElementById('pytorchTrainingChart').getContext('2d');
    if (pytorchChart) pytorchChart.destroy();

    pytorchChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'BCE Loss',
            data: lossData,
            borderColor: '#f97316',
            backgroundColor: 'rgba(249, 115, 22, 0.1)',
            yAxisID: 'yLoss',
            borderWidth: 2,
            tension: 0.2
          },
          {
            label: 'Accuracy (%)',
            data: accData,
            borderColor: '#38bdf8',
            borderDash: [4, 4],
            yAxisID: 'yAcc',
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
          x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.03)' } },
          yLoss: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Loss', color: '#f97316' },
            ticks: { color: '#94a3b8' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          yAcc: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'Accuracy (%)', color: '#38bdf8' },
            ticks: { color: '#94a3b8' },
            grid: { display: false }
          }
        }
      }
    });

  } catch (err) {
    console.error('PyTorch training failed:', err);
  } finally {
    btnTrain.disabled = false;
    btnTrain.innerHTML = '<span>⚡</span> Run PyTorch Loop';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. CNN Architecture & Filter Kernels
// ─────────────────────────────────────────────────────────────────────────────
async function loadCnnArchitecture() {
  try {
    const res = await fetch('/api/cnn/architecture');
    const json = await res.json();
    if (!json.success) return;

    const data = json.data;
    cnnFiltersCache = data.convolution_filters;

    // Populate stages table
    const tbody = document.querySelector('#cnn-stages-table tbody');
    tbody.innerHTML = '';
    data.pipeline_stages.forEach(s => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${s.layer}</td><td><code>${s.dim}</code></td><td>${s.details}</td>`;
      tbody.appendChild(tr);
    });

    // Wire kernel chips
    const kernelBtns = document.querySelectorAll('.kernel-selector button');
    kernelBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        kernelBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderKernelMatrix(btn.dataset.kernel);
      });
    });

    renderKernelMatrix('Sobel Horizontal (Edge Detection)');
  } catch (err) {
    console.error('Failed to load CNN architecture:', err);
  }
}

function renderKernelMatrix(kernelName) {
  if (!cnnFiltersCache) return;
  const matrix = cnnFiltersCache[kernelName];
  if (!matrix) return;

  const container = document.getElementById('kernel-matrix');
  container.innerHTML = '';
  matrix.forEach(row => {
    row.forEach(val => {
      const cell = document.createElement('div');
      cell.className = 'kernel-cell';
      cell.textContent = val;
      container.appendChild(cell);
    });
  });

  const descElem = document.getElementById('kernel-desc');
  if (kernelName.includes('Horizontal')) {
    descElem.textContent = 'Detects horizontal edges and intensity transitions along the vertical axis.';
  } else if (kernelName.includes('Vertical')) {
    descElem.textContent = 'Detects vertical edges and intensity transitions along the horizontal axis.';
  } else if (kernelName.includes('Sharpen')) {
    descElem.textContent = 'Enhances high-frequency spatial gradients, highlighting fine boundaries.';
  } else {
    descElem.textContent = 'Suppresses high-frequency noise using a normalized 2D Gaussian distribution.';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Sequence Modeling: RNN vs LSTM vs GRU
// ─────────────────────────────────────────────────────────────────────────────
async function loadSequenceComparison() {
  try {
    const res = await fetch('/api/sequences/compare');
    const json = await res.json();
    if (!json.success) return;

    const d = json.data;

    // 1. Loss trajectories
    const ctxLoss = document.getElementById('sequenceLossChart').getContext('2d');
    if (sequenceLossChart) sequenceLossChart.destroy();

    sequenceLossChart = new Chart(ctxLoss, {
      type: 'line',
      data: {
        labels: d.steps.map(s => `t=${s}`),
        datasets: [
          {
            label: 'Vanilla RNN',
            data: d.loss_progression.RNN,
            borderColor: '#3b82f6',
            borderWidth: 2,
            tension: 0.2
          },
          {
            label: 'LSTM',
            data: d.loss_progression.LSTM,
            borderColor: '#10b981',
            borderWidth: 2.5,
            tension: 0.2
          },
          {
            label: 'GRU',
            data: d.loss_progression.GRU,
            borderColor: '#8b5cf6',
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
          x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.03)' } },
          y: { title: { display: true, text: 'Loss', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });

    // 2. Parameter counts
    const ctxParams = document.getElementById('sequenceParamsChart').getContext('2d');
    if (sequenceParamsChart) sequenceParamsChart.destroy();

    const p = d.parameter_comparison;
    sequenceParamsChart = new Chart(ctxParams, {
      type: 'bar',
      data: {
        labels: ['Vanilla RNN', 'Gated Recurrent Unit (GRU)', 'LSTM'],
        datasets: [{
          label: 'Trainable Parameters',
          data: [p.RNN_params, p.GRU_params, p.LSTM_params],
          backgroundColor: ['rgba(59, 130, 246, 0.7)', 'rgba(139, 92, 246, 0.7)', 'rgba(16, 185, 129, 0.7)'],
          borderColor: ['#3b82f6', '#8b5cf6', '#10b981'],
          borderWidth: 1.5,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
          y: { title: { display: true, text: 'Parameter Count', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });

  } catch (err) {
    console.error('Failed to load sequences:', err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Regularization & Early Stopping
// ─────────────────────────────────────────────────────────────────────────────
async function loadRegularizationComparison() {
  try {
    const res = await fetch('/api/regularization/compare');
    const json = await res.json();
    if (!json.success) return;

    const d = json.data;
    const ctx = document.getElementById('regularizationChart').getContext('2d');
    if (regularizationChart) regularizationChart.destroy();

    regularizationChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: d.epochs.map(e => `Ep ${e}`),
        datasets: [
          {
            label: 'Unregularized Val Loss (Overfitting Divergence)',
            data: d.unregularized.val_loss,
            borderColor: '#f43f5e',
            borderWidth: 2,
            tension: 0.2
          },
          {
            label: 'Regularized Val Loss (Dropout + BatchNorm)',
            data: d.regularized.val_loss,
            borderColor: '#10b981',
            borderWidth: 2.5,
            tension: 0.2
          },
          {
            label: 'Regularized Train Loss',
            data: d.regularized.train_loss,
            borderColor: '#38bdf8',
            borderDash: [3, 3],
            borderWidth: 1.5,
            tension: 0.2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#cbd5e1' } } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.03)' } },
          y: { title: { display: true, text: 'Loss', color: '#64748b' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });

  } catch (err) {
    console.error('Failed to load regularization:', err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Hugging Face Transformers Form
// ─────────────────────────────────────────────────────────────────────────────
function setupTransformersForm() {
  const form = document.getElementById('hf-form');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = document.getElementById('hf-input-text').value;

    try {
      const res = await fetch('/api/transformers/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const json = await res.json();
      if (!json.success) throw new Error(json.error);

      const d = json.data;
      const chipsContainer = document.getElementById('token-chips');
      chipsContainer.innerHTML = '';
      d.tokenization.tokens.forEach((tok, idx) => {
        const chip = document.createElement('span');
        chip.className = 'token-chip';
        chip.textContent = `${tok} (${d.tokenization.input_ids[idx]})`;
        chipsContainer.appendChild(chip);
      });

      document.getElementById('hf-sentiment-label').textContent = d.classification.label;
      document.getElementById('hf-sentiment-label').className = 
        `stat-val ${d.classification.label === 'POSITIVE' ? 'stat-emerald' : 'stat-amber'}`;
      document.getElementById('hf-confidence').textContent = d.classification.confidence;

    } catch (err) {
      alert('Transformers pipeline analysis failed: ' + err.message);
    }
  });

  // Trigger initial analyze
  form.dispatchEvent(new Event('submit'));
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. LLM API Query Workbench
// ─────────────────────────────────────────────────────────────────────────────
function setupLlmWorkbench() {
  const form = document.getElementById('llm-form');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const provider = document.getElementById('llm-provider').value;
    const task_type = document.getElementById('llm-task').value;
    const prompt = document.getElementById('llm-prompt').value;
    const outputElem = document.getElementById('llm-response-text');

    outputElem.textContent = 'Dispatching prompt to LLM provider...';

    try {
      const res = await fetch('/api/llm/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, task_type, prompt })
      });
      const json = await res.json();
      if (!json.success) throw new Error(json.error);

      const d = json.data;
      document.getElementById('llm-provider-tag').textContent = d.provider;
      document.getElementById('llm-latency-tag').textContent = `${d.latency_ms} ms`;
      outputElem.textContent = d.response;

    } catch (err) {
      outputElem.textContent = 'Error querying LLM API: ' + err.message;
    }
  });
}
