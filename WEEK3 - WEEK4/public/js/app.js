// ─── Global State ──────────────────────────────────────────────────────────
let currentPlots = [];

// ─── DOM Elements ──────────────────────────────────────────────────────────
const navButtons = document.querySelectorAll('.nav-btn');
const tabPanes = document.querySelectorAll('.tab-content');

// KPIs
const elKpiStudents = document.getElementById('kpi-students');
const elKpiEnrollments = document.getElementById('kpi-enrollments');
const elKpiCgpa = document.getElementById('kpi-cgpa');
const elKpiCgpaSub = document.getElementById('kpi-cgpa-sub');
const elKpiAttendance = document.getElementById('kpi-attendance');

// Visualizer DOM
const plotsListContainer = document.getElementById('plots-list-container');
const elCurrentPlotTitle = document.getElementById('current-plot-title');
const elCurrentPlotImg = document.getElementById('current-plot-img');
const elCurrentPlotDesc = document.getElementById('current-plot-desc');

// GroupBy DOM
const elGroupbyDim = document.getElementById('groupby-dim');
const elGroupbyMetric = document.getElementById('groupby-metric');
const elGroupbyAgg = document.getElementById('groupby-agg');
const elBtnRunGroupby = document.getElementById('btn-run-groupby');
const elThGroupName = document.getElementById('th-group-name');
const elGroupbyTbody = document.getElementById('groupby-tbody');

// Table Switcher DOM
const elBtnTableStudents = document.getElementById('btn-table-students');
const elBtnTableEnrollments = document.getElementById('btn-table-enrollments');
const elBtnTableMerged = document.getElementById('btn-table-merged');
const elRecordsThead = document.getElementById('records-thead');
const elRecordsTbody = document.getElementById('records-tbody');

// NumPy DOM
const elCalcMath = document.getElementById('calc-math');
const elCalcReading = document.getElementById('calc-reading');
const elCalcWriting = document.getElementById('calc-writing');
const elBtnCalcNumpy = document.getElementById('btn-calc-numpy');
const elNumpyOutputCard = document.getElementById('numpy-output-card');

// ─── Tab Switching ─────────────────────────────────────────────────────────
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

// ─── API Calls & Initial Load ──────────────────────────────────────────────
async function loadSummary() {
  try {
    const res = await fetch('/api/dataset/summary');
    const data = await res.json();
    if (data.success) {
      elKpiStudents.textContent = data.students_count;
      elKpiEnrollments.textContent = data.enrollments_count;
      elKpiCgpa.textContent = data.cgpa_stats.mean;
      elKpiCgpaSub.textContent = `Std Dev: ± ${data.cgpa_stats.std}`;
      elKpiAttendance.textContent = `${data.attendance_stats.mean}%`;
    }
  } catch (err) {
    console.error('Failed to load summary:', err);
  }
}

async function loadPlots() {
  try {
    const res = await fetch('/api/plots/list');
    const data = await res.json();
    if (data.success) {
      currentPlots = data.plots;
      renderPlotSelector(data.plots);
    }
  } catch (err) {
    console.error('Failed to load plots list:', err);
  }
}

function renderPlotSelector(plots) {
  plotsListContainer.innerHTML = '';
  plots.forEach((p, idx) => {
    const btn = document.createElement('div');
    btn.className = `plot-item-btn ${idx === 0 ? 'active' : ''}`;
    btn.innerHTML = `
      <h4>${p.title}</h4>
      <p>${p.desc.substring(0, 65)}...</p>
    `;
    btn.addEventListener('click', () => {
      document.querySelectorAll('.plot-item-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectPlot(p);
    });
    plotsListContainer.appendChild(btn);
  });
}

function selectPlot(plot) {
  elCurrentPlotTitle.textContent = plot.title;
  elCurrentPlotImg.src = plot.file;
  elCurrentPlotDesc.textContent = plot.desc;
}

// ─── Pandas GroupBy Execution ──────────────────────────────────────────────
async function executeGroupBy() {
  const payload = {
    group_by: elGroupbyDim.value,
    metric: elGroupbyMetric.value,
    agg: elGroupbyAgg.value
  };

  elBtnRunGroupby.textContent = 'Calculating...';
  try {
    const res = await fetch('/api/pandas/groupby', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    elBtnRunGroupby.textContent = 'Compute GroupBy';

    if (data.success) {
      elThGroupName.textContent = data.group_by;
      elGroupbyTbody.innerHTML = '';
      data.records.forEach(r => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td style="font-weight: 600; color: #f8fafc;">${r[data.group_by]}</td>
          <td style="color: #38bdf8; font-weight: 700;">${r.Result}</td>
          <td>${r.Count}</td>
        `;
        elGroupbyTbody.appendChild(row);
      });
    }
  } catch (err) {
    elBtnRunGroupby.textContent = 'Compute GroupBy';
    console.error('Error running groupby:', err);
  }
}

elBtnRunGroupby.addEventListener('click', executeGroupBy);

// ─── Data Tables Loader ────────────────────────────────────────────────────
async function loadTable(tableType = 'merged') {
  try {
    const res = await fetch(`/api/dataset/records?table=${tableType}&limit=15`);
    const data = await res.json();
    if (data.success && data.data.length > 0) {
      const records = data.data;
      const headers = Object.keys(records[0]).filter(k => !['Skills', 'Projects', 'Activities'].includes(k));

      // Build Headers
      elRecordsThead.innerHTML = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>`;

      // Build Rows
      elRecordsTbody.innerHTML = '';
      records.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = headers.map(h => `<td>${row[h] !== undefined ? row[h] : ''}</td>`).join('');
        elRecordsTbody.appendChild(tr);
      });
    }
  } catch (err) {
    console.error('Failed to load table records:', err);
  }
}

elBtnTableStudents.addEventListener('click', () => {
  resetTableBtnStyles();
  elBtnTableStudents.style.background = 'var(--accent-cyan)';
  elBtnTableStudents.style.color = '#fff';
  loadTable('students');
});

elBtnTableEnrollments.addEventListener('click', () => {
  resetTableBtnStyles();
  elBtnTableEnrollments.style.background = 'var(--accent-cyan)';
  elBtnTableEnrollments.style.color = '#fff';
  loadTable('enrollments');
});

elBtnTableMerged.addEventListener('click', () => {
  resetTableBtnStyles();
  elBtnTableMerged.style.background = 'var(--accent-cyan)';
  elBtnTableMerged.style.color = '#fff';
  loadTable('merged');
});

function resetTableBtnStyles() {
  [elBtnTableStudents, elBtnTableEnrollments, elBtnTableMerged].forEach(btn => {
    btn.style.background = 'var(--bg-input)';
    btn.style.color = 'var(--text-primary)';
  });
}

// ─── NumPy Vectorization & Broadcasting Calculator ────────────────────────
async function calculateNumpy() {
  const payload = {
    math: parseFloat(elCalcMath.value) || 0,
    reading: parseFloat(elCalcReading.value) || 0,
    writing: parseFloat(elCalcWriting.value) || 0
  };

  elBtnCalcNumpy.textContent = 'Calculating...';
  try {
    const res = await fetch('/api/numpy/vectorize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    elBtnCalcNumpy.textContent = 'Calculate with NumPy';

    if (data.success) {
      elNumpyOutputCard.innerHTML = `
        <div style="display: flex; flex-direction: column; gap: 0.85rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 0.6rem;">
            <span style="font-size: 0.82rem; color: var(--text-muted); text-transform: uppercase;">Composite Score</span>
            <span style="font-size: 1.3rem; font-weight: 700; color: #38bdf8;">${data.composite_score} / 100</span>
          </div>
          <div>
            <span style="font-size: 0.78rem; text-transform: uppercase; color: var(--text-muted); display: block; margin-bottom: 0.35rem;">Broadcasted Z-Scores (Standard Deviations from &mu;):</span>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; text-align: center;">
              <div style="background: rgba(6, 182, 212, 0.1); padding: 0.5rem; border-radius: 6px;">
                <div style="font-size: 0.72rem; color: var(--text-muted);">Math Z</div>
                <div style="font-weight: 700; color: ${data.computed_z_scores.Math >= 0 ? '#34d399' : '#fb7185'};">${data.computed_z_scores.Math > 0 ? '+' : ''}${data.computed_z_scores.Math}&sigma;</div>
              </div>
              <div style="background: rgba(99, 102, 241, 0.1); padding: 0.5rem; border-radius: 6px;">
                <div style="font-size: 0.72rem; color: var(--text-muted);">Reading Z</div>
                <div style="font-weight: 700; color: ${data.computed_z_scores.Reading >= 0 ? '#34d399' : '#fb7185'};">${data.computed_z_scores.Reading > 0 ? '+' : ''}${data.computed_z_scores.Reading}&sigma;</div>
              </div>
              <div style="background: rgba(245, 158, 11, 0.1); padding: 0.5rem; border-radius: 6px;">
                <div style="font-size: 0.72rem; color: var(--text-muted);">Writing Z</div>
                <div style="font-weight: 700; color: ${data.computed_z_scores.Writing >= 0 ? '#34d399' : '#fb7185'};">${data.computed_z_scores.Writing > 0 ? '+' : ''}${data.computed_z_scores.Writing}&sigma;</div>
              </div>
            </div>
          </div>
          <div style="font-size: 0.85rem; color: var(--text-secondary);">
            Student performs higher than <strong>${data.percentile_rank}%</strong> of cohort in the Kaggle dataset.
          </div>
        </div>
      `;
    }
  } catch (err) {
    elBtnCalcNumpy.textContent = 'Calculate with NumPy';
    console.error('NumPy calc error:', err);
  }
}

elBtnCalcNumpy.addEventListener('click', calculateNumpy);

// ─── Initialization ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSummary();
  loadPlots();
  executeGroupBy();
  loadTable('merged');
  calculateNumpy();
});
