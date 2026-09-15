// ─── Global State & Configuration ──────────────────────────────────────────
const API_BASE = '';
let allStudents = [];
let currentStats = null;

// ─── DOM Elements ──────────────────────────────────────────────────────────
const navButtons = document.querySelectorAll('.nav-btn');
const tabPanes = document.querySelectorAll('.tab-content');

// Stats DOM
const elStatTotal = document.getElementById('stat-total-students');
const elStatCgpa = document.getElementById('stat-avg-cgpa');
const elStatAttendance = document.getElementById('stat-avg-attendance');
const elStatTop = document.getElementById('stat-top-student');
const elStatTopSub = document.getElementById('stat-top-sub');

// Charts DOM
const elChartBranch = document.getElementById('chart-branch-distribution');
const elChartCgpa = document.getElementById('chart-cgpa-distribution');
const elChartYear = document.getElementById('chart-year-distribution');
const elChartInsights = document.getElementById('chart-insights-summary');

// Directory DOM
const elStudentsGrid = document.getElementById('students-grid-container');
const elInputSearch = document.getElementById('input-search');
const elSelectBranch = document.getElementById('select-branch');
const elSelectYear = document.getElementById('select-year');
const elSelectStatus = document.getElementById('select-status');
const elSelectSort = document.getElementById('select-sort');
const elBtnResetFilters = document.getElementById('btn-reset-filters');
const elBtnRefresh = document.getElementById('btn-refresh-data');

// Modals
const modalAdd = document.getElementById('modal-add-student');
const modalEdit = document.getElementById('modal-edit-student');
const modalView = document.getElementById('modal-view-student');
const btnOpenAddModal = document.getElementById('btn-open-add-modal');
const formAddStudent = document.getElementById('form-add-student');
const formEditStudent = document.getElementById('form-edit-student');

// API Explorer DOM
const endpointCards = document.querySelectorAll('.endpoint-card');
const elTesterMethod = document.getElementById('tester-method');
const elTesterUrl = document.getElementById('tester-url');
const elTesterBtnSend = document.getElementById('tester-btn-send');
const elTesterBodyContainer = document.getElementById('tester-body-container');
const elTesterPayloadBody = document.getElementById('tester-payload-body');
const elTesterResponseOutput = document.getElementById('tester-response-output');
const elTesterStatusBadge = document.getElementById('tester-status-badge');
const elTesterResponseTime = document.getElementById('tester-response-time');

// Toast DOM
const toastContainer = document.getElementById('toast-container');

// ─── Toast System ──────────────────────────────────────────────────────────
function showToast(message, type = 'success') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  const icon = type === 'success' ? '✅' : '⚠️';
  toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ─── Tab Switching ─────────────────────────────────────────────────────────
navButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    const targetTab = btn.getAttribute('data-tab');
    navButtons.forEach(b => b.classList.remove('active'));
    tabPanes.forEach(p => p.classList.remove('active'));

    btn.classList.add('active');
    const targetPane = document.getElementById(`pane-${targetTab}`);
    if (targetPane) targetPane.classList.add('active');
  });
});

// ─── Modal Utility ─────────────────────────────────────────────────────────
function openModal(modal) {
  if (modal) modal.classList.add('open');
}

function closeModal(modal) {
  if (modal) modal.classList.remove('open');
}

document.querySelectorAll('[data-close]').forEach(btn => {
  btn.addEventListener('click', () => {
    const modalId = btn.getAttribute('data-close');
    closeModal(document.getElementById(modalId));
  });
});

window.addEventListener('click', (e) => {
  if (e.target.classList.contains('modal-backdrop')) {
    closeModal(e.target);
  }
});

btnOpenAddModal.addEventListener('click', () => {
  formAddStudent.reset();
  openModal(modalAdd);
});

// ─── API Client Service ───────────────────────────────────────────────────
async function fetchStats() {
  try {
    const res = await fetch(`${API_BASE}/api/stats`);
    const data = await res.json();
    if (data.success) {
      currentStats = data.stats;
      renderStats(data.stats);
    }
  } catch (err) {
    console.error('Error loading stats:', err);
  }
}

async function fetchStudents() {
  try {
    const search = elInputSearch.value.trim();
    const branch = elSelectBranch.value;
    const year = elSelectYear.value;
    const status = elSelectStatus.value;
    const sortVal = elSelectSort.value.split('-');
    const sortBy = sortVal[0];
    const order = sortVal[1];

    const params = new URLSearchParams();
    if (search) params.append('search', search);
    if (branch) params.append('branch', branch);
    if (year) params.append('year', year);
    if (status) params.append('status', status);
    if (sortBy) params.append('sort_by', sortBy);
    if (order) params.append('order', order);

    const res = await fetch(`${API_BASE}/api/students?${params.toString()}`);
    const data = await res.json();
    if (data.success) {
      allStudents = data.students;
      renderStudents(data.students);
    }
  } catch (err) {
    console.error('Error fetching students:', err);
    showToast('Failed to load students from API', 'error');
  }
}

// ─── Rendering Functions ───────────────────────────────────────────────────
function renderStats(stats) {
  if (!stats) return;

  elStatTotal.textContent = stats.total_students || 0;
  elStatCgpa.textContent = stats.average_cgpa ? `${stats.average_cgpa} / 10` : '--';
  elStatAttendance.textContent = stats.average_attendance ? `${stats.average_attendance}%` : '--';
  
  if (stats.top_student && stats.top_student.name) {
    elStatTop.textContent = `${stats.top_student.cgpa} CGPA`;
    elStatTopSub.textContent = `${stats.top_student.name} (${stats.top_student.branch})`;
  } else {
    elStatTop.textContent = '--';
  }

  // Render Branch Chart
  renderBarChart(elChartBranch, stats.branch_distribution || {}, stats.total_students, 'indigo');

  // Render CGPA Chart
  renderBarChart(elChartCgpa, stats.cgpa_distribution || {}, stats.total_students, 'cyan');

  // Render Year Chart
  renderBarChart(elChartYear, stats.year_distribution || {}, stats.total_students, 'emerald');

  // Render Demographics & Study Insights
  renderInsightsSummary(stats);
}

function renderBarChart(container, dataObj, total, colorClass = 'indigo') {
  container.innerHTML = '';
  const entries = Object.entries(dataObj);
  if (entries.length === 0) {
    container.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">No data available.</p>';
    return;
  }

  entries.forEach(([label, count]) => {
    const pct = total > 0 ? Math.round((count / total) * 100) : 0;
    const row = document.createElement('div');
    row.className = 'bar-chart-row';
    row.innerHTML = `
      <div class="bar-label" title="${label}">${label}</div>
      <div class="bar-track">
        <div class="bar-fill ${colorClass}" style="width: ${pct}%;"></div>
      </div>
      <div class="bar-count">${count} <span style="font-size: 0.72rem; color: var(--text-muted);">(${pct}%)</span></div>
    `;
    container.appendChild(row);
  });
}

function renderInsightsSummary(stats) {
  elChartInsights.innerHTML = '';
  const genders = stats.gender_distribution || {};
  const activeRate = stats.total_students > 0 ? Math.round((stats.active_students / stats.total_students) * 100) : 0;
  
  elChartInsights.innerHTML = `
    <div style="display: flex; flex-direction: column; gap: 0.9rem;">
      <div class="meta-item">
        <span class="meta-label">Enrollment Status Ratio</span>
        <div style="display: flex; gap: 1rem; align-items: center; margin-top: 0.2rem;">
          <span style="font-weight: 600; color: #34d399;">Active: ${stats.active_students || 0} (${activeRate}%)</span>
          <span style="font-weight: 600; color: #fb7185;">Inactive: ${stats.inactive_students || 0}</span>
        </div>
      </div>
      <div class="meta-item">
        <span class="meta-label">Average Weekly Study Hours</span>
        <span class="meta-val" style="color: #38bdf8; font-size: 1.1rem; margin-top: 0.2rem;">${stats.average_study_hours || 0} hrs/week</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Gender Diversity</span>
        <div style="display: flex; gap: 1rem; margin-top: 0.2rem;">
          ${Object.entries(genders).map(([g, c]) => `<span><strong>${g}:</strong> ${c}</span>`).join(' | ')}
        </div>
      </div>
    </div>
  `;
}

function renderStudents(students) {
  elStudentsGrid.innerHTML = '';

  if (!students || students.length === 0) {
    elStudentsGrid.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <h3 style="margin-bottom: 0.5rem;">No Students Matched</h3>
        <p style="color: var(--text-secondary); font-size: 0.9rem;">Try adjusting your search criteria or resetting filters.</p>
      </div>
    `;
    return;
  }

  students.forEach(s => {
    const card = document.createElement('div');
    card.className = 'student-card';

    const initials = s.name.split(' ').map(n => n[0]).slice(0, 2).join('').toUpperCase();
    const skillsHtml = (s.skills || []).slice(0, 3).map(sk => `<span class="skill-tag">${sk}</span>`).join('');
    const extraSkillsCount = (s.skills || []).length > 3 ? `<span class="skill-tag">+${(s.skills || []).length - 3}</span>` : '';
    const statusClass = (s.status || '').toLowerCase() === 'active' ? 'active' : 'inactive';

    card.innerHTML = `
      <div>
        <div class="student-card-header">
          <div class="student-avatar-wrap">
            <div class="student-avatar">${initials}</div>
            <div class="student-info">
              <h3>${s.name}</h3>
              <span class="student-id">${s.id} &bull; Year ${s.year}</span>
            </div>
          </div>
          <span class="status-badge ${statusClass}">${s.status || 'Active'}</span>
        </div>

        <div class="student-meta-grid">
          <div class="meta-item">
            <span class="meta-label">Department</span>
            <span class="meta-val" title="${s.branch}">${s.branch}</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">CGPA</span>
            <span class="meta-val cgpa-highlight">⭐ ${s.cgpa}</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">Attendance</span>
            <span class="meta-val">${s.attendance}%</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">City</span>
            <span class="meta-val">${s.city || 'N/A'}</span>
          </div>
        </div>

        <div class="skills-container">
          ${skillsHtml}
          ${extraSkillsCount}
        </div>
      </div>

      <div class="card-actions">
        <button class="btn-icon" title="View Full Profile" onclick="viewStudentDetails('${s.id}')">👁️</button>
        <button class="btn-icon" title="Edit Student" onclick="openEditModal('${s.id}')">✏️</button>
        <button class="btn-icon delete" title="Delete Student" onclick="deleteStudentRecord('${s.id}', '${s.name}')">🗑️</button>
      </div>
    `;

    elStudentsGrid.appendChild(card);
  });
}

// ─── CRUD Action Handlers ─────────────────────────────────────────────────
window.viewStudentDetails = function(studentId) {
  const student = allStudents.find(s => s.id === studentId);
  if (!student) return;

  const modalTitle = document.getElementById('view-student-title');
  const modalBody = document.getElementById('view-student-body');

  modalTitle.textContent = `${student.name} (${student.id})`;
  modalBody.innerHTML = `
    <div style="display: flex; flex-direction: column; gap: 1.25rem;">
      <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(255, 255, 255, 0.02); padding: 1rem; border-radius: var(--radius-md); border: 1px solid var(--border-color);">
        <div>
          <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase;">Department & Standing</div>
          <div style="font-size: 1.05rem; font-weight: 700; color: #a5b4fc;">${student.branch} - Year ${student.year} (Semester ${student.semester || (student.year * 2)})</div>
        </div>
        <span class="status-badge ${student.status.toLowerCase()}">${student.status}</span>
      </div>

      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; text-align: center;">
        <div style="background: rgba(99, 102, 241, 0.1); padding: 0.85rem; border-radius: var(--radius-md); border: 1px solid rgba(99, 102, 241, 0.2);">
          <div style="font-size: 0.75rem; color: var(--text-muted);">CGPA</div>
          <div style="font-size: 1.4rem; font-weight: 700; color: #38bdf8;">${student.cgpa}</div>
        </div>
        <div style="background: rgba(16, 185, 129, 0.1); padding: 0.85rem; border-radius: var(--radius-md); border: 1px solid rgba(16, 185, 129, 0.2);">
          <div style="font-size: 0.75rem; color: var(--text-muted);">Attendance</div>
          <div style="font-size: 1.4rem; font-weight: 700; color: #34d399;">${student.attendance}%</div>
        </div>
        <div style="background: rgba(245, 158, 11, 0.1); padding: 0.85rem; border-radius: var(--radius-md); border: 1px solid rgba(245, 158, 11, 0.2);">
          <div style="font-size: 0.75rem; color: var(--text-muted);">Study Time</div>
          <div style="font-size: 1.4rem; font-weight: 700; color: #fbbf24;">${student.study_hours_weekly || 0}h/wk</div>
        </div>
      </div>

      <div class="student-meta-grid">
        <div class="meta-item"><span class="meta-label">Email</span><span class="meta-val">${student.email}</span></div>
        <div class="meta-item"><span class="meta-label">Phone</span><span class="meta-val">${student.phone || 'N/A'}</span></div>
        <div class="meta-item"><span class="meta-label">Age / Gender</span><span class="meta-val">${student.age || 20} yrs &bull; ${student.gender || 'N/A'}</span></div>
        <div class="meta-item"><span class="meta-label">City</span><span class="meta-val">${student.city || 'N/A'}</span></div>
      </div>

      ${student.math_score ? `
      <div>
        <span class="meta-label" style="display: block; margin-bottom: 0.4rem;">Standardized Kaggle Exam Scores</span>
        <div style="display: flex; gap: 1rem; font-size: 0.9rem;">
          <span>Math: <strong>${student.math_score}/100</strong></span>
          <span>Reading: <strong>${student.reading_score}/100</strong></span>
          <span>Writing: <strong>${student.writing_score}/100</strong></span>
        </div>
      </div>` : ''}

      <div>
        <span class="meta-label" style="display: block; margin-bottom: 0.4rem;">Skills & Competencies</span>
        <div class="skills-container">${(student.skills || []).map(s => `<span class="skill-tag">${s}</span>`).join('')}</div>
      </div>

      <div>
        <span class="meta-label" style="display: block; margin-bottom: 0.4rem;">Projects & Capstones</span>
        <ul style="padding-left: 1.25rem; font-size: 0.88rem; color: var(--text-secondary);">
          ${(student.projects || []).map(p => `<li>${p}</li>`).join('')}
        </ul>
      </div>

      ${student.activities && student.activities.length ? `
      <div>
        <span class="meta-label" style="display: block; margin-bottom: 0.4rem;">Activities & Achievements</span>
        <ul style="padding-left: 1.25rem; font-size: 0.88rem; color: var(--text-secondary);">
          ${student.activities.map(a => `<li>${a}</li>`).join('')}
        </ul>
      </div>` : ''}
    </div>
  `;

  openModal(modalView);
};

window.openEditModal = function(studentId) {
  const student = allStudents.find(s => s.id === studentId);
  if (!student) return;

  document.getElementById('edit-id').value = student.id;
  document.getElementById('edit-name').value = student.name;
  document.getElementById('edit-email').value = student.email;
  document.getElementById('edit-branch').value = student.branch;
  document.getElementById('edit-year').value = student.year;
  document.getElementById('edit-cgpa').value = student.cgpa;
  document.getElementById('edit-attendance').value = student.attendance;
  document.getElementById('edit-status').value = student.status;
  document.getElementById('edit-city').value = student.city || '';
  document.getElementById('edit-skills').value = (student.skills || []).join(', ');

  openModal(modalEdit);
};

window.deleteStudentRecord = async function(studentId, studentName) {
  if (!confirm(`Are you sure you want to delete student ${studentName} (${studentId})?`)) {
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/students/${studentId}`, {
      method: 'DELETE'
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Student ${studentId} deleted successfully!`, 'success');
      await fetchStudents();
      await fetchStats();
    } else {
      showToast(data.message || 'Failed to delete student', 'error');
    }
  } catch (err) {
    console.error('Delete error:', err);
    showToast('Network error while deleting student', 'error');
  }
};

// ─── Form Submissions ──────────────────────────────────────────────────────
formAddStudent.addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    name: document.getElementById('add-name').value.trim(),
    email: document.getElementById('add-email').value.trim(),
    branch: document.getElementById('add-branch').value,
    year: parseInt(document.getElementById('add-year').value),
    cgpa: parseFloat(document.getElementById('add-cgpa').value),
    attendance: parseFloat(document.getElementById('add-attendance').value),
    city: document.getElementById('add-city').value.trim(),
    gender: document.getElementById('add-gender').value,
    skills: document.getElementById('add-skills').value.split(',').map(s => s.trim()).filter(Boolean),
    projects: document.getElementById('add-projects').value.split(',').map(s => s.trim()).filter(Boolean)
  };

  try {
    const res = await fetch(`${API_BASE}/api/students`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      closeModal(modalAdd);
      showToast(`Added new student ${data.student.name} (${data.student.id})!`, 'success');
      await fetchStudents();
      await fetchStats();
    } else {
      showToast(data.message || 'Error adding student', 'error');
    }
  } catch (err) {
    console.error('Error adding student:', err);
    showToast('Failed to connect to server', 'error');
  }
});

formEditStudent.addEventListener('submit', async (e) => {
  e.preventDefault();
  const studentId = document.getElementById('edit-id').value;
  const payload = {
    name: document.getElementById('edit-name').value.trim(),
    email: document.getElementById('edit-email').value.trim(),
    branch: document.getElementById('edit-branch').value,
    year: parseInt(document.getElementById('edit-year').value),
    cgpa: parseFloat(document.getElementById('edit-cgpa').value),
    attendance: parseFloat(document.getElementById('edit-attendance').value),
    status: document.getElementById('edit-status').value,
    city: document.getElementById('edit-city').value.trim(),
    skills: document.getElementById('edit-skills').value.split(',').map(s => s.trim()).filter(Boolean)
  };

  try {
    const res = await fetch(`${API_BASE}/api/students/${studentId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      closeModal(modalEdit);
      showToast(`Updated student ${studentId} successfully!`, 'success');
      await fetchStudents();
      await fetchStats();
    } else {
      showToast(data.message || 'Error updating student', 'error');
    }
  } catch (err) {
    console.error('Error updating student:', err);
    showToast('Failed to connect to server', 'error');
  }
});

// ─── Filter Events ─────────────────────────────────────────────────────────
elInputSearch.addEventListener('input', debounce(() => fetchStudents(), 300));
elSelectBranch.addEventListener('change', () => fetchStudents());
elSelectYear.addEventListener('change', () => fetchStudents());
elSelectStatus.addEventListener('change', () => fetchStudents());
elSelectSort.addEventListener('change', () => fetchStudents());

elBtnResetFilters.addEventListener('click', () => {
  elInputSearch.value = '';
  elSelectBranch.value = '';
  elSelectYear.value = '';
  elSelectStatus.value = '';
  elSelectSort.value = 'id-asc';
  fetchStudents();
});

elBtnRefresh.addEventListener('click', async () => {
  elBtnRefresh.innerHTML = '<span>⏳</span> Refreshing...';
  await fetchStats();
  await fetchStudents();
  elBtnRefresh.innerHTML = '<span>🔄</span> Refresh';
  showToast('All student records synchronized', 'success');
});

// ─── REST API Explorer Interactive Tester ──────────────────────────────────
endpointCards.forEach(card => {
  card.addEventListener('click', () => {
    endpointCards.forEach(c => c.classList.remove('active'));
    card.classList.add('active');

    const method = card.getAttribute('data-method');
    const url = card.getAttribute('data-url');
    const body = card.getAttribute('data-body');

    elTesterMethod.value = method;
    elTesterUrl.value = url;

    if (method === 'POST' || method === 'PUT') {
      elTesterBodyContainer.style.display = 'flex';
      elTesterPayloadBody.value = body ? body.replace(/\\n/g, '\n') : '{\n  \n}';
    } else {
      elTesterBodyContainer.style.display = 'none';
    }
  });
});

elTesterMethod.addEventListener('change', () => {
  const method = elTesterMethod.value;
  if (method === 'POST' || method === 'PUT') {
    elTesterBodyContainer.style.display = 'flex';
  } else {
    elTesterBodyContainer.style.display = 'none';
  }
});

elTesterBtnSend.addEventListener('click', async () => {
  const method = elTesterMethod.value;
  const url = elTesterUrl.value.trim();
  const startTime = performance.now();

  elTesterStatusBadge.textContent = 'Sending...';
  elTesterStatusBadge.style.color = '#fbbf24';
  elTesterResponseOutput.textContent = '// Dispatching request...';

  const options = { method };
  if (method === 'POST' || method === 'PUT') {
    try {
      options.headers = { 'Content-Type': 'application/json' };
      options.body = elTesterPayloadBody.value;
    } catch (e) {
      elTesterResponseOutput.textContent = `// JSON Syntax Error in request payload:\n${e.message}`;
      elTesterStatusBadge.textContent = 'Payload Error';
      return;
    }
  }

  try {
    const res = await fetch(url, options);
    const duration = Math.round(performance.now() - startTime);
    elTesterResponseTime.textContent = `${duration} ms`;
    elTesterStatusBadge.textContent = `${res.status} ${res.statusText}`;

    if (res.status >= 200 && res.status < 300) {
      elTesterStatusBadge.style.color = '#34d399';
    } else {
      elTesterStatusBadge.style.color = '#fb7185';
    }

    const json = await res.json();
    elTesterResponseOutput.textContent = JSON.stringify(json, null, 2);

    // If mutating, refresh table & stats
    if (['POST', 'PUT', 'DELETE'].includes(method)) {
      fetchStudents();
      fetchStats();
    }
  } catch (err) {
    const duration = Math.round(performance.now() - startTime);
    elTesterResponseTime.textContent = `${duration} ms`;
    elTesterStatusBadge.textContent = 'Network Error';
    elTesterStatusBadge.style.color = '#fb7185';
    elTesterResponseOutput.textContent = `// Error executing request:\n${err.message}`;
  }
});

// ─── Utility: Debounce ─────────────────────────────────────────────────────
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

// ─── Initialization ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  fetchStats();
  fetchStudents();
});
