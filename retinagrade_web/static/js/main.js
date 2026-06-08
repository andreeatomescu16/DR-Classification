'use strict';

// ── Sidebar mobile toggle ────────────────────────────────────────────────────
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebarOverlay = document.getElementById('sidebarOverlay');

if (sidebarToggle && sidebar) {
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    if (sidebarOverlay) sidebarOverlay.classList.toggle('active');
  });
}
if (sidebarOverlay) {
  sidebarOverlay.addEventListener('click', () => {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('active');
  });
}

// ── Auto-dismiss alerts ──────────────────────────────────────────────────────
document.querySelectorAll('.alert-close').forEach(btn => {
  btn.addEventListener('click', () => {
    btn.closest('.alert').remove();
  });
});
setTimeout(() => {
  document.querySelectorAll('.alert').forEach(el => {
    el.style.transition = 'opacity 0.5s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 500);
  });
}, 4000);

// ── Upload zone (analyse page) ───────────────────────────────────────────────
const uploadZone = document.getElementById('uploadZone');
const fileInput  = document.getElementById('fileInput');
const preview    = document.getElementById('uploadPreview');
const filename   = document.getElementById('uploadFilename');
const submitBtn  = document.getElementById('analyseSubmit');
const loadingOverlay = document.getElementById('loadingOverlay');

if (uploadZone && fileInput) {
  uploadZone.addEventListener('click', () => fileInput.click());

  uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) applyFile(file);
  });

  fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) applyFile(file);
  });
}

function applyFile(file) {
  if (!file.type.startsWith('image/')) {
    showUploadError('Only image files (JPG, PNG) are accepted.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showUploadError('File must be smaller than 10 MB.');
    return;
  }
  const reader = new FileReader();
  reader.onload = ev => {
    if (preview) { preview.src = ev.target.result; preview.classList.add('visible'); }
    if (filename) { filename.textContent = file.name; filename.classList.add('visible'); }
    if (uploadZone) uploadZone.classList.add('has-image');
    if (submitBtn) submitBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function showUploadError(msg) {
  const err = document.getElementById('uploadError');
  if (err) { err.textContent = msg; err.style.display = 'block'; }
}

// Show loading overlay on form submit
const analyseForm = document.getElementById('analyseForm');
if (analyseForm && loadingOverlay) {
  analyseForm.addEventListener('submit', () => {
    loadingOverlay.classList.add('active');
  });
}

// ── Animate probability bars (detail page) ──────────────────────────────────
document.querySelectorAll('.prob-fill[data-pct]').forEach(fill => {
  const pct = parseFloat(fill.dataset.pct);
  requestAnimationFrame(() => {
    setTimeout(() => { fill.style.width = pct + '%'; }, 80);
  });
});

// ── Bulk-select (history page) ───────────────────────────────────────────────
const selectAll = document.getElementById('selectAll');
const rowChecks = document.querySelectorAll('.row-checkbox');
const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');

if (selectAll) {
  selectAll.addEventListener('change', () => {
    rowChecks.forEach(cb => { cb.checked = selectAll.checked; });
    updateBulkBtn();
  });
}
rowChecks.forEach(cb => cb.addEventListener('change', updateBulkBtn));

function updateBulkBtn() {
  if (!bulkDeleteBtn) return;
  const anyChecked = Array.from(rowChecks).some(cb => cb.checked);
  bulkDeleteBtn.disabled = !anyChecked;
}

// ── Confirm dangerous actions ────────────────────────────────────────────────
document.querySelectorAll('[data-confirm]').forEach(el => {
  el.addEventListener('click', e => {
    if (!confirm(el.dataset.confirm)) e.preventDefault();
  });
});

// ── Dashboard chart (Chart.js) ───────────────────────────────────────────────
function initDashboardChart(canvasId, counts, labels) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === 'undefined') return;
  const colors = ['#51cf66', '#94d82d', '#ffd43b', '#ff922b', '#ff6b6b'];
  new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: counts,
        backgroundColor: colors.map(c => c + '33'),
        borderColor: colors,
        borderWidth: 1.5,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#8888aa', font: { size: 10 } },
          grid: { color: 'rgba(42,42,58,0.6)' },
        },
        y: {
          ticks: { color: '#8888aa', font: { size: 10 }, stepSize: 1 },
          grid: { color: 'rgba(42,42,58,0.6)' },
          beginAtZero: true,
        },
      },
    },
  });
}

// ── Detail page probability chart ────────────────────────────────────────────
function initProbChart(canvasId, probs, labels) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === 'undefined') return;
  const colors = ['#51cf66', '#94d82d', '#ffd43b', '#ff922b', '#ff6b6b'];
  new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: probs.map(p => (p * 100).toFixed(1)),
        backgroundColor: colors.map(c => c + '33'),
        borderColor: colors,
        borderWidth: 1.5,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          max: 100,
          ticks: { color: '#8888aa', font: { size: 10 }, callback: v => v + '%' },
          grid: { color: 'rgba(42,42,58,0.6)' },
        },
        y: {
          ticks: { color: '#8888aa', font: { size: 10 } },
          grid: { display: false },
        },
      },
    },
  });
}

// ── Admin grade distribution chart ───────────────────────────────────────────
function initDonutChart(canvasId, counts, labels) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === 'undefined') return;
  const colors = ['#51cf66', '#94d82d', '#ffd43b', '#ff922b', '#ff6b6b'];
  new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: counts,
        backgroundColor: colors.map(c => c + '99'),
        borderColor: colors,
        borderWidth: 1.5,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#8888aa', font: { size: 10 }, padding: 14 },
        },
      },
    },
  });
}
