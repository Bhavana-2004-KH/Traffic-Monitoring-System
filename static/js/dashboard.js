/**
 * dashboard.js
 * Simplified controller for Traffic Video Analyzer Dashboard.
 *
 * Exclusively manages 6 sections:
 *  1. Video Analysis
 *  2. Vehicle Counts
 *  3. Turning Analysis
 *  4. Traffic Forecast Graph
 *  5. Adaptive Green-Time Allocation
 *  6. Processed Video and Reports, if available
 *
 * Rules Compliance:
 *  - Uses existing Python program results without duplicate pipelines.
 *  - No fabricated or hardcoded counts/forecasts/green-times.
 *  - Total vehicle count is strictly the sum of distinct class counts.
 *  - No red-time allocations or red badges.
 *  - Safe video isolation (no mixing between videos or displaying stale data).
 */

let activeSection = 'video-analysis';
let statusPollInterval = null;
let forecastChartInstance = null;

let currentVideoSelection = null;
let currentVideoFolder = 'sample_videos';
let currentVideoMediaUrl = null;
let currentProcessedMediaUrl = null;
let currentPlayerMode = 'preview'; // 'preview' | 'processed'

const SECTION_METADATA = {
  'video-analysis': {
    title: '1. Video Analysis',
    subtitle: 'Select or upload traffic footage and execute the video analysis pipeline'
  },
  'vehicle-counts': {
    title: '2. Vehicle Counts',
    subtitle: 'Accurate distinct vehicle counts produced by existing YOLOv8 and BoT-SORT tracking'
  },
  'turning-analysis': {
    title: '3. Turning Analysis',
    subtitle: 'Actual movement patterns classified across approach zones by the existing program'
  },
  'traffic-forecast': {
    title: '4. Traffic Forecast Graph',
    subtitle: 'Compact comparison of actual historical traffic counts vs Auto-ARIMA predicted values'
  },
  'green-time': {
    title: '5. Adaptive Green-Time Allocation',
    subtitle: 'Actual road-wise traffic demand and allocated green light durations returned by the existing signal controller'
  },
  'processed-videos-reports': {
    title: '6. Processed Video & Reports',
    subtitle: 'Access generated annotated recordings and export data produced by the analysis programs'
  }
};

// --------------------------------------------------------------------------
// Navigation & Section Switching
// --------------------------------------------------------------------------

function switchSection(sectionId) {
  if (!SECTION_METADATA[sectionId]) return;
  activeSection = sectionId;

  document.querySelectorAll('.sidebar-nav .nav-link').forEach(link => {
    if (link.getAttribute('data-section') === sectionId) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  document.querySelectorAll('.dashboard-section').forEach(sec => {
    sec.classList.remove('active');
  });

  const targetSection = document.getElementById(`sec-${sectionId}`);
  if (targetSection) {
    targetSection.classList.add('active');
  }

  const meta = SECTION_METADATA[sectionId];
  const titleEl = document.getElementById('page-title');
  const subEl = document.getElementById('page-subtitle');
  if (titleEl) titleEl.textContent = meta.title;
  if (subEl) subEl.textContent = meta.subtitle;

  if (sectionId === 'processed-videos-reports') {
    loadOutputsDirectory();
  } else if (sectionId === 'traffic-forecast' && forecastChartInstance) {
    forecastChartInstance.resize();
  }
}

// --------------------------------------------------------------------------
// Initialization
// --------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  setupUploadHandlers();
  setupAnalysisForm();
  setupVideoErrorHandlers();
  loadSampleVideos();
  loadOutputsDirectory();
  checkCurrentStatusOnLoad();
});

function setupNavigation() {
  document.querySelectorAll('.sidebar-nav .nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const section = link.getAttribute('data-section');
      switchSection(section);
    });
  });
}

function setupVideoErrorHandlers() {
  const mainPlayer = document.getElementById('main-video-player');
  const procPlayer = document.getElementById('processed-video-element');
  const errBox = document.getElementById('video-error-message');
  const errText = document.getElementById('video-error-text');

  [mainPlayer, procPlayer].forEach(p => {
    if (p) {
      p.addEventListener('error', () => {
        if (p.src && p.src !== window.location.href) {
          if (errBox) errBox.style.display = 'flex';
          if (errText) errText.textContent = 'Video failed to load or unsupported format. Please verify the video file.';
        }
      });
      p.addEventListener('loadeddata', () => {
        if (errBox) errBox.style.display = 'none';
      });
    }
  });
}

// --------------------------------------------------------------------------
// Video Player Mode Switcher: Preview vs Processed Output
// --------------------------------------------------------------------------

function switchPlayerMode(mode) {
  currentPlayerMode = mode;
  const btnPrev = document.getElementById('btn-tab-preview');
  const btnProc = document.getElementById('btn-tab-processed');
  const subLabel = document.getElementById('player-mode-subtitle');
  const placeholder = document.getElementById('video-placeholder');
  const placeholderText = document.getElementById('video-placeholder-text');
  const player = document.getElementById('main-video-player');
  const errBox = document.getElementById('video-error-message');

  if (errBox) errBox.style.display = 'none';

  if (mode === 'preview') {
    if (btnPrev) btnPrev.classList.add('active');
    if (btnProc) btnProc.classList.remove('active');
    if (subLabel) subLabel.textContent = 'Previewing selected footage';

    if (currentVideoMediaUrl) {
      if (placeholder) placeholder.style.display = 'none';
      if (player) {
        player.style.display = 'block';
        if (player.src !== currentVideoMediaUrl) {
          player.src = currentVideoMediaUrl;
          player.load();
        }
      }
    } else {
      if (placeholder) placeholder.style.display = 'block';
      if (placeholderText) placeholderText.textContent = 'Select a traffic video above to play preview';
      if (player) player.style.display = 'none';
    }
  } else {
    // Processed mode
    if (btnPrev) btnPrev.classList.remove('active');
    if (btnProc) btnProc.classList.add('active');
    if (subLabel) subLabel.textContent = 'Playing processed footage with annotations';

    if (currentProcessedMediaUrl) {
      if (placeholder) placeholder.style.display = 'none';
      if (player) {
        player.style.display = 'block';
        player.src = currentProcessedMediaUrl;
        player.load();
      }
    } else {
      if (placeholder) placeholder.style.display = 'block';
      if (placeholderText) placeholderText.textContent = 'Processed output video is not available yet for this video. Please run an analysis first.';
      if (player) player.style.display = 'none';
    }
  }
}

// --------------------------------------------------------------------------
// Sample & Available Videos Loading
// --------------------------------------------------------------------------

async function loadSampleVideos() {
  try {
    const res = await fetch('/api/sample-videos');
    const data = await res.json();
    if (!data.videos) return;

    const selectEl = document.getElementById('video-select');
    if (!selectEl) return;

    selectEl.innerHTML = '<option value="">-- Select from available videos --</option>';

    data.videos.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v.name;
      opt.textContent = `${v.name} (${v.size_mb} MB) - ${v.type}`;
      opt.dataset.folder = v.folder;
      opt.dataset.mediaUrl = v.media_url;
      selectEl.appendChild(opt);
    });

    selectEl.addEventListener('change', (e) => {
      const val = e.target.value;
      const selectedOpt = selectEl.selectedOptions[0];
      onVideoSelected(val, selectedOpt ? selectedOpt.dataset.folder : 'sample_videos');
    });

    if (data.videos.length > 0 && !currentVideoSelection) {
      selectEl.value = data.videos[0].name;
      onVideoSelected(data.videos[0].name, data.videos[0].folder);
    }
  } catch (err) {
    console.error('Error loading sample videos:', err);
  }
}

function onVideoSelected(filename, folder = 'sample_videos') {
  if (!filename) {
    currentVideoSelection = null;
    currentVideoMediaUrl = null;
    currentProcessedMediaUrl = null;
    clearResultDisplays();
    updateVideoStatusNotice(null, false);
    switchPlayerMode('preview');
    const badge = document.getElementById('active-video-indicator');
    if (badge) {
      badge.textContent = 'No Video Selected';
      badge.className = 'badge badge-idle';
    }
    return;
  }

  currentVideoSelection = filename;
  currentVideoFolder = folder || 'sample_videos';
  currentVideoMediaUrl = `/media/${currentVideoFolder}/${filename}`;

  const stem = filename.replace(/\.[^/.]+$/, '');
  const fileLbl = document.getElementById('selected-file-label');
  if (fileLbl) fileLbl.textContent = filename;

  const activeBadge = document.getElementById('active-video-indicator');
  if (activeBadge) {
    activeBadge.textContent = filename;
    activeBadge.className = 'badge badge-completed';
  }

  // Clear previous results immediately to avoid showing stale data
  clearResultDisplays();

  // Reset to preview
  switchPlayerMode('preview');

  // Load results for this video stem
  loadResults(stem);
}

// --------------------------------------------------------------------------
// File Upload Handling
// --------------------------------------------------------------------------

function setupUploadHandlers() {
  const fileInput = document.getElementById('video-file-input');
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadFile(e.target.files[0]);
      }
    });
  }
}

async function uploadFile(file) {
  const allowedExts = ['.mp4', '.avi', '.mov', '.mkv'];
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  const statusMsg = document.getElementById('upload-status-msg');

  if (!allowedExts.includes(ext)) {
    if (statusMsg) {
      statusMsg.innerHTML = `<span style="color: var(--accent-red);">Error: Unsupported format '${ext}'. Allowed: .mp4, .avi, .mov, .mkv</span>`;
    }
    alert(`Unsupported file format '${ext}'. Allowed formats: MP4, AVI, MOV, MKV`);
    return;
  }

  if (statusMsg) {
    statusMsg.innerHTML = `<span style="color: var(--accent-blue);">Uploading ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)...</span>`;
  }

  const formData = new FormData();
  formData.append('video', file);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();

    if (data.success) {
      if (statusMsg) {
        statusMsg.innerHTML = `<span style="color: var(--primary-green-text); font-weight: 600;">Uploaded successfully: ${data.filename}</span>`;
      }
      document.getElementById('selected-file-label').textContent = data.filename;

      await loadSampleVideos();
      const selectEl = document.getElementById('video-select');
      if (selectEl) selectEl.value = data.filename;
      onVideoSelected(data.filename, 'uploads');
    } else {
      if (statusMsg) {
        statusMsg.innerHTML = `<span style="color: var(--accent-red);">${data.error || 'Upload failed.'}</span>`;
      }
    }
  } catch (err) {
    if (statusMsg) {
      statusMsg.innerHTML = `<span style="color: var(--accent-red);">Upload error: ${err.message}</span>`;
    }
  }
}

// --------------------------------------------------------------------------
// Start & Cancel Analysis Pipeline
// --------------------------------------------------------------------------

function setupAnalysisForm() {
  const modeRadios = document.querySelectorAll('input[name="road-mode"]');
  const roadCountContainer = document.getElementById('road-count-container');

  modeRadios.forEach(r => {
    r.addEventListener('change', (e) => {
      if (e.target.value === 'single') {
        if (roadCountContainer) roadCountContainer.style.display = 'none';
      } else {
        if (roadCountContainer) roadCountContainer.style.display = 'block';
      }
    });
  });

  const btnStart = document.getElementById('btn-start-analysis');
  if (btnStart) {
    btnStart.addEventListener('click', triggerStartAnalysis);
  }

  const btnCancel = document.getElementById('btn-cancel-analysis');
  if (btnCancel) {
    btnCancel.addEventListener('click', triggerCancelAnalysis);
  }
}

let isAnalysisStarting = false;

async function triggerStartAnalysis() {
  if (isAnalysisStarting) return;

  const selectEl = document.getElementById('video-select');
  const selectedVideo = currentVideoSelection || (selectEl ? selectEl.value : null);

  if (!selectedVideo) {
    alert('Please select or upload a traffic video first.');
    return;
  }

  isAnalysisStarting = true;
  setProcessingUIState(true);

  const modeRadio = document.querySelector('input[name="road-mode"]:checked');
  const mode = modeRadio ? modeRadio.value : 'intersection';
  const roads = mode === 'single' ? 1 : parseInt(document.getElementById('road-count-select').value || '4');

  try {
    const res = await fetch('/api/start-analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_name: selectedVideo,
        mode: mode,
        roads: roads
      })
    });

    const data = await res.json();
    if (!data.success) {
      alert(`Could not start analysis: ${data.error}`);
      setProcessingUIState(false);
      isAnalysisStarting = false;
      return;
    }

    startStatusPolling();
  } catch (err) {
    alert(`Failed to start analysis: ${err.message}`);
    setProcessingUIState(false);
  } finally {
    isAnalysisStarting = false;
  }
}

async function triggerCancelAnalysis() {
  if (!confirm('Are you sure you want to cancel the active traffic analysis?')) return;
  try {
    const res = await fetch('/api/cancel-analysis', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      stopStatusPolling();
      setProcessingUIState(false);
      document.getElementById('sidebar-status-text').textContent = 'Ready';
      document.getElementById('global-status-badge').className = 'badge badge-idle';
      document.getElementById('global-status-badge').textContent = 'System Ready';
    }
  } catch (err) {
    console.error('Cancel request error:', err);
  }
}

// --------------------------------------------------------------------------
// Status Polling Loop
// --------------------------------------------------------------------------

function startStatusPolling() {
  if (statusPollInterval) clearInterval(statusPollInterval);
  statusPollInterval = setInterval(pollStatus, 800);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkCurrentStatusOnLoad() {
  try {
    const res = await fetch('/api/status');
    const state = await res.json();
    if (state.status === 'processing') {
      setProcessingUIState(true);
      startStatusPolling();
    } else if (state.status === 'completed') {
      loadResults(state.video_stem);
    } else {
      setProcessingUIState(false);
    }
  } catch (err) {
    console.error('Error checking initial status:', err);
  }
}

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    const state = await res.json();

    updateStatusUI(state);

    if (state.status === 'completed') {
      stopStatusPolling();
      setProcessingUIState(false);
      onAnalysisCompleted(state);
    } else if (state.status === 'failed') {
      stopStatusPolling();
      setProcessingUIState(false);
      if (state.error) {
        alert(state.error);
      }
    } else if (state.status === 'cancelled') {
      stopStatusPolling();
      setProcessingUIState(false);
    }
  } catch (err) {
    console.error('Polling status error:', err);
  }
}

function updateStatusUI(state) {
  const pct = state.progress || 0;
  const progressText = `${pct}%`;

  const fill = document.getElementById('analysis-progress-fill');
  if (fill) fill.style.width = progressText;

  const pctLabel = document.getElementById('analysis-progress-percent');
  if (pctLabel) pctLabel.textContent = progressText;

  document.getElementById('analysis-stats-frames').textContent = `Frame: ${state.current_frame} / ${state.total_frames || '?'}`;
  document.getElementById('analysis-stats-active').textContent = `Active: ${state.active_vehicles}`;
  document.getElementById('analysis-stats-total').textContent = `Total: ${state.total_counted}`;
  document.getElementById('analysis-stats-elapsed').textContent = `Elapsed: ${state.elapsed_seconds || 0}s`;

  const pulse = document.getElementById('sidebar-pulse-dot');
  const statusTxt = document.getElementById('sidebar-status-text');
  const globalBadge = document.getElementById('global-status-badge');

  if (state.status === 'processing') {
    pulse.className = 'pulse-dot processing';
    statusTxt.textContent = `Processing (${pct}%)`;
    globalBadge.className = 'badge badge-processing';
    globalBadge.textContent = `Processing ${pct}%`;
  } else if (state.status === 'completed') {
    pulse.className = 'pulse-dot completed';
    statusTxt.textContent = 'Completed';
    globalBadge.className = 'badge badge-completed';
    globalBadge.textContent = 'Analysis Complete';
  } else if (state.status === 'failed') {
    pulse.className = 'pulse-dot failed';
    statusTxt.textContent = 'Failed';
    globalBadge.className = 'badge badge-failed';
    globalBadge.textContent = 'Failed';
  } else {
    pulse.className = 'pulse-dot';
    statusTxt.textContent = 'Ready';
    globalBadge.className = 'badge badge-idle';
    globalBadge.textContent = 'System Ready';
  }

  // Update live terminal log
  const terminal = document.getElementById('analysis-terminal-log');
  if (terminal && state.log_lines && state.log_lines.length > 0) {
    terminal.textContent = state.log_lines.join('\n');
    terminal.scrollTop = terminal.scrollHeight;
  }
}

function setProcessingUIState(isProcessing) {
  const btnStart = document.getElementById('btn-start-analysis');
  const btnCancel = document.getElementById('btn-cancel-analysis');
  const pContainer = document.getElementById('analysis-progress-container');

  if (isProcessing) {
    if (btnStart) btnStart.disabled = true;
    if (btnCancel) btnCancel.style.display = 'inline-flex';
    if (pContainer) pContainer.classList.add('active');
  } else {
    if (btnStart) btnStart.disabled = false;
    if (btnCancel) btnCancel.style.display = 'none';
  }
}

function onAnalysisCompleted(state) {
  loadResults(state.video_stem);
  loadOutputsDirectory();
  // Automatically switch to processed output video playback
  switchPlayerMode('processed');
}

// --------------------------------------------------------------------------
// Load & Render Actual Analysis Results
// --------------------------------------------------------------------------

async function loadResults(videoStem) {
  if (!videoStem) {
    clearResultDisplays();
    updateVideoStatusNotice(null, false);
    return;
  }
  try {
    const url = `/api/results?video=${encodeURIComponent(videoStem)}`;
    const res = await fetch(url);
    if (!res.ok) {
      clearResultDisplays();
      updateVideoStatusNotice(videoStem, false);
      return;
    }
    const data = await res.json();
    if (!data.success || !data.summary) {
      clearResultDisplays();
      updateVideoStatusNotice(videoStem, false);
      return;
    }

    renderSummaryData(data.summary, data.available_files, data.video_stem);
    renderForecastChart(data.forecast_chart);
    updateVideoStatusNotice(videoStem, true);
  } catch (err) {
    console.error('Error loading results for', videoStem, err);
    clearResultDisplays();
    updateVideoStatusNotice(videoStem, false);
  }
}

function updateVideoStatusNotice(videoStem, hasResults) {
  const noticeBox = document.getElementById('video-analysis-status-notice');
  if (!noticeBox) return;

  if (!videoStem) {
    noticeBox.style.display = 'none';
    return;
  }

  noticeBox.style.display = 'block';
  if (hasResults) {
    noticeBox.className = 'alert alert-info';
    noticeBox.innerHTML = `
      <strong>Analysis results loaded for ${videoStem}.</strong>
      Displaying actual vehicle counts, turning patterns, forecast graph, and adaptive green-time allocation.
    `;
  } else {
    noticeBox.className = 'alert alert-warning';
    noticeBox.innerHTML = `
      <strong>No completed analysis found for ${videoStem}.</strong>
      Click <em>Start Analysis</em> to execute vehicle detection and tracking.
    `;
  }
}

function clearResultDisplays() {
  // 1. Vehicle Counts
  const kpiIds = [
    'kpi-total-vehicles', 'kpi-cars', 'kpi-twowheelers', 'kpi-buses',
    'kpi-trucks', 'kpi-threewheelers', 'kpi-lcvs', 'kpi-bicycles'
  ];
  kpiIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '—';
  });

  const cntBadge = document.getElementById('counts-total-badge');
  if (cntBadge) cntBadge.textContent = 'Total Vehicle Count: —';

  const tableBody = document.getElementById('vehicle-counts-table-body');
  if (tableBody) {
    tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; color: var(--text-muted); padding: 18px;">Waiting for analysis results. Please select and analyze a video.</td></tr>';
  }

  // 2. Turning Analysis
  const turnIds = ['turning-val-left', 'turning-val-straight', 'turning-val-right'];
  turnIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '—';
  });

  const turnBadge = document.getElementById('turning-status-badge');
  if (turnBadge) turnBadge.textContent = 'Waiting for Analysis';

  const turnEmpty = document.getElementById('turning-empty-notice');
  if (turnEmpty) turnEmpty.style.display = 'none';
  const turnSingle = document.getElementById('turning-single-road-notice');
  if (turnSingle) turnSingle.style.display = 'none';
  const turnContent = document.getElementById('turning-content-area');
  if (turnContent) turnContent.style.display = 'block';

  // 3. Auto-ARIMA Forecast
  const fnv = document.getElementById('forecast-next-value');
  if (fnv) fnv.textContent = 'Waiting for analysis...';

  const fod = document.getElementById('forecast-order-display');
  if (fod) fod.textContent = 'Fitted ARIMA Model: N/A';

  const fBadge = document.getElementById('forecast-status-badge');
  if (fBadge) { fBadge.textContent = 'Waiting'; fBadge.className = 'badge badge-idle'; }

  renderForecastChart({
    has_data: false,
    has_comparison: false,
    reason: 'Auto-ARIMA forecasting requires at least 2 completed observation intervals (10 seconds of video time) to fit and predict traffic counts.'
  });

  // 4. Adaptive Green-Time
  renderGreenTimeTable({}, 'intersection');

  // 5. Processed Video Media and Download Buttons
  currentProcessedMediaUrl = null;
  const pPlayer = document.getElementById('processed-video-element');
  const pPlaceholder = document.getElementById('processed-video-placeholder');
  const btnDownload = document.getElementById('btn-download-processed-video');
  if (pPlayer) { pPlayer.src = ''; pPlayer.style.display = 'none'; }
  if (pPlaceholder) pPlaceholder.style.display = 'block';
  if (btnDownload) btnDownload.style.display = 'none';
}

function renderSummaryData(summary, availableFiles, videoStem) {
  const metadata = summary.video_metadata || {};
  const counts = summary.cumulative_vehicle_counts || {};
  const turning = summary.turning_patterns || {};
  const forecasting = summary.forecasting || {};
  const signal = summary.signal_timing || {};
  const mode = metadata.mode || 'intersection';

  // ------------------------------------------------------------------------
  // 1. Vehicle Counts
  // Supported classes: Cars, Two-Wheelers, Buses, Trucks, Three-Wheelers, LCVs, Bicycles
  // ------------------------------------------------------------------------
  const classList = [
    { name: 'Car', label: 'Cars', kpiId: 'kpi-cars' },
    { name: 'Two Wheeler', label: 'Two-Wheelers', kpiId: 'kpi-twowheelers' },
    { name: 'Bus', label: 'Buses', kpiId: 'kpi-buses' },
    { name: 'Truck', label: 'Trucks', kpiId: 'kpi-trucks' },
    { name: 'Three Wheeler', label: 'Three-Wheelers', kpiId: 'kpi-threewheelers' },
    { name: 'LCV', label: 'LCVs', kpiId: 'kpi-lcvs' },
    { name: 'Bicycle', label: 'Bicycles', kpiId: 'kpi-bicycles' }
  ];

  // Calculate TOTAL VEHICLE COUNT strictly as the sum of distinct class counts
  let totalVehicleCount = 0;
  classList.forEach(item => {
    const cnt = (counts[item.name] !== undefined && counts[item.name] !== null) ? Number(counts[item.name]) : 0;
    totalVehicleCount += cnt;

    const kpiEl = document.getElementById(item.kpiId);
    if (kpiEl) kpiEl.textContent = cnt;
  });

  // Include any additional supported classes present in output
  Object.keys(counts).forEach(cname => {
    if (!classList.some(item => item.name === cname)) {
      totalVehicleCount += Number(counts[cname] || 0);
    }
  });

  // Display Total Vehicle Count
  const totalKpiEl = document.getElementById('kpi-total-vehicles');
  if (totalKpiEl) totalKpiEl.textContent = totalVehicleCount;

  const totalBadgeEl = document.getElementById('counts-total-badge');
  if (totalBadgeEl) totalBadgeEl.textContent = `Total Vehicle Count: ${totalVehicleCount}`;

  const tableBody = document.getElementById('vehicle-counts-table-body');
  if (tableBody) {
    tableBody.innerHTML = '';
    classList.forEach(item => {
      const cnt = counts[item.name] !== undefined ? counts[item.name] : 0;
      const pct = totalVehicleCount > 0 ? ((cnt / totalVehicleCount) * 100).toFixed(1) : '0.0';

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${item.name}</strong></td>
        <td>${item.label}</td>
        <td style="text-align: right; font-weight: 700; color: var(--text-primary); font-size: 15px;">${cnt}</td>
        <td style="text-align: right;">
          <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
            <div style="width: 80px; height: 6px; background-color: #e2e8f0; border-radius: 3px; overflow: hidden;">
              <div style="height: 100%; width: ${pct}%; background-color: var(--primary-green);"></div>
            </div>
            <span style="font-size: 12px; color: var(--text-secondary); width: 42px;">${pct}%</span>
          </div>
        </td>
      `;
      tableBody.appendChild(tr);
    });
  }

  // ------------------------------------------------------------------------
  // 2. Turning Analysis
  // ------------------------------------------------------------------------
  const turningArea = document.getElementById('turning-content-area');
  const turningSingleNotice = document.getElementById('turning-single-road-notice');
  const turningEmptyNotice = document.getElementById('turning-empty-notice');
  const turningBadge = document.getElementById('turning-status-badge');

  if (mode === 'single') {
    if (turningArea) turningArea.style.display = 'none';
    if (turningSingleNotice) turningSingleNotice.style.display = 'flex';
    if (turningEmptyNotice) turningEmptyNotice.style.display = 'none';
    if (turningBadge) turningBadge.textContent = 'Single Road Mode';
  } else {
    if (turningSingleNotice) turningSingleNotice.style.display = 'none';
    if (turningBadge) turningBadge.textContent = 'Intersection Mode';

    const hasTurningData = turning && (
      turning.straight !== undefined ||
      turning.left_turn !== undefined ||
      turning.right_turn !== undefined
    );

    if (hasTurningData) {
      if (turningArea) turningArea.style.display = 'block';
      if (turningEmptyNotice) turningEmptyNotice.style.display = 'none';

      const left = turning.left_turn !== undefined ? turning.left_turn : (turning.left || 0);
      const straight = turning.straight !== undefined ? turning.straight : 0;
      const right = turning.right_turn !== undefined ? turning.right_turn : (turning.right || 0);

      document.getElementById('turning-val-left').textContent = left;
      document.getElementById('turning-val-straight').textContent = straight;
      document.getElementById('turning-val-right').textContent = right;
    } else {
      if (turningArea) turningArea.style.display = 'none';
      if (turningEmptyNotice) turningEmptyNotice.style.display = 'block';
    }
  }

  // ------------------------------------------------------------------------
  // 3. Auto-ARIMA Forecast Header
  // ------------------------------------------------------------------------
  const forecastVal = forecasting.final_forecast;
  const forecastStatus = forecasting.status || 'Preparing forecast...';
  const arimaOrder = forecasting.arima_order || 'N/A';

  const valText = (forecastVal !== null && forecastVal !== undefined)
    ? `Next Traffic: ${forecastVal} vehicles`
    : forecastStatus;

  document.getElementById('forecast-next-value').textContent = valText;
  document.getElementById('forecast-order-display').textContent = `Fitted ARIMA Model: ${arimaOrder}`;

  const fBadge = document.getElementById('forecast-status-badge');
  if (forecastVal !== null && forecastVal !== undefined) {
    if (fBadge) { fBadge.textContent = 'Forecast Ready'; fBadge.className = 'badge badge-completed'; }
  } else {
    if (fBadge) { fBadge.textContent = 'Warming Up'; fBadge.className = 'badge badge-idle'; }
  }

  // ------------------------------------------------------------------------
  // 4. Adaptive Green-Time Allocation (NO RED BADGES)
  // ------------------------------------------------------------------------
  renderGreenTimeTable(signal.road_signals || {}, mode, signal.recommended_green_seconds);

  // ------------------------------------------------------------------------
  // 5. Setup Processed Video URL
  // ------------------------------------------------------------------------
  const pPlayer = document.getElementById('processed-video-element');
  const pPlaceholder = document.getElementById('processed-video-placeholder');
  const btnDownload = document.getElementById('btn-download-processed-video');

  if (availableFiles && availableFiles.video) {
    currentProcessedMediaUrl = `/media/outputs/${availableFiles.video}`;
    const dlUrl = `/download/output/${availableFiles.video}`;

    if (pPlayer) {
      pPlayer.src = currentProcessedMediaUrl;
      pPlayer.style.display = 'block';
    }
    if (pPlaceholder) pPlaceholder.style.display = 'none';

    if (btnDownload) {
      btnDownload.style.display = 'inline-flex';
      btnDownload.href = dlUrl;
    }
  } else {
    currentProcessedMediaUrl = null;
    if (pPlayer) {
      pPlayer.src = '';
      pPlayer.style.display = 'none';
    }
    if (pPlaceholder) pPlaceholder.style.display = 'block';
    if (btnDownload) btnDownload.style.display = 'none';
  }
}

// --------------------------------------------------------------------------
// Render Clean Green-Time Allocation Table (NO RED BADGES)
// --------------------------------------------------------------------------

function renderGreenTimeTable(roadSignals, mode = 'intersection', singleGreenSec = 20) {
  const tableBody = document.getElementById('signal-table-body');
  const badge = document.getElementById('green-time-mode-badge');
  if (!tableBody) return;

  if (mode === 'single') {
    if (badge) badge.textContent = 'Single Road Mode';
    const gSec = singleGreenSec || (roadSignals['Single Road'] ? roadSignals['Single Road'].green_seconds : 20);
    const demand = roadSignals['Single Road'] ? roadSignals['Single Road'].vehicles : 0;

    tableBody.innerHTML = `
      <tr>
        <td><strong>Single Road</strong></td>
        <td style="text-align: right; font-weight: 600;">${demand} vehicles</td>
        <td style="text-align: right; font-weight: 700; color: var(--primary-green-hover); font-size: 15px;">${gSec} sec</td>
      </tr>
    `;
    return;
  }

  // Intersection mode
  if (badge) badge.textContent = 'Intersection Mode';
  const roads = Object.keys(roadSignals);

  if (roads.length === 0) {
    tableBody.innerHTML = '<tr><td colspan="3" style="text-align:center; color: var(--text-muted); padding: 20px;">No green-time allocation data available. Please run video analysis.</td></tr>';
    return;
  }

  let rowsHtml = '';
  roads.forEach(rname => {
    const rinfo = roadSignals[rname];
    const demand = (rinfo && rinfo.vehicles !== undefined) ? rinfo.vehicles : '—';
    const gSec = (rinfo && rinfo.green_seconds !== undefined) ? rinfo.green_seconds : 20;

    rowsHtml += `
      <tr>
        <td><strong>${rname}</strong></td>
        <td style="text-align: right; font-weight: 600;">${demand} ${demand !== '—' ? 'vehicles' : ''}</td>
        <td style="text-align: right; font-weight: 700; color: var(--primary-green-hover); font-size: 15px;">${gSec} sec</td>
      </tr>
    `;
  });

  tableBody.innerHTML = rowsHtml;
}

// --------------------------------------------------------------------------
// Forecast Chart.js Rendering (Real Data Only - Compact Line Graph)
// --------------------------------------------------------------------------

function renderForecastChart(chartData) {
  const emptyBox = document.getElementById('forecast-chart-empty');
  const emptyMsg = document.getElementById('forecast-chart-empty-msg');
  const chartWrapper = document.getElementById('forecast-chart-wrapper');

  if (!chartData || !chartData.has_data || !chartData.has_comparison) {
    const reasonText = (chartData && chartData.reason)
      ? chartData.reason
      : 'Forecast data is unavailable. Auto-ARIMA forecasting requires at least 2 completed observation intervals (10 seconds of video time) to fit and predict traffic counts.';

    if (emptyBox) emptyBox.style.display = 'block';
    if (emptyMsg) emptyMsg.textContent = reasonText;
    if (chartWrapper) chartWrapper.style.display = 'none';

    if (forecastChartInstance) {
      forecastChartInstance.destroy();
      forecastChartInstance = null;
    }
    return;
  }

  if (emptyBox) emptyBox.style.display = 'none';
  if (chartWrapper) chartWrapper.style.display = 'block';

  const labels = chartData.labels || [];
  const actualValues = chartData.actual || [];
  const predictedValues = chartData.predicted || [];

  const ctx = document.getElementById('trafficForecastChart');
  if (!ctx) return;

  if (forecastChartInstance) {
    forecastChartInstance.destroy();
  }

  forecastChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Actual Traffic',
          data: actualValues,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.08)',
          borderWidth: 2.5,
          fill: true,
          tension: 0.25,
          pointBackgroundColor: '#10b981',
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: 'Predicted Traffic (Auto-ARIMA)',
          data: predictedValues,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.05)',
          borderWidth: 2.5,
          borderDash: [5, 4],
          fill: false,
          tension: 0.25,
          pointBackgroundColor: '#2563eb',
          pointRadius: 4,
          pointHoverRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: 'Traffic Flow Forecast',
          font: { size: 14, weight: 'bold' },
          padding: { top: 6, bottom: 12 }
        },
        legend: {
          position: 'top',
          labels: { boxWidth: 14, font: { size: 12, weight: 600 } }
        },
        tooltip: {
          callbacks: {
            afterLabel: function(context) {
              const idx = context.dataIndex;
              if (chartData.arima_orders && chartData.arima_orders[idx] && chartData.arima_orders[idx] !== 'N/A') {
                return `Model: ARIMA ${chartData.arima_orders[idx]}`;
              }
              return '';
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Time / Forecast Intervals' },
          grid: { color: '#f1f5f9' }
        },
        y: {
          title: { display: true, text: 'Number of Vehicles' },
          beginAtZero: true,
          grid: { color: '#f1f5f9' }
        }
      }
    }
  });
}

// --------------------------------------------------------------------------
// Outputs & Reports Directory Explorer
// --------------------------------------------------------------------------

async function loadOutputsDirectory() {
  try {
    const res = await fetch('/api/outputs');
    const data = await res.json();
    if (!data.files) return;

    const rTable = document.getElementById('reports-table-body');
    if (!rTable) return;

    rTable.innerHTML = '';
    if (data.files.length === 0) {
      rTable.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 20px;">No report files found in outputs/ directory.</td></tr>';
      return;
    }

    data.files.forEach(f => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${f.name}</strong></td>
        <td><span class="badge badge-idle" style="font-size:11px;">${f.category}</span></td>
        <td>${f.size_str}</td>
        <td>
          <a href="${f.download_url}" class="btn btn-secondary btn-sm" download>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            Download
          </a>
        </td>
      `;
      rTable.appendChild(tr);
    });
  } catch (err) {
    console.error('Error loading outputs directory:', err);
  }
}
