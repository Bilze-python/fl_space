const state = {
  overview: null,
  session: null,
  experiments: [],
  selectedExperiment: null,
  library: [],
  orbit: null,
  orbitSlot: 0,
  orbitPlaying: false,
  orbitTimer: null,
  jobPoll: null,
  satelliteProfiles: {},
  studyTask: null,
  studyLoading: false,
};

const viewTitles = {
  overview: ["SPACE FEDERATED LEARNING", "实验总览"],
  config: ["EXPERIMENT CONTROL", "实验配置"],
  study: ["LEARNING TASK MODEL", "学习任务"],
  experiments: ["RESULT ARCHIVE", "实验档案"],
  orbit: ["CONSTELLATION VIEW", "轨道视图"],
  library: ["RESEARCH LIBRARY", "文献库"],
  ai: ["AI EXPERIMENT ANALYST", "AI 实验解读"],
};

function $(selector, root = document) { return root.querySelector(selector); }
function $$(selector, root = document) { return [...root.querySelectorAll(selector)]; }
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    let payload = null;
    try {
      payload = await response.json();
      const detail = payload?.detail;
      message = typeof detail === "string" ? detail : detail?.message || message;
    } catch (_) {}
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return response.json();
}

function toast(message, type = "success") {
  const element = document.createElement("div");
  element.className = `toast ${type}`;
  element.textContent = message;
  $("#toast-stack").appendChild(element);
  setTimeout(() => element.remove(), 3600);
}

function formatPercent(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "--";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatDate(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  }).format(date);
}

function experimentStatusBadge(status) {
  const normalized = String(status || "").toUpperCase();
  const tone = normalized === "DONE" ? "green" : normalized === "FAILED" ? "red" : "amber";
  const label = normalized === "PENDING" ? "待运行" : normalized;
  return `<span class="status-badge ${tone}">${label}</span>`;
}

function setView(name) {
  $$(".nav-item").forEach(button => button.classList.toggle("active", button.dataset.view === name));
  $$(".view").forEach(view => view.classList.toggle("active", view.id === `view-${name}`));
  $("#view-eyebrow").textContent = viewTitles[name][0];
  $("#view-title").textContent = viewTitles[name][1];
  if (name === "orbit" && !state.orbit && !state.orbitLoading) loadOrbit();
  if (name === "library" && !state.library.length) loadLibrary();
  if (name === "study" && !state.studyTask && !state.studyLoading) loadStudyTask();
  requestAnimationFrame(resizeCanvases);
}

function studyTaskData() {
  return state.studyTask?.task || null;
}

function studySetValue(id, value) {
  const element = $(`#${id}`);
  if (element) element.value = value ?? "";
}

function renderStudyTask() {
  const envelope = state.studyTask;
  const task = studyTaskData();
  if (!envelope || !task) return;
  studySetValue("study-name", task.name);
  studySetValue("study-dataset-name", task.dataset.name);
  studySetValue("study-dataset-type", task.dataset.data_type);
  studySetValue("study-dataset-distribution", task.dataset.distribution);
  studySetValue("study-model-name", task.model.name);
  studySetValue("study-model-framework", task.model.framework);
  studySetValue("study-scheduler-policy", task.scheduler.policy);
  studySetValue("study-round-duration", task.scheduler.round_duration_s);
  studySetValue("study-participant-fraction", task.scheduler.participant_fraction);

  const summary = envelope.summary;
  $("#study-sat-count").textContent = summary.satellite_count;
  $("#study-gs-count").textContent = summary.ground_station_count;
  $("#study-link-count").textContent = summary.link_count;
  $("#study-update-volume").textContent = `${summary.per_round_update_volume_mb} MB`;
  $("#study-summary-lines").innerHTML = [
    ["数据集", `${summary.dataset} · ${task.dataset.distribution}`],
    ["模型", `${summary.model} · ${task.model.framework}`],
    ["调度", `${summary.scheduler} · ${Number(task.scheduler.participant_fraction * 100).toFixed(0)}% 参与`],
    ["版本", `${envelope.revision} · ${formatDate(envelope.updated_at)}`],
  ].map(([label, value]) => `
    <div class="study-summary-line"><span>${escapeHtml(label)}</span><span>${escapeHtml(value)}</span></div>
  `).join("");
  $("#study-json").textContent = JSON.stringify(envelope, null, 2);

  $("#study-node-rows").innerHTML = (task.nodes || []).map(node => {
    const orbit = node.orbit || {};
    const visual = node.satellite?.visual || {};
    return `
      <tr>
        <td><strong>${escapeHtml(node.node_id)}</strong><small>${escapeHtml(node.satellite?.name || "")}</small></td>
        <td>${Number(orbit.semi_major_axis_km || 0).toFixed(2)}</td>
        <td>${Number(orbit.eccentricity || 0).toFixed(4)}</td>
        <td>${Number(orbit.inclination_deg || 0).toFixed(2)}</td>
        <td>${Number(orbit.raan_deg || 0).toFixed(2)}</td>
        <td>${Number(orbit.arg_perigee_deg || 0).toFixed(2)}</td>
        <td>${Number(orbit.anomaly_deg || 0).toFixed(2)} ${escapeHtml(orbit.anomaly_kind || "true")}</td>
        <td>${escapeHtml(visual.primitive || "box")}</td>
      </tr>
    `;
  }).join("");

  $("#study-link-rows").innerHTML = (task.infrastructure?.links || []).map(link => `
    <tr>
      <td><strong>${escapeHtml(link.name)}</strong></td>
      <td>${escapeHtml(link.link_kind)}</td>
      <td>${escapeHtml(link.band)}<small>${formatFrequency(link.frequency_hz)}</small></td>
      <td>${Number(link.bandwidth_mhz || 0).toLocaleString("zh-CN")} MHz</td>
      <td>${Number(link.data_rate_mbps || 0).toLocaleString("zh-CN")} Mb/s</td>
      <td>${Number(link.max_range_km || 0).toLocaleString("zh-CN")} km</td>
      <td>${escapeHtml(link.source || "unknown")}</td>
    </tr>
  `).join("");
  $("#study-stations").innerHTML = (task.infrastructure?.ground_stations || []).map(station => `
    <span class="study-station-tag">${escapeHtml(station.name)}
      <small>${Number(station.latitude_deg).toFixed(1)}°, ${Number(station.longitude_deg).toFixed(1)}°</small>
    </span>
  `).join("");

  const assumptions = [
    ...(task.assumptions || []),
    `调度策略：${task.scheduler.policy}`,
    ...(task.nodes || []).flatMap(node => node.satellite?.assumptions || []),
  ];
  $("#study-assumptions").innerHTML = assumptions.length
    ? assumptions.map(item => `<div class="study-assumption">${escapeHtml(item)}</div>`).join("")
    : '<div class="empty-state compact">当前任务没有额外假设</div>';
  drawStudyOrbit();
}

function formatFrequency(value) {
  const frequency = Number(value);
  if (!Number.isFinite(frequency) || frequency <= 0) return "--";
  if (frequency >= 1e12) return `${(frequency / 1e12).toFixed(2)} THz`;
  if (frequency >= 1e9) return `${(frequency / 1e9).toFixed(2)} GHz`;
  if (frequency >= 1e6) return `${(frequency / 1e6).toFixed(2)} MHz`;
  return `${frequency.toFixed(0)} Hz`;
}

function drawStudyOrbit() {
  const canvas = $("#study-orbit-canvas");
  const task = studyTaskData();
  if (!canvas || !task) return;
  const width = 640;
  const height = 340;
  const center = { x: width / 2, y: height / 2 };
  const earthRadius = 62;
  const nodes = task.nodes || [];
  canvas.innerHTML = `
    <circle class="study-orbit-earth" cx="${center.x}" cy="${center.y}" r="${earthRadius}"></circle>
    <text class="study-orbit-label" x="${center.x}" y="${center.y + 4}">地球</text>
  `;
  const colors = ["#4fd18b", "#55b8cf", "#e4b04b", "#ef6b6b", "#a98cdb", "#f08e62"];
  nodes.forEach((node, index) => {
    const orbit = node.orbit || {};
    const altitude = Math.max(80, Math.min(180, Number(orbit.semi_major_axis_km || 6928) - 6378));
    const angle = (Number(orbit.raan_deg || 0) + Number(orbit.anomaly_deg || 0)) * Math.PI / 180;
    const rx = altitude;
    const ry = altitude * Math.max(0.38, Math.cos(Number(orbit.inclination_deg || 0) * Math.PI / 360));
    const path = document.createElementNS("http://www.w3.org/2000/svg", "ellipse");
    path.setAttribute("class", "study-orbit-path");
    path.setAttribute("cx", center.x);
    path.setAttribute("cy", center.y);
    path.setAttribute("rx", rx);
    path.setAttribute("ry", ry);
    path.setAttribute("transform", `rotate(${Number(orbit.raan_deg || 0)} ${center.x} ${center.y})`);
    canvas.appendChild(path);
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "study-orbit-node");
    circle.setAttribute("fill", colors[index % colors.length]);
    circle.setAttribute("cx", center.x + rx * Math.cos(angle));
    circle.setAttribute("cy", center.y + ry * Math.sin(angle));
    circle.setAttribute("r", 6);
    canvas.appendChild(circle);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "study-orbit-label");
    label.setAttribute("x", center.x + rx * Math.cos(angle));
    label.setAttribute("y", center.y + ry * Math.sin(angle) - 11);
    label.textContent = node.node_id;
    canvas.appendChild(label);
  });
}

async function loadStudyTask() {
  state.studyLoading = true;
  $("#study-sync-note").textContent = "正在读取服务端任务快照…";
  try {
    state.studyTask = await api("/api/study/task");
    renderStudyTask();
    $("#study-sync-note").textContent = `已同步 · revision ${state.studyTask.revision}`;
  } catch (error) {
    $("#study-sync-note").textContent = `读取失败：${error.message}`;
    toast(error.message, "error");
  } finally {
    state.studyLoading = false;
  }
}

function collectStudyTask() {
  const task = structuredClone(studyTaskData());
  task.name = $("#study-name").value.trim();
  task.dataset.name = $("#study-dataset-name").value.trim();
  task.dataset.data_type = $("#study-dataset-type").value.trim() || "image";
  task.dataset.distribution = $("#study-dataset-distribution").value;
  task.model.name = $("#study-model-name").value.trim();
  task.model.framework = $("#study-model-framework").value.trim() || "pytorch";
  task.scheduler.policy = $("#study-scheduler-policy").value;
  task.scheduler.round_duration_s = Number($("#study-round-duration").value);
  task.scheduler.participant_fraction = Number($("#study-participant-fraction").value);
  return task;
}

async function saveStudyTask(event) {
  event.preventDefault();
  if (!state.studyTask) return;
  const button = $("#study-task-form button[type=submit]");
  button.disabled = true;
  try {
    const response = await api("/api/study/task", {
      method: "PUT",
      body: JSON.stringify({
        expected_revision: state.studyTask.revision,
        task: collectStudyTask(),
      }),
    });
    state.studyTask = response;
    renderStudyTask();
    $("#study-sync-note").textContent = `已保存并重新同步 · revision ${response.revision}`;
    toast("学习任务已保存，所有视图已从服务端快照刷新");
  } catch (error) {
    if (error.message.includes("学习任务已被其他页面更新")) {
      await loadStudyTask();
      $("#study-sync-note").textContent = "检测到版本冲突，已加载最新服务端快照；请确认后再保存";
    }
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function resetStudyTask() {
  const button = $("#study-reset-button");
  button.disabled = true;
  try {
    state.studyTask = await api("/api/study/task/reset", { method: "POST" });
    renderStudyTask();
    $("#study-sync-note").textContent = `默认任务已恢复 · revision ${state.studyTask.revision}`;
    toast("已恢复默认学习任务");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function loadOverview() {
  $("#sync-state").textContent = "正在同步本地数据";
  try {
    const [overview, experiments] = await Promise.all([
      api("/api/overview"),
      api("/api/experiments"),
    ]);
    state.overview = overview;
    state.session = overview.session;
    state.experiments = experiments;
    $("#metric-experiments").textContent = overview.experiments;
    $("#metric-passed").textContent = overview.validations_passed;
    $("#metric-accuracy").textContent = formatPercent(overview.mean_accuracy_delta);
    $("#metric-algorithm").textContent = overview.session.mount.algo.toUpperCase();
    $("#supported-list").innerHTML = overview.implementation.supported.map(item => `<li>${item}</li>`).join("");
    $("#limitations-list").innerHTML = overview.implementation.limitations.map(item => `<li>${item}</li>`).join("");
    renderRecentExperiments(experiments.slice(0, 5));
    populateSessionForm();
    renderExperimentList();
    populateAiExperiments();
    populateOrbitExperiments();
    if (experiments.length) {
      const latest = experiments.find(item => item.kind === "fedleo_validation") || experiments[0];
      const detail = await api(`/api/experiments/${encodeURIComponent(latest.id)}`);
      drawAccuracyChart($("#overview-chart"), detail.history_on, detail.history_off);
    }
    $("#sync-state").textContent = "已同步本地数据";
  } catch (error) {
    $("#sync-state").textContent = "同步失败";
    toast(error.message, "error");
  }
}

function populateAiExperiments() {
  const select = $("#ai-experiment-select");
  if (!select) return;
  select.innerHTML = state.experiments.map(item => `
    <option value="${escapeHtml(item.id)}">${escapeHtml(item.name)} · ${item.seed ?? "--"}</option>
  `).join("");
}

function populateOrbitExperiments() {
  const select = $("#orbit-experiment-select");
  if (!select) return;
  const current = select.value;
  select.innerHTML = `<option value="">当前 Session 参数</option>${state.experiments.map(item => `
    <option value="${escapeHtml(item.id)}">${escapeHtml(item.name)} · ${formatDate(item.created_at)}</option>
  `).join("")}`;
  if ([...select.options].some(option => option.value === current)) select.value = current;
}

function renderRecentExperiments(experiments) {
  $("#recent-experiments").innerHTML = experiments.map(item => `
    <tr>
      <td><strong>${escapeHtml(item.name)}</strong><br><small>${formatDate(item.created_at)}</small></td>
      <td>${String(item.dataset || "--").toUpperCase()}</td>
      <td>${item.rounds ?? "--"}</td>
      <td>${formatPercent(item.accuracy)}</td>
      <td class="${item.accuracy_delta >= 0 ? "positive" : ""}">${item.accuracy_delta == null ? "--" : formatPercent(item.accuracy_delta)}</td>
      <td>${experimentStatusBadge(item.status)}</td>
    </tr>
  `).join("");
}

function resizeCanvas(canvas) {
  if (!canvas || !canvas.clientWidth || !canvas.clientHeight) return;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.floor(canvas.clientWidth * ratio);
  const height = Math.floor(canvas.clientHeight * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function drawAccuracyChart(canvas, historyOn = [], historyOff = []) {
  if (!canvas) return;
  resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  const ratio = window.devicePixelRatio || 1;
  const pad = { left: 46 * ratio, right: 18 * ratio, top: 18 * ratio, bottom: 34 * ratio };
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#14181d";
  ctx.fillRect(0, 0, w, h);
  const values = [...historyOn, ...historyOff].map(row => Number(row.accuracy || 0));
  const maxY = Math.max(0.2, ...values) * 1.08;
  const maxRound = Math.max(1, ...historyOn.map(row => Number(row.round || 0)), ...historyOff.map(row => Number(row.round || 0)));
  ctx.strokeStyle = "#293039";
  ctx.lineWidth = ratio;
  ctx.fillStyle = "#78838c";
  ctx.font = `${10 * ratio}px Segoe UI`;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (h - pad.top - pad.bottom) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    const label = `${((maxY * (4 - i) / 4) * 100).toFixed(0)}%`;
    ctx.fillText(label, 4 * ratio, y + 3 * ratio);
  }
  function path(history, color) {
    if (!history.length) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2 * ratio; ctx.beginPath();
    history.forEach((row, index) => {
      const x = pad.left + (w - pad.left - pad.right) * Number(row.round || 0) / maxRound;
      const y = h - pad.bottom - (h - pad.top - pad.bottom) * Number(row.accuracy || 0) / maxY;
      if (!index) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  path(historyOff, "#55b8cf");
  path(historyOn, "#4fd18b");
  ctx.fillStyle = "#78838c";
  ctx.fillText("0", pad.left, h - 10 * ratio);
  ctx.fillText(String(maxRound), w - pad.right - 14 * ratio, h - 10 * ratio);
}

function populateSessionForm() {
  if (!state.session) return;
  $$("[name]", $("#session-form")).forEach(input => {
    const [section, key] = input.name.split(".");
    if (!key || state.session[section]?.[key] == null) return;
    if (input.type === "checkbox") input.checked = Boolean(state.session[section][key]);
    else input.value = state.session[section][key];
  });
  state.satelliteProfiles = structuredClone(state.session.tune.satellite_data_profiles || {});
  renderSatelliteProfiles();
}

function renderSatelliteProfiles() {
  const body = $("#satellite-profile-rows");
  if (!body) return;
  const satelliteCount = Number($('[name="mount.sats"]')?.value || state.session?.mount?.sats || 1);
  const entries = Object.entries(state.satelliteProfiles)
    .filter(([id]) => Number(id) >= 0 && Number(id) < satelliteCount)
    .sort((left, right) => Number(left[0]) - Number(right[0]));
  body.innerHTML = entries.map(([id, profile]) => `
    <tr data-satellite-id="${id}">
      <td><strong>SAT-${String(Number(id) + 1).padStart(2, "0")}</strong><small>#${id}</small></td>
      <td><input data-profile-field="preferred_classes" value="${(profile.preferred_classes || []).join(",")}" placeholder="0,3,7"></td>
      <td><input type="number" min="0" max="1" step="0.05" data-profile-field="preference_probability" value="${profile.preference_probability ?? 0.8}"></td>
      <td><input type="number" min="0" data-profile-field="max_samples" value="${profile.max_samples ?? 0}"></td>
      <td><button type="button" class="icon-button remove-profile" title="移除卫星画像">×</button></td>
    </tr>
  `).join("");
  $$(".remove-profile", body).forEach(button => button.addEventListener("click", () => {
    const row = button.closest("tr");
    delete state.satelliteProfiles[row.dataset.satelliteId];
    renderSatelliteProfiles();
  }));
}

function addSatelliteProfile() {
  syncSatelliteProfilesFromRows();
  const satelliteCount = Number($('[name="mount.sats"]')?.value || 1);
  const nextId = [...Array(satelliteCount).keys()].find(id => !state.satelliteProfiles[String(id)]);
  if (nextId == null) return toast("当前所有卫星都已配置", "error");
  state.satelliteProfiles[String(nextId)] = {
    preferred_classes: [], preference_probability: 0.8, max_samples: 0,
  };
  renderSatelliteProfiles();
}

function syncSatelliteProfilesFromRows() {
  $$("#satellite-profile-rows tr").forEach(row => {
    const read = field => row.querySelector(`[data-profile-field="${field}"]`).value;
    const classes = read("preferred_classes").split(",")
      .map(value => value.trim()).filter(Boolean).map(Number)
      .filter(value => Number.isInteger(value) && value >= 0);
    state.satelliteProfiles[row.dataset.satelliteId] = {
      preferred_classes: [...new Set(classes)],
      preference_probability: Number(read("preference_probability")),
      max_samples: Number(read("max_samples")),
    };
  });
}

function applyProtocolPreset() {
  if ($("#protocol-mode").value !== "paper_approx") return;
  $('[name="tune.selection_strategy"]').value = "earliest_return";
  $('[name="tune.contact_adaptive_epochs"]').checked = $('[name="mount.algo"]').value === "fedprox";
  if (!Number($('[name="tune.fedbuff_mu"]').value)) $('[name="tune.fedbuff_mu"]').value = 0.01;
  if ($('[name="tune.max_staleness"]').value === "") $('[name="tune.max_staleness"]').value = 4;
}

function collectSessionForm() {
  const data = { tune: {}, mount: {} };
  $$("[name]", $("#session-form")).forEach(input => {
    const [section, key] = input.name.split(".");
    if (!key) return;
    let value = input.type === "checkbox" ? input.checked : input.value;
    if (input.type === "number") {
      value = input.dataset.nullable === "true" && value === "" ? null : Number(value);
    }
    data[section][key] = value;
  });
  syncSatelliteProfilesFromRows();
  data.tune.satellite_data_profiles = state.satelliteProfiles;
  return data;
}

async function saveSession(event) {
  event.preventDefault();
  try {
    const result = await api("/api/session", {
      method: "PUT",
      body: JSON.stringify(collectSessionForm()),
    });
    state.session = result.session;
    $("#metric-algorithm").textContent = state.session.mount.algo.toUpperCase();
    await loadOverview();
    toast(`参数已保存；实验档案已刷新（${state.experiments.length} 份）`);
  } catch (error) { toast(error.message, "error"); }
}

async function resetSession() {
  try {
    const result = await api("/api/session/reset", { method: "POST" });
    state.session = result.session;
    populateSessionForm();
    toast("已恢复默认参数");
  } catch (error) { toast(error.message, "error"); }
}

async function runValidation() {
  const rounds = Number($("#validation-rounds").value);
  const seed = Number($("#validation-seed").value);
  const monitor = $("#job-monitor");
  monitor.classList.remove("hidden");
  monitor.innerHTML = `<strong>正在创建任务</strong><div class="progress-track"><div class="progress-bar"></div></div>`;
  try {
    const job = await api("/api/experiments/fedleo-validation", {
      method: "POST",
      body: JSON.stringify({ rounds, seed }),
    });
    toast("FedLEO 验证任务已启动");
    pollJob(job.id);
  } catch (error) {
    monitor.innerHTML = `<strong>任务启动失败</strong><p>${error.message}</p>`;
    toast(error.message, "error");
  }
}

async function pollJob(jobId) {
  clearInterval(state.jobPoll);
  const update = async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      const monitor = $("#job-monitor");
      if (job.status === "RUNNING" || job.status === "PENDING") {
        monitor.innerHTML = `<strong>${job.message}</strong><div class="progress-track"><div class="progress-bar"></div></div><span>Seed ${job.seed} · ${job.rounds} 轮</span>`;
        return;
      }
      clearInterval(state.jobPoll);
      const passed = job.result?.passed;
      monitor.innerHTML = `<strong>${job.message}</strong><p>${passed ? `准确率增益 ${formatPercent(job.result.deltas_on_minus_off.final_accuracy)}，真实卸载 ${job.result.offload_on.total_offloaded} 个样本。` : "请查看任务日志。"}</p>`;
      toast(passed ? "验证完成，全部门槛通过" : "验证未通过", passed ? "success" : "error");
      await loadOverview();
    } catch (error) {
      clearInterval(state.jobPoll);
      toast(error.message, "error");
    }
  };
  await update();
  state.jobPoll = setInterval(update, 1800);
}

function renderExperimentList() {
  const list = $("#experiment-list");
  if (!list) return;
  if (!state.experiments.length) {
    list.innerHTML = '<div class="empty-state compact">还没有实验档案，点击右上角“＋”新建，或先运行一次 FedLEO 验证。</div>';
    return;
  }
  list.innerHTML = state.experiments.map(item => `
    <article class="experiment-item ${state.selectedExperiment === item.id ? "active" : ""}" data-experiment-id="${escapeHtml(item.id)}">
      <div class="experiment-item-head">
        <strong>${escapeHtml(item.name)}</strong>
        ${experimentStatusBadge(item.status)}
      </div>
      <div class="experiment-meta"><span>${String(item.dataset || "--").toUpperCase()} · ${item.rounds ?? "--"} 轮</span><span>${formatDate(item.created_at)}</span></div>
      <div class="experiment-meta" style="margin-top:8px"><span>${item.offloaded} 样本卸载</span><span class="${item.accuracy_delta >= 0 ? "positive" : ""}">${item.accuracy_delta == null ? "--" : formatPercent(item.accuracy_delta)}</span></div>
      ${item.note ? `<div class="experiment-note">${escapeHtml(item.note)}</div>` : ""}
    </article>
  `).join("");
  $$(".experiment-item").forEach(element => element.addEventListener("click", () => selectExperiment(element.dataset.experimentId)));
}

async function selectExperiment(id) {
  state.selectedExperiment = id;
  renderExperimentList();
  $("#experiment-detail").innerHTML = `<div class="empty-state">正在读取实验结果</div>`;
  try {
    const detail = await api(`/api/experiments/${encodeURIComponent(id)}`);
    const raw = detail.raw || {};
    const on = raw.offload_on || raw.fedleo || {};
    const off = raw.offload_off || raw.baseline || {};
    const historyOn = detail.history_on || [];
    const historyOff = detail.history_off || [];
    const hasCurve = Boolean(historyOn.length || historyOff.length);
    $("#experiment-detail").innerHTML = `
      <div class="panel-head experiment-detail-heading">
      <div><p class="panel-kicker">${escapeHtml(String(detail.kind || "").replaceAll("_", " ").toUpperCase())}</p><h2>${escapeHtml(detail.name)}</h2></div>
        ${experimentStatusBadge(detail.status)}
        <div class="experiment-detail-actions">
          <button class="button secondary" id="rename-experiment" type="button">重命名</button>
          <button class="button danger" id="delete-experiment" type="button">删除</button>
          <button class="button primary" id="replay-experiment" type="button">载入实时演示</button>
        </div>
      </div>
      <div class="detail-metrics">
        <div class="detail-metric"><span>卸载准确率</span><strong>${formatPercent(on.final_accuracy)}</strong></div>
        <div class="detail-metric"><span>控制组准确率</span><strong>${formatPercent(off.final_accuracy)}</strong></div>
        <div class="detail-metric"><span>真实卸载</span><strong>${on.total_offloaded ?? detail.offloaded ?? 0}</strong></div>
        <div class="detail-metric"><span>耗时</span><strong>${Number(raw.elapsed_sec || on.elapsed_sec || 0).toFixed(1)}s</strong></div>
      </div>
      ${hasCurve
        ? `<canvas id="detail-chart"></canvas>
           <div class="chart-legend"><span><i class="legend-line on"></i>卸载开启</span><span><i class="legend-line off"></i>卸载关闭/基线</span></div>`
        : `<div class="experiment-empty-curve">
             <strong>该档案还没有逐轮曲线</strong>
             <p>${detail.kind === "manual"
               ? "这是一条手工/占位档案：可用当前 Session 参数直接运行一次 FedLEO 验证，结果会写回这条档案。"
               : "结果目录里没有找到 offload_on / offload_off 的逐轮记录。"}</p>
             ${detail.kind === "manual"
               ? '<button class="button primary" id="run-into-experiment" type="button">运行 FedLEO 验证并写回本档案</button>'
               : ""}
           </div>`}
      ${detail.note ? `<div class="experiment-callout"><strong>档案备注</strong><p>${escapeHtml(detail.note)}</p></div>` : ""}
    `;
    $("#replay-experiment").addEventListener("click", () => replayExperiment(id));
    $("#rename-experiment").addEventListener("click", () => renameExperiment(id, detail.name));
    $("#delete-experiment").addEventListener("click", () => deleteExperiment(id, detail.name));
    $("#run-into-experiment")?.addEventListener("click", () => runValidationForArchive(detail));
    if (hasCurve) drawAccuracyChart($("#detail-chart"), historyOn, historyOff);
  } catch (error) {
    $("#experiment-detail").innerHTML = `<div class="empty-state">${error.message}</div>`;
  }
}

async function renameExperiment(id, currentName) {
  const name = window.prompt("请输入新的实验名称", currentName || "");
  if (name === null || !name.trim() || name.trim() === currentName) return;
  try {
    await api(`/api/experiments/${encodeURIComponent(id)}/name`, {
      method: "PUT",
      body: JSON.stringify({ name: name.trim() }),
    });
    toast("实验已重命名", "success");
    await loadOverview();
    await selectExperiment(id);
  } catch (error) { toast(error.message, "error"); }
}

async function deleteExperiment(id, name) {
  if (!window.confirm(`确定删除实验“${name}”吗？此操作不可撤销。`)) return;
  try {
    await api(`/api/experiments/${encodeURIComponent(id)}`, { method: "DELETE" });
    state.selectedExperiment = null;
    toast("实验档案已删除", "success");
    await loadOverview();
    $("#experiment-detail").innerHTML = '<div class="empty-state">选择一个实验查看逐轮曲线与卸载记录</div>';
  } catch (error) { toast(error.message, "error"); }
}

async function openCreateExperiment() {
  const modal = $("#experiment-create-modal");
  if (!modal) return;
  modal.classList.remove("hidden");
  $("#experiment-create-name").value = "";
  $("#experiment-create-note").value = "";
  $("#experiment-create-path").value = "";
  $("#experiment-create-source").value = "run";
  const hint = $("#experiment-create-hint");
  hint.textContent = "正在检索可登记的结果目录…";
  $("#experiment-create-rounds").value = Math.min(30, Math.max(2, Number(state.session?.tune?.rounds) || 10));
  $("#experiment-create-seed").value = Number(state.session?.tune?.seed) || 20260810;
  const candidateSelect = $("#experiment-create-candidate");
  candidateSelect.innerHTML = '<option value="">（暂无候选，可在上方直接填写路径）</option>';
  try {
    const candidates = await api("/api/experiments/candidates");
    if (candidates.length) {
      candidateSelect.innerHTML = ['<option value="">请选择…</option>']
        .concat(candidates.map(item => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.path)} · ${formatPercent(item.accuracy)}</option>`))
        .join("");
      hint.textContent = `发现 ${candidates.length} 个尚未入档的结果目录，可直接登记。`;
    } else {
      hint.textContent = "没有发现未入档的结果目录：可“立即运行验证并归档”，或手动填写目录路径。";
    }
  } catch (error) {
    hint.textContent = error.message;
  }
  syncCreateExperimentRows();
  $("#experiment-create-name").focus();
}

function syncCreateExperimentRows() {
  const source = $("#experiment-create-source").value;
  $("#experiment-create-run-row").classList.toggle("hidden", source !== "run");
  $("#experiment-create-path-row").classList.toggle("hidden", source !== "directory");
}

function closeCreateExperiment() {
  $("#experiment-create-modal")?.classList.add("hidden");
}

async function submitCreateExperiment(event) {
  event.preventDefault();
  const name = $("#experiment-create-name").value.trim();
  if (!name) return toast("请填写档案名称", "error");
  const note = $("#experiment-create-note").value.trim();
  const source = $("#experiment-create-source").value;
  const button = $("#experiment-create-form button[type=submit]");
  button.disabled = true;
  try {
    if (source === "run") {
      const rounds = Math.min(30, Math.max(2, Number($("#experiment-create-rounds").value) || 10));
      const seed = Number($("#experiment-create-seed").value) || 20260810;
      const job = await api("/api/experiments/fedleo-validation", {
        method: "POST",
        body: JSON.stringify({ rounds, seed, name, note }),
      });
      closeCreateExperiment();
      setView("experiments");
      await loadOverview();
      if (job.output) await selectExperiment(job.output);
      toast("档案已创建，FedLEO 验证任务已启动");
      pollJob(job.id);
      return;
    }
    const result = await api("/api/experiments", {
      method: "POST",
      body: JSON.stringify({
        name, note, source,
        path: $("#experiment-create-path").value.trim(),
      }),
    });
    closeCreateExperiment();
    setView("experiments");
    await loadOverview();
    await selectExperiment(result.id);
    toast(`实验档案已创建：${result.id}`);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function runValidationForArchive(detail) {
  const rounds = Math.min(30, Math.max(2, Number(detail.rounds) || 10));
  const seed = Number(detail.seed) || 20260810;
  try {
    const job = await api("/api/experiments/fedleo-validation", {
      method: "POST",
      body: JSON.stringify({ rounds, seed, name: detail.name, archive_id: detail.id }),
    });
    toast(`已启动 FedLEO 验证，结果将写回档案「${detail.name}」`);
    setView("config");
    const monitor = $("#job-monitor");
    if (monitor) {
      monitor.classList.remove("hidden");
      monitor.innerHTML = `<strong>正在创建任务</strong><div class="progress-track"><div class="progress-bar"></div></div>`;
    }
    pollJob(job.id);
  } catch (error) { toast(error.message, "error"); }
}

function replayExperiment(id) {
  const select = $("#orbit-experiment-select");
  if (select) select.value = id;
  setView("orbit");
  loadOrbit(id, true);
}

async function loadLibrary(query = "") {
  try {
    state.library = await api(`/api/library?q=${encodeURIComponent(query)}`);
    $("#library-list").innerHTML = state.library.map(item => `
      <article class="library-item" data-library-path="${escapeHtml(item.path)}" data-library-type="${escapeHtml(item.type)}">
        <strong>${escapeHtml(item.title)}</strong>
        <small>${escapeHtml(item.type.toUpperCase())} · ${(item.size / 1024).toFixed(0)} KB · <span class="library-source">${escapeHtml(item.source || "项目")}</span></small>
      </article>
    `).join("");
    $$(".library-item").forEach(element => element.addEventListener("click", () => openDocument(element.dataset.libraryPath, element.dataset.libraryType, element)));
  } catch (error) { toast(error.message, "error"); }
}

async function loadLibrarySettings() {
  try {
    const settings = await api("/api/library/settings");
    $("#library-default-path").value = settings.default_path;
  } catch (error) { toast(error.message, "error"); }
}

async function saveLibraryPath() {
  try {
    const result = await api("/api/library/settings", {
      method: "PUT",
      body: JSON.stringify({ default_path: $("#library-default-path").value.trim() }),
    });
    $("#library-default-path").value = result.default_path;
    $("#library-path-editor").classList.add("hidden");
    await loadLibrary($("#library-search").value);
    toast("默认文献目录已更新");
  } catch (error) { toast(error.message, "error"); }
}

async function importLibraryFiles(files) {
  if (!files.length) return;
  let imported = 0;
  for (const file of files) {
    try {
      await api(`/api/library/import?filename=${encodeURIComponent(file.name)}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: file,
      });
      imported += 1;
    } catch (error) {
      toast(`${file.name}: ${error.message}`, "error");
    }
  }
  $("#library-import-input").value = "";
  if (imported) {
    await loadLibrary($("#library-search").value);
    toast(`已导入 ${imported} 份文献`);
  }
}

async function loadAiSettings() {
  try {
    const settings = await api("/api/settings");
    const badge = $("#ai-provider-status");
    badge.textContent = settings.ai_configured ? `${settings.ai_provider} 已连接` : "AI 未连接";
    badge.className = `status-badge ${settings.ai_configured ? "green" : "amber"}`;
    $("#cc-switch-detail").textContent = settings.ai_configured
      ? `${settings.ai_provider} · ${settings.ai_model}`
      : (settings.ai_error || "未配置当前供应商");
  } catch (error) {
    $("#ai-provider-status").textContent = "检测失败";
  }
}

async function importCcSwitch() {
  const button = $("#cc-switch-import");
  button.disabled = true;
  try {
    const settings = await api("/api/settings/cc-switch/import", { method: "POST" });
    toast(`已导入 CC Switch: ${settings.ai_provider}`);
    await loadAiSettings();
  } catch (error) { toast(error.message, "error"); }
  finally { button.disabled = false; }
}

function appendAiMessage(content, role, extraClass = "") {
  const message = document.createElement("div");
  message.className = `ai-message ${role} ${extraClass}`.trim();
  message.textContent = content;
  $("#ai-messages").appendChild(message);
  $("#ai-messages").scrollTop = $("#ai-messages").scrollHeight;
  return message;
}

async function askAi(event) {
  event?.preventDefault();
  const question = $("#ai-question").value.trim();
  if (!question) return;
  appendAiMessage(question, "user");
  $("#ai-question").value = "";
  const loading = appendAiMessage("正在读取实验数据并分析…", "assistant", "loading");
  try {
    const result = await api("/api/ai/analyze", {
      method: "POST",
      body: JSON.stringify({
        experiment_id: $("#ai-experiment-select").value,
        question,
      }),
    });
    loading.remove();
    appendAiMessage(result.content, "assistant");
  } catch (error) {
    loading.remove();
    appendAiMessage(`分析失败：${error.message}`, "assistant");
    toast(error.message, "error");
  }
}

async function openDocument(path, type, element) {
  $$(".library-item").forEach(item => item.classList.toggle("active", item === element));
  const title = element.querySelector("strong").textContent;
  if (type === "pdf") {
    $("#reader-panel").innerHTML = `<div class="reader-head"><h2>${escapeHtml(title)}</h2></div><iframe class="reader-frame" title="${escapeHtml(title)}" src="/api/library/file?path=${encodeURIComponent(path)}"></iframe>`;
    return;
  }
  try {
    const data = await api(`/api/library/content?path=${encodeURIComponent(path)}`);
    if (type === "md" && window.marked && window.DOMPurify) {
      const html = DOMPurify.sanitize(marked.parse(data.content, { gfm: true }));
      $("#reader-panel").innerHTML = `<div class="reader-head"><h2>${escapeHtml(title)}</h2></div><article class="reader-content markdown-body">${html}</article>`;
    } else {
      $("#reader-panel").innerHTML = `<div class="reader-head"><h2>${escapeHtml(title)}</h2></div><div class="reader-content"><pre></pre></div>`;
      $(".reader-content pre").textContent = data.content;
    }
  } catch (error) { toast(error.message, "error"); }
}

async function loadOrbit(experimentId = $("#orbit-experiment-select")?.value || "", autoplay = false) {
  pauseOrbit();
  state.orbitLoading = true;
  const session = state.session || { mount: {} };
  const mount = session.mount || {};
  const params = new URLSearchParams({
    sats: Math.min(Number(mount.sats || 12), 30),
    gs: Math.min(Number(mount.stations || 5), 13),
    altitude_km: Number(mount.altitude || 500),
    inclination_deg: Number(mount.inclination || 53),
    sim_hours: Math.min(Number(mount.sim_hours || 2), 4),
    timeslot_min: Math.max(Number(mount.timeslot_min || 2), 1),
    isl_enabled: mount.isl !== "disabled",
    seed: Number(state.session?.tune?.seed || 42),
  });
  $("#orbit-status").textContent = "生成中";
  try {
    const profile = $("#orbit-projection-profile")?.value || "standard";
    const projection = profile === "stress15"
      ? "projection_days=15&projection_step_min=30&satellites=24&ground_stations=6&isl_enabled=true"
      : "projection_days=1&projection_step_min=10";
    const endpoint = experimentId
      ? `/api/experiments/${encodeURIComponent(experimentId)}/orbit_data?${projection}`
      : `/api/orbit_data?${params}`;
    state.orbit = await api(endpoint);
    state.orbitSlot = 0;
    $("#orbit-slider").max = state.orbit.timeslots.length - 1;
    $("#orbit-slider").value = 0;
    $("#orbit-sats").textContent = state.orbit.satellites;
    $("#orbit-gs").textContent = state.orbit.ground_stations.length;
    $("#orbit-status").textContent = "轨道已加载";
    drawOrbit();
    syncCesiumSource(experimentId);
    if (autoplay) playOrbit();
  } catch (error) {
    $("#orbit-status").textContent = "生成失败";
    toast(error.message, "error");
  } finally {
    state.orbitLoading = false;
  }
}

function drawOrbit() {
  updateOrbitExperimentMetrics();
  // 调用2D轨道绘制模块
  if (window.draw2DOrbit && state.orbit) {
    window.draw2DOrbit("orbit-canvas-2d", state.orbit, state.orbitSlot);
    const slot = state.orbit.timeslots[state.orbitSlot];
    $("#orbit-time").textContent = new Date(slot.time).toLocaleString("zh-CN");
    $("#orbit-links").textContent = (slot.contacts?.length || 0) + (slot.isl_links?.length || 0);
    return;
  }

  // 降级到旧版本绘制
  const canvas = $("#orbit-canvas");
  if (!canvas || !state.orbit) return;
  resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  const ratio = window.devicePixelRatio || 1;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#07090b"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#1d242a"; ctx.lineWidth = ratio;
  for (let lon = -180; lon <= 180; lon += 30) {
    const x = (lon + 180) / 360 * w;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let lat = -90; lat <= 90; lat += 30) {
    const y = (90 - lat) / 180 * h;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  ctx.strokeStyle = "#2e3941"; ctx.lineWidth = 1.5 * ratio;
  ctx.beginPath();
  const coast = [[-168,62],[-140,56],[-125,45],[-118,30],[-95,18],[-82,26],[-68,45],[-55,52],[-40,70],[10,70],[30,58],[55,55],[80,62],[110,50],[135,35],[150,5],[130,-15],[115,-35],[145,-42],[165,-45],[178,-20],[-175,-8],[-150,-18],[-120,-50],[-80,-52],[-70,-30],[-58,-12],[-45,2],[-18,12],[5,35],[-10,50],[-40,60],[-80,70],[-120,72],[-168,62]];
  coast.forEach(([lon,lat], index) => {
    const x = (lon + 180) / 360 * w, y = (90 - lat) / 180 * h;
    if (!index) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
  const slot = state.orbit.timeslots[state.orbitSlot];
  const mapPoint = point => ({ x: (point.lon + 180) / 360 * w, y: (90 - point.lat) / 180 * h });
  ctx.lineWidth = ratio;
  for (const link of slot.isl_links || []) {
    const a = mapPoint(slot.positions[link.a_id]), b = mapPoint(slot.positions[link.b_id]);
    if (Math.abs(a.x - b.x) > w * .5) continue;
    ctx.strokeStyle = "rgba(228,176,75,.34)"; ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  }
  for (const link of slot.contacts || []) {
    const a = mapPoint(slot.positions[link.sat_id]);
    const station = state.orbit.ground_stations[link.gs_id];
    const b = mapPoint(station);
    ctx.strokeStyle = "rgba(79,209,139,.42)"; ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  }
  for (const station of state.orbit.ground_stations) {
    const p = mapPoint(station);
    ctx.fillStyle = "#55b8cf"; ctx.fillRect(p.x - 3*ratio, p.y - 3*ratio, 6*ratio, 6*ratio);
  }
  for (const satellite of slot.positions) {
    const p = mapPoint(satellite);
    ctx.beginPath(); ctx.arc(p.x,p.y,4*ratio,0,Math.PI*2);
    ctx.fillStyle = "#4fd18b"; ctx.fill();
    ctx.strokeStyle = "#07150e"; ctx.lineWidth = 2*ratio; ctx.stroke();
  }
  $("#orbit-time").textContent = new Date(slot.time).toLocaleString("zh-CN");
  $("#orbit-links").textContent = (slot.contacts?.length || 0) + (slot.isl_links?.length || 0);
}

function cesiumWindow() {
  return $("#orbit-canvas-3d iframe")?.contentWindow || null;
}

function syncCesiumSlot(slot) {
  const target = cesiumWindow();
  if (target?.SpaceFLOrbitViewer) target.SpaceFLOrbitViewer.renderSlot(slot);
}

function updateOrbitExperimentMetrics() {
  const slot = state.orbit?.timeslots?.[state.orbitSlot];
  const metrics = slot?.experiment || {};
  $("#orbit-round").textContent = metrics.round ?? "--";
  $("#orbit-accuracy").textContent = metrics.accuracy == null ? "--" : formatPercent(metrics.accuracy);
  $("#orbit-offloaded").textContent = metrics.total_offloaded_samples ?? "--";
  $("#orbit-projection-days").textContent = state.orbit?.projection_days?.toFixed?.(1) ?? "--";
  $("#orbit-frame").textContent = state.orbit ? `${state.orbitSlot + 1}/${state.orbit.timeslots.length}` : "--";
  if (state.orbit?.experiment) {
    $("#orbit-status").textContent = `存档 · ${state.orbit.experiment.dataset || "实验"}`;
  }
}

function syncCesiumSource(experimentId = "") {
  const container = $("#orbit-canvas-3d");
  if (!container) return;
  const profile = $("#orbit-projection-profile")?.value || "standard";
  const query = experimentId
    ? `?experiment_id=${encodeURIComponent(experimentId)}&projection=${encodeURIComponent(profile)}`
    : "";
  const src = `/cesium_orbit_viewer.html${query}`;
  const frame = container.querySelector("iframe");
  if (!frame || frame.getAttribute("src") !== src) {
    container.innerHTML = `<iframe src="${src}" title="Cesium 3D 轨道演示"></iframe>`;
  }
}

function setOrbitMode(mode) {
  pauseOrbit();
  const is3d = mode === "3d";
  $("#orbit-canvas-2d").classList.toggle("hidden", is3d);
  $("#orbit-canvas-3d").classList.toggle("hidden", !is3d);
  if (is3d) syncCesiumSource($("#orbit-experiment-select").value);
  else drawOrbit();
}

function playOrbit() {
  if (!state.orbit?.timeslots?.length || state.orbitLoading) return;
  pauseOrbit();
  state.orbitPlaying = true;
  const lastSlot = state.orbit.timeslots.length - 1;
  state.orbitTimer = setInterval(() => {
    if (!state.orbit?.timeslots?.length) {
      pauseOrbit();
      return;
    }
    if (state.orbitSlot >= lastSlot) {
      pauseOrbit();
      return;
    }
    state.orbitSlot += 1;
    $("#orbit-slider").value = state.orbitSlot;
    drawOrbit();
    syncCesiumSlot(state.orbitSlot);
  }, Math.max(40, Math.round(500 / Number($("#orbit-playback-speed")?.value || 1))));
}

function pauseOrbit() {
  state.orbitPlaying = false;
  if (state.orbitTimer != null) clearInterval(state.orbitTimer);
  state.orbitTimer = null;
  cesiumWindow()?.SpaceFLOrbitViewer?.pause();
}

function resizeCanvases() {
  const overview = $("#overview-chart");
  if (overview && state.experiments.length) loadOverviewChartOnly();
  if ($("#detail-chart") && state.selectedExperiment) selectExperiment(state.selectedExperiment);
  if (state.orbit) drawOrbit();
}

async function loadOverviewChartOnly() {
  const latest = state.experiments.find(item => item.kind === "fedleo_validation") || state.experiments[0];
  if (!latest) return;
  const detail = await api(`/api/experiments/${encodeURIComponent(latest.id)}`);
  drawAccuracyChart($("#overview-chart"), detail.history_on, detail.history_off);
}

function bindEvents() {
  $$(".nav-item").forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
  $$("[data-route]").forEach(button => button.addEventListener("click", () => setView(button.dataset.route)));
  $$(".config-tab").forEach(button => button.addEventListener("click", () => {
    $$(".config-tab").forEach(item => item.classList.toggle("active", item === button));
    $$(".config-section").forEach(section => section.classList.toggle("active", section.dataset.configSection === button.dataset.configTab));
  }));
  $("#session-form").addEventListener("submit", saveSession);
  $("#study-task-form").addEventListener("submit", saveStudyTask);
  $("#study-reset-button").addEventListener("click", resetStudyTask);
  $("#study-refresh-button").addEventListener("click", loadStudyTask);
  $("#add-satellite-profile").addEventListener("click", addSatelliteProfile);
  $("#protocol-mode").addEventListener("change", applyProtocolPreset);
  $('[name="mount.sats"]').addEventListener("change", () => {
    syncSatelliteProfilesFromRows();
    const count = Number($('[name="mount.sats"]').value || 1);
    Object.keys(state.satelliteProfiles).forEach(id => {
      if (Number(id) >= count) delete state.satelliteProfiles[id];
    });
    renderSatelliteProfiles();
  });
  $("#reset-session").addEventListener("click", resetSession);
  $("#run-validation-button").addEventListener("click", runValidation);
  $("#quick-run-button").addEventListener("click", () => { setView("config"); $('[data-config-tab="fedleo"]').click(); });
  $("#refresh-overview").addEventListener("click", loadOverview);
  $("#refresh-experiments").addEventListener("click", loadOverview);
  $("#create-experiment").addEventListener("click", openCreateExperiment);
  $("#experiment-create-close").addEventListener("click", closeCreateExperiment);
  $("#experiment-create-cancel").addEventListener("click", closeCreateExperiment);
  $("#experiment-create-form").addEventListener("submit", submitCreateExperiment);
  $("#experiment-create-source").addEventListener("change", syncCreateExperimentRows);
  $("#experiment-create-candidate").addEventListener("change", event => {
    if (event.target.value) $("#experiment-create-path").value = event.target.value;
  });
  $("#experiment-create-modal").addEventListener("click", event => {
    if (event.target.id === "experiment-create-modal") closeCreateExperiment();
  });
  $("#library-search").addEventListener("input", event => loadLibrary(event.target.value));
  $("#library-import-button").addEventListener("click", () => $("#library-import-input").click());
  $("#library-import-input").addEventListener("change", event => importLibraryFiles([...event.target.files]));
  $("#library-settings-button").addEventListener("click", async () => {
    const editor = $("#library-path-editor");
    editor.classList.toggle("hidden");
    if (!editor.classList.contains("hidden")) await loadLibrarySettings();
  });
  $("#library-save-path").addEventListener("click", saveLibraryPath);
  $("#ai-form").addEventListener("submit", askAi);
  $("#cc-switch-import").addEventListener("click", importCcSwitch);
  $$("[data-ai-question]").forEach(button => button.addEventListener("click", () => {
    $("#ai-question").value = button.dataset.aiQuestion;
    askAi();
  }));
  $("#orbit-load").addEventListener("click", loadOrbit);
  $("#orbit-experiment-select").addEventListener("change", event => loadOrbit(event.target.value));
  $("#orbit-projection-profile").addEventListener("change", () => loadOrbit());
  $$("[name='orbit-mode']").forEach(input => input.addEventListener("change", event => setOrbitMode(event.target.value)));
  $("#orbit-play").addEventListener("click", playOrbit);
  $("#orbit-pause").addEventListener("click", pauseOrbit);
  $("#orbit-playback-speed").addEventListener("change", () => {
    if (state.orbitPlaying) playOrbit();
  });
  $("#orbit-slider").addEventListener("input", event => {
    pauseOrbit();
    state.orbitSlot = Number(event.target.value);
    drawOrbit();
    syncCesiumSlot(state.orbitSlot);
  });
  window.addEventListener("resize", () => { clearTimeout(window.__resizeTimer); window.__resizeTimer = setTimeout(resizeCanvases, 140); });
}

bindEvents();
loadAiSettings();
loadOverview();
loadStudyTask();
