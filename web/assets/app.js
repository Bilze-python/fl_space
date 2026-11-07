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
  localClusterResult: null,
  datasets: [],
  trainingPoll: null,
  trainingAllowed: false,
  trainingActive: false,
};

const viewTitles = {
  overview: ["SPACE FEDERATED LEARNING", "实验总览"],
  agent: ["RESEARCH AGENT", "研究助手"],
  library: ["JOURNAL LIBRARY", "期刊文献库"],
  intel: ["SIGNAL RADAR", "前沿情报"],
  config: ["EXPERIMENT CONTROL", "实验配置"],
  "local-cluster": ["ISOLATED CLUSTER LAB", "局部星簇"],
  study: ["LEARNING TASK MODEL", "学习任务"],
  experiments: ["RESULT ARCHIVE", "实验档案"],
  orbit: ["CONSTELLATION VIEW", "轨道视图"],
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
  if (!viewTitles[name]) return;
  $$(".nav-item").forEach(button => button.classList.toggle("active", button.dataset.view === name));
  $$(".view").forEach(view => view.classList.toggle("active", view.id === `view-${name}`));
  $("#view-eyebrow").textContent = viewTitles[name][0];
  $("#view-title").textContent = viewTitles[name][1];
  try {
    if (window.history && window.history.replaceState) {
      window.history.replaceState(null, "", `#${name}`);
    }
  } catch (_) { /* file:// 场景忽略 */ }
  if (name === "orbit" && !state.orbit && !state.orbitLoading) loadOrbit();
  if (name === "study" && !state.studyTask && !state.studyLoading) loadStudyTask();
  if (name === "local-cluster" && !state.localClusterDefaults) loadLocalClusterDefaults();
  document.dispatchEvent(new CustomEvent("spacefl:view", { detail: { name } }));
  requestAnimationFrame(resizeCanvases);
}

async function loadLocalClusterDefaults() {
  try {
    const response = await api("/api/local-cluster/default");
    state.localClusterDefaults = response.config;
    state.localClusterProfiles = response.datasets;
    applyLocalClusterConfig(response.config);
    renderClusterTopology({ config: response.config, num_satellites: response.config.num_satellites });
  } catch (error) {
    toast(`局部星簇默认值读取失败：${error.message}`, "error");
  }
}

function applyLocalClusterConfig(config) {
  const link = config.link || {};
  const values = {
    "cluster-satellites": config.num_satellites,
    "cluster-radius": config.cluster_radius_km,
    "cluster-altitude": config.altitude_km,
    "cluster-topology": config.topology,
    "cluster-gflops": config.compute_gflops,
    "cluster-dataset": config.dataset,
    "cluster-samples": config.samples_per_satellite,
    "cluster-rounds": config.rounds,
    "cluster-aperture": link.aperture_diameter_cm,
    "cluster-distance": link.link_distance_km,
    "cluster-parallel": link.parallel_links,
    "cluster-channels": link.channels_per_link,
    "cluster-rate": link.rate_per_channel_gbps,
    "cluster-pointing": link.pointing_accuracy_urad,
  };
  Object.entries(values).forEach(([id, value]) => {
    const element = $(`#${id}`);
    if (element && value != null) element.value = value;
  });
  syncClusterDatasetFields();
}

function collectLocalClusterConfig() {
  const config = {
    enabled: true,
    name: "web-local-cluster",
    num_satellites: Number($("#cluster-satellites").value),
    cluster_radius_km: Number($("#cluster-radius").value),
    altitude_km: Number($("#cluster-altitude").value),
    topology: $("#cluster-topology").value,
    compute_gflops: Number($("#cluster-gflops").value),
    dataset: $("#cluster-dataset").value,
    samples_per_satellite: Number($("#cluster-samples").value),
    rounds: Number($("#cluster-rounds").value),
    link: {
      aperture_diameter_cm: Number($("#cluster-aperture").value),
      link_distance_km: Number($("#cluster-distance").value),
      parallel_links: Number($("#cluster-parallel").value),
      channels_per_link: Number($("#cluster-channels").value),
      rate_per_channel_gbps: Number($("#cluster-rate").value),
      pointing_accuracy_urad: Number($("#cluster-pointing").value),
    },
  };
  if ($("#cluster-dataset").value.startsWith("imported:")) {
    config.model_size_mb = Number($("#cluster-model-size").value);
    config.flops_per_sample = Number($("#cluster-flops").value);
  }
  return config;
}

function syncClusterDatasetFields() {
  const imported = $("#cluster-dataset").value.startsWith("imported:");
  $("#cluster-model-size-field").classList.toggle("hidden", !imported);
  $("#cluster-flops-field").classList.toggle("hidden", !imported);
  $("#cluster-model-size").disabled = !imported;
  $("#cluster-flops").disabled = !imported;
  $("#cluster-workload-note").classList.toggle("hidden", !imported);
}

function renderLocalClusterResult(result) {
  state.localClusterResult = result;
  $("#local-cluster-empty").classList.add("hidden");
  $("#local-cluster-result").classList.remove("hidden");
  $("#local-cluster-status").className = "status-badge green";
  $("#local-cluster-status").textContent = "已完成";
  $("#cluster-score").textContent = Number(result.cluster_score).toFixed(1);
  $("#cluster-bandwidth").textContent = `${Number(result.effective_bandwidth_gbps).toFixed(2)} Gbps`;
  $("#cluster-latency").textContent = `${Number(result.per_round_latency_s).toFixed(2)} s`;
  $("#cluster-throughput").textContent = Number(result.samples_per_second).toFixed(1);
  const rows = [
    ["名义单向带宽", `${Number(result.nominal_bandwidth_gbps).toFixed(2)} Gbps`],
    ["接收功率", `${Number(result.received_power_mw).toFixed(6)} mW`],
    ["链路余量", `${Number(result.link_margin_db).toFixed(2)} dB`],
    ["训练 / 轮", `${Number(result.training_time_per_round_s).toFixed(4)} s`],
    ["聚合 / 轮", `${Number(result.aggregation_time_per_round_s).toFixed(4)} s`],
    ["总实验时长", `${Number(result.total_time_s).toFixed(2)} s`],
  ];
  $("#local-cluster-summary").innerHTML = rows.map(([label, value]) => `<tr><th>${label}</th><td>${value}</td></tr>`).join("");
  $("#local-cluster-json").textContent = JSON.stringify(result, null, 2);
  $("#local-cluster-profile-note").textContent = "profile 只在用户显式传入全局实验时生效";
  renderClusterTopology(result);
}

function renderClusterTopology(result) {
  const canvas = $("#cluster-topology-canvas");
  if (!canvas || !result) return;
  const config = result.config || {};
  const topology = config.topology || "ring";
  const total = Number(result.num_satellites || 0);
  const count = topology === "mesh" ? Math.min(total, 28) : total;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 800;
  const height = canvas.clientHeight || 440;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.fillStyle = "#080b0d";
  ctx.fillRect(0, 0, width, height);
  const padding = 34;
  const nodes = Array.from({ length: count }, (_, index) => {
    if (topology === "grid") {
      const columns = Math.ceil(Math.sqrt(total));
      const rows = Math.ceil(total / columns);
      return {
        x: padding + (index % columns) * (width - 2 * padding) / Math.max(1, columns - 1),
        y: padding + Math.floor(index / columns) * (height - 2 * padding) / Math.max(1, rows - 1),
      };
    }
    const angle = -Math.PI / 2 + index * 2 * Math.PI / count;
    return {
      x: width / 2 + Math.cos(angle) * Math.min((width - 2 * padding) / 2, (height - 2 * padding) / 2),
      y: height / 2 + Math.sin(angle) * Math.min((width - 2 * padding) / 2, (height - 2 * padding) / 2),
    };
  });
  ctx.strokeStyle = topology === "mesh" ? "rgba(94,143,181,.13)" : "rgba(94,143,181,.48)";
  ctx.lineWidth = 1;
  const edge = (a, b) => {
    ctx.moveTo(nodes[a].x, nodes[a].y);
    ctx.lineTo(nodes[b].x, nodes[b].y);
  };
  ctx.beginPath();
  if (topology === "ring") {
    for (let i = 0; i < count; i++) edge(i, (i + 1) % count);
  } else if (topology === "grid") {
    const columns = Math.ceil(Math.sqrt(total));
    for (let i = 0; i < count; i++) {
      if (i % columns < columns - 1 && i + 1 < count) edge(i, i + 1);
      if (i + columns < count) edge(i, i + columns);
    }
  } else {
    for (let i = 0; i < count; i++) for (let j = i + 1; j < count; j++) edge(i, j);
  }
  ctx.stroke();
  nodes.forEach((node, index) => {
    ctx.beginPath();
    ctx.arc(node.x, node.y, count > 120 ? 2.2 : 3.8, 0, Math.PI * 2);
    ctx.fillStyle = index === 0 ? "#4fd18b" : "#e8eef2";
    ctx.fill();
  });
  const names = { ring: "环形链路", grid: "网格链路", mesh: "全连接链路" };
  $("#cluster-topology-label").textContent = names[topology];
  $("#cluster-node-count").textContent = count === total ? `${total} 颗卫星` : `显示 ${count}/${total} 颗卫星`;
}

async function runLocalCluster(event) {
  event.preventDefault();
  const button = $("#local-cluster-run");
  button.disabled = true;
  $("#local-cluster-status").textContent = "计算中";
  try {
    const result = await api("/api/local-cluster/run", {
      method: "POST",
      body: JSON.stringify(collectLocalClusterConfig()),
    });
    renderLocalClusterResult(result);
    toast("局部星簇测试完成；结果未写入主训练 Session");
  } catch (error) {
    $("#local-cluster-status").textContent = "失败";
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
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
    if (!SpaceFLLocal.isLocal() || !state.formPopulated) {
      populateSessionForm();
      state.formPopulated = true;
    }
    renderExperimentList();
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
  ctx.fillStyle = "#0a0a0a";
  ctx.fillRect(0, 0, w, h);
  const values = [...historyOn, ...historyOff].map(row => Number(row.accuracy || 0));
  const maxY = Math.max(0.2, ...values) * 1.08;
  const maxRound = Math.max(1, ...historyOn.map(row => Number(row.round || 0)), ...historyOff.map(row => Number(row.round || 0)));
  ctx.strokeStyle = "#232323";
  ctx.lineWidth = ratio;
  ctx.fillStyle = "#8a8a8a";
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
  path(historyOff, "#5e8fb5");
  path(historyOn, "#ffffff");
  ctx.fillStyle = "#8a8a8a";
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
  const selected = state.datasets.find(item => item.data_dir === state.session.tune.data_dir);
  if (state.session.tune.dataset === "imagefolder" && selected) {
    $("#training-dataset").value = `imported:${selected.id}`;
  }
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
  if (data.tune.dataset.startsWith("local:")) {
    data.tune.dataset = "imagefolder";
    data.tune.data_dir = "";
  } else if (data.tune.dataset.startsWith("imported:")) {
    const selected = state.datasets.find(item => `imported:${item.id}` === data.tune.dataset);
    if (!selected) throw new Error("所选导入数据集已不存在，请刷新页面");
    data.tune.dataset = "imagefolder";
    data.tune.data_dir = selected.data_dir;
  } else if (data.tune.dataset !== "imagefolder") {
    data.tune.data_dir = "./data";
  }
  data.tune.satellite_data_profiles = state.satelliteProfiles;
  return data;
}

async function loadDatasets() {
  try {
    const catalog = await api("/api/datasets");
    state.datasets = catalog.imported;
    const select = $("#training-dataset");
    $$("option[data-imported]", select).forEach(option => option.remove());
    catalog.imported.forEach(item => {
      const option = document.createElement("option");
      option.value = `imported:${item.id}`;
      option.textContent = `${item.name} (${item.classes.length} 类)`;
      option.dataset.imported = "true";
      select.appendChild(option);
    });
    const clusterSelect = $("#cluster-dataset");
    const previousCluster = clusterSelect.value;
    $$("option[data-imported]", clusterSelect).forEach(option => option.remove());
    catalog.imported.forEach(item => {
      const option = document.createElement("option");
      option.value = `imported:${item.id}`;
      option.textContent = item.name;
      option.dataset.imported = "true";
      clusterSelect.appendChild(option);
    });
    if ([...clusterSelect.options].some(option => option.value === previousCluster)) clusterSelect.value = previousCluster;
    syncClusterDatasetFields();
    const status = $("#dataset-import-status");
    state.trainingAllowed = !catalog.public_demo || (catalog.public_training && catalog.authenticated);
    for (const id of ["dataset-import-name", "dataset-import-file", "dataset-import-button", "start-training-button"]) {
      $(`#${id}`).disabled = !state.trainingAllowed || (id === "start-training-button" && state.trainingActive);
    }
    $('#session-form button[type="submit"]').disabled = !state.trainingAllowed;
    $("#reset-session").disabled = !state.trainingAllowed;
    $("#training-access-panel").classList.toggle("hidden", !catalog.public_training);
    $("#training-password").classList.toggle("hidden", catalog.authenticated);
    $("#training-login").classList.toggle("hidden", catalog.authenticated);
    $("#training-logout").classList.toggle("hidden", !catalog.authenticated);
    if (catalog.public_training) {
      const limits = catalog.limits;
      $("#training-access-status").textContent = `${catalog.authenticated ? "已登录共享空间" : "尚未登录"}。CPU 训练，每次一个任务；最多 ${limits.max_rounds} 轮、${limits.max_epochs} Epoch、${limits.max_satellites} 颗卫星、每星 ${limits.max_samples} 张样本，任务限时 ${Math.round(limits.max_seconds / 60)} 分钟。`;
      for (const [name, value] of Object.entries({"tune.rounds": limits.max_rounds, "tune.epochs": limits.max_epochs, "tune.max_contact_epochs": limits.max_epochs, "mount.sats": limits.max_satellites, "tune.batch_size": limits.max_batch_size, "tune.max_samples": limits.max_samples})) {
        $(`[name="${name}"]`).max = value;
      }
      for (const [name, value] of [["mount.algo", "fedleo"], ["tune.dataset", "femnist"], ["mount.backend", "skyfield"], ["mount.isl", "wgs84"]]) {
        [...$(`[name="${name}"]`).options].filter(option => option.value === value).forEach(option => { option.disabled = true; });
      }
      $('[name="tune.max_samples"]').min = 1;
      status.textContent = catalog.authenticated ? "导入 ImageFolder ZIP 后选择数据集即可启动训练。数据和结果保存在课题组共享磁盘。" : "登录课题组训练空间后可以上传数据集并运行训练。";
    } else if (catalog.public_demo) {
      status.textContent = "公网演示只读。数据集导入和训练请使用完整训练环境。";
    }
    $("#dataset-sources").innerHTML = catalog.sources.map(item => `
      <div class="dataset-source">
        <a href="${escapeHtml(item.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.name)} ↗</a>
        <span>${escapeHtml(item.task)}</span>
        <small>${escapeHtml(item.compatibility)}</small>
      </div>
    `).join("");
    state.publicSources = catalog.sources;
    const publicSelect = $("#public-dataset-source");
    const previousSource = publicSelect.value;
    publicSelect.replaceChildren();
    catalog.sources.forEach(item => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = `${item.name}${item.importable ? " RGB · 可导入" : " · 查看来源"}`;
      publicSelect.appendChild(option);
    });
    if ([...publicSelect.options].some(option => option.value === previousSource)) publicSelect.value = previousSource;
    updatePublicDatasetAction();
    if (!SpaceFLLocal.isLocal() || !state.formPopulated) populateSessionForm();
    SpaceFLLocal.decorate(catalog);
  } catch (error) { toast(`数据集目录读取失败：${error.message}`, "error"); }
}

function updatePublicDatasetAction() {
  const source = (state.publicSources || []).find(item => item.id === $("#public-dataset-source").value);
  $("#public-dataset-add").textContent = source?.importable ? "添加到数据集" : "查看来源";
  $("#public-dataset-status").textContent = source?.compatibility || "";
}

async function addPublicDataset() {
  const source = (state.publicSources || []).find(item => item.id === $("#public-dataset-source").value);
  if (!source) return;
  if (!source.importable) {
    window.open(source.url, "_blank", "noopener,noreferrer");
    return;
  }
  const button = $("#public-dataset-add");
  button.disabled = true;
  try {
    if (SpaceFLLocal.isLocal()) {
      const job = await SpaceFLLocal.importPublic(source.id);
      if (job.status !== "DONE") $("#public-dataset-status").textContent = "已提交给本机连接器；下载、校验和划分完成后会自动选中。";
    } else {
      if (!state.trainingAllowed) throw new Error("请先登录课题组训练空间，或切换到我的电脑并配对连接器");
      $("#public-dataset-status").textContent = "正在从官方来源下载并导入，请保持页面打开。";
      const imported = await api(`/api/datasets/public/${source.id}`, {method: "POST", body: "{}"});
      await loadDatasets();
      $("#training-dataset").value = `imported:${imported.id}`;
      $("#public-dataset-status").textContent = `${imported.name} 已导入并选中：训练 ${imported.train_images} 张，测试 ${imported.test_images} 张。`;
    }
  } catch (error) {
    $("#public-dataset-status").textContent = `添加失败：${error.message}`;
    toast(`添加失败：${error.message}`, "error");
  } finally { button.disabled = false; }
}

function renderTrainingJob(job) {
  const monitor = $("#training-monitor");
  monitor.classList.remove("hidden");
  const active = job.status === "PENDING" || job.status === "RUNNING";
  state.trainingActive = active;
  const percent = Math.round(100 * (job.progress_round || 0) / Math.max(1, job.rounds || 1));
  monitor.innerHTML = `
    <strong>${escapeHtml(job.message || job.status)}</strong>
    <div class="progress-track"><div class="progress-bar" style="width:${percent}%"></div></div>
    <span>${job.progress_round || 0}/${job.rounds || 0} 轮 · ${escapeHtml(job.algorithm || "")} · ${escapeHtml(job.dataset || "")}${job.accuracy == null ? "" : ` · 准确率 ${formatPercent(job.accuracy)}`}</span>
    ${job.status === "FAILED" && job.log ? `<details><summary>查看训练日志</summary><pre>${escapeHtml(job.log)}</pre></details>` : ""}
    ${job.status === "DONE" ? `<button type="button" class="button secondary" id="training-open-result">${job.kind === "local_training" ? "查看并下载本机训练摘要" : "查看实验档案"}</button>` : ""}
    ${active && job.kind === "local_training" ? '<button type="button" class="button secondary" id="local-cancel-job">停止本机训练</button>' : ""}
  `;
  $("#start-training-button").disabled = active || !state.trainingAllowed;
  $("#training-open-result")?.addEventListener("click", () => {
    if (job.kind === "local_training") { SpaceFLLocal.result(job); return; }
    setView("experiments");
    selectExperiment(job.id);
  });
  $("#local-cancel-job")?.addEventListener("click", SpaceFLLocal.cancel);
}

async function watchTrainingJob(jobId) {
  clearInterval(state.trainingPoll);
  const update = async () => {
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      renderTrainingJob(job);
      if (job.status === "DONE" || job.status === "FAILED") {
        clearInterval(state.trainingPoll);
        state.trainingPoll = null;
        if (job.status === "DONE") await loadOverview();
      }
      return job.status === "PENDING" || job.status === "RUNNING";
    } catch (error) {
      clearInterval(state.trainingPoll);
      state.trainingPoll = null;
      toast(`训练状态读取失败：${error.message}`, "error");
      return false;
    }
  };
  if (await update()) state.trainingPoll = setInterval(update, 1800);
}

async function startTraining() {
  const button = $("#start-training-button");
  button.disabled = true;
  try {
    if (SpaceFLLocal.isLocal()) { await SpaceFLLocal.start(); return; }
    const job = await api("/api/training/jobs", {
      method: "POST", body: JSON.stringify(collectSessionForm()),
    });
    renderTrainingJob(job);
    watchTrainingJob(job.id);
  } catch (error) {
    button.disabled = false;
    toast(`训练启动失败：${error.message}`, "error");
  }
}

async function restoreTrainingJob() {
  if (SpaceFLLocal.isLocal()) { await SpaceFLLocal.refresh(); return; }
  try {
    const jobs = await api("/api/jobs");
    const latest = jobs.find(job => job.kind === "fl_training");
    if (latest) {
      renderTrainingJob(latest);
      if (latest.status === "PENDING" || latest.status === "RUNNING") watchTrainingJob(latest.id);
    }
  } catch (_) { /* No prior job in this process. */ }
}

async function importDataset() {
  const name = $("#dataset-import-name").value.trim();
  const file = $("#dataset-import-file").files[0];
  if (!name || !file) { toast("请输入名称并选择 ZIP 文件", "error"); return; }
  if (file.size > 200 * 1024 * 1024) { toast("ZIP 不能超过 200 MB", "error"); return; }
  const button = $("#dataset-import-button");
  button.disabled = true;
  try {
    const imported = await api(`/api/datasets/import?name=${encodeURIComponent(name)}`, {
      method: "POST", headers: { "Content-Type": "application/zip" }, body: file,
    });
    await loadDatasets();
    $("#training-dataset").value = `imported:${imported.id}`;
    $("#dataset-import-status").textContent = `${imported.name}：训练 ${imported.train_images} 张，测试 ${imported.test_images} 张。已选中，可启动训练。`;
    $("#dataset-import-file").value = "";
    toast("数据集已导入");
  } catch (error) { toast(`导入失败：${error.message}`, "error"); }
  finally { button.disabled = false; }
}

async function changeTrainingAccess(login) {
  const input = $("#training-password");
  try {
    await api(`/api/training/access/${login ? "login" : "logout"}`, {
      method: "POST", body: JSON.stringify(login ? {password: input.value} : {}),
    });
    input.value = "";
    clearInterval(state.trainingPoll);
    state.trainingPoll = null;
    state.selectedExperiment = null;
    state.trainingActive = false;
    $("#experiment-detail").innerHTML = '<div class="empty-state">选择一个实验查看逐轮曲线与卸载记录</div>';
    $("#training-monitor").classList.add("hidden");
    await loadOverview();
    await loadDatasets();
    await restoreTrainingJob();
    toast(login ? "已登录课题组训练空间" : "已退出训练空间");
  } catch (error) {
    input.value = "";
    toast(error.message, "error");
  }
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
  if (state.localClusterDefaults) renderClusterTopology({
    config: collectLocalClusterConfig(),
    num_satellites: Number($("#cluster-satellites").value),
  });
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
  $("#dataset-import-button").addEventListener("click", importDataset);
  $("#public-dataset-source").addEventListener("change", updatePublicDatasetAction);
  $("#public-dataset-add").addEventListener("click", addPublicDataset);
  $("#training-login").addEventListener("click", () => changeTrainingAccess(true));
  $("#training-logout").addEventListener("click", () => changeTrainingAccess(false));
  $("#training-password").addEventListener("keydown", event => {
    if (event.key === "Enter") { event.preventDefault(); changeTrainingAccess(true); }
  });
  $("#start-training-button").addEventListener("click", startTraining);
  $("#study-task-form").addEventListener("submit", saveStudyTask);
  $("#study-reset-button").addEventListener("click", resetStudyTask);
  $("#study-refresh-button").addEventListener("click", loadStudyTask);
  $("#local-cluster-form").addEventListener("submit", runLocalCluster);
  $("#cluster-dataset").addEventListener("change", syncClusterDatasetFields);
  ["cluster-satellites", "cluster-topology"].forEach(id => $("#" + id).addEventListener("change", () => {
    renderClusterTopology({ config: collectLocalClusterConfig(), num_satellites: Number($("#cluster-satellites").value) });
  }));
  $("#local-cluster-reset").addEventListener("click", () => {
    if (state.localClusterDefaults) {
      applyLocalClusterConfig(state.localClusterDefaults);
      renderClusterTopology({ config: state.localClusterDefaults, num_satellites: state.localClusterDefaults.num_satellites });
    }
  });
  $("#local-cluster-copy-profile").addEventListener("click", async () => {
    const profile = state.localClusterResult?.federated_profile;
    if (!profile) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(profile, null, 2));
      toast("全局调用 profile 已复制");
    } catch (error) {
      toast(`复制失败：${error.message}`, "error");
    }
  });
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
const initialViewHash = (window.location.hash || "").replace(/^#/, "");
if (viewTitles[initialViewHash]) setView(initialViewHash);
async function initializeTrainingWorkspace() {
  await loadOverview();
  await SpaceFLLocal.initialize();
  await loadDatasets();
  await restoreTrainingJob();
}
initializeTrainingWorkspace();
loadStudyTask();
