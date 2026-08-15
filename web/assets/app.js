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
};

const viewTitles = {
  overview: ["SPACE FEDERATED LEARNING", "实验总览"],
  config: ["EXPERIMENT CONTROL", "实验配置"],
  experiments: ["RESULT ARCHIVE", "实验档案"],
  orbit: ["CONSTELLATION VIEW", "轨道视图"],
  library: ["RESEARCH LIBRARY", "文献库"],
  ai: ["AI EXPERIMENT ANALYST", "AI 实验解读"],
};

function $(selector, root = document) { return root.querySelector(selector); }
function $$(selector, root = document) { return [...root.querySelectorAll(selector)]; }

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
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
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  }).format(new Date(value));
}

function setView(name) {
  $$(".nav-item").forEach(button => button.classList.toggle("active", button.dataset.view === name));
  $$(".view").forEach(view => view.classList.toggle("active", view.id === `view-${name}`));
  $("#view-eyebrow").textContent = viewTitles[name][0];
  $("#view-title").textContent = viewTitles[name][1];
  if (name === "orbit" && !state.orbit && !state.orbitLoading) loadOrbit();
  if (name === "library" && !state.library.length) loadLibrary();
  requestAnimationFrame(resizeCanvases);
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
    <option value="${item.id}">${item.name} · ${item.seed ?? "--"}</option>
  `).join("");
}

function populateOrbitExperiments() {
  const select = $("#orbit-experiment-select");
  if (!select) return;
  const current = select.value;
  select.innerHTML = `<option value="">当前 Session 参数</option>${state.experiments.map(item => `
    <option value="${item.id}">${item.name} · ${formatDate(item.created_at)}</option>
  `).join("")}`;
  if ([...select.options].some(option => option.value === current)) select.value = current;
}

function renderRecentExperiments(experiments) {
  $("#recent-experiments").innerHTML = experiments.map(item => `
    <tr>
      <td><strong>${item.name}</strong><br><small>${formatDate(item.created_at)}</small></td>
      <td>${String(item.dataset || "--").toUpperCase()}</td>
      <td>${item.rounds ?? "--"}</td>
      <td>${formatPercent(item.accuracy)}</td>
      <td class="${item.accuracy_delta >= 0 ? "positive" : ""}">${item.accuracy_delta == null ? "--" : formatPercent(item.accuracy_delta)}</td>
      <td><span class="status-badge ${item.status === "DONE" ? "green" : "red"}">${item.status}</span></td>
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
    toast("参数已保存到 .fls_session.json");
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
  $("#experiment-list").innerHTML = state.experiments.map(item => `
    <article class="experiment-item ${state.selectedExperiment === item.id ? "active" : ""}" data-experiment-id="${item.id}">
      <strong>${item.name}</strong>
      <div class="experiment-meta"><span>${String(item.dataset).toUpperCase()} · ${item.rounds ?? "--"} 轮</span><span>${formatDate(item.created_at)}</span></div>
      <div class="experiment-meta" style="margin-top:8px"><span>${item.offloaded} 样本卸载</span><span class="${item.accuracy_delta >= 0 ? "positive" : ""}">${item.accuracy_delta == null ? "--" : formatPercent(item.accuracy_delta)}</span></div>
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
    const raw = detail.raw;
    const on = raw.offload_on || raw.fedleo || {};
    const off = raw.offload_off || raw.baseline || {};
    $("#experiment-detail").innerHTML = `
      <div class="panel-head">
        <div><p class="panel-kicker">${detail.kind.replaceAll("_", " ").toUpperCase()}</p><h2>${detail.name}</h2></div>
        <span class="status-badge ${detail.status === "DONE" ? "green" : "red"}">${detail.status}</span>
      </div>
      <div class="detail-metrics">
        <div class="detail-metric"><span>卸载准确率</span><strong>${formatPercent(on.final_accuracy)}</strong></div>
        <div class="detail-metric"><span>控制组准确率</span><strong>${formatPercent(off.final_accuracy)}</strong></div>
        <div class="detail-metric"><span>真实卸载</span><strong>${on.total_offloaded ?? detail.offloaded ?? 0}</strong></div>
        <div class="detail-metric"><span>耗时</span><strong>${Number(raw.elapsed_sec || on.elapsed_sec || 0).toFixed(1)}s</strong></div>
      </div>
      <div class="experiment-detail-actions">
        <button class="button primary" id="replay-experiment" type="button">载入实时演示</button>
      </div>
      <canvas id="detail-chart"></canvas>
      <div class="chart-legend"><span><i class="legend-line on"></i>卸载开启</span><span><i class="legend-line off"></i>卸载关闭/基线</span></div>
    `;
    $("#replay-experiment").addEventListener("click", () => replayExperiment(id));
    drawAccuracyChart($("#detail-chart"), detail.history_on, detail.history_off);
  } catch (error) {
    $("#experiment-detail").innerHTML = `<div class="empty-state">${error.message}</div>`;
  }
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
      <article class="library-item" data-library-path="${item.path}" data-library-type="${item.type}">
        <strong>${item.title}</strong>
        <small>${item.type.toUpperCase()} · ${(item.size / 1024).toFixed(0)} KB</small>
      </article>
    `).join("");
    $$(".library-item").forEach(element => element.addEventListener("click", () => openDocument(element.dataset.libraryPath, element.dataset.libraryType, element)));
  } catch (error) { toast(error.message, "error"); }
}

async function loadAiSettings() {
  try {
    const settings = await api("/api/settings");
    const badge = $("#ai-provider-status");
    badge.textContent = settings.ai_configured ? "DeepSeek 已连接" : "未配置密钥";
    badge.className = `status-badge ${settings.ai_configured ? "green" : "amber"}`;
  } catch (error) {
    $("#ai-provider-status").textContent = "检测失败";
  }
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
    $("#reader-panel").innerHTML = `<div class="reader-head"><h2>${title}</h2></div><iframe class="reader-frame" src="/api/library/file?path=${encodeURIComponent(path)}"></iframe>`;
    return;
  }
  try {
    const data = await api(`/api/library/content?path=${encodeURIComponent(path)}`);
    $("#reader-panel").innerHTML = `<div class="reader-head"><h2>${title}</h2></div><div class="reader-content"><pre></pre></div>`;
    $(".reader-content pre").textContent = data.content;
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
    window.draw2DOrbit("orbit-canvas-2d", state.orbit);
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
  }, Number($("#orbit-playback-speed")?.value || 250));
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
  $("#library-search").addEventListener("input", event => loadLibrary(event.target.value));
  $("#ai-form").addEventListener("submit", askAi);
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
