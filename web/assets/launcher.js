const state = { session: null, preset: "quick", activeJob: null, pollTimer: null, orbitPhase: 0 };
const presets = {
  quick: { sats: 3, rounds: 5, epochs: 2, isl: false, label: "快速试跑" },
  standard: { sats: 8, rounds: 10, epochs: 2, isl: false, label: "标准验证" },
  link: { sats: 16, rounds: 15, epochs: 1, isl: true, label: "链路压力" },
};
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `请求失败 (${response.status})`);
  return payload;
}

function showToast(message, isError = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("visible"), 3200);
}

function setService(online) {
  $("#service-dot").className = `service-dot ${online ? "online" : "offline"}`;
  $("#service-label").textContent = online ? "本地服务在线" : "本地服务不可用";
  $("#launch-button").disabled = !online || Boolean(state.activeJob);
}

function clampInput(input) {
  const min = Number(input.min || Number.NEGATIVE_INFINITY);
  const max = Number(input.max || Number.POSITIVE_INFINITY);
  input.value = String(Math.max(min, Math.min(max, Number(input.value || min))));
}

function applyPreset(name) {
  const preset = presets[name];
  state.preset = name;
  $("#sats").value = preset.sats;
  $("#rounds").value = preset.rounds;
  $("#epochs").value = preset.epochs;
  $("#isl").checked = preset.isl;
  $$(".preset").forEach((button) => {
    const selected = button.dataset.preset === name;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  syncSummary();
}

function syncSummary() {
  const sats = Number($("#sats").value);
  const rounds = Number($("#rounds").value);
  const epochs = Number($("#epochs").value);
  const algorithm = $("#algorithm").selectedOptions[0].text;
  const isl = $("#isl").checked;
  const minutes = Math.max(2, Math.ceil(rounds * epochs * (sats / 4) * 0.45));
  $("#telemetry-sats").textContent = sats;
  $("#telemetry-rounds").textContent = rounds;
  $("#telemetry-isl").textContent = isl ? "GS + ISL" : "GS";
  $("#mission-window").textContent = `00:${String(Math.min(minutes, 59)).padStart(2, "0")}:00`;
  $("#ready-detail").textContent = `预计 ${minutes}–${minutes + 2} 分钟 · CPU 本地执行`;
  $("#launch-caption").textContent = `${presets[state.preset].label} / ${algorithm}`;
  drawOrbit();
}

function sessionFromForm() {
  const current = state.session || { tune: {}, mount: {} };
  return {
    tune: { ...current.tune, dataset: $("#dataset").value, rounds: Number($("#rounds").value), epochs: Number($("#epochs").value), seed: Number($("#seed").value), non_iid: $("#non-iid").checked },
    mount: { ...current.mount, algo: $("#algorithm").value, sats: Number($("#sats").value), isl: $("#isl").checked ? "wgs84" : "disabled" },
  };
}

function setJobState(kind, label) {
  $("#mission-state-dot").className = kind;
  $("#mission-state-text").textContent = label;
  $("#launch-button").disabled = kind === "running";
}

function updateProgress(job) {
  const running = ["PENDING", "RUNNING"].includes(job.status);
  const done = job.status === "DONE";
  const failed = job.status === "FAILED";
  const progress = done || failed ? 100 : job.status === "RUNNING" ? 62 : 20;
  $("#stage-progress").classList.add("visible");
  $("#progress-bar").style.width = `${progress}%`;
  $("#progress-value").textContent = `${progress}%`;
  $("#progress-label").textContent = job.message || (running ? "任务执行中" : "任务已结束");
  if (running) setJobState("running", "运行中");
  else if (done) setJobState("done", "已完成");
  else if (failed) setJobState("failed", "运行失败");
  else setJobState("", "待命");
}

async function launch(event) {
  event.preventDefault();
  if (state.activeJob) return;
  try {
    $("#launch-button").disabled = true;
    $("#ready-title").textContent = "正在写入会话";
    state.session = (await api("/api/session", { method: "PUT", body: JSON.stringify(sessionFromForm()) })).session;
    const job = await api("/api/experiments/fedleo-validation", { method: "POST", body: JSON.stringify({ rounds: Number($("#rounds").value), seed: Number($("#seed").value) }) });
    state.activeJob = job.id;
    $("#ready-title").textContent = "任务已进入队列";
    updateProgress(job);
    renderJobs([job]);
    showToast(`任务 ${job.id} 已启动`);
    pollJob(job.id);
  } catch (error) {
    state.activeJob = null;
    $("#ready-title").textContent = "启动失败";
    setJobState("failed", "启动失败");
    $("#launch-button").disabled = false;
    showToast(error.message, true);
  }
}

async function pollJob(jobId) {
  clearInterval(state.pollTimer);
  const check = async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      updateProgress(job);
      await loadJobs(false);
      if (!["PENDING", "RUNNING"].includes(job.status)) {
        clearInterval(state.pollTimer);
        state.activeJob = null;
        $("#launch-button").disabled = false;
        $("#ready-title").textContent = job.status === "DONE" ? "任务验证完成" : "任务需要检查";
        showToast(job.status === "DONE" ? "联邦学习任务已完成" : "任务运行失败，请查看完整操作台", job.status !== "DONE");
      }
    } catch (error) {
      clearInterval(state.pollTimer);
      state.activeJob = null;
      setJobState("failed", "连接中断");
      showToast(error.message, true);
    }
  };
  await check();
  if (state.activeJob) state.pollTimer = setInterval(check, 1800);
}

function renderJobs(jobs) {
  const list = $("#job-list");
  if (!jobs.length) {
    list.innerHTML = '<div class="empty-job">暂无运行记录</div>';
    return;
  }
  list.innerHTML = jobs.slice(0, 3).map((job) => {
    const stateClass = ["PENDING", "RUNNING"].includes(job.status) ? "running" : job.status === "DONE" ? "done" : "failed";
    const statusLabel = { PENDING: "等待执行", RUNNING: "正在运行", DONE: "已完成", FAILED: "失败" }[job.status] || job.status;
    const created = job.created_at ? new Date(job.created_at).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }) : "--:--";
    return `<div class="job-item"><span class="job-status ${stateClass}"></span><span class="job-copy"><strong>${statusLabel} · ${job.rounds || "--"} 轮</strong><small>${job.id}</small></span><time class="job-time">${created}</time></div>`;
  }).join("");
}

async function loadJobs(resume = true) {
  try {
    const jobs = await api("/api/jobs");
    renderJobs(jobs);
    const running = resume && jobs.find((job) => ["PENDING", "RUNNING"].includes(job.status));
    if (running && !state.activeJob) {
      state.activeJob = running.id;
      updateProgress(running);
      pollJob(running.id);
    }
  } catch (_) {
    renderJobs([]);
  }
}

function resizeCanvas() {
  const canvas = $("#orbit-canvas");
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  return { canvas, width, height, ratio };
}

function drawOrbit() {
  const { canvas, width, height, ratio } = resizeCanvas();
  const ctx = canvas.getContext("2d");
  const sats = Math.min(Number($("#sats")?.value || 3), 24);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(86,101,115,.16)";
  ctx.lineWidth = ratio;
  for (let x = 0; x < width; x += 42 * ratio) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
  for (let y = 0; y < height; y += 42 * ratio) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }

  const cx = width * .5;
  const cy = height * .53;
  const earthR = Math.min(width, height) * .15;
  const orbitRx = Math.min(width * .42, height * .54);
  const orbitRy = Math.min(height * .32, width * .22);
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(-0.12);
  ctx.strokeStyle = "rgba(109,187,208,.38)";
  ctx.beginPath(); ctx.ellipse(0, 0, orbitRx, orbitRy, 0, 0, Math.PI * 2); ctx.stroke();
  ctx.restore();
  ctx.fillStyle = "#111d25";
  ctx.strokeStyle = "#3f6b78";
  ctx.beginPath(); ctx.arc(cx, cy, earthR, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.strokeStyle = "rgba(109,187,208,.28)";
  ctx.beginPath(); ctx.ellipse(cx, cy, earthR * .95, earthR * .34, -.25, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.ellipse(cx, cy, earthR * .42, earthR, .15, 0, Math.PI * 2); ctx.stroke();

  const points = [];
  for (let index = 0; index < sats; index += 1) {
    const angle = state.orbitPhase + (index / sats) * Math.PI * 2;
    const x0 = orbitRx * Math.cos(angle);
    const y0 = orbitRy * Math.sin(angle);
    const rotation = -0.12;
    const x = cx + x0 * Math.cos(rotation) - y0 * Math.sin(rotation);
    const y = cy + x0 * Math.sin(rotation) + y0 * Math.cos(rotation);
    points.push({ x, y });
    ctx.fillStyle = index === 0 ? "#70d69a" : "#d7ad5c";
    ctx.fillRect(x - 2.5 * ratio, y - 2.5 * ratio, 5 * ratio, 5 * ratio);
  }
  if ($("#isl")?.checked && points.length > 1) {
    ctx.strokeStyle = "rgba(112,214,154,.2)";
    points.forEach((point, index) => { const next = points[(index + 1) % points.length]; ctx.beginPath(); ctx.moveTo(point.x, point.y); ctx.lineTo(next.x, next.y); ctx.stroke(); });
  }
}

function animateOrbit() {
  state.orbitPhase += state.activeJob ? 0.004 : 0.0015;
  drawOrbit();
  requestAnimationFrame(animateOrbit);
}

async function initialize() {
  const updateClock = () => { $("#local-time").textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false }); };
  updateClock();
  setInterval(updateClock, 1000);
  try {
    await api("/api/health");
    setService(true);
    state.session = await api("/api/session");
    $("#algorithm").value = state.session.mount?.algo || "fedavg";
    $("#dataset").value = state.session.tune?.dataset || "mnist";
    $("#seed").value = state.session.tune?.seed || 20260815;
    $("#non-iid").checked = Boolean(state.session.tune?.non_iid);
  } catch (_) {
    setService(false);
    showToast("无法连接 SpaceFL 本地服务", true);
  }
  applyPreset("quick");
  await loadJobs();
  animateOrbit();
}

$$(".preset").forEach((button) => button.addEventListener("click", () => applyPreset(button.dataset.preset)));
$$('[data-step]').forEach((button) => button.addEventListener("click", () => {
  const input = $(`#${button.dataset.step}`);
  input.value = String(Number(input.value) + Number(button.dataset.delta));
  clampInput(input);
  syncSummary();
}));
$$("#launch-form input, #launch-form select").forEach((control) => control.addEventListener("change", syncSummary));
$("#launch-form").addEventListener("submit", launch);
$("#reset-config").addEventListener("click", () => applyPreset("quick"));
$("#refresh-jobs").addEventListener("click", () => loadJobs());
window.addEventListener("resize", drawOrbit);
initialize();
