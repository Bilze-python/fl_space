/* Outbound worker pairing; the browser never connects to localhost. */
window.SpaceFLLocal = (() => {
  let connection = null;
  let worker = null;
  let catalog = null;
  let timer = null;
  let initializedBackend = false;
  let datasetSignature = "";
  let polling = false;
  let selectedImportJobId = null;
  const storageKey = "spacefl-local-connection";
  const isLocal = () => document.querySelector("#compute-backend").value === "local";
  const selectedDatasetId = () => {
    const value = document.querySelector("#training-dataset").value;
    return value.startsWith("local:") ? value.slice(6) : null;
  };
  const allowed = () => Boolean(connection && worker?.online);

  async function request(suffix = "", body) {
    if (!connection) throw new Error("请先连接本地程序");
    return api(`/api/local-compute/${encodeURIComponent(connection.id)}${suffix}`, {
      method: body === undefined ? "GET" : "POST",
      headers: {"Content-Type": "application/json", "Authorization": `Bearer ${connection.token}`},
      ...(body === undefined ? {} : {body: JSON.stringify(body)}),
    });
  }

  function updateControls() {
    if (!catalog) return;
    const local = isLocal();
    $("#local-compute-controls").classList.toggle("hidden", !local);
    $("#training-access-panel").classList.toggle("hidden", local || !catalog.public_training);
    state.trainingAllowed = local ? allowed() : !catalog.public_demo || (catalog.public_training && catalog.authenticated);
    $("#start-training-button").disabled = !state.trainingAllowed || state.trainingActive;
    for (const id of ["dataset-import-name", "dataset-import-file", "dataset-import-button"]) {
      $(`#${id}`).disabled = local || !state.trainingAllowed;
    }
    $('#session-form button[type="submit"]').disabled = local || !state.trainingAllowed;
    $("#reset-session").disabled = local || !state.trainingAllowed;
    if (local) $("#dataset-import-status").textContent = "自定义 ZIP 可按上方本机命令导入；EuroSAT RGB 可直接点“添加到数据集”。原始图片留在本机。";
    const limits = local ? {max_rounds: 1000, max_epochs: 100, max_satellites: 100, max_batch_size: 512, max_samples: 20000} : catalog.limits;
    if (local || catalog.public_training) {
      for (const [name, value] of Object.entries({"tune.rounds": limits.max_rounds, "tune.epochs": limits.max_epochs, "tune.max_contact_epochs": limits.max_epochs, "mount.sats": limits.max_satellites, "tune.batch_size": limits.max_batch_size, "tune.max_samples": limits.max_samples})) {
        $(`[name="${name}"]`).max = value;
      }
      $('[name="tune.max_samples"]').min = 1;
    }
    for (const [name, value] of [["mount.algo", "fedleo"], ["tune.dataset", "femnist"], ["mount.backend", "skyfield"], ["mount.isl", "wgs84"]]) {
      [...$(`[name="${name}"]`).options].filter(option => option.value === value).forEach(option => { option.disabled = local || catalog.public_training; });
    }
    $("#local-disconnect").disabled = !connection;
    $("#local-pair").disabled = Boolean(connection);
  }

  function renderDatasets() {
    const signature = JSON.stringify([isLocal(), worker?.datasets || [], catalog?.imported || []]);
    if (signature === datasetSignature) return;
    datasetSignature = signature;
    const select = $("#training-dataset");
    const previous = select.value;
    $$("option[data-imported]", select).forEach(option => option.remove());
    const items = isLocal() ? worker?.datasets || [] : catalog?.imported || [];
    for (const item of items) {
      const option = document.createElement("option");
      option.dataset.imported = "true";
      option.value = `${isLocal() ? "local" : "imported"}:${item.id}`;
      option.textContent = `${isLocal() ? "本机 · " : ""}${item.name} (${item.classes.length} 类)`;
      select.appendChild(option);
    }
    if ([...select.options].some(option => option.value === previous)) select.value = previous;
  }

  function renderWorker() {
    $("#local-compute-status").textContent = worker ? `${worker.name} · ${worker.device.toUpperCase()} · ${worker.online ? "在线" : "离线，请保持本地程序运行"}。每台电脑同时运行一个训练任务；时间和线程上限由本地程序决定。` : "尚未连接电脑；请先启动本地程序并输入配对码。";
    const container = $("#local-compute-jobs");
    container.replaceChildren();
    for (const job of [...(worker?.jobs || [])].reverse()) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "button secondary";
      button.textContent = job.kind === "dataset_import"
        ? `${job.dataset} · ${job.message}`
        : `${job.algorithm} · ${job.dataset} · ${job.status} · ${job.progress_round}/${job.rounds} 轮`;
      button.addEventListener("click", () => {
        if (job.kind === "dataset_import") $("#public-dataset-status").textContent = job.message;
        else renderTrainingJob(job);
      });
      container.appendChild(button);
    }
    renderDatasets();
    const activeImport = [...(worker?.jobs || [])].reverse().find(job => job.kind === "dataset_import" && ["PENDING", "RUNNING", "FAILED"].includes(job.status));
    if (isLocal() && activeImport && $("#public-dataset-source").value === "eurosat_rgb") {
      $("#public-dataset-status").textContent = activeImport.message;
    }
    const imported = [...(worker?.jobs || [])].reverse().find(job => job.kind === "dataset_import" && job.status === "DONE" && job.imported_id);
    if (isLocal() && imported && imported.id !== selectedImportJobId) {
      selectedImportJobId = imported.id;
      $("#training-dataset").value = `local:${imported.imported_id}`;
      $("#public-dataset-status").textContent = `${imported.dataset} 已导入本机并选中，可启动训练。`;
    }
    updateControls();
  }

  async function refresh() {
    if (!connection || polling) return;
    polling = true;
    try {
      worker = await request();
      if (isLocal()) {
        const latest = [...worker.jobs].reverse().find(job => job.kind === "local_training");
        if (latest) renderTrainingJob(latest);
        state.trainingActive = worker.jobs.some(job => ["PENDING", "RUNNING"].includes(job.status));
      }
      renderWorker();
    } catch (error) {
      if (worker) worker.online = false;
      renderWorker();
      $("#local-compute-status").textContent = `连接不可用：${error.message}。本机已运行的任务和文件可在电脑上查看。`;
    } finally { polling = false; }
  }

  async function pair() {
    try {
      connection = await api("/api/local-compute/pair", {method: "POST", body: JSON.stringify({code: $("#local-pair-code").value.trim()})});
      try { sessionStorage.setItem(storageKey, JSON.stringify(connection)); } catch (_) {}
      $("#local-pair-code").value = "";
      await refresh();
      toast("已连接本机，训练将使用该电脑的算力");
    } catch (error) { toast(error.message, "error"); }
  }

  async function disconnect() {
    try { await request("/disconnect", {}); } catch (_) {}
    connection = worker = null;
    try { sessionStorage.removeItem(storageKey); } catch (_) {}
    state.trainingActive = false;
    $("#training-monitor").classList.add("hidden");
    renderWorker();
    toast("连接已解除；已启动的本机训练继续运行，按本地终端 Ctrl+C 可停止");
  }

  function decorate(value) {
    catalog = value;
    if (!initializedBackend) {
      initializedBackend = true;
      $("#compute-backend").value = connection || catalog.public_demo ? "local" : "server";
    }
    datasetSignature = "";
    renderWorker();
  }

  async function initialize() {
    try { connection = JSON.parse(sessionStorage.getItem(storageKey) || "null"); } catch (_) {}
    $("#local-server-example").textContent = `--server ${location.origin}`;
    $("#local-pair").addEventListener("click", pair);
    $("#local-disconnect").addEventListener("click", disconnect);
    $("#local-pair-code").addEventListener("keydown", event => {
      if (event.key === "Enter") { event.preventDefault(); pair(); }
    });
    $("#compute-backend").addEventListener("change", async () => {
      clearInterval(state.trainingPoll);
      state.trainingActive = false;
      $("#training-monitor").classList.add("hidden");
      await loadDatasets();
      if (isLocal()) await refresh(); else await restoreTrainingJob();
    });
    if (connection) await refresh();
    clearInterval(timer);
    timer = setInterval(refresh, 3000);
  }

  async function start() {
    if (!allowed()) throw new Error("本地程序未连接，请先配对并保持在线");
    const job = await request("/jobs", {session: collectSessionForm(), dataset_id: selectedDatasetId()});
    renderTrainingJob(job);
    await refresh();
  }

  async function importPublic(source) {
    if (!allowed()) throw new Error("请先启动本地连接器并完成配对");
    const job = await request("/datasets/import", {source});
    if (job.status === "DONE" && job.imported_id) {
      $("#training-dataset").value = `local:${job.imported_id}`;
      $("#public-dataset-status").textContent = "EuroSAT RGB 已在本机数据集列表中，已选中。";
    }
    await refresh();
    return job;
  }

  function result(job) {
    const rows = job.history || [];
    const blob = new Blob([JSON.stringify(job, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `spacefl-local-${job.id}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    const detail = document.createElement("details");
    detail.open = true;
    const summary = document.createElement("summary");
    summary.textContent = "本机训练逐轮准确率（摘要已下载）";
    detail.appendChild(summary);
    const pre = document.createElement("pre");
    pre.textContent = rows.map(row => `第 ${row.round} 轮：${formatPercent(row.accuracy)}`).join("\n");
    detail.appendChild(pre);
    $("#local-result-viewer").replaceChildren(detail);
    const canvas = document.createElement("canvas");
    canvas.style.width = "100%";
    canvas.style.height = "220px";
    canvas.setAttribute("aria-label", "本机训练逐轮准确率曲线");
    detail.prepend(canvas);
    requestAnimationFrame(() => drawAccuracyChart(canvas, rows, []));
    detail.scrollIntoView({behavior: "smooth", block: "nearest"});
  }

  async function cancel() {
    try { await request("/cancel", {}); await refresh(); }
    catch (error) { toast(error.message, "error"); }
  }

  return {initialize, decorate, isLocal, allowed, start, importPublic, refresh, result, cancel, selectedDatasetId};
})();
