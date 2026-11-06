/* SpaceFL 科研智能平台 — 研究助手 / 期刊文献库 / 前沿情报
 *
 * 与 app.js 共存：app.js 负责既有实验视图，本文件负责新增的科研视图
 * 与 /api/research/* 接口。两者通过 "spacefl:view" 自定义事件同步路由。
 */
(() => {
  "use strict";

  const rstate = {
    config: null,
    overview: null,
    experiments: [],
    agentMessages: [],
    agentMode: "auto",
    library: [],
    libraryQuery: "",
    libraryFilter: "all",
    selectedDocId: "",
    readerTab: "digest",
    digestCache: {},
    intel: null,
    intelBrief: "",
    jobs: {},
    jobTimers: {},
    heroFrame: null,
    heroCanvas: null,
    booted: false,
    viewsLoaded: {},
  };

  const $r = (selector, root = document) => root.querySelector(selector);
  const $$r = (selector, root = document) => [...root.querySelectorAll(selector)];

  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>'"]/g, ch => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
    })[ch]);
  }

  function pct(value, digits = 1) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "--";
    return `${(number * 100).toFixed(digits)}%`;
  }

  function num(value, digits = 1) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "--";
    return number.toFixed(digits);
  }

  function when(value) {
    if (!value) return "--";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "--";
    return new Intl.DateTimeFormat("zh-CN", {
      month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
    }).format(date);
  }

  function md(text) {
    const raw = String(text == null ? "" : text);
    if (window.marked && window.DOMPurify) {
      try {
        return DOMPurify.sanitize(marked.parse(raw, { gfm: true, breaks: true }));
      } catch (_) { /* fallthrough */ }
    }
    return `<p>${esc(raw).replace(/\n{2,}/g, "</p><p>").replace(/\n/g, "<br>")}</p>`;
  }

  function badge(text, tone = "") {
    return `<span class="status-badge ${tone}">${esc(text)}</span>`;
  }

  function toastR(message, type = "success") {
    if (typeof window.toast === "function") window.toast(message, type);
    else console.log(message);
  }

  async function request(path, options = {}) {
    if (typeof window.api === "function") return window.api(path, options);
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const payload = await response.json();
        const detail = payload && payload.detail;
        message = typeof detail === "string" ? detail : (detail && detail.message) || message;
      } catch (_) { /* ignore */ }
      throw new Error(message);
    }
    return response.json();
  }

  function activateView(name) {
    if (typeof window.setView === "function") window.setView(name);
    else {
      $$r(".nav-item").forEach(btn => btn.classList.toggle("active", btn.dataset.view === name));
      $$r(".view").forEach(view => view.classList.toggle("active", view.id === `view-${name}`));
    }
  }

  // ── 概览 / 研究目标 ──────────────────────────────────────

  async function loadOverview() {
    try {
      const [overview, config, experiments] = await Promise.all([
        request("/api/research/overview"),
        request("/api/research/config"),
        request("/api/research/experiments").catch(() => []),
      ]);
      rstate.overview = overview;
      rstate.config = config;
      rstate.experiments = Array.isArray(experiments) ? experiments : [];
      renderOverview(overview);
      renderConfig(config);
      fillExperimentSelect();
    } catch (error) {
      console.warn("研究概览加载失败", error);
    }
  }

  function renderOverview(overview) {
    const literature = overview.literature || {};
    const intel = overview.intel || {};
    const experiments = overview.experiments || {};
    setText("#hero-stat-library", literature.total ?? "--");
    setText("#hero-stat-parsed", literature.parsed ?? "--");
    setText("#hero-stat-intel", intel.count ?? "--");
    setText("#hero-stat-experiments", experiments.count ?? "--");
    setText("#overview-goal", overview.goal || "尚未设置研究目标");
    setText("#context-library", `${literature.total ?? 0} 篇 · 已解析 ${literature.parsed ?? 0}`);
    setText("#context-intel", `${intel.count ?? 0} 条`);
    setText("#context-experiments", `${experiments.count ?? 0} 个`);
    const mini = $("#overview-intel");
    if (mini) {
      const items = intel.items || [];
      mini.innerHTML = items.length
        ? items.map(item => `
            <article class="intel-mini-item">
              <div class="intel-mini-meta">${esc(item.source || "前沿")} · ${esc(when(item.published))}</div>
              <strong>${esc(item.title)}</strong>
              <p>${esc((item.summary || "").slice(0, 120))}</p>
            </article>`).join("")
        : '<div class="empty-state compact">尚未抓取情报，前往「前沿情报」刷新。</div>';
    }
  }

  function setText(selector, value) {
    const el = $r(selector);
    if (el) el.textContent = value == null ? "--" : String(value);
  }

  function demoMode() {
    return Boolean(rstate.config && rstate.config.demo_mode);
  }

  function ensureDemoBanner(show) {
    const thread = $("#agent-messages");
    if (!thread) return;
    const existing = thread.querySelector(".demo-banner");
    if (!show) {
      if (existing) existing.remove();
      return;
    }
    if (existing) return;
    const banner = document.createElement("div");
    banner.className = "agent-message assistant demo-banner";
    banner.innerHTML = `<p><strong>公网演示模式</strong>：可完整体验文献精读、知识库问答、前沿情报、自然语言实验规划与结果解读；为保证服务稳定，<b>训练任务、文献导入与配置写入已禁用</b>。完整功能请本地部署体验（<code>start_spacefl.bat</code>）。</p>`;
    thread.insertBefore(banner, thread.firstChild);
  }

  function renderConfig(config) {
    const goalInput = $("#research-goal-input");
    if (goalInput) goalInput.value = config.goal || "";
    const goalBadge = $("#agent-goal-state");
    if (goalBadge) {
      goalBadge.textContent = config.goal ? "已设置" : "未设置";
      goalBadge.className = `status-badge ${config.goal ? "green" : "amber"}`;
    }
    const ai = config.ai || {};
    const status = $("#agent-provider-status");
    if (status) {
      status.textContent = ai.configured ? `${ai.provider || "LLM"} 已配置` : "离线规则引擎";
      status.className = `status-badge ${ai.configured ? "green" : "amber"}`;
    }
    const detail = $("#agent-provider-detail");
    if (detail) {
      detail.innerHTML = ai.configured
        ? `<strong>${esc(ai.provider || "--")}</strong><span>${esc(ai.model || "--")}</span><small>${esc(ai.base_url || "")} · ${esc(ai.wire_api || "chat")} · 调用失败会自动降级为离线规则引擎</small>`
        : `<strong>离线规则引擎</strong><span>未配置 API Key</span><small>规划、分析、精读仍可完整演示，配置后可获得大模型综合结论。</small>`;
    }
    const nameInput = $("#provider-name");
    if (nameInput && document.activeElement !== nameInput) nameInput.value = ai.provider || "";
    const modelInput = $("#provider-model");
    if (modelInput && document.activeElement !== modelInput) modelInput.value = ai.model || "";
    const baseInput = $("#provider-base-url");
    if (baseInput && document.activeElement !== baseInput) baseInput.value = ai.base_url || "";
    const wireSelect = $("#provider-wire-api");
    if (wireSelect && ai.wire_api) wireSelect.value = ai.wire_api;
    const topicsInput = $("#intel-topic-input");
    if (topicsInput && document.activeElement !== topicsInput) {
      topicsInput.value = (config.topics || []).join(", ");
    }
    const demo = demoMode();
    const providerConfig = $(".provider-config");
    if (providerConfig) providerConfig.classList.toggle("hidden", demo);
    const providerNote = $("#provider-note");
    if (providerNote && demo) providerNote.textContent = ai.configured
      ? "公网模型由服务端配置，访客无需填写 API Key；调用失败时自动使用离线规则引擎。"
      : "公网当前使用离线规则引擎。";
    const goalSaveButton = $("#research-goal-save");
    if (goalSaveButton) {
      goalSaveButton.disabled = demo;
      goalSaveButton.textContent = demo ? "演示模式只读" : "保存研究目标";
    }
    ["#provider-save", "#provider-test", "#cc-switch-import"].forEach(selector => {
      const button = $r(selector);
      if (button) button.disabled = demo;
    });
    if (demo) {
      const statusBadge = $("#agent-provider-status");
      if (statusBadge) {
        statusBadge.textContent = ai.configured ? `${ai.provider || "在线模型"} · 演示只读` : "离线演示 · 只读";
        statusBadge.className = `status-badge ${ai.configured ? "green" : "amber"}`;
      }
    }
    ensureDemoBanner(demo);
    renderTopicChips(config.topics || []);
  }

  function renderTopicChips(topics) {
    const container = $("#intel-topic-chips");
    if (!container) return;
    container.innerHTML = (topics || []).map(topic => `<span class="topic-chip">${esc(topic)}</span>`).join("");
  }

  async function loadConfigOnly() {
    try {
      const config = await request("/api/research/config");
      rstate.config = config;
      renderConfig(config);
      return config;
    } catch (error) {
      toastR(`配置读取失败：${error.message}`, "error");
      return null;
    }
  }

  async function saveResearchGoal() {
    if (demoMode()) { toastR("公网演示模式为只读，请在本地部署中修改研究目标", "error"); return; }
    const input = $("#research-goal-input");
    const goal = (input && input.value || "").trim();
    if (!goal) { toastR("研究目标不能为空", "error"); return; }
    const button = $("#research-goal-save");
    if (button) button.disabled = true;
    try {
      const config = await request("/api/research/config", {
        method: "PUT",
        body: JSON.stringify({ goal }),
      });
      rstate.config = config;
      renderConfig(config);
      setText("#overview-goal", goal);
      toastR("研究目标已保存，智能体分析与文献召回将围绕该目标进行");
    } catch (error) {
      toastR(error.message, "error");
    } finally {
      if (button) button.disabled = false;
    }
  }

  function fillExperimentSelect() {
    const select = $("#agent-experiment-select");
    if (!select) return;
    const current = select.value;
    const options = ['<option value="">自动选择最新实验档案</option>'];
    for (const item of rstate.experiments) {
      options.push(`<option value="${esc(item.id)}">${esc(item.name)} · ${esc(item.dataset || "--")} · seed ${esc(item.seed ?? "--")}</option>`);
    }
    select.innerHTML = options.join("");
    if ([...select.options].some(option => option.value === current)) select.value = current;
  }

  // ── AI 供应商配置 ────────────────────────────────────────

  async function saveProvider() {
    if (demoMode()) { toastR("公网演示模式为只读，请在本地部署中修改 AI 配置", "error"); return; }
    const payload = {
      ai: {
        provider: ($("#provider-name")?.value || "").trim(),
        model: ($("#provider-model")?.value || "").trim(),
        base_url: ($("#provider-base-url")?.value || "").trim(),
        api_key: ($("#provider-api-key")?.value || "").trim(),
        wire_api: $("#provider-wire-api")?.value || "chat",
      },
    };
    try {
      const config = await request("/api/research/config", {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      rstate.config = config;
      renderConfig(config);
      const keyInput = $("#provider-api-key");
      if (keyInput) keyInput.value = "";
      toastR("AI 供应商配置已保存");
    } catch (error) {
      toastR(error.message, "error");
    }
  }

  async function testProvider() {
    if (demoMode()) { toastR("公网演示模式为只读，请在本地部署中测试模型连通性", "error"); return; }
    const button = $("#provider-test");
    if (button) button.disabled = true;
    try {
      const result = await request("/api/research/ai/test", { method: "POST" });
      if (result.ok) toastR(`连通性正常 · ${result.latency_ms} ms`);
      else toastR(`未连接：${result.error || "未知错误"}`, "error");
    } catch (error) {
      toastR(error.message, "error");
    } finally {
      if (button) button.disabled = false;
    }
  }

  async function importCcSwitch() {
    if (demoMode()) { toastR("公网演示模式为只读，请在本地部署中导入供应商配置", "error"); return; }
    try {
      await request("/api/settings/cc-switch/import", { method: "POST" });
      await loadConfigOnly();
      toastR("已从 CC Switch 导入供应商配置");
    } catch (error) {
      toastR(error.message, "error");
    }
  }

  // ── 研究智能体 ────────────────────────────────────────────

  function appendMessage(role, html, extraClass = "") {
    const thread = $("#agent-messages");
    if (!thread) return null;
    const wrapper = document.createElement("div");
    wrapper.className = `agent-message ${role} ${extraClass}`.trim();
    wrapper.innerHTML = html;
    thread.appendChild(wrapper);
    requestAnimationFrame(() => {
      const top = wrapper.offsetTop - 12;
      if (Number.isFinite(top) && top > 0) thread.scrollTop = top;
      else thread.scrollTop = thread.scrollHeight;
    });
    return wrapper;
  }

  function stepsHtml(steps) {
    if (!steps || !steps.length) return "";
    return `<ol class="agent-steps">${steps.map(step => `
      <li class="agent-step ${esc(step.status || "done")}">
        <span class="step-dot"></span>
        <span class="step-label">${esc(step.label || "")}</span>
        <span class="step-detail">${esc(step.detail || "")}</span>
      </li>`).join("")}</ol>`;
  }

  function cleanQuote(text) {
    return String(text || "")
      .replace(/\r/g, " ")
      .replace(/[#>*`|]+/g, " ")
      .replace(/-{2,}/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 150);
  }

  function citationsHtml(citations) {
    if (!citations || !citations.length) return "";
    return `<div class="agent-citations">
      <div class="agent-citations-title">文献依据</div>
      <ul>${citations.map((item, index) => `
        <li><span class="cite-index">[${index + 1}]</span>
          <span class="cite-title">《${esc(item.title || "未知文献")}》</span>
          ${item.page ? `<span class="cite-page">p.${esc(item.page)}</span>` : ""}
          ${item.quote ? `<blockquote>${esc(cleanQuote(item.quote))}</blockquote>` : ""}
        </li>`).join("")}
      </ul>
    </div>`;
  }

  function planCardHtml(plan, review) {
    if (!plan) return "";
    const conditions = plan.conditions || [];
    const rows = conditions.map(condition => {
      const params = condition.params || {};
      return `<tr>
        <td>${esc(condition.name || condition.id)}</td>
        <td>${esc(params.planes)} × ${esc(params.sats_per_plane)}</td>
        <td>${esc(params.gs)}</td>
        <td>${esc(params.rounds)}</td>
        <td>${esc(params.seed)}</td>
      </tr>`;
    }).join("");
    const score = review && review.metrics ? review.metrics.score : null;
    return `<div class="plan-card" data-plan-id="${esc(plan.id)}">
      <div class="plan-card-head">
        <div>
          <div class="plan-kicker">EXPERIMENT PLAN · ${plan.mode === "llm" ? "LLM" : "OFFLINE"}</div>
          <h3>${esc(plan.title)}</h3>
        </div>
        <div class="plan-card-badges">
          ${score != null ? badge(`评审 ${score}/100`, score >= 70 ? "green" : "amber") : ""}
          ${badge(`预计 ${Math.ceil((plan.expected_duration_sec || 0) / 60)} 分钟`)}
        </div>
      </div>
      <p class="plan-hypothesis">${esc(plan.hypothesis || "")}</p>
      <div class="table-wrap">
        <table class="plan-table">
          <thead><tr><th>条件</th><th>轨面 × 每轨星</th><th>地面站</th><th>轮次</th><th>seed</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div class="plan-card-actions">
        <button class="button primary" data-plan-action="run" ${demoMode() ? 'disabled title="公网演示模式已禁用训练任务"' : ""}>运行计划</button>
        <button class="button secondary" data-plan-action="review">重新评审</button>
        <button class="button secondary" data-plan-action="json">查看 JSON</button>
      </div>
      <pre class="plan-json hidden">${esc(JSON.stringify(plan, null, 2))}</pre>
    </div>`;
  }

  function analysisCardHtml(analysis) {
    if (!analysis || !analysis.content) return "";
    return `<div class="analysis-card">
      <div class="analysis-head">
        ${badge(analysis.mode === "llm" ? `大模型 · ${analysis.provider || ""}` : "离线规则引擎", analysis.mode === "llm" ? "green" : "amber")}
        <span class="field-note">${esc(when(analysis.generated_at))}</span>
      </div>
      <article class="markdown-body">${md(analysis.content)}</article>
      ${citationsHtml(analysis.citations)}
    </div>`;
  }

  function jobCardHtml(job) {
    const conditions = job.conditions || [];
    const rows = conditions.map(item => `<tr>
      <td>${esc(item.name || item.id)}</td>
      <td>${badge(item.status || "PENDING", item.status === "DONE" ? "green" : item.status === "FAILED" ? "red" : "amber")}</td>
      <td>${item.final_accuracy != null ? pct(item.final_accuracy) : "--"}</td>
      <td>${item.accuracy_delta != null ? pct(item.accuracy_delta) : "--"}</td>
      <td>${item.elapsed_sec != null ? num(item.elapsed_sec) + " s" : "--"}</td>
    </tr>`).join("");
    return `<div class="job-card" data-job-id="${esc(job.id)}">
      <div class="job-head">
        <div>
          <div class="plan-kicker">BATCH JOB · ${esc(job.id)}</div>
          <h3>${esc(job.plan_title || "批量实验")}</h3>
        </div>
        ${badge(job.status || "PENDING", job.status === "DONE" ? "green" : job.status === "FAILED" ? "red" : "amber")}
      </div>
      <div class="job-progress"><div class="job-progress-bar" style="width:${Math.round((job.progress || 0) * 100)}%"></div></div>
      <p class="job-message">${esc(job.message || "")}</p>
      ${rows ? `<div class="table-wrap"><table class="job-table"><thead><tr><th>条件</th><th>状态</th><th>最终准确率</th><th>准确率差值</th><th>耗时</th></tr></thead><tbody>${rows}</tbody></table></div>` : ""}
      <div class="job-actions">
        <button class="button secondary" data-job-action="analyze" ${job.status === "DONE" ? "" : "disabled"}>解读这批结果</button>
        <button class="button secondary" data-job-action="refresh">刷新进度</button>
        <button class="button danger" data-job-action="cancel" ${["DONE", "FAILED", "CANCELLED"].includes(job.status) ? "disabled" : ""}>终止任务</button>
      </div>
    </div>`;
  }

  function bindPlanActions(root, plan, review) {
    $$r("[data-plan-action]", root).forEach(button => {
      button.addEventListener("click", async () => {
        const action = button.dataset.planAction;
        if (action === "json") {
          const pre = $r(".plan-json", root);
          if (pre) pre.classList.toggle("hidden");
          return;
        }
        if (action === "review") {
          button.disabled = true;
          try {
            const result = await request("/api/research/agent/review", {
              method: "POST", body: JSON.stringify({ plan }),
            });
            appendMessage("assistant", analysisCardHtml(result));
          } catch (error) { toastR(error.message, "error"); }
          finally { button.disabled = false; }
          return;
        }
        if (action === "run") {
          if (demoMode()) {
            toastR("公网演示模式已禁用后台训练任务；可在本地部署中运行计划", "error");
            return;
          }
          button.disabled = true;
          button.textContent = "提交中…";
          try {
            const job = await request("/api/research/agent/run", {
              method: "POST", body: JSON.stringify({ plan }),
            });
            appendJobCard(job);
            toastR("批量实验已提交后台执行");
          } catch (error) {
            toastR(error.message, "error");
            button.disabled = false;
            button.textContent = "运行计划";
          }
        }
      });
    });
  }

  function appendJobCard(job) {
    const wrapper = appendMessage("assistant", jobCardHtml(job), "job-message-wrap");
    if (!wrapper) return;
    rstate.jobs[job.id] = job;
    bindJobActions(wrapper, job.id);
    pollJob(job.id);
  }

  function bindJobActions(root, jobId) {
    $$r("[data-job-action]", root).forEach(button => {
      if (button.dataset.bound === "1") return;
      button.dataset.bound = "1";
      button.addEventListener("click", async () => {
        const action = button.dataset.jobAction;
        if (action === "refresh") {
          await refreshJobCard(jobId);
        } else if (action === "cancel") {
          try {
            await request(`/api/research/agent/jobs/${jobId}/cancel`, { method: "POST" });
            await refreshJobCard(jobId);
            toastR("已请求终止任务");
          } catch (error) { toastR(error.message, "error"); }
        } else if (action === "analyze") {
          button.disabled = true;
          button.textContent = "解读中…";
          try {
            const analysis = await request("/api/research/agent/analyze", {
              method: "POST", body: JSON.stringify({ job_id: jobId, question: "" }),
            });
            appendAnalysisMessage(analysis);
          } catch (error) {
            toastR(error.message, "error");
          } finally {
            button.disabled = false;
            button.textContent = "解读这批结果";
          }
        }
      });
    });
  }

  async function refreshJobCard(jobId) {
    try {
      const job = await request(`/api/research/agent/jobs/${jobId}`);
      rstate.jobs[jobId] = job;
      const card = $r(`.job-card[data-job-id="${jobId}"]`);
      if (!card) return job;
      card.outerHTML = jobCardHtml(job);
      const fresh = $r(`.job-card[data-job-id="${jobId}"]`);
      bindJobActions(fresh, jobId);
      if (job.status === "DONE" || job.status === "FAILED" || job.status === "CANCELLED") {
        clearInterval(rstate.jobTimers[jobId]);
        delete rstate.jobTimers[jobId];
      }
      return job;
    } catch (error) {
      clearInterval(rstate.jobTimers[jobId]);
      delete rstate.jobTimers[jobId];
      return null;
    }
  }

  function pollJob(jobId) {
    if (rstate.jobTimers[jobId]) return;
    rstate.jobTimers[jobId] = setInterval(async () => {
      const job = await refreshJobCard(jobId);
      if (job && job.status === "DONE") {
        toastR("批量实验完成，可解读结果");
      }
    }, 2500);
  }

  function appendAnalysisMessage(analysis) {
    const wrapper = appendMessage("assistant", analysisCardHtml(analysis));
    if (wrapper) bindAnalysis(wrapper, analysis);
    return wrapper;
  }

  function bindAnalysis(root, analysis) {
    if (!root || !analysis || !analysis.metrics) return;
    const jobId = analysis.job_id;
    if (!jobId) return;
    const button = document.createElement("button");
    button.className = "button secondary";
    button.textContent = "查看逐条件明细";
    button.addEventListener("click", () => {
      const rows = (analysis.metrics.conditions || []).map(item => `<tr>
        <td>${esc(item.name)}</td>
        <td>${esc(item.status)}</td>
        <td>${item.final_accuracy != null ? pct(item.final_accuracy) : "--"}</td>
        <td>${item.accuracy_delta != null ? pct(item.accuracy_delta) : "--"}</td>
      </tr>`).join("");
      appendMessage("assistant", `<div class="analysis-card"><div class="table-wrap"><table><thead><tr><th>条件</th><th>状态</th><th>准确率</th><th>差值</th></tr></thead><tbody>${rows}</tbody></table></div></div>`);
    });
    root.appendChild(button);
  }

  async function sendAgentMessage() {
    const input = $("#agent-input");
    const text = (input && input.value || "").trim();
    if (!text) return;
    if (rstate.agentMode === "analyze") {
      input.value = "";
      appendMessage("user", `<p>${esc(text)}</p>`);
      const loading = appendMessage("assistant", `<p class="agent-loading">正在读取实验档案与文献并分析…</p>`);
      try {
        const analysis = await request("/api/research/agent/analyze", {
          method: "POST",
          body: JSON.stringify({ question: text, experiment_id: $("#agent-experiment-select")?.value || "" }),
        });
        loading.remove();
        appendAnalysisMessage(analysis);
      } catch (error) {
        loading.remove();
        appendMessage("assistant", `<p class="agent-error">分析失败：${esc(error.message)}</p>`);
      }
      return;
    }
    input.value = "";
    appendMessage("user", `<p>${esc(text)}</p>`);
    const loading = appendMessage("assistant", `<p class="agent-loading">研究智能体工作中：检索文献 → 规划 / 分析 …</p>`);
    try {
      const payload = { message: text, experiment_id: $("#agent-experiment-select")?.value || "" };
      const result = await request("/api/research/agent/chat", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      loading.remove();
      const hasAnalysis = Boolean(result.analysis && result.analysis.content);
      const html = [
        stepsHtml(result.steps),
        hasAnalysis ? "" : `<article class="markdown-body">${md(result.reply || "")}</article>`,
        result.plan ? planCardHtml(result.plan, result.review) : "",
        result.job ? jobCardHtml(result.job) : "",
        hasAnalysis ? analysisCardHtml(result.analysis) : "",
        citationsHtml(result.citations),
      ].filter(Boolean).join("");
      const wrapper = appendMessage("assistant", html);
      if (wrapper) {
        if (result.plan) bindPlanActions(wrapper, result.plan, result.review);
        if (result.job) {
          rstate.jobs[result.job.id] = result.job;
          bindJobActions(wrapper, result.job.id);
          pollJob(result.job.id);
        }
        if (result.analysis) bindAnalysis(wrapper, result.analysis);
      }
    } catch (error) {
      loading.remove();
      appendMessage("assistant", `<p class="agent-error">请求失败：${esc(error.message)}</p>`);
    }
  }

  async function analyzeLatestArchive() {
    appendMessage("user", "<p>分析最新实验档案</p>");
    const loading = appendMessage("assistant", `<p class="agent-loading">正在读取最新实验档案…</p>`);
    try {
      const analysis = await request("/api/research/agent/analyze", {
        method: "POST", body: JSON.stringify({ question: "" }),
      });
      loading.remove();
      appendAnalysisMessage(analysis);
    } catch (error) {
      loading.remove();
      appendMessage("assistant", `<p class="agent-error">${esc(error.message)}</p>`);
    }
  }

  // ── 期刊文献库 ────────────────────────────────────────────

  async function loadLibrary(force = false) {
    if (rstate.viewsLoaded.library && !force) return;
    rstate.viewsLoaded.library = true;
    try {
      const data = await request("/api/research/library");
      rstate.library = data.items || [];
      renderLibraryStats(data.stats || {});
      renderLibraryList();
    } catch (error) {
      toastR(`文献索引读取失败：${error.message}`, "error");
    }
  }

  function renderLibraryStats(stats) {
    const container = $("#library-stats");
    if (!container) return;
    const warm = stats.warm_up || {};
    const warmText = warm.status === "running" ? " · 正在后台解析 PDF…" : "";
    container.innerHTML = `<div class="library-stat"><strong>${stats.total ?? 0}</strong><span>篇文献</span></div>
      <div class="library-stat"><strong>${stats.parsed ?? 0}</strong><span>全文已解析</span></div>
      <div class="library-stat"><strong>${Math.round((stats.chars || 0) / 1000)}</strong><span>千字索引</span></div>
      <p class="field-note">${esc(stats.roots ? stats.roots.map(root => root.label).join(" · ") : "")}${esc(warmText)}</p>`;
  }

  function libraryItemMatches(item, query) {
    if (rstate.libraryFilter === "pdf" && item.type !== "pdf") return false;
    if (rstate.libraryFilter === "md" && !["md", "markdown", "txt"].includes(item.type)) return false;
    if (rstate.libraryFilter === "parsed" && !item.extracted) return false;
    if (!query) return true;
    const haystack = `${item.title} ${item.abstract || ""} ${(item.keywords || []).join(" ")} ${(item.tags || []).join(" ")}`.toLowerCase();
    return haystack.includes(query.toLowerCase());
  }

  function renderLibraryList(items) {
    const container = $("#library-list");
    if (!container) return;
    const source = items || rstate.library.filter(item => libraryItemMatches(item, rstate.libraryQuery));
    if (!source.length) {
      container.innerHTML = '<div class="empty-state compact">没有匹配的文献。可导入 PDF 或调整检索词。</div>';
      return;
    }
    container.innerHTML = source.map(item => `
      <article class="library-item ${item.id === rstate.selectedDocId ? "active" : ""}" data-doc-id="${esc(item.id)}">
        <div class="library-item-head">
          <strong>${esc(item.title)}</strong>
          ${badge(String(item.type || "").toUpperCase(), item.extracted ? "green" : "amber")}
        </div>
        <div class="library-item-meta">${esc(item.venue || "未识别出处")}${item.year ? " · " + esc(item.year) : ""} · ${esc(item.pages || 0)} 页</div>
        <p class="library-item-abstract">${esc((item.abstract || "").slice(0, 150))}</p>
        <div class="library-item-tags">${(item.keywords || []).slice(0, 4).map(keyword => `<span>${esc(keyword)}</span>`).join("")}</div>
      </article>`).join("");
    $$r(".library-item", container).forEach(element => {
      element.addEventListener("click", () => openDocument(element.dataset.docId));
    });
  }

  async function openDocument(docId) {
    rstate.selectedDocId = docId;
    $$r(".library-item").forEach(item => item.classList.toggle("active", item.dataset.docId === docId));
    const panel = $("#reader-panel");
    panel.innerHTML = '<div class="empty-state compact">正在读取文献…</div>';
    try {
      const detail = await request(`/api/research/library/${encodeURIComponent(docId)}`);
      rstate.selectedDoc = detail;
      renderReader(detail);
    } catch (error) {
      panel.innerHTML = `<div class="empty-state compact">读取失败：${esc(error.message)}</div>`;
    }
  }

  function renderReader(detail) {
    const panel = $("#reader-panel");
    const item = detail.item || {};
    const pdfLink = item.type === "pdf"
      ? `<a class="button secondary" href="/api/library/file?path=${encodeURIComponent(item.path)}" target="_blank" rel="noopener">打开原文 PDF</a>`
      : "";
    panel.innerHTML = `<div class="reader-head">
        <div class="reader-title">
          <p class="panel-kicker">${esc(item.source || "文献")} · ${esc(String(item.type || "").toUpperCase())}</p>
          <h2>${esc(item.title)}</h2>
          <div class="reader-meta">${esc((item.authors || []).join("、") || "作者未识别")} · ${esc(item.venue || "出处未识别")}${item.year ? " · " + esc(item.year) : ""} · ${esc(item.pages || 0)} 页 · ${esc(item.chunks || 0)} 索引片段</div>
        </div>
        <div class="reader-actions">
          <button class="button primary" id="reader-digest">AI 精读</button>
          <button class="button secondary" id="reader-fulltext">全文</button>
          ${pdfLink}
        </div>
      </div>
      <div class="reader-tabs" id="reader-tabs">
        <button class="active" data-reader-tab="digest">结构化精读</button>
        <button data-reader-tab="pages">抽取全文</button>
        <button data-reader-tab="meta">元数据</button>
      </div>
      <div class="reader-body" id="reader-body"><div class="empty-state compact">点击「AI 精读」生成结构化摘要，或切换到「抽取全文」直接阅读。</div></div>`;
    $$r("[data-reader-tab]").forEach(button => {
      button.addEventListener("click", () => {
        $$r("[data-reader-tab]").forEach(item2 => item2.classList.toggle("active", item2 === button));
        rstate.readerTab = button.dataset.readerTab;
        renderReaderTab(detail);
      });
    });
    const digestButton = $("#reader-digest");
    if (digestButton) digestButton.addEventListener("click", () => digestDocument(item.id, true));
    const fulltextButton = $("#reader-fulltext");
    if (fulltextButton) fulltextButton.addEventListener("click", () => {
      $$r("[data-reader-tab]").forEach(btn => btn.classList.toggle("active", btn.dataset.readerTab === "pages"));
      rstate.readerTab = "pages";
      renderReaderTab(detail);
    });
    renderReaderTab(detail);
  }

  async function renderReaderTab(detail) {
    const body = $("#reader-body");
    if (!body) return;
    const item = detail.item || {};
    const tab = rstate.readerTab;
    if (tab === "pages") {
      const pages = detail.pages || [];
      body.innerHTML = pages.length
        ? pages.map(page => `<section class="reader-page"><div class="reader-page-no">PAGE ${esc(page.page)}</div><pre>${esc((page.text || "").slice(0, 4000))}</pre></section>`).join("")
        : '<div class="empty-state compact">该文件尚未抽取到正文，可尝试重建索引或查看原文 PDF。</div>';
      return;
    }
    if (tab === "meta") {
      const meta = {
        id: item.id, type: item.type, source: item.source, path: item.path,
        size: item.size, pages: item.pages, chunks: item.chunks,
        keywords: item.keywords, extract_note: item.extract_note,
        modified_at: item.modified_at,
      };
      body.innerHTML = `<pre class="reader-meta">${esc(JSON.stringify(meta, null, 2))}</pre>`;
      return;
    }
    if (!rstate.digestCache[item.id]) {
      body.innerHTML = '<div class="empty-state compact">正在生成结构化精读…</div>';
      await digestDocument(item.id, false);
      return;
    }
    body.innerHTML = digestHtml(rstate.digestCache[item.id]);
  }

  function digestHtml(digest) {
    const relevance = digest.relevance || {};
    const list = (items) => (items && items.length ? `<ul>${items.map(item => `<li>${esc(item)}</li>`).join("")}</ul>` : "<p>—</p>");
    return `<div class="digest">
      <div class="digest-head">
        ${badge(digest.mode === "llm" ? `大模型 · ${digest.provider || ""}` : "离线规则精读", digest.mode === "llm" ? "green" : "amber")}
        ${digest.cached ? badge("缓存") : ""}
        <div class="relevance-score"><span>与研究目标相关度</span><strong>${esc(relevance.score ?? "--")}</strong><small>/100</small></div>
      </div>
      <p class="digest-relevance">${esc(relevance.reason || "")}</p>
      <section><h3>摘要</h3><p>${esc(digest.summary || "—")}</p></section>
      <section><h3>主要贡献</h3>${list(digest.contributions)}</section>
      <section><h3>方法要点</h3><p>${esc(digest.method || "—")}</p></section>
      <section><h3>关键发现</h3>${list(digest.findings)}</section>
      <section><h3>局限与未来工作</h3>${list(digest.limitations)}</section>
      <section><h3>可引用原文</h3>${list((digest.quotes || []).map(quote => `p.${quote.page}：${cleanQuote(quote.text)}`))}</section>
    </div>`;
  }

  async function digestDocument(docId, showToast) {
    try {
      const digest = await request(`/api/research/library/${encodeURIComponent(docId)}/digest`, { method: "POST" });
      rstate.digestCache[docId] = digest;
      if (rstate.selectedDoc && rstate.selectedDoc.item && rstate.selectedDoc.item.id === docId) {
        rstate.readerTab = "digest";
        $$r("[data-reader-tab]").forEach(btn => btn.classList.toggle("active", btn.dataset.readerTab === "digest"));
        const body = $("#reader-body");
        if (body) body.innerHTML = digestHtml(digest);
      }
      if (showToast) toastR("精读完成");
      return digest;
    } catch (error) {
      toastR(`精读失败：${error.message}`, "error");
      return null;
    }
  }

  async function searchLibrary(query) {
    rstate.libraryQuery = query;
    if (!query) { renderLibraryList(); return; }
    try {
      const data = await request(`/api/research/library/search?q=${encodeURIComponent(query)}&limit=10`);
      renderLibraryList(data.items || []);
    } catch (error) {
      toastR(error.message, "error");
    }
  }

  async function importLibraryFiles(files) {
    if (!files || !files.length) return;
    let imported = 0;
    for (const file of files) {
      try {
        await request(`/api/library/import?filename=${encodeURIComponent(file.name)}`, {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: file,
        });
        imported += 1;
      } catch (error) {
        toastR(`${file.name}：${error.message}`, "error");
      }
    }
    const input = $("#library-import-input");
    if (input) input.value = "";
    if (imported) {
      toastR(`已导入 ${imported} 份文献，正在重建索引…`);
      await scanLibrary(true);
    }
  }

  async function scanLibrary(showToast = false) {
    try {
      const result = await request("/api/research/library/scan", { method: "POST" });
      toastR(`索引完成：${result.indexed ?? 0} 篇，解析 ${result.parsed ?? 0} 篇`);
      await loadLibrary(true);
    } catch (error) {
      toastR(`索引失败：${error.message}`, "error");
    }
  }

  async function askLibrary() {
    const input = $("#library-ask-input");
    const question = (input && input.value || "").trim();
    if (!question) return;
    input.value = "";
    const thread = $("#library-ask-messages");
    const userMsg = document.createElement("div");
    userMsg.className = "ask-message user";
    userMsg.innerHTML = `<p>${esc(question)}</p>`;
    thread.appendChild(userMsg);
    const loading = document.createElement("div");
    loading.className = "ask-message assistant loading";
    loading.textContent = "正在检索文献并回答…";
    thread.appendChild(loading);
    thread.scrollTop = thread.scrollHeight;
    try {
      const result = await request("/api/research/library/ask", {
        method: "POST", body: JSON.stringify({ question }),
      });
      loading.remove();
      const answer = document.createElement("div");
      answer.className = "ask-message assistant";
      answer.innerHTML = `<article class="markdown-body">${md(result.answer || "")}</article>${citationsHtml(result.citations)}`;
      thread.appendChild(answer);
      thread.scrollTop = thread.scrollHeight;
    } catch (error) {
      loading.remove();
      const answer = document.createElement("div");
      answer.className = "ask-message assistant";
      answer.textContent = `回答失败：${error.message}`;
      thread.appendChild(answer);
    }
  }

  // ── 前沿情报 ──────────────────────────────────────────────

  async function loadIntel(force = false) {
    if (rstate.viewsLoaded.intel && !force) return;
    rstate.viewsLoaded.intel = true;
    try {
      const data = await request(`/api/research/intel?refresh=${force ? "true" : "false"}`);
      rstate.intel = data;
      renderIntel(data);
    } catch (error) {
      toastR(`情报加载失败：${error.message}`, "error");
    }
  }

  function renderIntel(feed) {
    const updated = $("#intel-updated");
    if (updated) updated.textContent = feed.updated_at ? `更新于 ${when(feed.updated_at)}` : "尚未抓取";
    const sources = $("#intel-sources");
    if (sources) {
      const list = feed.sources || [];
      sources.innerHTML = list.map(source => `
        <span class="intel-source ${source.status === "ok" ? "ok" : "error"}">
          <b>${esc(source.name)}</b>
          <span>${source.status === "ok" ? `${source.count} 条` : "抓取失败"}</span>
        </span>`).join("");
    }
    const container = $("#intel-list");
    if (!container) return;
    const items = feed.items || [];
    if (!items.length) {
      container.innerHTML = '<div class="empty-state compact">暂未获取到情报。点击「刷新情报」主动抓取，或检查网络代理设置。</div>';
      return;
    }
    container.innerHTML = items.map(item => `
      <article class="intel-item" data-intel-id="${esc(item.id)}">
        <div class="intel-item-head">
          <div class="intel-item-meta">
            <span class="intel-source-tag">${esc(item.source)}</span>
            <span>${esc(when(item.published))}</span>
            ${item.relevance ? `<span class="intel-relevance">相关度 ${esc(item.relevance)}</span>` : ""}
            ${(item.tags || []).slice(0, 3).map(tag => `<span class="intel-tag">${esc(tag)}</span>`).join("")}
          </div>
          ${badge(item.kind === "paper" ? "论文" : "资讯", item.kind === "paper" ? "green" : "")}
        </div>
        <h3>${esc(item.title)}</h3>
        <p>${esc((item.summary || "").slice(0, 260))}</p>
        ${(item.matched_keywords || []).length ? `<div class="intel-matched">${item.matched_keywords.map(word => `<span>${esc(word)}</span>`).join("")}</div>` : ""}
        <div class="intel-item-actions">
          <a class="button secondary" href="${esc(item.url)}" target="_blank" rel="noopener">查看原文</a>
          <button class="button secondary" data-ingest="${esc(item.id)}" ${item.ingested ? "disabled" : ""}>${item.ingested ? "已入库" : "加入文献库"}</button>
        </div>
      </article>`).join("");
    $$r("[data-ingest]", container).forEach(button => {
      button.addEventListener("click", () => ingestIntel(button.dataset.ingest, button));
    });
  }

  async function refreshIntel() {
    const button = $("#intel-refresh");
    if (button) { button.disabled = true; button.textContent = "抓取中…"; }
    try {
      const feed = await request("/api/research/intel?refresh=true");
      rstate.intel = feed;
      renderIntel(feed);
      toastR(`情报已更新：${(feed.items || []).length} 条`);
    } catch (error) {
      toastR(error.message, "error");
    } finally {
      if (button) { button.disabled = false; button.textContent = "刷新情报"; }
    }
  }

  async function ingestIntel(itemId, button) {
    if (!itemId) return;
    if (button) { button.disabled = true; button.textContent = "归档中…"; }
    try {
      const result = await request("/api/research/intel/ingest", {
        method: "POST", body: JSON.stringify({ item_id: itemId }),
      });
      toastR(`已归档《${result.name || result.title || "情报"}》并更新文献索引`);
      if (button) button.textContent = "已入库";
      await loadLibrary(true);
    } catch (error) {
      toastR(error.message, "error");
      if (button) { button.disabled = false; button.textContent = "加入文献库"; }
    }
  }

  async function saveIntelTopics() {
    const input = $("#intel-topic-input");
    const topics = (input && input.value || "").split(/[,，、]/).map(item => item.trim()).filter(Boolean);
    try {
      const config = await request("/api/research/config", {
        method: "PUT", body: JSON.stringify({ topics }),
      });
      rstate.config = config;
      renderConfig(config);
      toastR("订阅主题已保存");
    } catch (error) {
      toastR(error.message, "error");
    }
  }

  async function generateIntelBrief() {
    const button = $("#intel-brief-run");
    if (button) { button.disabled = true; button.textContent = "生成中…"; }
    try {
      const result = await request("/api/research/intel/brief", { method: "POST" });
      const container = $("#intel-brief");
      if (container) container.innerHTML = `<article class="markdown-body">${md(result.content || "")}</article>`;
    } catch (error) {
      toastR(error.message, "error");
    } finally {
      if (button) { button.disabled = false; button.textContent = "让研究助手生成简报"; }
    }
  }

  // ── Hero 星图动画 ─────────────────────────────────────────

  function startHeroCanvas() {
    const canvas = $("#hero-canvas");
    if (!canvas || rstate.heroFrame) return;
    rstate.heroCanvas = canvas;
    const ctx = canvas.getContext("2d");
    const stars = [];
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let width = 0;
    let height = 0;
    let start = performance.now();

    function resize() {
      const ratio = window.devicePixelRatio || 1;
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      stars.length = 0;
      const count = Math.min(140, Math.round((width * height) / 12000));
      for (let i = 0; i < count; i++) {
        stars.push({
          x: Math.random() * width,
          y: Math.random() * height,
          r: Math.random() * 1.2 + 0.2,
          speed: Math.random() * 0.02 + 0.004,
          phase: Math.random() * Math.PI * 2,
        });
      }
    }
    resize();
    window.addEventListener("resize", resize);

    function frame(now) {
      const elapsed = (now - start) / 1000;
      ctx.clearRect(0, 0, width, height);
      // 星点
      for (const star of stars) {
        const alpha = 0.25 + 0.55 * Math.abs(Math.sin(elapsed * star.speed + star.phase));
        ctx.beginPath();
        ctx.fillStyle = `rgba(255,255,255,${alpha.toFixed(3)})`;
        ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2);
        ctx.fill();
      }
      // 轨道弧线
      const cx = width * 0.78;
      const cy = height * 0.62;
      const tilt = 0.32;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-0.35);
      for (let ring = 0; ring < 3; ring++) {
        const radius = Math.min(width, height) * (0.28 + ring * 0.12);
        ctx.beginPath();
        ctx.strokeStyle = `rgba(255,255,255,${0.1 - ring * 0.02})`;
        ctx.lineWidth = 1;
        ctx.ellipse(0, 0, radius, radius * tilt, 0, 0, Math.PI * 2);
        ctx.stroke();
        const angle = elapsed * (0.18 + ring * 0.05) + ring * 1.7;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius * tilt;
        ctx.beginPath();
        ctx.fillStyle = "#ffffff";
        ctx.arc(x, y, 2.4 - ring * 0.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.strokeStyle = "rgba(255,255,255,.35)";
        ctx.moveTo(x, y);
        ctx.lineTo(x - Math.cos(angle) * 26, y - Math.sin(angle) * tilt * 26);
        ctx.stroke();
      }
      ctx.restore();
      if (!reduced) rstate.heroFrame = requestAnimationFrame(frame);
    }
    rstate.heroFrame = requestAnimationFrame(frame);
  }

  // ── 视图路由与事件绑定 ────────────────────────────────────

  function bindResearchEvents() {
    const on = (selector, event, handler) => {
      const element = $r(selector);
      if (element) element.addEventListener(event, handler);
    };
    on("#research-goal-save", "click", saveResearchGoal);
    $$r("[data-goal]").forEach(button => {
      button.addEventListener("click", () => {
        const input = $("#research-goal-input");
        if (input) input.value = button.dataset.goal || "";
      });
    });
    on("#agent-context-refresh", "click", () => { loadOverview(); loadConfigOnly(); });
    on("#agent-form", "submit", event => { event.preventDefault(); sendAgentMessage(); });
    on("#agent-input", "keydown", event => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        sendAgentMessage();
      }
    });
    $$r("[data-agent-mode]").forEach(button => {
      button.addEventListener("click", () => {
        rstate.agentMode = button.dataset.agentMode || "auto";
        $$r("[data-agent-mode]").forEach(item => item.classList.toggle("active", item === button));
      });
    });
    $$r("[data-agent-quick]").forEach(button => {
      button.addEventListener("click", () => {
        const input = $("#agent-input");
        if (input) input.value = button.dataset.agentQuick || "";
        sendAgentMessage();
      });
    });
    on("#provider-save", "click", saveProvider);
    on("#provider-test", "click", testProvider);
    on("#cc-switch-import", "click", importCcSwitch);

    on("#library-search", "input", event => {
      clearTimeout(window.__librarySearchTimer);
      window.__librarySearchTimer = setTimeout(() => searchLibrary(event.target.value.trim()), 320);
    });
    on("#library-import-button", "click", () => $("#library-import-input")?.click());
    on("#library-import-input", "change", event => importLibraryFiles([...event.target.files]));
    on("#library-scan-button", "click", () => scanLibrary(true));
    on("#library-settings-button", "click", async () => {
      const editor = $("#library-path-editor");
      if (!editor) return;
      editor.classList.toggle("hidden");
      if (editor.classList.contains("hidden")) return;
      try {
        const settings = await request("/api/library/settings");
        const input = $("#library-default-path");
        if (input) input.value = settings.default_path || "";
      } catch (error) { toastR(error.message, "error"); }
    });
    on("#library-save-path", "click", async () => {
      const value = ($("#library-default-path")?.value || "").trim();
      try {
        await request("/api/library/settings", {
          method: "PUT", body: JSON.stringify({ default_path: value }),
        });
        $("#library-path-editor")?.classList.add("hidden");
        toastR("默认文献目录已更新，正在重建索引…");
        await scanLibrary(true);
      } catch (error) { toastR(error.message, "error"); }
    });
    $$r("[data-lib-filter]").forEach(button => {
      button.addEventListener("click", () => {
        rstate.libraryFilter = button.dataset.libFilter || "all";
        $$r("[data-lib-filter]").forEach(item => item.classList.toggle("active", item === button));
        renderLibraryList();
      });
    });
    on("#library-ask-form", "submit", event => { event.preventDefault(); askLibrary(); });

    on("#intel-refresh", "click", refreshIntel);
    on("#intel-topic-save", "click", saveIntelTopics);
    on("#intel-brief-run", "click", generateIntelBrief);

    document.addEventListener("spacefl:view", event => {
      const name = event.detail?.name;
      if (name === "agent") { loadConfigOnly(); fillExperimentSelect(); }
      if (name === "library") loadLibrary();
      if (name === "intel") loadIntel();
      if (name === "overview") { startHeroCanvas(); loadOverview(); }
    });
  }

  async function boot() {
    if (rstate.booted) return;
    rstate.booted = true;
    bindResearchEvents();
    startHeroCanvas();
    await loadOverview();
    // 若打开时不在总览，按当前激活视图懒加载
    const active = $r(".view.active");
    if (active && active.id === "view-library") loadLibrary();
    if (active && active.id === "view-intel") loadIntel();
    // 文献预热状态轮询：后台解析 PDF 时刷新一次统计
    setInterval(async () => {
      if (!rstate.overview) return;
      const traffic = rstate.overview.literature || {};
      if ((traffic.warm_up || {}).status === "running") await loadOverview();
    }, 12000);
  }

  window.SpaceFLResearch = {
    state: rstate,
    loadOverview,
    loadLibrary,
    loadIntel,
    analyzeLatestArchive,
    refreshIntel,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
