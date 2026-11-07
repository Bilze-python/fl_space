"""科研智能平台编排层。

对 web 层只暴露一个 :class:`ResearchService`，把文献库、情报抓取、大模型供应商、
实验规划器、结果分析器和后台批量实验任务串成完整的研究闭环：

    研究目标 → 文献检索 → 计划生成 → 计划评审 → 后台执行 → 结果解读 → 引用溯源

设计要点：

- 所有能力都有离线兜底，``mode`` 字段如实标注 ``llm`` / ``offline``；
- 后台实验通过子进程运行 ``scripts/run_research_plan.py``，以 ``progress.json``
  为准同步进度，任务失败不会拖垮 web 服务；
- 文献索引与情报缓存落盘在项目根目录，重启后无需重新解析 PDF。
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from uuid import uuid4

from fl_space.research import analyzer, bilingual, planner
from fl_space.research.config import DEFAULT_GOAL, ResearchConfigStore
from fl_space.research.intel import IntelFeed
from fl_space.research.literature import LiteratureIndex

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LITERATURE_CONFIG_FILE = ".literature_config.json"
INDEX_CACHE_FILE = ".spacefl_literature_index.json"
DIGEST_CACHE_FILE = ".spacefl_digests.json"
INTEL_CACHE_FILE = ".spacefl_intel_cache.json"
INTEL_INGESTED_FILE = ".spacefl_intel_ingested.json"
DIGEST_CACHE_VERSION = "v3"
RESULTS_DIR = "research_results"
RUNNER_SCRIPT = Path("scripts") / "run_research_plan.py"

MAX_JOB_SECONDS = 3600
JOB_POLL_INTERVAL = 1.5
LOG_TAIL_CHARS = 6000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _step(label: str, detail: str, status: str = "done") -> dict[str, str]:
    return {"label": label, "detail": detail, "status": status}


def _tail(text: str, limit: int = LOG_TAIL_CHARS) -> str:
    text = str(text or "")
    return text[-limit:] if len(text) > limit else text


class ResearchService:
    """研究平台编排服务（供 FastAPI 调用）。"""

    def __init__(
        self,
        project_dir: Path | str = PROJECT_DIR,
        *,
        list_experiments: Callable[[], list[dict[str, Any]]] | None = None,
        get_experiment: Callable[[str], dict[str, Any]] | None = None,
        cc_switch_reader: Callable[[], dict[str, Any] | None] | None = None,
        public_demo: bool | None = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        if public_demo is None:
            public_demo = os.environ.get("SPACEFL_PUBLIC_DEMO", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.public_demo = bool(public_demo)
        self._last_intel_force = 0.0
        self.config = ResearchConfigStore(self.project_dir)
        self._list_experiments = list_experiments
        self._get_experiment = get_experiment
        self._cc_switch_reader = cc_switch_reader
        self._lock = threading.RLock()
        self._library_lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._procs: dict[str, subprocess.Popen[Any]] = {}
        self._warm_state = {"status": "idle", "started_at": None, "finished_at": None}
        self._roots_signature: tuple[tuple[str, str], ...] = ()
        self._digests = _read_json(self.project_dir / DIGEST_CACHE_FILE, {}) or {}
        self._intel_ingested = set(_read_json(self.project_dir / INTEL_INGESTED_FILE, []) or [])
        self._intel = IntelFeed(self.project_dir / INTEL_CACHE_FILE)
        self._index = self._build_index(self._library_roots())

    # ── 文献库 ────────────────────────────────────────────────

    def _library_roots(self) -> list[tuple[str, Path]]:
        if self.public_demo:
            return [("公开演示文献", (self.project_dir / "deploy" / "demo_data" / "library").resolve())]
        raw = _read_json(self.project_dir / LITERATURE_CONFIG_FILE, {}) or {}
        default_path = str(raw.get("default_path") or "").strip()
        configured = Path(default_path) if default_path else self.project_dir / "文献"
        if not configured.is_absolute():
            configured = self.project_dir / configured
        # 顺序必须与 web/server.py 的 _library_roots 保持一致：
        # 索引路径 "0:..."/"1:..." 会被旧版 /api/library/content 接口复用。
        candidates = [
            ("项目文档", (self.project_dir / "docs").resolve()),
            ("默认文献", configured.resolve()),
        ]
        roots: list[tuple[str, Path]] = []
        seen: set[Path] = set()
        for label, path in candidates:
            if path in seen or not path.exists() or not path.is_dir():
                continue
            seen.add(path)
            roots.append((label, path))
        if not roots:
            roots.append(("项目文档", (self.project_dir / "docs").resolve()))
        return roots

    def _roots_signature_of(self, roots: list[tuple[str, Path]]) -> tuple[tuple[str, str], ...]:
        return tuple((label, str(path)) for label, path in roots)

    def _build_index(self, roots: list[tuple[str, Path]]) -> LiteratureIndex:
        default_root = roots[0][1] if roots else self.project_dir / "文献"
        extract_dir = self.project_dir / "文献" / "extracted"
        if not extract_dir.exists():
            extract_dir = default_root / "extracted"
        index = LiteratureIndex(
            roots,
            cache_path=self.project_dir / INDEX_CACHE_FILE,
            extract_dir=extract_dir,
        )
        index.load()
        self._roots_signature = self._roots_signature_of(roots)
        self._extract_dir = extract_dir
        return index

    def _ensure_index(self) -> LiteratureIndex:
        roots = self._library_roots()
        signature = self._roots_signature_of(roots)
        if signature != self._roots_signature:
            with self._library_lock:
                self._index = self._build_index(roots)
        return self._index

    def warm_up(self) -> None:
        """预热：后台重建文献索引并拉取一次情报。"""
        if self._warm_state.get("status") == "running":
            return
        self._warm_state = {"status": "running", "started_at": _now_iso(), "finished_at": None}
        threading.Thread(target=self._warm_up_worker, name="research-warmup", daemon=True).start()

    def _warm_up_worker(self) -> None:
        error = ""
        try:
            index = self._ensure_index()
            with self._library_lock:
                index.rebuild()
            try:
                self._intel.fetch(topics=self.config.as_dict()["topics"])
            except Exception as exc:
                error = f"情报抓取失败：{exc}"
        except Exception as exc:
            error = f"文献索引失败：{exc}"
        self._warm_state = {
            "status": "error" if error else "done",
            "started_at": self._warm_state.get("started_at"),
            "finished_at": _now_iso(),
            "error": error,
        }

    def warm_state(self) -> dict[str, Any]:
        """返回预热状态。"""
        return dict(self._warm_state)

    def library(self) -> dict[str, Any]:
        """返回文献索引条目与统计。"""
        index = self._ensure_index()
        with self._library_lock:
            items = [doc.to_public() for doc in index.items()]
            stats = index.stats()
        stats["warm_up"] = self.warm_state()
        return {"items": items, "stats": stats}

    def rebuild_library(self, *, force: bool = False) -> dict[str, Any]:
        """重建（增量）文献索引。"""
        if self.public_demo:
            raise ValueError("公网演示模式已禁用索引重建")
        index = self._ensure_index()
        with self._library_lock:
            result = index.rebuild(force=force)
            index.save()
        return {"rebuilt": True, **result, "stats": index.stats()}

    def search_library(self, query: str, limit: int = 8) -> dict[str, Any]:
        """全文检索文献库。"""
        query = str(query or "").strip()
        if not query:
            return {"query": "", "items": []}
        index = self._ensure_index()
        expanded = bilingual.expand_query(query)
        with self._library_lock:
            items = index.search(expanded, limit=max(1, min(int(limit), 20)))
        return {"query": query, "items": items}

    def literature_detail(self, doc_id: str) -> dict[str, Any]:
        """单篇文献元数据与全文分页。"""
        index = self._ensure_index()
        with self._library_lock:
            doc = index.get(doc_id)
            if doc is None:
                raise KeyError("文献不存在")
            pages = index.pages(doc_id)
        return {"item": doc.to_public(), "pages": pages}

    def digest_literature(self, doc_id: str) -> dict[str, Any]:
        """AI 精读（带缓存）。"""
        index = self._ensure_index()
        with self._library_lock:
            doc = index.get(doc_id)
            if doc is None:
                raise KeyError("文献不存在")
            pages = index.pages(doc_id)
            public = doc.to_public()
        goal = self.config.as_dict()["goal"]
        goal_hash = hashlib.sha1(goal.encode("utf-8")).hexdigest()[:8]
        cache_key = f"{doc_id}@{public.get('modified_at')}@{goal_hash}@{DIGEST_CACHE_VERSION}"
        cached = self._digests.get(cache_key)
        if isinstance(cached, dict):
            return {**cached, "cached": True}
        provider = self._provider()
        result = analyzer.digest_document(
            public,
            pages,
            goal=goal,
            provider=provider,
        )
        self._digests[cache_key] = result
        _write_json(self.project_dir / DIGEST_CACHE_FILE, self._digests)
        return {**result, "cached": False}

    def library_ask(self, question: str) -> dict[str, Any]:
        """基于文献库的 RAG 问答。"""
        question = str(question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        index = self._ensure_index()
        goal = self.config.as_dict()["goal"]
        query = bilingual.expand_query(f"{question} {goal}")
        with self._library_lock:
            context_text, citations = index.context(query, max_chars=6000, limit=5)
        result = analyzer.answer_literature_question(
            question,
            context_text=context_text,
            citations=citations,
            goal=goal,
            provider=self._provider(),
        )
        return {
            **result,
            "question": question,
            "steps": [
                _step("检索文献库", f"命中 {len(citations)} 条片段"),
                _step("生成回答", "已附引用" if citations else "未命中文献，给出检索建议"),
            ],
        }

    # ── 前沿情报 ──────────────────────────────────────────────

    def intel(self, *, refresh: bool = False) -> dict[str, Any]:
        """返回情报流（缓存优先），并按研究目标相关度重排。"""
        topics = self.config.as_dict()["topics"]
        if self.public_demo and refresh:
            now = time.time()
            if now - self._last_intel_force < 120:
                refresh = False
            else:
                self._last_intel_force = now
        feed = self._intel.fetch(topics=topics, force=refresh)
        return self._rank_intel(feed)

    def _goal_keywords(self) -> list[str]:
        """把研究目标 + 订阅主题扩展为中英关键词，用于资讯相关度打分。"""
        try:
            from fl_space.research import bilingual

            payload = self.config.as_dict()
            text = " ".join([payload["goal"], *payload["topics"]])
            expanded = bilingual.expand_text(text).lower()
        except Exception:
            expanded = self.config.as_dict()["goal"].lower()
        words = re.findall(r"[a-z]{3,}", expanded)
        seen: list[str] = []
        for word in words:
            if word not in seen:
                seen.append(word)
        return seen

    def _rank_intel(self, feed: dict[str, Any]) -> dict[str, Any]:
        """按「与研究目标的相关度」重排情报，避免通用航天新闻淹没领域论文。"""
        if not isinstance(feed, dict):
            return feed
        keywords = self._goal_keywords()
        items = [item for item in feed.get("items") or [] if isinstance(item, dict)]
        for item in items:
            haystack = " ".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("summary") or ""),
                    " ".join(str(tag) for tag in (item.get("tags") or [])),
                ]
            ).lower()
            matched = [word for word in keywords if word in haystack]
            item["matched_keywords"] = matched[:6]
            score = 0
            if keywords:
                score = min(100, round(100 * len(matched) / len(keywords) * 3.5))
            item["relevance"] = score
            item["ingested"] = bool(item.get("ingested")) or item.get("id") in self._intel_ingested
        items.sort(
            key=lambda item: (int(item.get("relevance") or 0), str(item.get("published") or "")),
            reverse=True,
        )
        feed["items"] = items
        return feed

    def ingest_intel(self, item_id: str) -> dict[str, Any]:
        """把一条情报归档进文献库并重建索引。"""
        roots = self._library_roots()
        target_dir = roots[0][1]
        item_id = str(item_id or "").strip()
        if self.public_demo:
            raise ValueError("公网演示模式已禁用情报入库")
        if not item_id:
            raise ValueError("缺少情报 id")
        result = self._intel.ingest(item_id, target_dir, self._extract_dir)
        self._intel_ingested.add(item_id)
        _write_json(self.project_dir / INTEL_INGESTED_FILE, sorted(self._intel_ingested))
        rebuild = self.rebuild_library()
        return {"ingested": True, **result, "library": rebuild.get("stats", {})}

    def intel_brief(self) -> dict[str, Any]:
        """生成情报简报。"""
        feed = self._intel.cached() or self._intel.fetch(
            topics=self.config.as_dict()["topics"], force=False
        )
        items = feed.get("items") if isinstance(feed, dict) else []
        result = analyzer.intel_brief(
            items or [],
            goal=self.config.as_dict()["goal"],
            topics=self.config.as_dict()["topics"],
            provider=self._provider(),
        )
        return {
            **result,
            "items": (items or [])[:8],
            "steps": [
                _step("读取前沿情报", f"共 {len(items or [])} 条"),
                _step("对齐研究目标", "已生成情报简报"),
            ],
        }

    # ── 供应商 ────────────────────────────────────────────────

    def _cc_provider(self) -> dict[str, Any] | None:
        if self._cc_switch_reader is None:
            return None
        try:
            return self._cc_switch_reader()
        except Exception:
            return None

    def _provider(self) -> Any:
        try:
            from fl_space.research import providers

            if self.public_demo and os.environ.get("SPACEFL_PUBLIC_LLM", "").strip() != "1":
                return providers.ProviderConfig()
            if os.environ.get("SPACEFL_OFFLINE_ONLY", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                return providers.ProviderConfig()
            return providers.load_provider(
                self.project_dir,
                cc_switch_provider=self._cc_provider(),
            )
        except Exception:
            from fl_space.research import providers

            return providers.ProviderConfig()

    def ai_status(self) -> dict[str, Any]:
        """返回脱敏后的 AI 供应商状态。"""
        status = self._provider().public()
        if self.public_demo:
            status.pop("masked_api_key", None)
        return status

    def ai_test(self) -> dict[str, Any]:
        """测试当前 AI 供应商连通性。"""
        provider = self._provider()
        if not getattr(provider, "configured", False):
            return {"ok": False, "latency_ms": 0, "error": "尚未配置大模型 API Key，当前使用离线规则引擎"}
        try:
            from fl_space.research import providers

            return providers.test_provider(provider)
        except Exception as exc:
            return {"ok": False, "latency_ms": 0, "error": str(exc)}

    # ── 配置 ──────────────────────────────────────────────────

    def config_payload(self) -> dict[str, Any]:
        """返回完整研究配置。"""
        payload = self.config.as_dict()
        index = self._ensure_index()
        with self._library_lock:
            stats = index.stats()
        payload["ai"] = self.ai_status()
        roots = self._library_roots()
        default_root = next(
            (path for label, path in roots if label == "默认文献"),
            roots[-1][1] if roots else self.project_dir / "文献",
        )
        payload["literature"] = {
            "root": str(default_root),
            "indexed": stats.get("total", 0),
            "parsed": stats.get("parsed", 0),
        }
        payload["warm_up"] = self.warm_state()
        payload["demo_mode"] = self.public_demo
        return payload

    def update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        """更新研究目标/订阅主题，可选更新 AI 供应商配置。"""
        if not isinstance(payload, dict):
            raise ValueError("配置必须是 JSON 对象")
        update = {
            key: payload[key]
            for key in ("goal", "topics")
            if key in payload and payload[key] is not None
        }
        if update:
            self.config.update(update)
        ai = payload.get("ai")
        if isinstance(ai, dict):
            try:
                from fl_space.research import providers

                providers.save_provider_config(self.project_dir / ".spacefl_ai_config.json", ai)
            except Exception as exc:
                raise ValueError(f"AI 配置保存失败：{exc}") from exc
        return self.config_payload()

    # ── 实验档案与计划 ────────────────────────────────────────

    def experiments(self) -> list[dict[str, Any]]:
        """返回可分析的实验档案列表。"""
        if self._list_experiments is None:
            return []
        try:
            items = self._list_experiments()
        except Exception:
            return []
        return [item for item in items if isinstance(item, dict)]

    def _experiment_detail(self, experiment_id: str) -> dict[str, Any] | None:
        if not experiment_id or self._get_experiment is None:
            return None
        try:
            detail = self._get_experiment(experiment_id)
        except Exception:
            return None
        return detail if isinstance(detail, dict) else None

    def _latest_experiment_id(self) -> str:
        experiments = self.experiments()
        for item in experiments:
            if item.get("kind") == "fedleo_validation":
                return str(item.get("id") or "")
        return str(experiments[0].get("id") or "") if experiments else ""

    def _literature_context(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        try:
            index = self._ensure_index()
            expanded = bilingual.expand_query(query)
            with self._library_lock:
                return index.context(expanded, max_chars=6000, limit=4)
        except Exception:
            return "", []

    def generate_plan(self, instruction: str) -> dict[str, Any]:
        """自然语言 → 实验计划（含文献检索与计划评审）。"""
        instruction = str(instruction or "").strip()
        if not instruction:
            raise ValueError("请输入实验指令")
        if len(instruction) > 2000:
            raise ValueError("实验指令请控制在 2000 字以内")
        goal = self.config.as_dict()["goal"]
        context_text, citations = self._literature_context(f"{goal} {instruction}")
        literature = [
            {
                "id": item.get("doc_id"),
                "title": item.get("title"),
                "page": item.get("page"),
                "why": "为计划中的变量与指标定义提供文献依据",
            }
            for item in citations
        ]
        provider = self._provider()
        plan = planner.generate_plan(
            instruction,
            goal=goal,
            provider=provider,
            literature=literature,
        )
        # 公网演示可只对"计划生成"调用大模型，评审走离线规则以控制响应时延
        # （Render 免费实例的 HTTP 请求有超时限制）。
        fast_public = os.environ.get("SPACEFL_PUBLIC_LLM_FAST", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        review = analyzer.review_plan(
            plan,
            goal=goal,
            context_text=context_text,
            citations=citations,
            provider=None if fast_public else provider,
        )
        steps = [
            _step("解析研究指令", f"识别到 {len(plan.get('conditions') or [])} 个实验条件"),
            _step("检索文献库", f"命中 {len(citations)} 篇相关文献"),
            _step("生成实验计划", f"计划模式：{'大模型' if plan.get('mode') == 'llm' else '离线规则'}"),
            _step("评审计划设计", f"结构化评分 {review['metrics'].get('score', '--')}/100"),
        ]
        reply = self._plan_reply(plan, review)
        return {
            "reply": reply,
            "plan": plan,
            "review": review,
            "citations": citations,
            "steps": steps,
            "mode": plan.get("mode", "offline"),
        }

    @staticmethod
    def _plan_reply(plan: dict[str, Any], review: dict[str, Any]) -> str:
        lines = [f"## 实验计划：{plan.get('title')}", ""]
        if plan.get("hypothesis"):
            lines.append(f"**假设**：{plan['hypothesis']}")
            lines.append("")
        lines.append("**实验条件**")
        for condition in plan.get("conditions") or []:
            params = condition.get("params") or {}
            lines.append(
                f"- {condition.get('name')}：{params.get('planes')} 轨面 × "
                f"{params.get('sats_per_plane')} 星，{params.get('gs')} 地面站，"
                f"{params.get('rounds')} 轮，seed {params.get('seed')}"
            )
        lines.append("")
        lines.append(
            f"**预计耗时**：{planner.plan_runtime_hint(plan)}；"
            f"**计划评审**：{review['metrics'].get('score', '--')}/100。"
        )
        notes = plan.get("notes") or []
        if notes:
            lines.append("")
            lines.append("**说明**")
            lines.extend(f"- {item}" for item in notes[:4])
        return "\n".join(lines)

    def review_plan(self, plan: Any) -> dict[str, Any]:
        """评审前端提交的计划。"""
        validated = planner.validate_plan(plan)
        goal = self.config.as_dict()["goal"]
        title = str(validated.get("title") or "")
        context_text, citations = self._literature_context(f"{goal} {title}")
        result = analyzer.review_plan(
            validated,
            goal=goal,
            context_text=context_text,
            citations=citations,
            provider=self._provider(),
        )
        return {
            **result,
            "plan": validated,
            "citations": citations,
            "steps": [
                _step("校验计划结构", f"{len(validated['conditions'])} 个条件"),
                _step("检索文献依据", f"命中 {len(citations)} 篇"),
                _step("生成评审意见", "已输出改进建议"),
            ],
        }

    # ── 批量实验任务 ──────────────────────────────────────────

    def start_plan_job(self, plan: Any) -> dict[str, Any]:
        """启动后台批量实验任务。"""
        if self.public_demo:
            raise ValueError("公网演示模式已禁用后台批量实验；完整功能请在本地部署体验")
        validated = planner.validate_plan(plan)
        job_id = f"rjob_{uuid4().hex[:10]}"
        out_dir = self.project_dir / RESULTS_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        plan_path = out_dir / "plan.json"
        _write_json(plan_path, validated)
        job: dict[str, Any] = {
            "id": job_id,
            "plan_id": validated.get("id"),
            "plan_title": validated.get("title"),
            "status": "PENDING",
            "progress": 0.0,
            "message": "任务已创建，等待执行器启动",
            "started_at": _now_iso(),
            "finished_at": None,
            "conditions": [],
            "result_path": f"{RESULTS_DIR}/{job_id}",
            "plan": validated,
            "log_tail": "",
        }
        with self._lock:
            self._jobs[job_id] = job
        threading.Thread(
            target=self._run_job,
            args=(job_id, plan_path, out_dir),
            name=f"research-job-{job_id}",
            daemon=True,
        ).start()
        return self.job(job_id) or job

    def _run_job(self, job_id: str, plan_path: Path, out_dir: Path) -> None:
        log_path = out_dir / "runner.log"
        started = time.time()
        try:
            with self._lock:
                self._jobs[job_id].update(
                    {"status": "RUNNING", "message": "实验执行器启动中"}
                )
            command = [
                sys.executable,
                str(self.project_dir / RUNNER_SCRIPT),
                "--plan",
                str(plan_path),
                "--output",
                str(out_dir),
            ]
            with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
                proc = subprocess.Popen(
                    command,
                    cwd=str(self.project_dir),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                with self._lock:
                    self._procs[job_id] = proc
                while proc.poll() is None:
                    time.sleep(JOB_POLL_INTERVAL)
                    self._sync_job_progress(job_id, out_dir)
                    if time.time() - started > MAX_JOB_SECONDS:
                        proc.terminate()
                        break
                returncode = proc.poll()
            self._sync_job_progress(job_id, out_dir)
            result = _read_json(out_dir / "research_result.json", None)
            log_tail = _tail(_read_text(log_path))
            if not log_tail:
                log_tail = _tail(_progress_logs(out_dir))
            with self._lock:
                job = self._jobs[job_id]
                job["finished_at"] = _now_iso()
                job["log_tail"] = log_tail
                if returncode == 0 and isinstance(result, dict):
                    job["status"] = "DONE"
                    job["progress"] = 1.0
                    job["message"] = "批量实验完成，可开始结果解读"
                    job["result"] = result
                elif returncode == 0:
                    job["status"] = "DONE"
                    job["message"] = "执行器已结束，但未找到结果文件"
                else:
                    job["status"] = "FAILED"
                    job["message"] = f"执行器退出码 {returncode}，请查看日志"
                self._procs.pop(job_id, None)
        except Exception as exc:
            with self._lock:
                self._jobs[job_id].update(
                    {
                        "status": "FAILED",
                        "message": f"任务执行异常：{exc}",
                        "finished_at": _now_iso(),
                        "log_tail": _tail(_read_text(log_path)),
                    }
                )
                self._procs.pop(job_id, None)

    def _sync_job_progress(self, job_id: str, out_dir: Path) -> None:
        progress = _read_json(out_dir / "progress.json", None)
        if not isinstance(progress, dict):
            return
        conditions = progress.get("conditions") if isinstance(progress.get("conditions"), list) else []
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["progress"] = float(progress.get("progress") or 0.0)
            job["conditions"] = conditions
            job["message"] = str(progress.get("message") or progress.get("current") or job["message"])
            logs = progress.get("logs")
            if isinstance(logs, list) and logs:
                job["log_tail"] = _tail("\n".join(str(item) for item in logs))

    def job(self, job_id: str) -> dict[str, Any] | None:
        """返回任务快照（含结果文件路径）。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return _public_job(job)

    def jobs(self) -> list[dict[str, Any]]:
        """返回任务列表（新→旧）。"""
        with self._lock:
            items = [_public_job(job) for job in self._jobs.values()]
        return sorted(items, key=lambda item: item.get("started_at") or "", reverse=True)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """请求终止正在运行的任务。"""
        with self._lock:
            proc = self._procs.get(job_id)
            job = self._jobs.get(job_id)
        if proc is None or job is None:
            raise KeyError("任务不存在或尚未运行")
        proc.terminate()
        with self._lock:
            job["status"] = "CANCELLED"
            job["message"] = "任务已被用户终止"
            job["finished_at"] = _now_iso()
        return _public_job(job)

    def _job_result(self, job_id: str) -> dict[str, Any]:
        job = self.job(job_id)
        if job is None:
            raise KeyError("任务不存在")
        if isinstance(job.get("result"), dict):
            return job["result"]
        path = self.project_dir / str(job.get("result_path") or "") / "research_result.json"
        result = _read_json(path, None)
        if not isinstance(result, dict):
            raise ValueError("该任务还没有生成结果文件，请等待执行完成")
        with self._lock:
            self._jobs[job_id]["result"] = result
        return result

    # ── 智能体对话 / 分析 ─────────────────────────────────────

    def analyze(
        self,
        *,
        job_id: str = "",
        experiment_id: str = "",
        question: str = "",
    ) -> dict[str, Any]:
        """分析实验档案或批量任务结果。"""
        goal = self.config.as_dict()["goal"]
        provider = self._provider()
        steps: list[dict[str, str]] = []
        if job_id:
            result = self._job_result(job_id)
            plan = result.get("plan") if isinstance(result.get("plan"), dict) else {}
            query = f"{goal} {plan.get('title') or ''} {question}".strip()
            context_text, citations = self._literature_context(query)
            steps.append(_step("读取批量结果", f"{len(result.get('conditions') or [])} 个条件"))
            steps.append(_step("检索文献依据", f"命中 {len(citations)} 篇"))
            analysis = analyzer.analyze_batch_result(
                result,
                question=question,
                goal=goal,
                context_text=context_text,
                citations=citations,
                provider=provider,
            )
            steps.append(_step("生成结果解读", f"模式：{analysis.get('mode')}"))
            return {**analysis, "job_id": job_id, "steps": steps}

        target_id = experiment_id or self._latest_experiment_id()
        detail = self._experiment_detail(target_id)
        if detail is None:
            raise ValueError("没有找到可分析的实验档案，请先运行或登记一个实验")
        query = f"{goal} {detail.get('name') or ''} {question}".strip()
        context_text, citations = self._literature_context(query)
        steps.append(_step("读取实验档案", f"{detail.get('name') or target_id}"))
        steps.append(_step("检索文献依据", f"命中 {len(citations)} 篇"))
        analysis = analyzer.analyze_experiment(
            detail,
            question=question,
            goal=goal,
            context_text=context_text,
            citations=citations,
            provider=provider,
        )
        steps.append(_step("生成结果解读", f"模式：{analysis.get('mode')}"))
        return {**analysis, "experiment_id": target_id, "steps": steps}

    def chat(self, message: str, *, experiment_id: str = "") -> dict[str, Any]:
        """研究智能体对话入口：自动识别规划 / 分析 / 情报 / 文献问答意图。"""
        message = str(message or "").strip()
        if not message:
            raise ValueError("请输入研究指令或问题")
        if len(message) > 4000:
            raise ValueError("消息请控制在 4000 字以内")
        text = message.lower()

        wants_intel = bool(re.search(r"资讯|新闻|前沿|情报|最新动态|arxiv", text))
        wants_analyze = bool(
            re.search(r"分析|解读|评审|证明|为什么|能否|结论|评估|档案", text)
        ) and not re.search(r"规划|计划|批量|运行|跑|执行", text)
        wants_plan = bool(
            re.search(r"规划|计划|批量|运行|跑|执行|对比|比较|实验设计|设计实验", text)
        )
        if wants_intel:
            return self._chat_intel()
        if wants_analyze:
            analysis = self.analyze(experiment_id=experiment_id, question=message)
            return {
                "reply": analysis.get("content") or "",
                "steps": analysis.get("steps") or [],
                "citations": analysis.get("citations") or [],
                "analysis": analysis,
                "mode": analysis.get("mode"),
                "plan": None,
                "job": None,
            }
        if wants_plan:
            plan_result = self.generate_plan(message)
            reply = plan_result["reply"]
            job: dict[str, Any] | None = None
            if re.search(r"运行|跑|执行|开始|启动", text):
                job = self.start_plan_job(plan_result["plan"])
                reply += (
                    f"\n\n已提交后台批量实验任务：`{job['id']}`，"
                    f"预计 {planner.plan_runtime_hint(plan_result['plan'])}。"
                    "任务完成后可在本页直接解读结果。"
                )
                plan_result["steps"].append(
                    _step("提交后台任务", f"{job['id']} · {job['status']}")
                )
            return {
                "reply": reply,
                "steps": plan_result["steps"],
                "citations": plan_result["citations"],
                "plan": plan_result["plan"],
                "review": plan_result["review"],
                "job": job,
                "analysis": None,
                "mode": plan_result["mode"],
            }
        # 其余情况按文献知识库问答处理
        answer = self.library_ask(message)
        steps = [
            _step("识别意图", "文献知识库问答"),
            *answer.get("steps", []),
        ]
        return {
            "reply": answer.get("answer") or "",
            "steps": steps,
            "citations": answer.get("citations") or [],
            "analysis": None,
            "plan": None,
            "job": None,
            "mode": answer.get("mode"),
        }

    def _chat_intel(self) -> dict[str, Any]:
        brief = self.intel_brief()
        return {
            "reply": brief.get("content") or "",
            "steps": brief.get("steps") or [],
            "citations": [],
            "analysis": None,
            "plan": None,
            "job": None,
            "mode": brief.get("mode"),
        }

    # ── 概览 ──────────────────────────────────────────────────

    def overview(self) -> dict[str, Any]:
        """研究平台首页所需的聚合数据。"""
        config = self.config.as_dict()
        index = self._ensure_index()
        with self._library_lock:
            stats = index.stats()
        intel_cached = self._intel.cached() or {}
        intel_items = intel_cached.get("items") if isinstance(intel_cached, dict) else []
        experiments = self.experiments()
        return {
            "demo_mode": self.public_demo,
            "goal": config["goal"],
            "topics": config["topics"],
            "ai": self.ai_status(),
            "literature": {
                "total": stats.get("total", 0),
                "parsed": stats.get("parsed", 0),
                "chars": stats.get("chars", 0),
                "root": str(self._library_roots()[0][1]),
                "warm_up": self.warm_state(),
            },
            "intel": {
                "count": len(intel_items or []),
                "updated_at": intel_cached.get("updated_at") if isinstance(intel_cached, dict) else None,
                "items": (intel_items or [])[:5],
            },
            "experiments": {
                "count": len(experiments),
                "latest": experiments[0] if experiments else None,
            },
            "jobs": self.jobs()[:3],
            "updated_at": _now_iso(),
        }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _progress_logs(out_dir: Path) -> str:
    """执行器的 stdout 被重定向捕获进 progress.json，日志读取做一次兜底。"""
    progress = _read_json(out_dir / "progress.json", None)
    if not isinstance(progress, dict):
        return ""
    logs = progress.get("logs")
    if not isinstance(logs, list):
        return ""
    return "\n".join(str(item) for item in logs)


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job)
    payload.pop("result", None)
    plan = payload.get("plan")
    if isinstance(plan, dict):
        payload["plan"] = plan
    return payload


__all__ = ["DEFAULT_GOAL", "ResearchService"]
