"""科研智能平台 API 路由。

由 ``web/server.py`` 的 :func:`create_app` 调用 :func:`register_research_routes`
注册。所有编排逻辑都在 :class:`fl_space.research.service.ResearchService` 中，
这里只负责 HTTP 语义（参数校验、错误码映射、异步线程调度）。
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from fastapi import Body, FastAPI, HTTPException, Query

from fl_space.research.service import ResearchService


async def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """把同步的编排调用放到线程池，并统一映射错误码。"""
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except KeyError as exc:
        detail = str(exc).strip("'\"") or "资源不存在"
        raise HTTPException(status_code=404, detail=detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def register_research_routes(app: FastAPI, service: ResearchService) -> None:
    """注册 ``/api/research/*`` 路由。"""

    @app.get("/api/research/overview")
    async def research_overview() -> dict[str, Any]:
        return await _call(service.overview)

    # ── 配置与供应商 ──────────────────────────────────────────

    @app.get("/api/research/config")
    async def research_config() -> dict[str, Any]:
        return await _call(service.config_payload)

    @app.put("/api/research/config")
    async def research_update_config(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return await _call(service.update_config, payload)

    @app.post("/api/research/ai/test")
    async def research_ai_test() -> dict[str, Any]:
        return await _call(service.ai_test)

    # ── 文献库 ────────────────────────────────────────────────

    @app.get("/api/research/library")
    async def research_library() -> dict[str, Any]:
        return await _call(service.library)

    @app.post("/api/research/library/scan")
    async def research_library_scan(
        force: bool = Query(default=False),
    ) -> dict[str, Any]:
        return await _call(service.rebuild_library, force=force)

    @app.get("/api/research/library/search")
    async def research_library_search(
        q: str = Query(default="", max_length=200),
        limit: int = Query(default=8, ge=1, le=20),
    ) -> dict[str, Any]:
        return await _call(service.search_library, q, limit)

    @app.get("/api/research/library/{doc_id}")
    async def research_library_detail(doc_id: str) -> dict[str, Any]:
        return await _call(service.literature_detail, doc_id)

    @app.post("/api/research/library/{doc_id}/digest")
    async def research_library_digest(doc_id: str) -> dict[str, Any]:
        return await _call(service.digest_literature, doc_id)

    @app.post("/api/research/library/ask")
    async def research_library_ask(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        question = str(payload.get("question") or "").strip()
        return await _call(service.library_ask, question)

    # ── 前沿情报 ──────────────────────────────────────────────

    @app.get("/api/research/intel")
    async def research_intel(
        refresh: bool = Query(default=False),
    ) -> dict[str, Any]:
        return await _call(service.intel, refresh=refresh)

    @app.post("/api/research/intel/refresh")
    async def research_intel_refresh() -> dict[str, Any]:
        return await _call(service.intel, refresh=True)

    @app.post("/api/research/intel/ingest")
    async def research_intel_ingest(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        item_id = str(payload.get("item_id") or "").strip()
        return await _call(service.ingest_intel, item_id)

    @app.post("/api/research/intel/brief")
    async def research_intel_brief() -> dict[str, Any]:
        return await _call(service.intel_brief)

    # ── 实验档案 / 智能体 ─────────────────────────────────────

    @app.get("/api/research/experiments")
    async def research_experiments() -> list[dict[str, Any]]:
        return await _call(service.experiments)

    @app.post("/api/research/agent/plan")
    async def research_agent_plan(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        instruction = str(payload.get("instruction") or "").strip()
        return await _call(service.generate_plan, instruction)

    @app.post("/api/research/agent/review")
    async def research_agent_review(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        plan = payload.get("plan")
        return await _call(service.review_plan, plan)

    @app.post("/api/research/agent/run")
    async def research_agent_run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        plan = payload.get("plan")
        return await _call(service.start_plan_job, plan)

    @app.get("/api/research/agent/jobs")
    async def research_agent_jobs() -> list[dict[str, Any]]:
        return await _call(service.jobs)

    @app.get("/api/research/agent/jobs/{job_id}")
    async def research_agent_job(job_id: str) -> dict[str, Any]:
        job = await _call(service.job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return job

    @app.post("/api/research/agent/jobs/{job_id}/cancel")
    async def research_agent_job_cancel(job_id: str) -> dict[str, Any]:
        return await _call(service.cancel_job, job_id)

    @app.post("/api/research/agent/analyze")
    async def research_agent_analyze(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return await _call(
            service.analyze,
            job_id=str(payload.get("job_id") or "").strip(),
            experiment_id=str(payload.get("experiment_id") or "").strip(),
            question=str(payload.get("question") or "").strip(),
        )

    @app.post("/api/research/agent/chat")
    async def research_agent_chat(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return await _call(
            service.chat,
            str(payload.get("message") or "").strip(),
            experiment_id=str(payload.get("experiment_id") or "").strip(),
        )


__all__ = ["register_research_routes"]
