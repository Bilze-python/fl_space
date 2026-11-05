"""科研平台编排层（ResearchService）离线测试。

覆盖：文献库索引/检索、配置持久化、离线规划与分析、情报简报、任务进度同步。
全部使用临时目录与假实验档案，不启动子进程、不访问网络。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fl_space.research import providers
from fl_space.research.service import ResearchService

DETAIL = {
    "id": "fedleo_fixture",
    "name": "FedLEO 卸载对照测试档案",
    "dataset": "mnist",
    "rounds": 6,
    "seed": 42,
    "note": "测试档案",
    "history_on": [
        {"round": i, "accuracy": 0.1 + 0.08 * i, "train_loss": 2.0 - 0.2 * i,
         "total_delay": i, "total_offloaded_samples": 30 * i,
         "data_balance_entropy": 0.86 + 0.01 * i, "data_sizes": [100, 100, 100]}
        for i in range(6)
    ],
    "history_off": [
        {"round": i, "accuracy": 0.1 + 0.05 * i, "train_loss": 2.0 - 0.15 * i,
         "total_delay": i, "total_offloaded_samples": 0,
         "data_balance_entropy": 0.86, "data_sizes": [100, 100, 100]}
        for i in range(6)
    ],
    "raw": {
        "config": {"num_rounds": 6, "dataset": "mnist", "seed": 42},
        "offload_on": {"final_accuracy": 0.5, "peak_accuracy": 0.5, "total_offloaded": 450},
        "offload_off": {"final_accuracy": 0.35, "peak_accuracy": 0.35},
        "deltas_on_minus_off": {"final_accuracy": 0.15},
        "gates": {"same_initial_model": True, "offload_samples_conserved": True},
        "passed": True,
    },
}

EXPERIMENTS = [
    {
        "id": "fedleo_fixture",
        "name": "FedLEO 卸载对照测试档案",
        "kind": "fedleo_validation",
        "dataset": "mnist",
        "rounds": 6,
        "seed": 42,
        "accuracy": 0.5,
        "accuracy_delta": 0.15,
        "status": "DONE",
        "created_at": "2026-11-01T00:00:00+00:00",
    }
]


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    literature = tmp_path / "文献"
    literature.mkdir()
    (literature / "fedleo_note.md").write_text(
        "# FedLEO 精读笔记\n\n"
        "FedLEO 通过数据卸载降低通信时延，并提升数据均衡度。\n\n"
        "实验表明卸载可以改善非 IID 数据下的收敛速度。\n\n"
        "## 局限\n\n未考虑动态链路竞争。\n",
        encoding="utf-8",
    )
    (literature / "satellite_fl.md").write_text(
        "# Satellite Federated Learning Survey\n\n"
        "This survey reviews federated learning in satellite constellations, "
        "covering offloading, scheduling and latency.\n",
        encoding="utf-8",
    )
    (tmp_path / ".literature_config.json").write_text(
        json.dumps({"default_path": str(literature)}, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture()
def service(project: Path) -> ResearchService:
    return ResearchService(
        project,
        list_experiments=lambda: list(EXPERIMENTS),
        get_experiment=lambda experiment_id: dict(DETAIL) if experiment_id == "fedleo_fixture" else {},
    )


# ── 文献库 ────────────────────────────────────────────────────


def test_rebuild_library_indexes_markdown(service: ResearchService):
    result = service.rebuild_library()
    assert result["indexed"] == 2
    assert result["stats"]["total"] == 2
    library = service.library()
    assert len(library["items"]) == 2
    assert all(item["extracted"] for item in library["items"])


def test_search_library_with_bilingual_expansion(service: ResearchService):
    service.rebuild_library()
    result = service.search_library("卸载 时延", limit=5)
    assert result["items"]
    assert any("FedLEO" in item["title"] for item in result["items"])


def test_literature_detail_and_digest(service: ResearchService):
    service.rebuild_library()
    doc = service.library()["items"][0]
    detail = service.literature_detail(doc["id"])
    assert detail["item"]["id"] == doc["id"]
    assert detail["pages"]
    digest = service.digest_literature(doc["id"])
    assert digest["mode"] == "offline"
    assert digest["summary"]
    assert "relevance" in digest
    cached = service.digest_literature(doc["id"])
    assert cached.get("cached") is True


def test_library_ask_offline(service: ResearchService):
    service.rebuild_library()
    result = service.library_ask("卸载能否提升数据均衡度？")
    assert result["mode"] == "offline"
    assert result["answer"]
    assert result["steps"]


def test_library_detail_missing(service: ResearchService):
    service.rebuild_library()
    with pytest.raises(KeyError):
        service.literature_detail("lib_not_exists")


# ── 配置 ──────────────────────────────────────────────────────


def test_update_goal_and_topics(project: Path, service: ResearchService):
    updated = service.update_config({"goal": "研究卫星调度", "topics": ["调度", "星座"]})
    assert updated["goal"] == "研究卫星调度"
    assert updated["topics"] == ["调度", "星座"]
    reloaded = ResearchService(project, list_experiments=lambda: [], get_experiment=lambda _: {})
    assert reloaded.config.as_dict()["goal"] == "研究卫星调度"


def test_update_config_rejects_empty_goal(service: ResearchService):
    with pytest.raises(ValueError):
        service.update_config({"goal": "  "})


def test_ai_provider_public_masks_key(project: Path, service: ResearchService):
    config = service.update_config(
        {
            "ai": {
                "provider": "DeepSeek",
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com",
                "api_key": "sk-test-12345678",
                "wire_api": "chat",
            }
        }
    )
    assert config["ai"]["configured"] is True
    assert config["ai"]["model"] == "deepseek-chat"
    assert "api_key" not in config["ai"]
    raw = json.loads((project / ".spacefl_ai_config.json").read_text(encoding="utf-8"))
    assert raw["source"] == "local"
    assert raw["api_key"] == "sk-test-12345678"


def test_ai_test_without_provider(service: ResearchService):
    result = service.ai_test()
    assert result["ok"] is False
    assert result["error"]


# ── 规划与分析 ────────────────────────────────────────────────


def test_generate_plan_offline(service: ResearchService):
    service.rebuild_library()
    result = service.generate_plan("对比地面站 2/3 下的准确率，3 轮，seed 42")
    assert result["plan"]["mode"] == "offline"
    assert len(result["plan"]["conditions"]) == 2
    assert result["review"]["metrics"]["score"] >= 0
    assert [step["label"] for step in result["steps"]][:2] == ["解析研究指令", "检索文献库"]
    assert result["reply"].startswith("## 实验计划")


def test_generate_plan_falls_back_when_llm_fails(project: Path, monkeypatch, service: ResearchService):
    service.update_config(
        {"ai": {"provider": "x", "model": "x", "base_url": "http://localhost", "api_key": "sk-x", "wire_api": "chat"}}
    )

    def boom(*args, **kwargs):
        raise providers.ProviderError("网络不可用")

    monkeypatch.setattr(providers, "chat_json", boom)
    result = service.generate_plan("3 轨面 2 星")
    assert result["plan"]["mode"] == "offline"
    assert any("离线" in note for note in result["plan"]["notes"])


def test_analyze_latest_experiment(service: ResearchService):
    service.rebuild_library()
    result = service.analyze(question="这组实验能证明什么？")
    assert result["experiment_id"] == "fedleo_fixture"
    assert result["mode"] == "offline"
    assert "尚不能支持的结论" in result["content"]
    assert result["steps"][0]["label"] == "读取实验档案"


def test_analyze_without_experiment(project: Path):
    empty = ResearchService(project, list_experiments=lambda: [], get_experiment=lambda _: {})
    with pytest.raises(ValueError):
        empty.analyze()


def test_chat_plan_intent(service: ResearchService):
    result = service.chat("规划一组地面站 2/3 的卸载对照实验，3 轮")
    assert result["plan"] is not None
    assert result["job"] is None
    assert result["mode"] == "offline"


def test_chat_analyze_intent(service: ResearchService):
    result = service.chat("分析最新实验档案能证明什么")
    assert result["analysis"] is not None
    assert result["reply"]
    assert result["mode"] == "offline"


def test_chat_intel_intent(service: ResearchService, monkeypatch):
    monkeypatch.setattr(
        service._intel,
        "cached",
        lambda: {"items": [{"title": "LEO FL 新论文", "source": "arXiv", "published": "2026-11-01"}], "updated_at": "2026-11-01"},
    )
    result = service.chat("最新前沿资讯有什么")
    assert result["analysis"] is None
    assert result["reply"]


def test_chat_literature_fallback(service: ResearchService):
    service.rebuild_library()
    result = service.chat("FedLEO 的卸载策略是什么")
    assert result["mode"] == "offline"
    assert "文献" in result["reply"]


def test_review_plan_endpoint(service: ResearchService):
    plan = {"title": "测试", "conditions": [{"id": "c1", "params": {"rounds": 3, "seed": 42}}]}
    result = service.review_plan(plan)
    assert result["plan"]["title"] == "测试"
    assert "评审" in result["content"]


# ── 情报 ──────────────────────────────────────────────────────


def test_intel_and_brief(service: ResearchService, monkeypatch):
    feed = {
        "updated_at": "2026-11-05T00:00:00+00:00",
        "sources": [{"name": "arXiv", "status": "ok", "count": 1}],
        "items": [
            {
                "id": "arxiv_1",
                "source": "arXiv",
                "kind": "paper",
                "title": "Federated Learning for LEO Constellations",
                "url": "https://arxiv.org/abs/9999.00001",
                "published": "2026-11-01T00:00:00+00:00",
                "summary": "We study offloading in LEO satellite networks.",
            }
        ],
    }
    monkeypatch.setattr(service._intel, "fetch", lambda **kwargs: feed)
    monkeypatch.setattr(service._intel, "cached", lambda: feed)
    assert service.intel()["items"][0]["title"].startswith("Federated")
    brief = service.intel_brief()
    assert brief["mode"] == "offline"
    assert "Federated Learning" in brief["content"]
    overview = service.overview()
    assert overview["intel"]["count"] == 1


# ── 后台任务（不真正训练） ───────────────────────────────────


def test_start_plan_job_rejects_invalid_plan(service: ResearchService):
    with pytest.raises(ValueError):
        service.start_plan_job({"conditions": []})


def test_job_progress_sync(project: Path, service: ResearchService):
    out_dir = Path(project) / "research_results" / "rjob_test"
    out_dir.mkdir(parents=True)
    (out_dir / "progress.json").write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "progress": 0.5,
                "current": "c1",
                "message": "正在运行条件 1/2",
                "conditions": [
                    {"id": "c1", "name": "3 轨面 × 2 星", "status": "DONE", "final_accuracy": 0.5}
                ],
                "logs": ["line-1", "line-2"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    service._jobs["rjob_test"] = {
        "id": "rjob_test",
        "status": "RUNNING",
        "progress": 0.0,
        "message": "",
        "conditions": [],
        "log_tail": "",
        "result_path": "research_results/rjob_test",
        "started_at": "2026-11-05T00:00:00+00:00",
    }
    service._sync_job_progress("rjob_test", out_dir)
    job = service.job("rjob_test")
    assert job["progress"] == 0.5
    assert job["conditions"][0]["final_accuracy"] == 0.5
    assert "line-2" in job["log_tail"]

    (out_dir / "research_result.json").write_text(
        json.dumps(
            {
                "plan_id": "plan_1",
                "title": "批量",
                "conditions": [],
                "summary": {"mean_final_accuracy": 0.5},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = service._job_result("rjob_test")
    assert result["title"] == "批量"


def test_jobs_listing_and_cancel_missing(service: ResearchService):
    assert service.jobs() == []
    with pytest.raises(KeyError):
        service.cancel_job("rjob_unknown")
