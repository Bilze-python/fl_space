"""科研平台：自然语言规划器与分析器离线测试。

所有测试不依赖网络与大模型 Key，覆盖比赛现场「离线兜底」路径。
"""

from __future__ import annotations

import json

import pytest

from fl_space.research import analyzer, planner

# ── 规划器 ────────────────────────────────────────────────────


def test_parse_single_experiment_scalars():
    plan = planner.build_default_plan("跑一个 5 颗卫星 3 个地面站 10 轮的实验")
    assert len(plan["conditions"]) == 1
    params = plan["conditions"][0]["params"]
    assert params["planes"] * params["sats_per_plane"] == 5
    assert params["gs"] == 3
    assert params["rounds"] == 10
    assert params["seed"] == 42
    assert plan["backend"] == "fedleo"


def test_parse_ground_station_grid():
    plan = planner.build_default_plan("比较地面站 2/3/5 下的准确率，3 轮，seed 42")
    assert len(plan["conditions"]) == 3
    assert [item["params"]["gs"] for item in plan["conditions"]] == [2, 3, 5]
    assert all(item["params"]["seed"] == 42 for item in plan["conditions"])
    assert all(item["params"]["rounds"] == 3 for item in plan["conditions"])


def test_parse_explicit_topology_pairs():
    plan = planner.build_default_plan(
        "对比 FedLEO 卸载开启/关闭在 3 轨面 2 星与 3 轨面 3 星下的准确率差异，3 轮、seed 42"
    )
    topologies = [
        (item["params"]["planes"], item["params"]["sats_per_plane"]) for item in plan["conditions"]
    ]
    assert topologies == [(3, 2), (3, 3)]
    assert [item["params"]["rounds"] for item in plan["conditions"]] == [3, 3]
    assert "3 轨面 × 2 星" in plan["title"]


def test_multi_seed_dimension():
    plan = planner.build_default_plan("多种子重复实验，3 轨面 2 星")
    assert len(plan["conditions"]) == 3
    seeds = [item["params"]["seed"] for item in plan["conditions"]]
    assert seeds == [42, 43, 44]


def test_grid_is_capped_and_notes_truncation():
    plan = planner.build_default_plan("地面站 2/3/5/7/9/11/13/15 全扫描")
    assert len(plan["conditions"]) == planner.MAX_CONDITIONS
    assert any("截断" in note for note in plan["notes"])


def test_quick_mode_compresses_budget():
    normal = planner.build_default_plan("3 轨面 2 星，8 轮")
    quick = planner.build_default_plan("快速演示 3 轨面 2 星，8 轮")
    assert quick["conditions"][0]["params"]["rounds"] <= normal["conditions"][0]["params"]["rounds"]
    assert any("快速" in note for note in quick["notes"])


def test_unsupported_algorithm_is_explained():
    plan = planner.build_default_plan("规划 FedProx 与 FedAvg 的对比实验")
    assert any("FedProx" in note for note in plan["notes"])


def test_validate_plan_clamps_ranges():
    raw = {
        "title": "越界计划",
        "conditions": [
            {
                "id": "c1",
                "name": "大配置",
                "params": {"planes": 9, "sats_per_plane": 9, "gs": 99, "rounds": 999, "lr": 5},
            }
        ],
    }
    plan = planner.validate_plan(raw)
    params = plan["conditions"][0]["params"]
    assert params["planes"] == 6
    assert params["planes"] * params["sats_per_plane"] <= 24
    assert params["gs"] == 20
    assert params["rounds"] == planner.MAX_ROUNDS_V1
    assert params["lr"] == 1.0
    assert plan["backend"] == "fedleo"


def test_validate_plan_rejects_empty_and_too_many():
    with pytest.raises(ValueError):
        planner.validate_plan({"conditions": []})
    with pytest.raises(ValueError):
        planner.validate_plan(
            {"conditions": [{"id": f"c{i}", "params": {}} for i in range(planner.MAX_CONDITIONS + 1)]}
        )


def test_validate_plan_clamps_oversized_constellation():
    plan = planner.validate_plan({"conditions": [{"params": {"planes": 6, "sats_per_plane": 6}}]})
    params = plan["conditions"][0]["params"]
    assert params["planes"] * params["sats_per_plane"] <= 24
    assert any("超过 24" in note for note in plan["notes"])


def test_plan_runtime_hint():
    plan = planner.build_default_plan("3 轨面 2 星")
    assert "秒" in planner.plan_runtime_hint(plan) or "分钟" in planner.plan_runtime_hint(plan)


def test_repair_fullwidth_and_chinese_numbers():
    plan = planner.build_default_plan("三个地面站，三轨道面，两颗星，３轮")
    params = plan["conditions"][0]["params"]
    assert params["gs"] == 3
    assert params["planes"] == 3
    assert params["sats_per_plane"] == 2
    assert params["rounds"] == 3


# ── 分析器 ────────────────────────────────────────────────────


def _history(start: float, end: float, rounds: int = 6) -> list[dict]:
    rows = []
    for index in range(rounds):
        progress = index / max(rounds - 1, 1)
        rows.append(
            {
                "round": index,
                "accuracy": start + (end - start) * progress,
                "train_loss": 2.4 - progress * 1.5,
                "total_delay": 2.0 * index,
                "total_offloaded_samples": 40 * index if index % 2 else 0,
                "data_balance_entropy": 0.86 + 0.02 * index,
                "data_sizes": [100, 120, 90],
            }
        )
    return rows


def _detail() -> dict:
    return {
        "id": "fedleo_test",
        "name": "FedLEO 卸载开关测试",
        "dataset": "mnist",
        "rounds": 6,
        "seed": 42,
        "note": "冒烟档案",
        "history_on": _history(0.1, 0.62),
        "history_off": _history(0.1, 0.55),
        "raw": {
            "config": {"num_rounds": 6, "dataset": "mnist", "seed": 42},
            "offload_on": {"final_accuracy": 0.62, "peak_accuracy": 0.62, "total_offloaded": 200},
            "offload_off": {"final_accuracy": 0.55, "peak_accuracy": 0.55},
            "deltas_on_minus_off": {"final_accuracy": 0.07},
            "gates": {"same_initial_model": True, "offload_samples_conserved": True},
            "passed": True,
        },
    }


def test_experiment_metrics_extracts_curves():
    metrics = analyzer.experiment_metrics(_detail())
    assert metrics["on"]["final_accuracy"] == pytest.approx(0.62)
    assert metrics["off"]["final_accuracy"] == pytest.approx(0.55)
    assert metrics["deltas"]["final_accuracy"] == pytest.approx(0.07)
    assert metrics["samples_conserved"] is True
    assert metrics["on"]["offload_rounds"] > 0
    assert metrics["gates"]


def test_offline_experiment_report_has_required_sections():
    result = analyzer.analyze_experiment(
        _detail(),
        question="这组实验是否真正触发了卸载？能证明什么？",
        goal="研究数据卸载对准确率的影响",
        citations=[
            {"index": 1, "doc_id": "lib_1", "title": "FedLEO", "page": 3, "quote": "卸载可降低时延"}
        ],
    )
    assert result["mode"] == "offline"
    content = result["content"]
    for section in ("结论摘要", "关键指标对照", "证据链", "尚不能支持的结论", "下一步建议"):
        assert section in content
    assert "FedLEO" in content
    assert result["citations"][0]["title"] == "FedLEO"


def test_offline_report_warns_small_rounds():
    detail = _detail()
    detail["history_on"] = _history(0.1, 0.3, rounds=2)
    detail["history_off"] = _history(0.1, 0.25, rounds=2)
    result = analyzer.analyze_experiment(detail)
    assert "轮" in result["content"]
    assert result["metrics"]["on"]["rounds"] == 2


def test_series_metrics_conserves_samples():
    metrics = analyzer.series_metrics(_history(0.1, 0.5))
    assert metrics["samples_conserved"] is True
    assert metrics["rounds"] == 6
    assert metrics["peak_accuracy"] >= metrics["final_accuracy"]


def test_review_plan_flags_weak_design():
    plan = planner.build_default_plan("3 轨面 2 星")
    review = analyzer.review_plan(plan, goal="测试目标")
    assert review["mode"] == "offline"
    assert 0 <= review["metrics"]["score"] <= 100
    content = review["content"]
    assert "实验计划评审" in content
    assert "随机种子" in content


def test_digest_document_extracts_sections():
    pages = [
        {
            "page": 1,
            "text": (
                "We propose a decentralized offloading framework for LEO satellites. "
                "Experiments show that the proposed method improves accuracy by 5%. "
                "However, the current work has limitations on dynamic channel models. "
                "Future work will extend the evaluation to multi-hop scenarios."
            ),
        }
    ]
    doc = {
        "id": "lib_test",
        "title": "Offloading in LEO satellite networks",
        "abstract": "We propose a decentralized offloading framework.",
        "authors": ["A"],
        "venue": "IEEE",
        "year": 2024,
        "keywords": ["offloading"],
    }
    digest = analyzer.digest_document(doc, pages, goal="研究卫星数据卸载")
    assert digest["mode"] == "offline"
    assert digest["summary"]
    assert digest["contributions"]
    assert digest["limitations"]
    assert digest["relevance"]["score"] > 0


def test_literature_answer_uses_citations():
    result = analyzer.answer_literature_question(
        "卸载能否提升准确率",
        context_text="[1] 《FedLEO》 p.3:\n卸载可以降低时延。",
        citations=[{"index": 1, "doc_id": "lib_1", "title": "FedLEO", "page": 3, "quote": "卸载可以降低时延"}],
    )
    assert result["mode"] == "offline"
    assert "FedLEO" in result["answer"]
    assert "文献库" in result["answer"]


def test_intel_brief_offline():
    result = analyzer.intel_brief(
        [{"title": "New LEO FL paper", "source": "arXiv", "published": "2026-11-01T00:00:00Z", "summary": "satellite federated learning"}],
        goal="研究太空联邦学习",
        topics=["太空联邦学习"],
    )
    assert result["mode"] == "offline"
    assert "New LEO FL paper" in result["content"]


def test_format_citations():
    text = analyzer.format_citations([{"index": 1, "title": "FedLEO", "page": 3}])
    assert text == "[1] 《FedLEO》 p.3"


def test_analyze_batch_result_offline(tmp_path):
    result = {
        "plan_id": "plan_1",
        "title": "批量对照",
        "plan": {"title": "批量对照", "hypothesis": "卸载提升准确率"},
        "generated_at": "2026-11-05T00:00:00Z",
        "conditions": [
            {
                "id": "c1",
                "name": "3 轨面 × 2 星",
                "status": "DONE",
                "final_accuracy": 0.62,
                "accuracy_delta": 0.04,
                "total_offloaded": 120,
                "total_delay_slots": 10,
                "elapsed_sec": 95,
            },
            {
                "id": "c2",
                "name": "地面站 3",
                "status": "DONE",
                "final_accuracy": 0.58,
                "accuracy_delta": -0.01,
                "total_offloaded": 80,
                "total_delay_slots": 8,
                "elapsed_sec": 90,
            },
        ],
        "summary": {"mean_final_accuracy": 0.6, "mean_delta_accuracy": 0.015},
    }
    analysis = analyzer.analyze_batch_result(result, question="批量结果如何？")
    assert analysis["mode"] == "offline"
    assert "批量实验解读" in analysis["content"]
    assert "逐条件结果" in analysis["content"]
    assert "尚不能支持的结论" in analysis["content"]
    assert analysis["metrics"]["conditions"][0]["name"] == "3 轨面 × 2 星"


def test_metrics_json_serialisable():
    metrics = analyzer.experiment_metrics(_detail())
    json.dumps(metrics)
