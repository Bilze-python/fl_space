#!/usr/bin/env python3
"""Generate selected charts from a completed SpaceFL result JSON/directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _history(data: object) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        value = data.get("history", data.get("records", []))
        if isinstance(value, dict):
            value = list(value.values())
        return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []
    return []


def main() -> int:
    p = argparse.ArgumentParser(description="从实验结果生成指定可视化图表")
    p.add_argument("result", help="结果 JSON 文件或包含结果文件的目录")
    p.add_argument("--output", "-o", default=None, help="图表输出目录，默认写回结果目录")
    p.add_argument("--plots", default="accuracy,time,summary", help="逗号分隔: accuracy,time,summary")
    args = p.parse_args()

    root = Path(args.result)
    if root.is_dir():
        candidates = [root / "experiment_results.json", root / "result.json", root / "history.json"]
        source = next((x for x in candidates if x.exists()), None)
        if source is None:
            jsons = sorted(root.glob("*.json"))
            source = jsons[0] if jsons else None
    else:
        source = root
    if source is None or not source.exists():
        print(f"[失败] 找不到结果 JSON: {root}")
        return 2

    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[失败] 读取结果失败: {exc}")
        return 2
    hist = _history(data)
    if not hist:
        print("[失败] 结果中没有可绘制的 history 记录")
        return 2

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.output) if args.output else source.parent
    out.mkdir(parents=True, exist_ok=True)
    plots = {x.strip().lower() for x in args.plots.split(",") if x.strip()}
    rounds = [x.get("round", i) for i, x in enumerate(hist)]
    accuracy = [x.get("accuracy") for x in hist]
    loss = [x.get("loss") for x in hist]
    made = []

    if "accuracy" in plots and any(x is not None for x in accuracy):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(rounds, accuracy, color="#1976d2", linewidth=2)
        ax.set(xlabel="Round", ylabel="Accuracy", title="Training Accuracy")
        ax.grid(alpha=0.25)
        path = out / "accuracy_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(path)

    if "time" in plots:
        keys = ("wait_distribution", "download", "train", "wait_return", "upload")
        totals = {k: sum(float((x.get("time_breakdown") or {}).get(k, 0) or 0) for x in hist) for k in keys}
        if any(totals.values()):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(list(totals), list(totals.values()), color="#00897b")
            ax.set(ylabel="Time", title="Time Breakdown")
            fig.savefig(out / "time_breakdown.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            made.append(out / "time_breakdown.png")

    if "summary" in plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        values = [x for x in accuracy if x is not None]
        if values:
            ax.bar(["final", "max"], [values[-1], max(values)], color=["#5e35b1", "#ef6c00"])
            ax.set_ylim(0, 1)
        ax.set_title("Experiment Summary")
        path = out / "summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(path)

    if not made:
        print(f"[失败] 未生成图表，请检查 --plots 或结果字段: {args.plots}")
        return 2
    print("[成功] 已生成:")
    for path in made:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
