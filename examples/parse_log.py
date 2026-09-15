"""
SpaceFL 日志解析工具 — 将终端输出转为 JSON

用法:
    # 从文件解析
    python examples/parse_log.py experiment.log --output results/

    # 从管道解析
    cat experiment.log | python examples/parse_log.py --output results/
"""
import argparse
from datetime import datetime
import json
import os
import re
import sys

LOG_PATTERN = re.compile(
    r"轮次\s+(\d+)/(\d+)\s*\|\s*TS=\s*(\d+)→\s*(\d+)\s*\((.+?)\)\s*\|\s*"
    r"在线:(\d+)\s*\|\s*选中:\s*(\d+)\s*\|\s*训练:\d+\s*\|\s*准确率:([\d.]+)"
)

EARLY_STOP_PATTERN = re.compile(r"早停触发.*准确率\s+([\d.]+).*第\s+(\d+)\s*轮")

SUMMARY_PATTERN = re.compile(
    r"总虚拟时间:\s*(\d+)\s*timeslots\s*=\s*([\d.]+)\s*小时\s*\(\s*([\d.]+)\s*天\)"
)

FINAL_PATTERN = re.compile(r"完成:\s*(\d+)\s*轮,\s*准确率\s*([\d.]+)")


def parse_log(text: str) -> dict:
    """解析日志文本为结构化 JSON。"""
    rounds = []
    early_stop = None
    summary = {}
    final = {}

    for line in text.split("\n"):
        m = LOG_PATTERN.search(line)
        if m:
            rounds.append({
                "round": int(m.group(1)),
                "total_rounds": int(m.group(2)),
                "ts_start": int(m.group(3)),
                "ts_end": int(m.group(4)),
                "duration": m.group(5),
                "online_clients": int(m.group(6)),
                "selected_clients": int(m.group(7)),
                "accuracy": float(m.group(8)),
            })
            continue

        m = EARLY_STOP_PATTERN.search(line)
        if m:
            early_stop = {"accuracy": float(m.group(1)), "round": int(m.group(2))}
            continue

        m = SUMMARY_PATTERN.search(line)
        if m:
            summary = {
                "total_timeslots": int(m.group(1)),
                "total_hours": float(m.group(2)),
                "total_days": float(m.group(3)),
            }
            continue

        m = FINAL_PATTERN.search(line)
        if m:
            final = {"rounds": int(m.group(1)), "accuracy": float(m.group(2))}

    result = {
        "parsed_at": datetime.now().isoformat(),
        "total_rounds_parsed": len(rounds),
        "early_stop": early_stop,
        "final": final,
        "summary": summary,
        "history": rounds,
    }

    if rounds:
        accs = [r["accuracy"] for r in rounds]
        result["stats"] = {
            "max_accuracy": max(accs),
            "max_round": rounds[accs.index(max(accs))]["round"],
            "min_accuracy": min(accs),
            "min_round": rounds[accs.index(min(accs))]["round"],
            "mean_accuracy": sum(accs) / len(accs),
            "std_accuracy": (
                (sum((a - sum(accs) / len(accs)) ** 2 for a in accs) / len(accs)) ** 0.5
            ),
        }

    return result


def main():
    p = argparse.ArgumentParser(description="解析 SpaceFL 终端输出为 JSON")
    p.add_argument("logfile", nargs="?", help="日志文件路径 (不指定则从 stdin 读取)")
    p.add_argument("--output", "-o", default="results", help="输出目录")
    args = p.parse_args()

    if args.logfile:
        with open(args.logfile, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    result = parse_log(text)
    os.makedirs(args.output, exist_ok=True)

    out_path = os.path.join(args.output, "parsed_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    n = result["total_rounds_parsed"]
    print(f"解析完成: {n} 轮 → {out_path}")
    if "stats" in result:
        s = result["stats"]
        print(f"  max={s['max_accuracy']:.4f} @ R{s['max_round']}")
        print(f"  min={s['min_accuracy']:.4f} @ R{s['min_round']}")
        print(f"  mean={s['mean_accuracy']:.4f} ± {s['std_accuracy']:.4f}")


if __name__ == "__main__":
    main()
