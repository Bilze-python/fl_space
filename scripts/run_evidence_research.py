"""科研决策协作本地入口；默认不调用任何在线模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from fl_space.research.decision import JevDecisionProvider, OfflineDecisionProvider  # noqa: E402
from fl_space.research.service import ResearchService  # noqa: E402


def main() -> None:
    """重建本地文献索引并输出证据账本；外部模型需分别显式启用。"""
    parser = argparse.ArgumentParser(description="只读科研证据工作流（默认离线）")
    parser.add_argument("--question", required=True)
    parser.add_argument("--decision", choices=("offline", "jev"), default="offline")
    parser.add_argument("--llm", action="store_true", help="允许已配置的 LLM 选择摘录")
    parser.add_argument("--tools", action="store_true", help="使用只读代码/文献工具调查")
    parser.add_argument("--output", type=Path, default=Path("output/research-evidence.json"))
    args = parser.parse_args()
    decision = JevDecisionProvider() if args.decision == "jev" else OfflineDecisionProvider()
    service = ResearchService(PROJECT_DIR, public_demo=False, decision_provider=decision)
    service.rebuild_library()
    result = (
        service.tool_research(args.question, use_llm=args.llm)
        if args.tools
        else service.research_evidence(args.question, use_llm=args.llm)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    if args.tools:
        print(
            f"mode={result['mode']}, calls={len(result['tool_trace'])}, excerpts={len(result['excerpts'])}"
        )
    else:
        print(
            f"mode={result['mode']}, excerpts={len(result['claims'])}, "
            f"steps={result['decision']['steps']}, citation={result['verification']['citation']}"
        )
    print(f"结果已写入 {args.output.resolve()}")


if __name__ == "__main__":
    main()
