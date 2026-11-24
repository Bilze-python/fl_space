"""科研协作审计：候选schema、工具参数和带来源文本的确定性检查。"""

from __future__ import annotations

import re


def validate_candidates(payload, tools):
    """参谋提案经校验后才成为指挥可选项；路径还会在执行器复核。"""
    if not isinstance(payload, dict) or set(payload) != {"candidates"}:
        raise ValueError("candidate_schema")
    raw = payload["candidates"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= 4:
        raise ValueError("candidate_count")
    result = []
    for candidate in raw:
        if not isinstance(candidate, dict):
            raise ValueError("candidate_type")
        kind = candidate.get("kind")
        if kind == "tool":
            if set(candidate) != {"kind", "tool", "arguments"}:
                raise ValueError("tool_candidate_schema")
            arguments = tools._validate(candidate["tool"], candidate["arguments"])
            if candidate["tool"].startswith("code.") and "path" in arguments:
                tools._path(arguments["path"])
            result.append({"kind": kind, "tool": candidate["tool"], "arguments": arguments})
        elif kind in {"compose", "stop", "ask", "refuse"}:
            if set(candidate) != {"kind"}:
                raise ValueError("terminal_candidate_schema")
            result.append({"kind": kind})
        else:
            raise ValueError("unknown_candidate")
    return result


def verify_source_draft(draft, sources):
    """严格来源模式：只允许已有来源ID及逐字摘录，不伪装语义验证。

    以来源子串保证新数字/单位不会由模型自由生成。不能证明摘录相关性、
    论文真实或原始计算正确。unsupported 可用代码固定文案补充。
    """
    if not isinstance(draft, dict) or set(draft) != {"claims"}:
        raise ValueError("draft_schema")
    claims = draft["claims"]
    if not isinstance(claims, list) or not 1 <= len(claims) <= 5:
        raise ValueError("claim_count")
    result = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"source_id", "text"}:
            raise ValueError("claim_schema")
        source = sources[claim["source_id"]]
        text = claim["text"]
        if not isinstance(text, str) or not text.strip() or text not in source["text"]:
            raise ValueError("unsupported_text")
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?%?", text)
        if any(number not in source["text"] for number in numbers):
            raise ValueError("unsupported_number")
        result.append(
            {
                **claim,
                "evidence": [claim["source_id"]],
                "verified": True,
                "verification_scope": "source_copy_only",
            }
        )
    return result
