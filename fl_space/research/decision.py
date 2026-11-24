"""科研行动决策接口；外部评分不构成事实或执行授权。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import json
import math
import os
from typing import Any
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    """拒绝认证请求重定向，不向第三方域名转发凭据。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class DecisionState:
    """最小决策状态；不包含问题原文、文献正文、路径、凭据。"""

    evidence_count: int
    rejected_count: int
    retrievals: int
    steps_remaining: int
    candidates: tuple[str, ...]
    candidate_summaries: tuple[dict[str, Any], ...] = ()
    budget: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value in (
            self.evidence_count,
            self.rejected_count,
            self.retrievals,
            self.steps_remaining,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("决策计数必须是非负整数")
        if (
            not self.candidates
            or len(set(self.candidates)) != len(self.candidates)
            or any(
                action not in {"retrieve", "compose", "stop", "ask", "refuse"}
                and not (action.startswith("option_") and action[7:].isdigit())
                for action in self.candidates
            )
        ):
            raise ValueError("决策动作必须为合法的唯一候选 ID")
        if self.candidate_summaries:
            if len(self.candidate_summaries) != len(self.candidates):
                raise ValueError("候选摘要必须逐一对应")
            for summary in self.candidate_summaries:
                if (
                    set(summary) != {"id", "kind", "tool", "cost_units"}
                    or summary["id"] not in self.candidates
                    or summary["kind"] not in {"tool", "compose", "stop", "ask", "refuse"}
                    or type(summary["cost_units"]) is not int
                    or not 0 <= summary["cost_units"] <= 1
                    or summary["tool"]
                    not in {
                        "",
                        "code.list_files",
                        "code.search",
                        "code.read_lines",
                        "code.python_outline",
                        "code.check_syntax",
                        "literature.search",
                        "literature.read_page",
                    }
                ):
                    raise ValueError("候选摘要含非法字段")
            if len({summary["id"] for summary in self.candidate_summaries}) != len(self.candidates):
                raise ValueError("候选摘要ID重复")
        if any(
            key not in {"tools_remaining", "llm_remaining", "decisions_remaining"}
            or type(value) is not int
            or value < 0
            for key, value in self.budget.items()
        ):
            raise ValueError("预算必须为合法非负计数")

    def to_dict(self) -> dict[str, Any]:
        """返回可审计的最小状态。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionState:
        """恢复候选动作元组。"""
        return cls(
            **{
                **data,
                "candidates": tuple(data["candidates"]),
                "candidate_summaries": tuple(data.get("candidate_summaries", ())),
            }
        )


@dataclass(frozen=True)
class DecisionAdvice:
    """建议与未校准分数，禁止解释为结论正确率。"""

    action: str
    provider: str = "offline"
    probabilities: dict[str, float] = field(default_factory=dict)

    def validate(self, state: DecisionState) -> None:
        """建议动作和概率键均不得超出调用方给定集合。"""
        if self.action not in state.candidates:
            raise ValueError("决策模型选择了未授权候选动作")
        if self.probabilities:
            if set(self.probabilities) != set(state.candidates):
                raise ValueError("决策概率集合不匹配")
            if any(
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                for value in self.probabilities.values()
            ):
                raise ValueError("决策概率非法")
            if abs(sum(self.probabilities.values()) - 1) > 0.02:
                raise ValueError("决策概率不归一")

    def to_dict(self) -> dict[str, Any]:
        """导出建议。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionAdvice:
        """恢复建议；使用前还须 validate(state)。"""
        return cls(**data)


class DecisionProvider(ABC):
    """供应商只返回建议，不持有工具或执行器。"""

    @abstractmethod
    def advise(self, state: DecisionState) -> DecisionAdvice:
        """对允许动作提供建议；网络适配器须自行限制请求超时。"""


class OfflineDecisionProvider(DecisionProvider):
    """确定性规则基线：有来源则编排，否则有限检索或停止。"""

    def advise(self, state: DecisionState) -> DecisionAdvice:
        if state.candidate_summaries:
            priority = {"compose": 0, "tool": 1, "ask": 2, "refuse": 3, "stop": 4}
            selected = min(state.candidate_summaries, key=lambda item: priority[item["kind"]])
            return DecisionAdvice(selected["id"])
        action = (
            "compose"
            if state.evidence_count and "compose" in state.candidates
            else ("retrieve" if "retrieve" in state.candidates else "stop")
        )
        return DecisionAdvice(action)


def _setting(name: str) -> str:
    """Windows 用户级实时全局开关优先于继承环境；不读取或写入配置文件。"""
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                return str(winreg.QueryValueEx(key, name)[0])
        except FileNotFoundError:
            pass
    return os.environ.get(name, "")


class JevDecisionProvider(DecisionProvider):
    """显式实例化才调用 Jev；每次调用复核全局开关，不重试计费请求。

    使用与本机 Jev 集成一致的固定 HTTPS 协议。仅传结构化计数/候选动作。
    transport/setting_reader 可注入测试，导入此模块绝不会调用网络。
    """

    endpoint = "https://api.defapi.org/api/v1/decisions"
    model = "typesafe/jev-1.13"

    def __init__(
        self,
        *,
        timeout_s: float = 10,
        transport: Callable[..., Any] | None = None,
        setting_reader: Callable[[str], str] = _setting,
    ) -> None:
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 30:
            raise ValueError("timeout_s 必须在 (0,30] 秒")
        self.timeout_s = timeout_s
        self._transport = transport if transport is not None else build_opener(_NoRedirect()).open
        self._setting = setting_reader

    def advise(self, state: DecisionState) -> DecisionAdvice:
        if self._setting("JEV_ENABLED").lower() not in {"1", "true", "on"}:
            raise RuntimeError("decision_disabled")
        key = self._setting("JEV_API_KEY")
        if not key:
            raise RuntimeError("decision_not_configured")
        question = {
            "type": "choice",
            "instructions": "Select the next permitted research workflow action using validated summaries and budgets. "
            "compose organizes source excerpts, not verified scientific conclusions; "
            "retrieve seeks additional evidence; ask requests clarification; refuse reports unavailable capability; "
            "stop abstains. Tool cost_units consumes a remaining tool grant. Prefer useful progress but stop "
            "when remaining budget cannot support it. Never infer factual truth or extend a budget.",
            "criteria": {
                action: next(
                    (
                        f"{item['kind']} {item['tool']} cost={item['cost_units']}"
                        for item in state.candidate_summaries
                        if item["id"] == action
                    ),
                    action,
                )
                for action in state.candidates
            },
        }
        body = {"model": self.model, "state": state.to_dict(), "questions": {"decision": question}}
        request = Request(
            self.endpoint,
            data=json.dumps(body, allow_nan=False).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._transport(request, timeout=self.timeout_s) as response:
                payload = response.read(65537)
                if len(payload) > 65536:
                    raise ValueError("oversized_response")
                data = json.loads(payload)
            answer = data["answers"]["decision"]
            if answer["type"] != "choice" or not answer.get("probabilities"):
                raise ValueError("wrong_type")
            advice = DecisionAdvice(answer["choice"], "jev", answer["probabilities"])
            advice.validate(state)
            return advice
        except Exception:
            # 上游正文/异常可能含凭据，不允许进入 trace 或回复。
            raise RuntimeError("decision_provider_failed") from None
