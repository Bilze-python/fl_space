"""三角色分权：参谋不能执行、指挥不能越权、审计阻止伪造输出。"""

import json

import pytest

from fl_space.research.commander import ResearchBudget, run_commanded_research
from fl_space.research.decision import DecisionAdvice, DecisionProvider
from fl_space.research.tools import ResearchTools


@pytest.fixture
def tools(tmp_path):
    (tmp_path / "fl_space").mkdir()
    (tmp_path / "fl_space" / "a.py").write_text("def train():\n    return 42\n", encoding="utf-8")
    return ResearchTools(
        tmp_path,
        code_enabled=True,
        search_literature=lambda *args: [],
        read_literature=lambda *args: "",
    )


class ChooseKind(DecisionProvider):
    def __init__(self, kind):
        self.kind = kind
        self.states = []

    def advise(self, state):
        self.states.append(state)
        selected = next(item for item in state.candidate_summaries if item["kind"] == self.kind)
        return DecisionAdvice(selected["id"], "test_commander")


@pytest.mark.parametrize(
    "kind,status", [("ask", "needs_input"), ("refuse", "refused"), ("stop", "stopped")]
)
def test_commander_overrides_advisor_without_executing(tools, kind, status):
    decision = ChooseKind(kind)
    result = run_commanded_research(
        "train",
        tools,
        decision=decision,
        advisor=lambda *args: {
            "candidates": [
                {"kind": "tool", "tool": "code.read_lines", "arguments": {"path": "fl_space/a.py"}}
            ]
        },
    )
    assert result["status"] == status
    assert result["tool_trace"] == []
    assert "path" not in json.dumps(decision.states[0].to_dict())
    assert result["budget"]["remaining"]["tools_remaining"] == 5


def test_commander_selects_second_llm_candidate(tools):
    class Second(DecisionProvider):
        def advise(self, state):
            target = next(
                (item for item in state.candidate_summaries if item["kind"] == "compose"), None
            )
            return DecisionAdvice(target["id"] if target else "option_1", "second")

    result = run_commanded_research(
        "train",
        tools,
        decision=Second(),
        advisor=lambda *args: {
            "candidates": [
                {"kind": "tool", "tool": "code.list_files", "arguments": {}},
                {"kind": "tool", "tool": "code.read_lines", "arguments": {"path": "fl_space/a.py"}},
            ]
        },
    )
    assert result["tool_trace"][0]["tool"] == "code.read_lines"
    assert len(result["tool_trace"]) == 1
    assert result["claims"]


def test_offline_vertical_flow_and_budget(tools):
    result = run_commanded_research("train", tools)
    assert result["status"] == "completed"
    assert [record["tool"] for record in result["tool_trace"]] == ["code.search", "code.read_lines"]
    assert result["roles"]["decision"] == "commander"
    assert result["verification"]["semantic_support"] == "not_assessed"
    assert result["budget"]["remaining"]["tools_remaining"] == 3
    assert result["budget"]["remaining"]["decisions_remaining"] == 2
    json.dumps(result, allow_nan=False)


def test_no_budget_extension_and_no_duplicate_calls(tools):
    class AlwaysTool(DecisionProvider):
        def advise(self, state):
            item = next(
                (item for item in state.candidate_summaries if item["kind"] == "tool"),
                next(item for item in state.candidate_summaries if item["kind"] == "stop"),
            )
            return DecisionAdvice(item["id"])

    result = run_commanded_research(
        "train",
        tools,
        decision=AlwaysTool(),
        budget=ResearchBudget(tool_calls=1),
        advisor=lambda *args: {
            "candidates": [{"kind": "tool", "tool": "code.list_files", "arguments": {}}]
        },
    )
    assert len(result["tool_trace"]) == 1
    assert result["budget"]["remaining"]["tools_remaining"] == 0
    with pytest.raises(ValueError):
        ResearchBudget(tool_calls=6)


def test_hallucinated_numbers_citations_and_extra_schema_are_rejected(tools):
    result = run_commanded_research(
        "train",
        tools,
        writer=lambda *args: {"claims": [{"source_id": "call2:L2", "text": "return 99.9"}]},
    )
    assert result["command_trace"][-1]["audit"]["draft"] == "fail_offline_fallback"
    assert all("99.9" not in claim["text"] for claim in result["claims"])
    assert result["verification"]["numeric"] == "source_copy_only"


def test_advisor_invalid_tool_cannot_expand_permissions(tools):
    result = run_commanded_research(
        "train",
        tools,
        advisor=lambda *args: {"candidates": [{"kind": "tool", "tool": "shell", "arguments": {}}]},
    )
    assert result["command_trace"][0]["audit"]["candidates"] == "fail"
    assert result["claims"]
    assert not any(call["tool"] == "shell" for call in result["tool_trace"])


def test_unknown_commander_action_switches_to_offline_once(tools):
    class Broken(DecisionProvider):
        calls = 0

        def advise(self, state):
            self.calls += 1
            return DecisionAdvice("execute_experiment")

    decision = Broken()
    result = run_commanded_research("train", tools, decision=decision)
    assert result["claims"] and decision.calls == 1
    assert result["command_trace"][0]["decision_fallback"]


def test_writer_selection_accepted_and_llm_budget_charged(tools):
    def write(question, sources):
        source = next(item for item in sources if "return" in item["text"])
        return {"claims": [{"source_id": source["source_id"], "text": "return 42"}]}

    result = run_commanded_research("train", tools, writer=write)
    assert result["claims"][0]["text"] == "return 42"
    assert result["budget"]["remaining"]["llm_remaining"] == 3
    assert result["command_trace"][-1]["audit"]["draft"] == "pass"
