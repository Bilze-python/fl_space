"""只读工具、模型调用边界和路径隔离回归；不联网。"""

import json

import pytest

from fl_space.research.tools import ResearchTools, run_tool_research


@pytest.fixture
def tools(tmp_path):
    folder = tmp_path / "fl_space"
    folder.mkdir()
    (folder / "sample.py").write_text(
        "import os\n\nclass Model:\n    def train(self):\n        return 42\n", encoding="utf-8"
    )
    (folder / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=hidden", encoding="utf-8")
    return ResearchTools(
        tmp_path,
        code_enabled=True,
        search_literature=lambda q, limit: [
            {"doc_id": "d1", "page": 1, "quote": "research source"}
        ],
        read_literature=lambda doc, page: "research source" if (doc, page) == ("d1", 1) else "",
    )


def test_catalog_and_file_search_read_outline(tools):
    assert len(tools.catalog()) == 7
    files = tools.call("code.list_files", {})["data"]["files"]
    assert set(files) == {"fl_space/sample.py", "fl_space/broken.py"}
    match = tools.call("code.search", {"query": "train"})["data"]["matches"][0]
    assert match["line"] == 4
    read = tools.call("code.read_lines", {"path": match["path"], "start_line": 3, "line_count": 3})[
        "data"
    ]
    assert read["lines"][0]["line"] == 3
    assert len(read["sha256"]) == 64
    outline = tools.call("code.python_outline", {"path": match["path"]})["data"]
    assert {item["name"] for item in outline["definitions"]} == {"Model", "train"}
    assert outline["executed"] is False
    assert len(tools.trace) == 4
    assert all(record["latency_ms"] >= 0 for record in tools.trace)


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "../.env",
        "fl_space/../../outside.py",
        "C:/secret.py",
        "/etc/passwd",
        "fl_space\\sample.py",
        "fl_space/.key.py",
        "web/secrets/key.py",
    ],
)
def test_path_traversal_and_hidden_resources_denied(tools, path):
    assert tools.call("code.read_lines", {"path": path})["ok"] is False


def test_external_symlink_cannot_be_read_or_searched(tools, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside.py"
    outside.write_text("PRIVATE_EXTERNAL", encoding="utf-8")
    link = tmp_path / "fl_space" / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("host does not allow symlink creation")
    assert not tools.call("code.read_lines", {"path": "fl_space/linked.py"})["ok"]
    assert not tools.call("code.search", {"query": "PRIVATE_EXTERNAL"})["data"]["matches"]


def test_schema_rejects_unknown_fields_and_bool_line_numbers(tools):
    assert not tools.call("code.read_lines", {"path": "fl_space/sample.py", "start_line": True})[
        "ok"
    ]
    assert not tools.call("code.search", {"query": "x", "command": "whoami"})["ok"]
    assert not tools.call("shell", {"command": "whoami"})["ok"]


def test_syntax_only_never_executes(tools, tmp_path):
    path = tmp_path / "fl_space" / "danger.py"
    path.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
    assert tools.call("code.check_syntax", {"path": "fl_space/danger.py"})["data"]["syntax_valid"]
    invalid = tools.call("code.check_syntax", {"path": "fl_space/broken.py"})
    assert invalid["ok"] and invalid["data"]["syntax_valid"] is False


def test_public_catalog_hides_code_and_direct_invocation_is_denied(tools):
    public = ResearchTools(
        tools.project_dir,
        code_enabled=False,
        search_literature=tools.search_literature,
        read_literature=tools.read_literature,
    )
    assert len(public.catalog()) == 2
    assert not public.call("code.read_lines", {"path": "fl_space/sample.py"})["ok"]
    assert (
        public.call("literature.read_page", {"doc_id": "d1", "page": 1})["data"]["text"]
        == "research source"
    )


def test_offline_literature_search_then_reads_page(tools):
    public = ResearchTools(
        tools.project_dir,
        code_enabled=False,
        search_literature=tools.search_literature,
        read_literature=tools.read_literature,
    )
    result = run_tool_research("research", public)
    assert [item["tool"] for item in result["tool_trace"]] == [
        "literature.search",
        "literature.read_page",
    ]
    assert result["excerpts"][0]["doc_id"] == "d1"


def test_offline_search_then_reads_matching_lines(tools):
    result = run_tool_research("train", tools)
    assert [item["tool"] for item in result["tool_trace"]] == ["code.search", "code.read_lines"]
    assert result["excerpts"][0]["path"] == "fl_space/sample.py"
    assert result["verification"]["code_modified"] is False
    json.dumps(result, allow_nan=False)


def test_model_chooses_ast_and_read_tools_with_budget(tools):
    def choose(question, catalog, history):
        if not history:
            return {"tool": "code.python_outline", "arguments": {"path": "fl_space/sample.py"}}
        if len(history) == 1:
            return {"tool": "code.read_lines", "arguments": {"path": "fl_space/sample.py"}}
        return {"stop": True}

    result = run_tool_research("inspect", tools, chooser=choose)
    assert result["stop_reason"] == "model_stop"
    assert len(result["tool_trace"]) == 2
    assert result["excerpts"]


def test_model_cannot_invent_tool_or_execute_document_instructions(tools):
    result = run_tool_research(
        "train",
        tools,
        chooser=lambda *args: {"tool": "shell", "arguments": {"command": "malicious"}},
    )
    assert result["model_fallback"]
    assert all(record["tool"] != "shell" for record in result["tool_trace"])
    assert result["excerpts"]


def test_duplicate_and_total_call_budget(tools):
    result = run_tool_research(
        "train", tools, chooser=lambda *args: {"tool": "code.list_files", "arguments": {}}
    )
    assert result["stop_reason"] == "duplicate_call" and len(result["tool_trace"]) == 1
    with pytest.raises(ValueError):
        run_tool_research("train", tools, max_calls=6)


def test_long_lines_and_big_files_are_bounded(tools, tmp_path):
    (tmp_path / "fl_space" / "long.py").write_text(("#" * 2000 + "\n") * 80)
    result = tools.call("code.read_lines", {"path": "fl_space/long.py", "line_count": 160})["data"]
    assert sum(len(item["text"]) for item in result["lines"]) <= 10000
    assert result["truncated"]
    (tmp_path / "fl_space" / "huge.py").write_text("x" * (tools.max_bytes + 1))
    assert not tools.call("code.read_lines", {"path": "fl_space/huge.py"})["ok"]


def test_http_tool_mode_and_service_gates(tools, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from fl_space.research.service import ResearchService
    from web.research_routes import register_research_routes

    monkeypatch.setenv("SPACEFL_OFFLINE_ONLY", "1")
    service = ResearchService(tools.project_dir, public_demo=False)
    app = FastAPI()
    register_research_routes(app, service)
    with TestClient(app) as client:
        assert len(client.get("/api/research/agent/tools").json()["tools"]) == 7
        response = client.post(
            "/api/research/agent/chat", json={"message": "train", "tool_mode": True}
        )
        assert response.status_code == 200 and response.json()["excerpts"]
        assert (
            client.post(
                "/api/research/agent/chat", json={"message": "train", "tool_mode": "true"}
            ).status_code
            == 422
        )
    service.public_demo = True
    assert len(service.research_tools().catalog()) == 2
