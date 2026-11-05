"""Public demo must expose only the curated data bundle."""

from pathlib import Path

from fastapi import HTTPException
import pytest

from fl_space.research.service import ResearchService
from web import server


def test_public_demo_library_uses_curated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    demo = tmp_path / "deploy" / "demo_data" / "library"
    demo.mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "文献").mkdir()
    monkeypatch.setattr(server, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(server, "PUBLIC_DEMO", True)

    assert server._library_roots() == [("公开演示文献", demo.resolve())]
    service = ResearchService(tmp_path, public_demo=True)
    assert service._library_roots() == [("公开演示文献", demo.resolve())]


def test_public_demo_experiment_paths_stay_in_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    demo = tmp_path / "deploy" / "demo_data" / "experiments"
    selected = demo / "selected" / "validation_summary.json"
    selected.parent.mkdir(parents=True)
    selected.write_text("{}", encoding="utf-8")
    unrelated = tmp_path / "other" / "validation_summary.json"
    unrelated.parent.mkdir()
    unrelated.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(server, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(server, "PUBLIC_DEMO", True)

    assert server._experiment_summary_files() == [selected]
    assert server._safe_project_path("deploy/demo_data/experiments/selected/validation_summary.json") == selected
    with pytest.raises(HTTPException) as exc:
        server._safe_project_path("other/validation_summary.json")
    assert exc.value.status_code == 403
