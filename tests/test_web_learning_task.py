"""学习任务可视化 API 的快照一致性测试。"""

from copy import deepcopy

import pytest

from web import server


def test_learning_task_snapshot_has_one_canonical_shape(monkeypatch, tmp_path) -> None:
    """默认任务、摘要、轨道预览都来自同一份快照。"""
    task_path = tmp_path / ".spacefl_learning_task.json"
    monkeypatch.setattr(server, "LEARNING_TASK_FILE", task_path)

    response = server._load_learning_task_response()
    assert task_path.is_file()
    assert response["schema_version"] == "learning_task_v1"
    assert response["summary"]["satellite_count"] == len(response["task"]["nodes"])
    assert response["summary"]["link_count"] == len(response["task"]["infrastructure"]["links"])
    assert len(response["orbit_preview"]) == response["summary"]["satellite_count"]
    assert response["task"]["scheduler"]["policy"] == "contact_aware"


def test_learning_task_update_returns_new_revision_and_round_trips(monkeypatch, tmp_path) -> None:
    """保存后重新读取必须得到服务端规范化结果。"""
    task_path = tmp_path / ".spacefl_learning_task.json"
    monkeypatch.setattr(server, "LEARNING_TASK_FILE", task_path)

    original = server._load_learning_task_response()
    updated_task = deepcopy(original["task"])
    updated_task["name"] = "验证任务"
    updated_task["scheduler"]["participant_fraction"] = 0.5
    saved = server._update_learning_task_record(
        updated_task,
        expected_revision=original["revision"],
    )
    loaded = server._load_learning_task_response()

    assert saved["revision"] != original["revision"]
    assert loaded["revision"] == saved["revision"]
    assert loaded["task"]["name"] == "验证任务"
    assert loaded["summary"]["per_round_update_volume_mb"] == 3.0


def test_learning_task_rejects_stale_revision(monkeypatch, tmp_path) -> None:
    """旧页面保存不能覆盖新页面已经提交的任务。"""
    task_path = tmp_path / ".spacefl_learning_task.json"
    monkeypatch.setattr(server, "LEARNING_TASK_FILE", task_path)

    original = server._load_learning_task_response()
    updated_task = deepcopy(original["task"])
    updated_task["name"] = "先提交的页面"
    server._update_learning_task_record(
        updated_task,
        expected_revision=original["revision"],
    )

    stale_task = deepcopy(original["task"])
    stale_task["name"] = "过期页面"
    with pytest.raises(server.LearningTaskRevisionConflictError) as error:
        server._update_learning_task_record(
            stale_task,
            expected_revision=original["revision"],
        )
    assert error.value.current["task"]["name"] == "先提交的页面"
