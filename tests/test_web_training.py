"""A frozen imported dataset must complete the actual FL training CLI."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
from PIL import Image

from web import server


def test_imported_imagefolder_trains_and_archives(tmp_path: Path):
    data_dir = tmp_path / "dataset"
    for split, count in (("train", 4), ("test", 1)):
        for category, color in (("forest", (25, 90, 40)), ("urban", (125, 125, 135))):
            folder = data_dir / split / category
            folder.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (32, 32), color).save(folder / f"{index}.png")

    session = deepcopy(server._default_session())
    session["tune"].update({
        "dataset": "imagefolder", "data_dir": str(data_dir), "rounds": 1,
        "epochs": 1, "batch_size": 4, "max_samples": 0, "non_iid": False,
        "partition_strategy": "iid", "early_stop": 1.0,
    })
    session["mount"].update({"algo": "fedavg", "sats": 2})
    output_dir = tmp_path / "train_result"
    output_dir.mkdir()
    snapshot = output_dir / "session.json"
    snapshot.write_text(json.dumps(session), encoding="utf-8")
    summary = output_dir / "training_summary.json"

    completed = subprocess.run(
        [sys.executable, "-m", "fl_space.cli", "run", "train", "--session-file", str(snapshot),
         "--output", str(summary), "--quiet"],
        cwd=server.PROJECT_DIR, capture_output=True, text=True, timeout=180, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(summary.read_text(encoding="utf-8"))
    assert result["config"]["dataset"] == "imagefolder"
    assert len(server._training_rounds(result["history"])) == 1
    assert 0.0 <= result["history"][-1]["accuracy"] <= 1.0
    archived = server._experiment_from_summary(summary)
    assert archived is not None
    assert archived["kind"] == "fl_training"
    assert archived["rounds"] == 1


def test_web_training_freezes_session_and_blocks_duplicate_jobs(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "test").mkdir()
    session = server._default_session()
    session["tune"].update({"dataset": "imagefolder", "data_dir": str(data_dir), "rounds": 2})
    session["mount"]["sats"] = 2
    session_file = tmp_path / "session.json"
    session_file.write_text(json.dumps(session), encoding="utf-8")
    monkeypatch.setattr(server, "SESSION_FILE", session_file)
    monkeypatch.setattr(server, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(server, "PUBLIC_DEMO", False)
    monkeypatch.setattr(server, "JOBS", {})
    monkeypatch.setattr(server, "_run_training_job", lambda *_: None)
    with TestClient(server.create_app(), base_url="http://127.0.0.1:8721", client=("127.0.0.1", 50000)) as client:
        created = client.post("/api/training/jobs")
        assert created.status_code == 200
        job_id = created.json()["id"]
        snapshot = tmp_path / "experiments" / job_id / "session.json"
        session["tune"]["rounds"] = 100
        session_file.write_text(json.dumps(session), encoding="utf-8")
        assert json.loads(snapshot.read_text(encoding="utf-8"))["tune"]["rounds"] == 2
        assert client.post("/api/training/jobs").status_code == 409
        assert client.post(
            "/api/training/jobs", headers={"Origin": "https://other.example"}
        ).status_code == 403

    monkeypatch.setattr(server, "PUBLIC_DEMO", True)
    with TestClient(server.create_app()) as client:
        assert client.post("/api/training/jobs").status_code == 403
