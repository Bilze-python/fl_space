"""Public group access, bounded workloads, and durable training integration."""

from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from starlette.requests import Request

from web import server
from web.training_runtime import COOKIE_NAME, TrainingRuntime

PASSWORD = "research-group-test-key-0123456789abcdef"


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    runtime = TrainingRuntime(enabled=True, root=tmp_path / "persistent", access_key=PASSWORD)
    monkeypatch.setattr(server, "TRAINING_RUNTIME", runtime)
    monkeypatch.setattr(server, "PUBLIC_DEMO", True)
    monkeypatch.setattr(server, "JOBS", {})
    monkeypatch.setattr(server, "_run_training_job", lambda *_: None)
    return runtime


def login(client):
    response = client.post("/api/training/access/login", json={"password": PASSWORD})
    assert response.status_code == 200, response.text
    return response


def dataset_zip():
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for split in ("train", "test"):
            for label, color in (("forest", "green"), ("urban", "gray")):
                for index in range(4 if split == "train" else 1):
                    image = BytesIO()
                    Image.new("RGB", (32, 32), color).save(image, format="PNG")
                    archive.writestr(f"{split}/{label}/{index}.png", image.getvalue())
    return buffer.getvalue()


def test_cloud_requires_strong_key_and_demo_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "TRAINING_RUNTIME", TrainingRuntime(enabled=True, root=tmp_path, access_key="short"))
    with pytest.raises(RuntimeError, match="32"):
        server.create_app()
    monkeypatch.setattr(server, "TRAINING_RUNTIME", TrainingRuntime(enabled=True, root=tmp_path, access_key=PASSWORD))
    monkeypatch.setattr(server, "PUBLIC_DEMO", False)
    with pytest.raises(RuntimeError, match="SPACEFL_PUBLIC_DEMO"):
        server.create_app()


def test_signed_cookie_tamper_expiry(cloud, monkeypatch):
    import web.training_runtime as module

    monkeypatch.setattr(module.time, "time", lambda: 1000)
    cookie = cloud.issue_cookie()

    def request(value):
        return Request({"type": "http", "headers": [(b"cookie", f"{COOKIE_NAME}={value}".encode())]})

    assert cloud.authenticated(request(cookie))
    assert not cloud.authenticated(request(cookie + "x"))
    assert not cloud.authenticated(request("bad.cookie"))
    monkeypatch.setattr(module.time, "time", lambda: 1000 + 12 * 3600)
    assert not cloud.authenticated(request(cookie))


def test_login_security_and_visitor_scope(cloud):
    with TestClient(server.create_app(), base_url="https://spacefl.example") as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/training/access").json()["authenticated"] is False
        assert client.get("/api/jobs").json() == []
        assert client.get("/api/datasets").json()["imported"] == []
        for path in ("/api/training/jobs", "/api/datasets/import?name=x", "/api/session/reset"):
            assert client.post(path, json={}).status_code == 401
        assert client.put("/api/session", json={}).status_code == 401
        assert client.post("/api/training/access/login", json={"password": "wrong"}).status_code == 401
        assert client.post("/api/training/access/login", json={"password": PASSWORD}, headers={"Origin": "https://evil.example"}).status_code == 403
        response = login(client)
        cookie = response.headers["set-cookie"]
        assert all(flag in cookie for flag in ("HttpOnly", "Secure", "SameSite=strict", "Path=/api"))
        assert client.get("/api/training/access").json()["authenticated"] is True
        assert client.post("/api/training/jobs", json={}, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
        # Group access does not unlock arbitrary execution or server configuration.
        assert client.post("/api/validation", json={}).status_code == 403
        assert client.post("/api/settings/import-cc-switch", json={}).status_code == 403
        assert client.post("/api/training/access/logout").status_code == 200
        assert client.get("/api/training/access").json()["authenticated"] is False
        assert PASSWORD not in client.get("/api/datasets").text


def test_password_attempts_are_limited_and_http_rejected(cloud):
    with TestClient(server.create_app(), base_url="https://spacefl.example") as client:
        for _ in range(8):
            assert client.post("/api/training/access/login", json={"password": "wrong"}).status_code == 401
        assert client.post("/api/training/access/login", json={"password": PASSWORD}).status_code == 429
    with TestClient(server.create_app(), base_url="http://spacefl.example") as client:
        assert client.post("/api/training/access/login", json={"password": PASSWORD}).status_code == 403


@pytest.mark.parametrize("section,key,value", [
    ("tune", "rounds", 31), ("tune", "epochs", 6), ("mount", "sats", 11),
    ("tune", "max_samples", 0), ("tune", "batch_size", 129),
    ("tune", "data_workers", 2), ("tune", "workers", 5),
    ("tune", "device", "cuda"), ("tune", "dataset", "femnist"),
    ("tune", "max_contact_epochs", 6), ("mount", "config", "/etc/passwd"),
    ("mount", "time_model_args", {"file": "/etc/passwd"}), ("tune", "lr", "NaN"),
    ("tune", "extra", "unexpected"),
])
def test_public_training_rejects_unbounded_config(cloud, section, key, value):
    with TestClient(server.create_app(), base_url="https://spacefl.example") as client:
        login(client)
        response = client.post("/api/training/jobs", json={section: {key: value}})
        assert response.status_code == 422, response.text
        assert server.JOBS == {}


def test_managed_paths_and_job_snapshot(cloud):
    with TestClient(server.create_app(), base_url="https://spacefl.example") as client:
        login(client)
        assert client.post("/api/training/jobs", json={"tune": {"dataset": "imagefolder", "data_dir": str(cloud.root)}}).status_code == 422
        created = client.post("/api/training/jobs", json={"tune": {"rounds": 2, "data_dir": "/etc"}})
        assert created.status_code == 200, created.text
        job_id = created.json()["id"]
        folder = cloud.root / "experiments" / job_id
        frozen = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        assert frozen["tune"]["rounds"] == 2
        assert frozen["tune"]["data_dir"] == str(cloud.root / "cache")
        assert client.put("/api/session", json={"tune": {"rounds": 3}}).status_code == 200
        assert json.loads((folder / "session.json").read_text(encoding="utf-8")) == frozen
        assert client.post("/api/training/jobs", json={}).status_code == 409
        assert (folder / "job.json").is_file()
        client.post("/api/training/access/logout")
        assert client.get(f"/api/jobs/{job_id}").status_code == 401
        assert client.get("/api/session").json()["tune"]["rounds"] == 10
    server.JOBS.clear()
    with TestClient(server.create_app(), base_url="https://spacefl.example") as client:
        login(client)
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "FAILED"
        assert "重启" in job["message"]


def test_public_import_train_restart_archive(tmp_path, monkeypatch):
    """Use the real subprocess and trainer, with no external dataset download."""
    runtime = TrainingRuntime(enabled=True, root=tmp_path / "disk", access_key=PASSWORD)
    monkeypatch.setattr(server, "TRAINING_RUNTIME", runtime)
    monkeypatch.setattr(server, "PUBLIC_DEMO", True)
    monkeypatch.setattr(server, "JOBS", {})
    run_training = server._run_training_job
    monkeypatch.setattr(server, "_run_training_job", lambda *_: None)
    with TestClient(server.create_app(), base_url="https://spacefl.example") as client:
        login(client)
        upload = client.post("/api/datasets/import?name=satellite-test", content=dataset_zip(), headers={"Content-Type": "application/zip"})
        assert upload.status_code == 200, upload.text
        dataset = upload.json()
        session = client.get("/api/session").json()
        session["tune"].update({"dataset": "imagefolder", "data_dir": dataset["data_dir"],
                                "rounds": 1, "batch_size": 4, "non_iid": False,
                                "partition_strategy": "iid", "early_stop": 1.0})
        session["mount"]["sats"] = 2
        created = client.post("/api/training/jobs", json=session)
        assert created.status_code == 200, created.text
        job_id = created.json()["id"]
        run_training(job_id, runtime.root / "experiments" / job_id)
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "DONE", job
        detail = client.get(f"/api/experiments/{job_id}")
        assert detail.status_code == 200, detail.text
        assert detail.json()["path"].startswith("cloud-experiments/")
        assert 0 <= job["accuracy"] <= 1
        # Traversal cannot read the shared access key or arbitrary server files.
        token = server.TRAINING_MEMBER.set(True)
        try:
            with pytest.raises(HTTPException) as error:
                server._safe_project_path("cloud-experiments/../../session.json")
            assert error.value.status_code == 403
        finally:
            server.TRAINING_MEMBER.reset(token)
        client.post("/api/training/access/logout")
        assert client.get("/api/datasets").json()["imported"] == []
        assert client.get(f"/api/experiments/{job_id}").status_code == 404
        assert job_id not in client.get("/api/experiments").text
    server.JOBS.clear()
    with TestClient(server.create_app(), base_url="https://spacefl.example") as client:
        login(client)
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "DONE"
        assert client.get("/api/datasets").json()["imported"][0]["id"] == dataset["id"]
        assert client.get(f"/api/experiments/{job_id}").status_code == 200


def test_dataset_capacity_is_checked_under_import_lock(cloud, monkeypatch):
    from dataclasses import replace
    monkeypatch.setattr(server, "TRAINING_RUNTIME", replace(cloud, max_datasets=0))
    with TestClient(server.create_app(), base_url="https://spacefl.example") as client:
        login(client)
        response = client.post("/api/datasets/import?name=full", content=dataset_zip())
        assert response.status_code == 409
