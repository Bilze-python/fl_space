"""Local relay authorization, bounded jobs, and real on-device training."""

from copy import deepcopy
from io import BytesIO
import json
from zipfile import ZipFile

from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from fl_space.cli import DEFAULT_SESSION
from web import server
from web.dataset_store import import_imagefolder
from web.local_compute import PREFIX, validate_local_session
from web.local_worker import RelayClient, choose_device, dataset_metadata, execute_job, execute_dataset_import
from web.training_runtime import TrainingRuntime


@pytest.fixture
def portal(monkeypatch):
    monkeypatch.setattr(server, "PUBLIC_DEMO", True)
    monkeypatch.setattr(server, "TRAINING_RUNTIME", TrainingRuntime())
    app = server.create_app()
    with TestClient(app, base_url="https://portal.example") as client:
        yield client, app.state.local_compute


def register(client, datasets=None, capabilities=None):
    response = client.post(PREFIX + "/register", json={"name": "test-PC", "device": "cpu", "datasets": datasets or [],
                                                        "capabilities": capabilities or []})
    assert response.status_code == 200, response.text
    agent = response.json()
    response = client.post(PREFIX + "/pair", json={"code": agent["pair_code"]})
    assert response.status_code == 200
    return agent, response.json()


def auth(identity):
    return {"Authorization": "Bearer " + identity["token"]}


def test_demo_pairing_separate_credentials_and_private_results(portal):
    client, _ = portal
    agent, browser = register(client)
    path = PREFIX + "/" + agent["id"]
    assert client.get(path).status_code == 401
    assert client.get(path, headers=auth(agent)).status_code == 401
    response = client.get(path, headers=auth(browser))
    assert response.json()["online"]
    assert response.headers["cache-control"] == "private, no-store"
    assert "token" not in response.text and "pair_code" not in response.text
    assert client.post(PREFIX + "/pair", json={"code": agent["pair_code"]}).status_code == 401
    assert client.post(path + "/poll", json={}, headers=auth(browser)).status_code == 401
    other_agent, other_browser = register(client)
    assert client.get(path, headers=auth(other_browser)).status_code == 401
    assert client.post(path + "/poll", json={}, headers=auth(other_agent)).status_code == 401
    assert client.post(path + "/jobs", json={"session": {}}, headers={**auth(browser), "Origin": "https://evil.example"}).status_code == 403
    for route in ("/api/training/jobs", "/api/datasets/import", "/api/session/reset", "/api/settings/import-cc-switch"):
        assert client.post(route, json={}).status_code == 403
    assert client.get("/api/jobs").json() == []
    assert client.post(path + "/disconnect", json={}, headers=auth(browser)).status_code == 200
    assert client.post(path + "/poll", json={}, headers=auth(agent)).status_code == 401


def test_expiry_offline_and_rate_limit(portal):
    client, relay = portal
    relay.clock = lambda: 1000
    response = client.post(PREFIX + "/register", json={})
    code = response.json()["pair_code"]
    relay.clock = lambda: 1601
    assert client.post(PREFIX + "/pair", json={"code": code}).status_code == 401
    agent, browser = register(client)
    path = PREFIX + "/" + agent["id"]
    relay.clock = lambda: 1650
    assert not client.get(path, headers=auth(browser)).json()["online"]
    assert client.post(path + "/jobs", json={"session": {}}, headers=auth(browser)).status_code == 409
    relay.clock = lambda: 50000
    assert client.get(path, headers=auth(browser)).status_code == 401
    for _ in range(20):
        client.post(PREFIX + "/pair", json={"code": "wrong"})
    assert client.post(PREFIX + "/pair", json={"code": "wrong"}).status_code == 429


@pytest.mark.parametrize("section,key,value", [
    ("tune", "dataset", []), ("tune", "rounds", "3"), ("tune", "rounds", True),
    ("tune", "rounds", 1001), ("tune", "device", "cuda"), ("tune", "workers", 4),
    ("tune", "protocol_mode", "shell"), ("tune", "selection_strategy", "shell"),
    ("mount", "config", "C:/secret.json"), ("mount", "backend", "skyfield"),
    ("tune", "satellite_data_profiles", {"0": {"max_samples": "bad"}}),
    ("tune", "dataset", "imagefolder"), ("mount", "exec", "echo bad"),
])
def test_worker_independently_rejects_unsafe_config(tmp_path, section, key, value):
    payload = {section: {key: value}}
    with pytest.raises(HTTPException) as error:
        validate_local_session(payload, None, [], tmp_path)
    assert error.value.status_code == 422


def test_worker_local_paths_and_local_device(tmp_path):
    payload = deepcopy(DEFAULT_SESSION)
    payload["tune"].update(dataset="imagefolder", data_dir="C:/private")
    validated = validate_local_session(payload, "images", [{"id": "images"}], tmp_path, "cuda")
    assert validated["tune"]["data_dir"] == str(tmp_path / "datasets" / "images")
    assert validated["tune"]["device"] == "cuda"
    assert payload["tune"]["data_dir"] == "C:/private"
    payload["tune"].update(dataset="mnist")
    assert validate_local_session(payload, None, [], tmp_path)["tune"]["data_dir"] == str(tmp_path / "cache")


def test_job_claim_serialization_cancellation_and_bounded_body(portal):
    client, _ = portal
    agent, browser = register(client)
    path = PREFIX + "/" + agent["id"]
    response = client.post(path + "/jobs", json={"session": {}}, headers=auth(browser))
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["session"]["tune"]["data_dir"] == ""
    assert client.post(path + "/jobs", json={"session": {}}, headers=auth(browser)).status_code == 409
    claimed = client.post(path + "/poll", json={}, headers=auth(agent)).json()["job"]
    assert claimed["id"] == job["id"]
    assert client.post(path + "/poll", json={}, headers=auth(agent)).json()["job"]["id"] == job["id"]
    update = path + "/jobs/" + job["id"]
    assert client.post(update, json={}, headers=auth(agent)).status_code == 200
    assert client.post(path + "/poll", json={}, headers=auth(agent)).json()["job"] is None
    assert client.post(update, json={"status": "DONE"}, headers=auth(agent)).status_code == 422
    assert client.post(path + "/cancel", json={}, headers=auth(browser)).status_code == 200
    assert client.post(update, json={}, headers=auth(agent)).json()["cancel"]
    assert client.post(update, json={"status": "CANCELLED"}, headers=auth(agent)).status_code == 200
    assert client.post(update, json={}, headers=auth(agent)).status_code == 409
    assert client.post(path + "/jobs", content=b" " * (256 * 1024 + 1), headers=auth(browser)).status_code == 413


class InProcessAgent:
    def __init__(self, client, agent):
        self.client, self.agent = client, agent
        self.worker_id = agent["id"]

    def post(self, suffix, payload):
        response = self.client.post(PREFIX + suffix, json=payload, headers=auth(self.agent))
        assert response.status_code == 200, response.text
        return response.json()


def make_zip():
    raw = BytesIO()
    with ZipFile(raw, "w") as archive:
        for split in ("train", "test"):
            for category, color in (("forest", "green"), ("urban", "gray")):
                for index in range(4 if split == "train" else 1):
                    image = BytesIO()
                    Image.new("RGB", (32, 32), color).save(image, format="PNG")
                    archive.writestr(f"{split}/{category}/{index}.png", image.getvalue())
    return raw.getvalue()


def test_real_local_import_train_and_private_result_relay(portal, tmp_path):
    client, relay = portal
    imported = import_imagefolder(tmp_path / "datasets", "satellite images", make_zip())
    metadata = dataset_metadata(tmp_path)
    assert "data_dir" not in metadata[0]
    agent, browser = register(client, metadata)
    session = deepcopy(DEFAULT_SESSION)
    session["tune"].update(dataset="imagefolder", rounds=1, epochs=1, max_samples=4,
                           partition_strategy="iid", non_iid=False, early_stop=1.0)
    session["mount"].update(sats=2, sim_hours=0.1)
    path = PREFIX + "/" + agent["id"]
    response = client.post(path + "/jobs", json={"session": session, "dataset_id": imported["id"]}, headers=auth(browser))
    assert response.status_code == 200, response.text
    adapter = InProcessAgent(client, agent)
    job = adapter.post("/" + agent["id"] + "/poll", {"datasets": metadata})["job"]
    result = execute_job(adapter, job, tmp_path, "cpu", 1, 90)
    assert result["status"] == "DONE", (tmp_path / "experiments" / job["id"] / "training.log").read_text(encoding="utf-8")
    assert len(result["history"]) == 1
    relayed = client.get(path, headers=auth(browser)).json()["jobs"][-1]
    assert relayed["status"] == "DONE"
    assert relayed["history"] == result["history"]
    assert str(tmp_path) not in json.dumps(relayed)
    folder = tmp_path / "experiments" / job["id"]
    assert all((folder / name).is_file() for name in ("session.json", "training.log", "training_summary.json", "job.json"))
    assert client.get("/api/jobs").json() == []
    relay.workers.clear()  # Same semantics as a platform restart: local outputs survive.
    assert client.get(path, headers=auth(browser)).status_code == 401
    assert (folder / "training_summary.json").is_file()


def test_timeout_and_cancel_keep_local_job_record(tmp_path):
    class Agent:
        worker_id = "test"

        def post(self, suffix, payload):
            return {"cancel": True}

    for index, seconds in enumerate((0, 60)):
        job = {"id": f"{index:024x}", "session": {}, "dataset_id": None}
        state = execute_job(Agent(), job, tmp_path, "cpu", 1, seconds)
        assert state["status"] == ("FAILED" if seconds == 0 else "CANCELLED")
        assert (tmp_path / "experiments" / job["id"] / "job.json").exists()


def test_https_and_cuda_selection(monkeypatch):
    for url in ("http://remote.example", "https://user:password@remote.example", "https://remote.example/path"):
        with pytest.raises(ValueError):
            RelayClient(url)
    assert RelayClient("http://127.0.0.1:8722").server.startswith("http:")
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert choose_device("auto") == "cpu"
    with pytest.raises(ValueError, match="CUDA"):
        choose_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert choose_device("auto") == "cuda"


def test_public_dataset_import_requires_pairing_and_reports_completion(portal, tmp_path, monkeypatch):
    client, _ = portal
    old_agent, old_browser = register(client)
    assert client.post(PREFIX + "/" + old_agent["id"] + "/datasets/import",
                       json={"source": "eurosat_rgb"}, headers=auth(old_browser)).status_code == 409
    agent, browser = register(client, capabilities=["eurosat_rgb_import"])
    path = PREFIX + "/" + agent["id"]
    endpoint = path + "/datasets/import"
    assert client.post(endpoint, json={"source": "eurosat_rgb"}).status_code == 401
    assert client.post(endpoint, json={"source": "spacenet"}, headers=auth(browser)).status_code == 422
    response = client.post(endpoint, json={"source": "eurosat_rgb"}, headers=auth(browser))
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["kind"] == "dataset_import"
    assert client.post(endpoint, json={"source": "eurosat_rgb"}, headers=auth(browser)).status_code == 409
    adapter = InProcessAgent(client, agent)
    claimed = adapter.post("/" + agent["id"] + "/poll", {"datasets": []})["job"]
    assert claimed["id"] == job["id"]

    def fake_import(root, progress):
        progress("正在准备测试数据")
        return import_imagefolder(root, "EuroSAT RGB", make_zip())

    monkeypatch.setattr("web.local_worker.import_eurosat_rgb", fake_import)
    result = execute_dataset_import(adapter, claimed, tmp_path)
    status = client.get(path, headers=auth(browser)).json()
    assert status["jobs"][-1]["status"] == "DONE"
    assert status["jobs"][-1]["imported_id"] == result["id"]
    assert status["datasets"][0]["id"] == result["id"]
    assert client.post(endpoint, json={"source": "eurosat_rgb"}, headers=auth(browser)).json()["imported_id"] == result["id"]
