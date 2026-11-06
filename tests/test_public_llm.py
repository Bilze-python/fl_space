"""Public LLM opt-in keeps the shared secret server-side and bounds use."""

from fastapi.testclient import TestClient

from fl_space.research.service import ResearchService
from web import server


def test_public_provider_requires_explicit_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-private-test-key")
    monkeypatch.setenv("SPACEFL_OFFLINE_ONLY", "0")
    service = ResearchService(tmp_path, public_demo=True)
    assert service.ai_status()["configured"] is False
    monkeypatch.setenv("SPACEFL_PUBLIC_LLM", "1")
    status = service.ai_status()
    assert status["configured"] is True
    assert status["provider"] == "DeepSeek"
    assert status["model"] == "deepseek-chat"
    assert "masked_api_key" not in status
    assert "sk-private" not in str(status)
    monkeypatch.setenv("SPACEFL_OFFLINE_ONLY", "1")
    assert service.ai_status()["configured"] is False


def test_public_llm_quota_and_origin_guard(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-private-test-key")
    monkeypatch.setenv("SPACEFL_OFFLINE_ONLY", "0")
    monkeypatch.setenv("SPACEFL_PUBLIC_LLM", "1")
    monkeypatch.setattr(server, "PUBLIC_DEMO", True)
    monkeypatch.setattr(server, "TRAINING_RUNTIME", server.TrainingRuntime())
    monkeypatch.setattr(ResearchService, "generate_plan", lambda *_: {"mode": "llm", "plan": {}})
    with TestClient(server.create_app(), base_url="https://spacefl.example") as client:
        config = client.get("/api/research/config").json()
        assert config["ai"]["configured"] is True
        assert "masked_api_key" not in config["ai"]
        assert "sk-private" not in str(config)
        path = "/api/research/agent/plan"
        assert client.post(path, json={"instruction": "x"}, headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.post(path, json={"instruction": "x"}, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
        assert client.post(path, content=b"x" * 32769).status_code == 413
        for _ in range(8):
            assert client.post(path, json={"instruction": "x"}).status_code == 200
        assert client.post(path, json={"instruction": "x"}).status_code == 429
        assert client.post("/api/research/library/ask", json={"question": "x"}).status_code == 429
        assert client.post("/api/research/agent/run", json={}).status_code == 403
        assert client.put("/api/research/config", json={}).status_code == 403
