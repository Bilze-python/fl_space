"""Single-account access control for the public demo."""

from fastapi.testclient import TestClient

from fl_space.research.service import ResearchService
from web import server

USER = "spaceTS"
PASSWORD = "11451467"


def test_public_demo_requires_single_account_login(monkeypatch):
    monkeypatch.setattr(server, "PUBLIC_DEMO", True)
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setattr(server, "AUTH_USER", USER)
    monkeypatch.setattr(server, "AUTH_PASSWORD", PASSWORD)
    monkeypatch.setattr(server, "TRAINING_RUNTIME", server.TrainingRuntime())
    monkeypatch.setattr(
        ResearchService,
        "generate_plan",
        lambda *_: {"mode": "llm", "plan": {}, "review": {"metrics": {}}, "citations": [], "steps": []},
    )
    with TestClient(server.create_app(), base_url="https://spacefl.example") as client:
        # unauthenticated browser request -> login page
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"].startswith("/login")
        # unauthenticated API -> 401 JSON, health is public
        assert client.get("/api/research/config").status_code == 401
        assert client.get("/api/health").status_code == 200
        assert client.get("/login").status_code == 200
        # wrong account/password -> back to login with error
        response = client.post("/login", data={"username": USER, "password": "wrong"}, follow_redirects=False)
        assert response.status_code == 302
        assert "error=1" in response.headers["location"]
        # correct account -> session cookie and platform access
        response = client.post(
            "/login",
            data={"username": USER, "password": PASSWORD, "next": "/#agent"},
            follow_redirects=False,
        )
        assert response.status_code == 302
        assert response.headers["location"] == "/#agent"
        assert client.get("/").status_code == 200
        assert client.get("/api/research/config").status_code == 200
        # read-only demo guard still applies after login
        assert client.post("/api/research/agent/run", json={}).status_code == 403
        # logout clears the session
        assert client.get("/logout", follow_redirects=False).status_code == 302
        assert client.get("/api/research/config").status_code == 401
