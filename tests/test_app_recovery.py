"""Recovery behaviour: the app keeps working after errors, offline NCT fetch, restart persistence."""
import urllib.request

from fastapi.testclient import TestClient

from kormat.app.engine import FakeEngine
from kormat.app.server import create_app

NOTE = "1. Adult with diabetes.\n2. No allergies."
C = [{"type": "inclusion", "text": "diabetes"}]


def test_keeps_working_after_exception(tmp_path):
    class Flaky(FakeEngine):
        n = 0

        def screen_one(self, *a, **k):
            Flaky.n += 1
            if Flaky.n == 1:
                raise RuntimeError("transient")
            return super().screen_one(*a, **k)

    with TestClient(create_app(Flaky(), tmp_path / "t.db"), raise_server_exceptions=False) as c:
        assert c.post("/api/screen", json={"note": NOTE, "criteria": C}).status_code == 500
        assert c.post("/api/screen", json={"note": NOTE, "criteria": C}).status_code == 200
        assert c.get("/api/health").status_code == 200


def test_nct_offline_graceful(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with TestClient(create_app(FakeEngine(), tmp_path / "t.db")) as c:
        r = c.get("/api/nct/NCT01234567")
        assert r.status_code == 502 and "paste" in r.json()["error"]["message"]


def test_model_error_state(tmp_path):
    class Bad(FakeEngine):
        def ready(self):
            return False

        def state(self):
            return {"status": "error", "detail": "weights missing"}

    with TestClient(create_app(Bad(), tmp_path / "t.db")) as c:
        r = c.post("/api/screen", json={"note": NOTE, "criteria": C})
        assert r.status_code == 500 and "weights missing" in r.json()["error"]["message"]
