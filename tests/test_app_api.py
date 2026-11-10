import pytest
from fastapi.testclient import TestClient

from kormat.app.engine import FakeEngine
from kormat.app.server import create_app

NOTE = "1. 58-year-old woman with type 2 diabetes.\n2. HbA1c was 8.4%.\n3. She takes metformin."
C = [{"type": "inclusion", "text": "type 2 diabetes"}, {"type": "exclusion", "text": "pregnant"}]


class Loading(FakeEngine):
    def ready(self):
        return False

    def state(self):
        return {"status": "loading", "detail": "x"}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(FakeEngine(), tmp_path / "t.db")) as c:
        yield c


def err(r, status, code):
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code
    assert r.json()["error"]["message"]


def test_health_and_static(client):
    assert client.get("/api/health").json()["status"] == "ready"
    assert client.get("/").status_code == 200
    assert client.get("/logo.png").status_code == 200


def test_happy_path_and_history(client, tmp_path):
    r = client.post("/api/screen", json={"note": NOTE, "criteria": C})
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["criteria"]) == 2 and d["aggregate"]["outcome"] and d["provenance"]["git_sha"]
    assert "Educational screening prototype" in d["disclaimer"]
    rid = d["id"]
    assert client.get("/api/history").json()["reports"][0]["id"] == rid
    assert "Educational screening prototype" in client.get(f"/api/report/{rid}.html").text
    assert client.get(f"/api/report/{rid}.json").json()["id"] == rid
    # survives a restart: new app, same db
    with TestClient(create_app(FakeEngine(), tmp_path / "t.db")) as c2:
        assert c2.get(f"/api/history/{rid}").status_code == 200
    assert client.delete(f"/api/history/{rid}").status_code == 200
    err(client.get(f"/api/history/{rid}"), 404, "not_found")


def test_block_input(client):
    r = client.post("/api/screen", json={"note": NOTE, "criteria_block": "Inclusion Criteria:\n- diabetes\nExclusion Criteria:\n- pregnant"})
    assert r.status_code == 200 and len(r.json()["criteria"]) == 2


def test_validation(client):
    err(client.post("/api/screen", json={"note": "  ", "criteria": C}), 400, "empty_note")
    err(client.post("/api/screen", json={"note": "x" * 12001, "criteria": C}), 400, "note_too_long")
    assert "12000" in client.post("/api/screen", json={"note": "x" * 12001, "criteria": C}).json()["error"]["message"]
    err(client.post("/api/screen", json={"note": NOTE, "criteria": []}), 400, "no_criteria")
    err(client.post("/api/screen", json={"note": NOTE, "criteria": [C[0]] * 41}), 400, "too_many_criteria")
    err(client.post("/api/screen", json={"note": NOTE, "criteria": [{"type": "other", "text": "a"}]}), 400, "bad_type")
    err(client.post("/api/screen", json={"note": NOTE, "criteria": C, "mode": "magic"}), 400, "unknown_mode")
    err(client.post("/api/screen", json={"note": NOTE, "criteria": C, "mode": "speculative", "gamma": 99}), 400, "bad_gamma")
    err(client.post("/api/screen", json={"note": NOTE, "criteria": "nope"}), 422, "invalid_request")


def test_model_loading_503(tmp_path):
    with TestClient(create_app(Loading(), tmp_path / "t.db")) as c:
        r = c.post("/api/screen", json={"note": NOTE, "criteria": C})
        err(r, 503, "model_loading")
        assert r.headers["retry-after"] and r.json()["error"]["hint"]


def test_exception_to_json(tmp_path):
    class Boom(FakeEngine):
        def screen_one(self, *a, **k):
            raise RuntimeError("boom")

    with TestClient(create_app(Boom(), tmp_path / "t.db"), raise_server_exceptions=False) as c:
        err(c.post("/api/screen", json={"note": NOTE, "criteria": C}), 500, "internal_error")


def test_lab(client):
    r = client.post("/api/lab", json={"note": NOTE, "criterion": C[0], "modes": ["no_cache", "kv_cached", "speculative"], "gamma": 3})
    assert r.status_code == 200, r.text
    runs = {x["mode"]: x for x in r.json()["runs"]}
    assert runs["no_cache"]["equal_to_kv_cached"] and runs["speculative"]["timing"]["stats"]["gamma"] == 3
    err(client.post("/api/lab", json={"note": NOTE, "criterion": C[0], "modes": []}), 400, "no_modes")
    err(client.post("/api/lab", json={"note": NOTE, "modes": ["kv_cached"]}), 400, "no_criteria")


def test_parse_nct_samples_stored(client):
    d = client.post("/api/parse_criteria", json={"block": "Inclusion Criteria:\n- a\n- b"}).json()
    assert d["count"] == 2
    err(client.post("/api/parse_criteria", json={"block": ""}), 400, "no_criteria")
    err(client.get("/api/nct/abc"), 400, "bad_nct")
    assert client.get("/api/samples").json()["samples"]
    s = client.get("/api/stored").json()
    assert set(s) == {"kvcache", "speculative", "attention"}


def test_unseen_note_acceptance(client):
    note = "Mr. Okafor is a 71-year-old man. He was admitted with chest pain. Troponin was elevated. He has a pacemaker. He does not smoke."
    r = client.post("/api/screen", json={"note": note, "criteria": [
        {"type": "inclusion", "text": "Age over 65"}, {"type": "exclusion", "text": "Implanted cardiac device"},
        {"type": "exclusion", "text": "Current smoker"}]})
    assert r.status_code == 200
    for c in r.json()["criteria"]:
        assert c["label"] and c["explanation"] and set(c["evidence_ids"]) <= set(map(int, r.json()["sentences"]))
