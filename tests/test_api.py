"""HTTP API and batch-operation tests."""

from __future__ import annotations

import numpy as np
import threading
import time
from fastapi.testclient import TestClient

from vexor.api.app import create_app
from vexor.db import VectorDB


def test_vectordb_batch_add_and_search():
    rng = np.random.default_rng(123)
    vecs = rng.standard_normal((50, 16)).astype(np.float32)
    queries = rng.standard_normal((4, 16)).astype(np.float32)

    db = VectorDB(dim=16, index_type="flat", metric="l2")
    ids = db.add_batch(vecs)
    assert ids == list(range(50))

    batch_hits = db.search_batch(queries, k=5)
    assert len(batch_hits) == 4
    assert all(len(per_query) == 5 for per_query in batch_hits)


def test_api_create_add_search_batch_flow():
    rng = np.random.default_rng(77)
    vecs = rng.standard_normal((40, 8)).astype(np.float32)
    queries = rng.standard_normal((3, 8)).astype(np.float32)

    client = TestClient(create_app())

    create = client.post(
        "/indexes",
        json={"name": "demo", "dim": 8, "index_type": "flat", "metric": "l2"},
    )
    assert create.status_code == 200
    assert create.json()["size"] == 0

    add_batch = client.post(
        "/indexes/demo/vectors/batch",
        json={"vectors": vecs.tolist()},
    )
    assert add_batch.status_code == 200
    assert len(add_batch.json()["vec_ids"]) == 40

    search_batch = client.post(
        "/indexes/demo/search/batch",
        json={"queries": queries.tolist(), "k": 5},
    )
    assert search_batch.status_code == 200
    payload = search_batch.json()
    assert len(payload["results"]) == 3
    assert all(len(per_query) == 5 for per_query in payload["results"])

    listed = client.get("/indexes")
    assert listed.status_code == 200
    assert listed.json()[0]["size"] == 40


def test_api_background_ingestion_queue():
    rng = np.random.default_rng(99)
    vecs = rng.standard_normal((60, 8)).astype(np.float32)

    client = TestClient(create_app())
    assert client.post(
        "/indexes",
        json={"name": "bg", "dim": 8, "index_type": "flat", "metric": "l2"},
    ).status_code == 200

    queued = client.post(
        "/indexes/bg/ingest/queue",
        json={"vectors": vecs.tolist()},
    )
    assert queued.status_code == 200
    job_id = queued.json()["job_id"]

    deadline = time.time() + 3.0
    status = None
    while time.time() < deadline:
        resp = client.get(f"/indexes/bg/ingest/jobs/{job_id}")
        assert resp.status_code == 200
        status = resp.json()["status"]
        if status == "completed":
            break
        time.sleep(0.02)
    assert status == "completed"

    listed = client.get("/indexes")
    assert listed.status_code == 200
    assert listed.json()[0]["size"] == 60


def test_concurrent_search_during_ingestion():
    rng = np.random.default_rng(1234)
    initial = rng.standard_normal((80, 8)).astype(np.float32)
    queued = rng.standard_normal((120, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32).tolist()

    client = TestClient(create_app())
    assert client.post(
        "/indexes",
        json={"name": "conc", "dim": 8, "index_type": "flat", "metric": "l2"},
    ).status_code == 200
    assert client.post(
        "/indexes/conc/vectors/batch",
        json={"vectors": initial.tolist()},
    ).status_code == 200
    queued_resp = client.post(
        "/indexes/conc/ingest/queue",
        json={"vectors": queued.tolist()},
    )
    assert queued_resp.status_code == 200
    job_id = queued_resp.json()["job_id"]

    errors: list[str] = []

    def _search_worker() -> None:
        for _ in range(40):
            r = client.post("/indexes/conc/search", json={"query": query, "k": 5})
            if r.status_code != 200:
                errors.append(f"status={r.status_code}")
            else:
                body = r.json()
                if len(body["results"]) == 0:
                    errors.append("empty_results")

    threads = [threading.Thread(target=_search_worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    flush = client.post("/indexes/conc/ingest/flush?timeout_s=5")
    assert flush.status_code == 200
    assert flush.json()["drained"] is True

    st_resp = client.get(f"/indexes/conc/ingest/jobs/{job_id}")
    assert st_resp.status_code == 200
    assert st_resp.json()["status"] == "completed"
