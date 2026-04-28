"""
WAL recovery tests: WAL replay must reproduce correct index state after crash.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path

from vexor.storage.wal import WriteAheadLog
from vexor.db import VectorDB


N = 100
DIM = 16
K = 5

rng = np.random.default_rng(99)
_VECS = rng.standard_normal((N, DIM)).astype(np.float32)
_QUERIES = rng.standard_normal((10, DIM)).astype(np.float32)


class TestWriteAheadLog:
    def test_roundtrip_insert(self, tmp_path):
        wal = WriteAheadLog(tmp_path / "test.wal")
        vec = _VECS[0]
        meta = {"source": "test", "year": 2024}
        wal.append_insert(0, vec, meta)

        entries = wal.replay()
        assert len(entries) == 1
        entry = entries[0]
        assert entry["op"] == 1
        assert entry["vec_id"] == 0
        assert entry["metadata"] == meta
        np.testing.assert_allclose(entry["vector"], vec, rtol=1e-5)

    def test_roundtrip_delete(self, tmp_path):
        wal = WriteAheadLog(tmp_path / "test.wal")
        wal.append_delete(42)

        entries = wal.replay()
        assert len(entries) == 1
        assert entries[0]["op"] == 2
        assert entries[0]["vec_id"] == 42

    def test_multiple_entries(self, tmp_path):
        wal = WriteAheadLog(tmp_path / "test.wal")
        for i in range(10):
            wal.append_insert(i, _VECS[i], {"idx": i})

        entries = wal.replay()
        assert len(entries) == 10
        for i, e in enumerate(entries):
            assert e["vec_id"] == i

    def test_truncate_clears_wal(self, tmp_path):
        wal_path = tmp_path / "test.wal"
        wal = WriteAheadLog(wal_path)
        wal.append_insert(0, _VECS[0], {})
        wal.truncate()
        assert not wal_path.exists()
        assert wal.replay() == []

    def test_corrupt_entry_is_skipped(self, tmp_path):
        wal_path = tmp_path / "test.wal"
        wal = WriteAheadLog(wal_path)
        wal.append_insert(0, _VECS[0], {"ok": True})

        with open(wal_path, "ab") as f:
            f.write(b"\xff\xff\xff\xff\xff")

        wal.append_insert(1, _VECS[1], {"ok": True})

        entries = wal.replay()
        assert entries[0]["vec_id"] == 0


class TestVectorDBRecovery:
    def test_wal_replays_after_crash(self, tmp_path):
        wal_path = str(tmp_path / "db.wal")
        db = VectorDB(DIM, index_type="hnsw", metric="l2",
                      M=8, ef_construction=100, wal_path=wal_path)

        for v in _VECS[:50]:
            db.add(v)

        # Simulate crash: create fresh index from same WAL
        db2 = VectorDB(DIM, index_type="hnsw", metric="l2",
                       M=8, ef_construction=100, wal_path=wal_path)
        replayed = db2.recover_from_wal()
        assert replayed == 50

        for q in _QUERIES:
            results = db2.search(q, k=K)
            assert len(results) == K

    def test_save_truncates_wal(self, tmp_path):
        wal_path = tmp_path / "db.wal"
        snap_path = tmp_path / "db.snap"
        db = VectorDB(DIM, index_type="hnsw", metric="l2",
                      M=8, ef_construction=100, wal_path=str(wal_path))
        for v in _VECS[:20]:
            db.add(v)
        db.save(str(snap_path))
        assert not wal_path.exists()
