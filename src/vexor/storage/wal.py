"""
Write-Ahead Log (WAL) for crash recovery.

Entry layout:
  op:        1 byte  (INSERT=1, DELETE=2)
  vector_id: 8 bytes uint64
  dim:       4 bytes uint32
  vector:    dim * 4 bytes float32
  meta_len:  4 bytes uint32
  metadata:  meta_len bytes (JSON)
  checksum:  4 bytes CRC32

On startup: if a WAL file exists, replay it to recover inserts that
occurred after the last full snapshot. After a successful save, truncate.
"""

from __future__ import annotations
import json
import struct
import zlib
from pathlib import Path
from typing import Any
import numpy as np

_OP_INSERT: int = 1
_OP_DELETE: int = 2


class WriteAheadLog:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append_insert(self, vec_id: int, vector: np.ndarray, metadata: dict[str, Any]) -> None:
        self._write_entry(_OP_INSERT, vec_id, vector, metadata)

    def append_delete(self, vec_id: int) -> None:
        self._write_entry(_OP_DELETE, vec_id, np.empty(0, dtype=np.float32), {})

    def replay(self) -> list[dict[str, Any]]:
        """Return all valid entries in order. Entries with bad checksums are skipped."""
        if not self._path.exists():
            return []

        entries = []
        with open(self._path, "rb") as f:
            while True:
                try:
                    entry = self._read_entry(f)
                    if entry is None:
                        break
                    entries.append(entry)
                except (struct.error, EOFError, json.JSONDecodeError):
                    break
        return entries

    def truncate(self) -> None:
        if self._path.exists():
            self._path.unlink()

    def _write_entry(self, op: int, vec_id: int, vector: np.ndarray, metadata: dict) -> None:
        vec_bytes = vector.astype(np.float32).tobytes()
        meta_bytes = json.dumps(metadata).encode()
        payload = (
            struct.pack("!B Q I", op, vec_id, len(vector))
            + vec_bytes
            + struct.pack("!I", len(meta_bytes))
            + meta_bytes
        )
        checksum = struct.pack("!I", zlib.crc32(payload) & 0xFFFFFFFF)
        with open(self._path, "ab") as f:
            f.write(payload + checksum)

    def _read_entry(self, f) -> dict[str, Any] | None:
        header = f.read(13)
        if len(header) < 13:
            return None
        op, vec_id, dim = struct.unpack("!B Q I", header)
        vec_bytes = f.read(dim * 4)
        meta_len_bytes = f.read(4)
        if len(meta_len_bytes) < 4:
            return None
        meta_len = struct.unpack("!I", meta_len_bytes)[0]
        meta_bytes = f.read(meta_len)
        checksum_bytes = f.read(4)
        if len(checksum_bytes) < 4:
            return None

        payload = header + vec_bytes + meta_len_bytes + meta_bytes
        stored_crc = struct.unpack("!I", checksum_bytes)[0]
        if zlib.crc32(payload) & 0xFFFFFFFF != stored_crc:
            return None

        vector = np.frombuffer(vec_bytes, dtype=np.float32).copy() if dim > 0 else np.empty(0)
        metadata = json.loads(meta_bytes.decode())
        return {"op": op, "vec_id": vec_id, "vector": vector, "metadata": metadata}
