"""
Write-Ahead Log (WAL) for crash recovery.

Entry layout:
  magic:     4 bytes  b"VXL1"
  seq:       8 bytes  uint64 (monotonic sequence number)
  op:        1 byte   (INSERT=1, DELETE=2)
  vector_id: 8 bytes  uint64
  dim:       4 bytes  uint32
  meta_len:  4 bytes  uint32
  vector:    dim * 4 bytes float32
  metadata:  meta_len bytes (JSON)
  checksum:  4 bytes  CRC32 of all bytes before checksum

On startup: if a WAL file exists, replay it to recover inserts that
occurred after the last full snapshot. After a successful save, truncate.
"""

from __future__ import annotations
import json
import os
import struct
import zlib
from pathlib import Path
from typing import Any
import numpy as np

_OP_INSERT: int = 1
_OP_DELETE: int = 2
_MAGIC = b"VXL1"
_HEADER_FMT = "!4sQ B Q I I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class WriteAheadLog:
    def __init__(self, path: str | Path, max_segment_bytes: int = 64 * 1024 * 1024) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_segment_bytes = max_segment_bytes
        self._checkpoint_path = self._path.with_suffix(self._path.suffix + ".checkpoint")
        self._next_seq = self._last_sequence() + 1

    def append_insert(self, vec_id: int, vector: np.ndarray, metadata: dict[str, Any]) -> None:
        self._write_entry(_OP_INSERT, vec_id, vector, metadata)

    def append_delete(self, vec_id: int) -> None:
        self._write_entry(_OP_DELETE, vec_id, np.empty(0, dtype=np.float32), {})

    def replay(
        self,
        *,
        idempotent: bool = False,
        advance_checkpoint: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Return valid WAL entries in order.

        - Corrupt entries are skipped when possible.
        - Partial writes at EOF terminate replay cleanly.
        - When idempotent=True, entries at or before checkpoint are skipped.
        """
        segments = self._segments()
        if not segments:
            return []

        entries = []
        seen_seq: set[int] = set()
        checkpoint_seq = self._read_checkpoint() if idempotent else -1
        max_seen = checkpoint_seq

        for seg in segments:
            with open(seg, "rb") as f:
                while True:
                    try:
                        entry, status = self._read_entry(f)
                        if status == "eof":
                            break
                        if status == "partial":
                            break
                        if entry is None:
                            continue
                        seq = int(entry["seq"])
                        if seq <= checkpoint_seq or seq in seen_seq:
                            continue
                        seen_seq.add(seq)
                        entries.append(entry)
                        if seq > max_seen:
                            max_seen = seq
                    except (struct.error, EOFError, json.JSONDecodeError):
                        break

        if advance_checkpoint and max_seen >= 0:
            self._write_checkpoint(max_seen)
        return entries

    def truncate(self) -> None:
        for seg in self._segments():
            seg.unlink(missing_ok=True)
        self._checkpoint_path.unlink(missing_ok=True)
        self._next_seq = 0

    def _write_entry(self, op: int, vec_id: int, vector: np.ndarray, metadata: dict) -> None:
        vec_bytes = vector.astype(np.float32).tobytes()
        meta_bytes = json.dumps(metadata).encode()
        payload = struct.pack(
            _HEADER_FMT,
            _MAGIC,
            self._next_seq,
            op,
            vec_id,
            len(vector),
            len(meta_bytes),
        )
        payload += vec_bytes + meta_bytes
        checksum = struct.pack("!I", zlib.crc32(payload) & 0xFFFFFFFF)
        entry = payload + checksum
        target = self._append_target(len(entry))
        with open(target, "ab") as f:
            f.write(entry)
            f.flush()
            os.fsync(f.fileno())
        self._next_seq += 1

    def _read_entry(self, f) -> tuple[dict[str, Any] | None, str]:
        header = f.read(_HEADER_SIZE)
        if not header:
            return None, "eof"
        if len(header) < _HEADER_SIZE:
            return None, "partial"

        magic, seq, op, vec_id, dim, meta_len = struct.unpack(_HEADER_FMT, header)
        if magic != _MAGIC:
            return None, "corrupt"

        vec_bytes = f.read(dim * 4)
        if len(vec_bytes) < dim * 4:
            return None, "partial"

        meta_bytes = f.read(meta_len)
        if len(meta_bytes) < meta_len:
            return None, "partial"

        checksum_bytes = f.read(4)
        if len(checksum_bytes) < 4:
            return None, "partial"

        payload = header + vec_bytes + meta_bytes
        stored_crc = struct.unpack("!I", checksum_bytes)[0]
        if zlib.crc32(payload) & 0xFFFFFFFF != stored_crc:
            return None, "corrupt"

        vector = np.frombuffer(vec_bytes, dtype=np.float32).copy() if dim > 0 else np.empty(0)
        metadata = json.loads(meta_bytes.decode())
        return {
            "seq": seq,
            "op": op,
            "vec_id": vec_id,
            "vector": vector,
            "metadata": metadata,
        }, "ok"

    def _segments(self) -> list[Path]:
        segments: list[Path] = []
        if self._path.exists():
            segments.append(self._path)
        rotated = sorted(
            [p for p in self._path.parent.glob(f"{self._path.name}.*") if p != self._checkpoint_path],
            key=self._segment_order,
        )
        segments.extend(rotated)
        return segments

    def _append_target(self, next_entry_bytes: int) -> Path:
        segments = self._segments()
        if not segments:
            return self._path
        current = segments[-1]
        if current.stat().st_size + next_entry_bytes <= self._max_segment_bytes:
            return current
        next_idx = self._segment_order(current) + 1
        return self._path.parent / f"{self._path.name}.{next_idx:06d}"

    def _segment_order(self, path: Path) -> int:
        if path == self._path:
            return 0
        suffix = path.name.rsplit(".", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    def _last_sequence(self) -> int:
        last = -1
        for seg in self._segments():
            with open(seg, "rb") as f:
                while True:
                    entry, status = self._read_entry(f)
                    if status in {"eof", "partial"}:
                        break
                    if entry is None:
                        continue
                    seq = int(entry["seq"])
                    if seq > last:
                        last = seq
        return last

    def _read_checkpoint(self) -> int:
        if not self._checkpoint_path.exists():
            return -1
        try:
            return int(self._checkpoint_path.read_text().strip())
        except ValueError:
            return -1

    def _write_checkpoint(self, seq: int) -> None:
        self._checkpoint_path.write_text(str(int(seq)))
