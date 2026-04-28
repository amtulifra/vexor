"""
Vexor binary persistence format.

Layout:
  [ HEADER: 64 bytes ]
    magic:    8 bytes  b"VEXOR001"
    version:  4 bytes  uint32
    index:    4 bytes  uint32  (HNSW=1, IVF=2, IVFPQ=3, KDT=4, FLAT=5)
    n:        8 bytes  uint64  (number of vectors)
    dim:      4 bytes  uint32
    params:   36 bytes (index-specific packed params)

  [ VECTORS: n * dim * 4 bytes, float32 ]
    CRC32:    4 bytes

  [ GRAPH/STRUCTURE: variable, JSON ]
    length:   8 bytes  uint64
    payload:  <length> bytes
    CRC32:    4 bytes

  [ METADATA: variable, JSON ]
    length:   8 bytes  uint64
    payload:  <length> bytes
    CRC32:    4 bytes
"""

from __future__ import annotations
import json
import pickle
import struct
import zlib
from pathlib import Path
import numpy as np

_MAGIC = b"VEXOR001"
_VERSION = 1

INDEX_TYPE = {"hnsw": 1, "ivf": 2, "ivfpq": 3, "kdtree": 4, "flat": 5}
INDEX_TYPE_INV = {v: k for k, v in INDEX_TYPE.items()}

_HEADER_FMT = "!8sIIQ I 36s"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def save_index(index, path: str | Path, index_type: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vectors, structure, metadata = _extract_index_data(index)

    n, dim = (vectors.shape[0], vectors.shape[1]) if vectors.ndim == 2 else (0, 0)
    type_id = INDEX_TYPE.get(index_type, 0)

    header = struct.pack(
        _HEADER_FMT,
        _MAGIC,
        _VERSION,
        type_id,
        n,
        dim,
        b"\x00" * 36,
    )

    vec_bytes = vectors.astype(np.float32).tobytes()
    vec_crc = struct.pack("!I", zlib.crc32(vec_bytes) & 0xFFFFFFFF)

    struct_bytes = json.dumps(structure, default=_json_default).encode()
    struct_crc = struct.pack("!I", zlib.crc32(struct_bytes) & 0xFFFFFFFF)

    meta_bytes = json.dumps(metadata).encode()
    meta_crc = struct.pack("!I", zlib.crc32(meta_bytes) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(header)
        f.write(vec_bytes)
        f.write(vec_crc)
        f.write(struct.pack("!Q", len(struct_bytes)))
        f.write(struct_bytes)
        f.write(struct_crc)
        f.write(struct.pack("!Q", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(meta_crc)


def load_index(path: str | Path):
    path = Path(path)
    with open(path, "rb") as f:
        header_raw = f.read(_HEADER_SIZE)
        magic, version, type_id, n, dim, _ = struct.unpack(_HEADER_FMT, header_raw)

        if magic != _MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic!r}")

        vec_bytes = f.read(n * dim * 4)
        stored_crc = struct.unpack("!I", f.read(4))[0]
        if zlib.crc32(vec_bytes) & 0xFFFFFFFF != stored_crc:
            raise ValueError("Vector section CRC32 mismatch — file may be corrupt.")
        vectors = np.frombuffer(vec_bytes, dtype=np.float32).reshape(n, dim).copy()

        struct_len = struct.unpack("!Q", f.read(8))[0]
        struct_bytes = f.read(struct_len)
        struct_crc = struct.unpack("!I", f.read(4))[0]
        if zlib.crc32(struct_bytes) & 0xFFFFFFFF != struct_crc:
            raise ValueError("Structure section CRC32 mismatch.")
        structure = json.loads(struct_bytes.decode())

        meta_len = struct.unpack("!Q", f.read(8))[0]
        meta_bytes = f.read(meta_len)
        meta_crc = struct.unpack("!I", f.read(4))[0]
        if zlib.crc32(meta_bytes) & 0xFFFFFFFF != meta_crc:
            raise ValueError("Metadata section CRC32 mismatch.")
        metadata = json.loads(meta_bytes.decode())

    return {
        "index_type": INDEX_TYPE_INV.get(type_id, "unknown"),
        "vectors": vectors,
        "structure": structure,
        "metadata": metadata,
    }


def _extract_index_data(index) -> tuple[np.ndarray, dict, list]:
    from vexor.indexes.flat import FlatIndex
    from vexor.indexes.hnsw import HNSWIndex
    from vexor.indexes.ivf import IVFIndex
    from vexor.indexes.kdtree import KDTreeIndex

    if isinstance(index, FlatIndex):
        vectors = np.stack(index._vectors) if index._vectors else np.empty((0, 0), dtype=np.float32)
        return vectors, {"type": "flat"}, list(index._metadata)

    if isinstance(index, HNSWIndex):
        if not index._vectors:
            return np.empty((0, 0), dtype=np.float32), {}, []
        sorted_ids = sorted(index._vectors.keys())
        vectors = np.stack([index._vectors[i] for i in sorted_ids])
        structure = {
            "layers": {str(k): {str(l): v for l, v in lm.items()}
                       for k, lm in index._layers.items()},
            "max_layer": {str(k): v for k, v in index._max_layer.items()},
            "entry_point": index._entry_point,
            "top_layer": index._top_layer,
            "deleted": list(index._deleted),
            "M": index._M,
            "ef_construction": index._ef_construction,
            "ef_search": index._ef_search,
        }
        metadata = [index._metadata.get(i, {}) for i in sorted_ids]
        return vectors, structure, metadata

    if isinstance(index, IVFIndex):
        vectors = np.stack(index._vectors) if index._vectors else np.empty((0, 0), dtype=np.float32)
        structure = {
            "centroids": index._centroids.tolist() if index._centroids is not None else None,
            "inverted_lists": index._inverted_lists,
            "nlist": index._nlist,
            "nprobe": index._nprobe,
        }
        return vectors, structure, list(index._metadata)

    vectors = np.stack(index._vectors) if hasattr(index, "_vectors") and index._vectors else np.empty((0, 0), dtype=np.float32)
    return vectors, {}, []


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    return str(obj)
