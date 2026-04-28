"""
Roaring bitmap metadata index.

One BitMap per (field, value) pair. Predicate evaluation is a chain of
bitwise AND/OR operations — O(N/64) rather than O(N) linear scan.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Any
from pyroaring import BitMap


Filter = dict[str, Any]


class BitmapIndex:
    def __init__(self) -> None:
        # bitmaps[field][value] = BitMap of vector IDs
        self._bitmaps: dict[str, dict[Any, BitMap]] = defaultdict(dict)
        self._all_ids: BitMap = BitMap()

    def add(self, vec_id: int, metadata: dict[str, Any]) -> None:
        self._all_ids.add(vec_id)
        for field, value in metadata.items():
            key = _coerce(value)
            if key not in self._bitmaps[field]:
                self._bitmaps[field][key] = BitMap()
            self._bitmaps[field][key].add(vec_id)

    def remove(self, vec_id: int, metadata: dict[str, Any]) -> None:
        self._all_ids.discard(vec_id)
        for field, value in metadata.items():
            key = _coerce(value)
            bm = self._bitmaps.get(field, {}).get(key)
            if bm is not None:
                bm.discard(vec_id)

    def query(self, filter: Filter | None) -> BitMap:
        """Return a BitMap of IDs satisfying all predicates (AND semantics)."""
        if not filter:
            return BitMap(self._all_ids)

        result: BitMap | None = None
        for field, value in filter.items():
            key = _coerce(value)
            bm = self._bitmaps.get(field, {}).get(key, BitMap())
            result = bm if result is None else result & bm

        return result if result is not None else BitMap()

    def query_any(self, field: str, values: list[Any]) -> BitMap:
        """Return IDs where field equals any of the given values (OR semantics)."""
        result: BitMap = BitMap()
        for value in values:
            key = _coerce(value)
            bm = self._bitmaps.get(field, {}).get(key, BitMap())
            result |= bm
        return result

    @property
    def total(self) -> int:
        return len(self._all_ids)


def _coerce(value: Any) -> str | int | float:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
