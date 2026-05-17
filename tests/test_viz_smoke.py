"""Smoke checks for visualization assets."""

from __future__ import annotations

import py_compile
from pathlib import Path


def test_viz_pages_compile():
    root = Path(__file__).resolve().parents[1]
    pages = sorted((root / "viz" / "pages").glob("*.py"))
    assert pages, "No viz pages found."

    for page in pages:
        py_compile.compile(str(page), doraise=True)
