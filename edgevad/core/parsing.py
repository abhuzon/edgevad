"""CLI argument parsing helpers."""

from __future__ import annotations

import re
from typing import List, Optional


def parse_classes(s: str) -> Optional[List[int]]:
    s = str(s).strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None
    parts = [p.strip() for p in s.split(",")]
    out = []
    for p in parts:
        if p == "":
            continue
        out.append(int(p))
    return out if out else None


def extract_scene_id(name: str) -> Optional[str]:
    """Extract scene ID prefix from a ShanghaiTech video/clip name.

    Examples:
        "01_001.avi" -> "01"
        "01_0014"    -> "01"
        "12_003.avi" -> "12"
        "something"  -> None
    """
    stem = name.rsplit(".", 1)[0] if "." in name else name
    m = re.match(r"^(\d+)_", stem)
    if m:
        return m.group(1)
    return None
