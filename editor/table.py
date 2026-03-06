from __future__ import annotations

from typing import List, Dict, Tuple


def compute_grid_dimensions(cells: List[Dict]) -> Tuple[int, int]:
    rows = max((c.get("row", 0) for c in cells), default=-1) + 1
    cols = max((c.get("col", 0) for c in cells), default=-1) + 1
    return rows, cols
