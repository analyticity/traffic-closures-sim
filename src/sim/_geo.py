"""Shared low-level geometry helpers used across multiple packages."""
from __future__ import annotations

import math
from typing import Optional


def bearing_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Bearing in degrees (0 = north, 90 = east) from (x1, y1) to (x2, y2)."""
    return math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360.0


def bearing_diff(a: float, b: float) -> float:
    """Absolute angular difference in degrees (0–180)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def bearing_from_line(geom) -> Optional[float]:
    """Bearing (0–360) from first to last vertex of a LineString-like geometry."""
    try:
        coords = list(geom.coords)
        if len(coords) < 2:
            return None
        return bearing_deg(coords[0][0], coords[0][1], coords[-1][0], coords[-1][1])
    except Exception:
        return None
