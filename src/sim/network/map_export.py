"""Network map visualisation constants and PNG export helpers.

The palette, DPI, and figure size are loaded from
``SIM_DEFAULTS["network"]["map_export"]`` so every map in the pipeline
shares a consistent visual style.  Other modules (``zoning``,
``supernetwork``) import these constants for their own map figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt

from sim.defaults import SIM_DEFAULTS

_MAP_CFG = SIM_DEFAULTS["network"]["map_export"]

NETWORK_MAP_PALETTE: Dict[str, str] = _MAP_CFG["palette"]
NETWORK_MAP_EXPORT_DPI: int = _MAP_CFG["dpi"]
NETWORK_MAP_EXPORT_FIGSIZE: Tuple[float, float] = tuple(_MAP_CFG["figsize"])


def save_network_links_map_png(
    links_gdf: gpd.GeoDataFrame,
    bbox_gdf: gpd.GeoDataFrame,
    png_path: Path,
    *,
    title: str,
    links_color: str,
    dpi: int = NETWORK_MAP_EXPORT_DPI,
    figsize: Tuple[float, float] = NETWORK_MAP_EXPORT_FIGSIZE,
) -> None:
    """Render *links_gdf* with a *bbox_gdf* frame and save as PNG."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=NETWORK_MAP_PALETTE["figure"])
    ax.set_facecolor(NETWORK_MAP_PALETTE["figure"])
    title_pt = max(14.0, min(24.0, figsize[0] * 1.05))
    links_gdf.plot(ax=ax, color=links_color, linewidth=0.25, zorder=1)
    bbox_gdf.boundary.plot(ax=ax, color=NETWORK_MAP_PALETTE["bbox"], linewidth=3.5, zorder=3)
    ax.set_title(title, color=NETWORK_MAP_PALETTE["title"], fontsize=title_pt)
    ax.set_axis_off()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=NETWORK_MAP_PALETTE["figure"],
        pad_inches=0.05,
    )
    plt.close(fig)
