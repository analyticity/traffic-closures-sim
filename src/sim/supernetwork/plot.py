"""Supernetwork overview map export."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union

from sim.datasets.utils import ensure_dir
from sim.network.map_export import (
    NETWORK_MAP_EXPORT_DPI,
    NETWORK_MAP_EXPORT_FIGSIZE,
    NETWORK_MAP_PALETTE,
)


def _external_units_outside_model_area(units: gpd.GeoDataFrame, model_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Omit unit markers inside the model polygon to reduce map clutter."""
    if units.empty or model_area.empty:
        return units
    aoi = unary_union(model_area.geometry.values)
    if aoi is None or aoi.is_empty:
        return units
    inside = units.geometry.within(aoi)
    return units.loc[~inside].copy()


def plot_overview(edges_metric: gpd.GeoDataFrame, model_area: gpd.GeoDataFrame, gateways: gpd.GeoDataFrame, units: gpd.GeoDataFrame, out_png: Path) -> None:
    from matplotlib.lines import Line2D

    ensure_dir(out_png.parent)
    _base_w, _base_h = NETWORK_MAP_EXPORT_FIGSIZE
    fig_w = max(_base_w, 20.0)
    fig_h = max(_base_h, 15.0)
    title_pt = max(18.0, min(28.0, float(fig_w) * 1.1))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=NETWORK_MAP_PALETTE["figure"])
    ax.set_facecolor(NETWORK_MAP_PALETTE["figure"])

    model_area.boundary.plot(ax=ax, color=NETWORK_MAP_PALETTE["bbox"], linewidth=3.5, zorder=2)
    edges_metric.plot(
        ax=ax,
        color=NETWORK_MAP_PALETTE["links_after"],
        linewidth=0.35,
        alpha=0.92,
        zorder=1,
    )
    gateways.plot(ax=ax, color="#423E3A", markersize=36, marker="o", zorder=5)
    units_out = _external_units_outside_model_area(units, model_area)
    if not units_out.empty:
        units_out.plot(ax=ax, color="#70747D", markersize=8, marker="o", alpha=0.88, zorder=4)

    ax.set_title(
        "Supernetwork overview",
        color=NETWORK_MAP_PALETTE["title"],
        fontsize=title_pt,
        fontweight="bold",
        pad=12,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=NETWORK_MAP_PALETTE["links_after"],
            linewidth=2.5,
            label="Super-edges",
        ),
        Line2D(
            [0],
            [0],
            color=NETWORK_MAP_PALETTE["bbox"],
            linewidth=2.5,
            label="Model boundary",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#423E3A",
            linestyle="None",
            markersize=10,
            label="Gateways",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#70747D",
            linestyle="None",
            markersize=7,
            label="Connector units",
        ),
    ]
    legend_fs = max(13.0, title_pt * 0.58)
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        framealpha=0.96,
        fontsize=legend_fs,
        handlelength=2.0,
        labelspacing=0.6,
    )
    ax.set_axis_off()

    sx, sy, sx2, sy2 = edges_metric.total_bounds
    pad_x = (sx2 - sx) * 0.01
    pad_y = (sy2 - sy) * 0.01
    ax.set_xlim(sx - pad_x, sx2 + pad_x)
    ax.set_ylim(sy - pad_y, sy2 + pad_y)
    ax.set_aspect("equal", adjustable="box")

    fig.savefig(
        out_png,
        dpi=NETWORK_MAP_EXPORT_DPI,
        bbox_inches="tight",
        facecolor=NETWORK_MAP_PALETTE["figure"],
        pad_inches=0.15,
    )
    plt.close(fig)
