"""Zone map visualisation (matplotlib PNG export)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from aequilibrae import Project

from sim.network.map_export import (
    NETWORK_MAP_EXPORT_DPI,
    NETWORK_MAP_EXPORT_FIGSIZE,
    NETWORK_MAP_PALETTE,
)
from sim.zoning.geo import force_to_target_crs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (aligned with scripts/create_pipeline_diagram.py accents)
# ---------------------------------------------------------------------------

_ZONE_EDGE = "#148F77"
_NET_FG = "#3A4442"
_BRNO_MESTO_ORANGE = "#CA6F1E"
_EXIT_PURPLE = "#8E44AD"

_DISTRICT_BRNO_OUTLINE: Dict[str, str] = {
    "Brno-město": _BRNO_MESTO_ORANGE,
    "Brno-venkov": _ZONE_EDGE,
}
_DISTRICT_FALLBACK_OUTLINE: List[str] = [
    _BRNO_MESTO_ORANGE,
    _ZONE_EDGE,
    _NET_FG,
]


def _short_district_label(raw: Any) -> str:
    """Short legend text; recognizes Brno districts from OSM ``place`` strings."""
    s = str(raw).strip()
    sl = s.lower()
    if "brno-město" in sl or "brno-mesto" in sl:
        return "Brno-město"
    if "brno-venkov" in sl:
        return "Brno-venkov"
    if len(s) <= 44:
        return s
    return s[:41] + "…"


def _prepare_internal_district_groups(
    internal_z: gpd.GeoDataFrame,
) -> List[Tuple[str, gpd.GeoDataFrame, str]]:
    """Split internal zones by ``source_place`` (or ``source_rank``).

    Returns ``[(legend_label, subset_gdf, outline_hex), ...]``.
    """
    if internal_z.empty:
        return []

    has_place = "source_place" in internal_z.columns and bool(internal_z["source_place"].notna().any())
    work = internal_z.copy()
    if has_place:
        work["_dkey"] = work["source_place"].fillna(work["source_rank"].astype(str)).astype(str)
    else:
        work["_dkey"] = work["source_rank"].astype(str)

    order = work.groupby("_dkey")["source_rank"].min().sort_values()
    out: List[Tuple[str, gpd.GeoDataFrame, str]] = []
    for i, key in enumerate(order.index.tolist()):
        sub = work[work["_dkey"] == key].drop(columns=["_dkey"], errors="ignore")
        short = _short_district_label(key)
        if has_place and short in _DISTRICT_BRNO_OUTLINE:
            edge = _DISTRICT_BRNO_OUTLINE[short]
        else:
            edge = _DISTRICT_FALLBACK_OUTLINE[i % len(_DISTRICT_FALLBACK_OUTLINE)]
        label = _short_district_label(key) if has_place else f"Internal zones (source rank {key})"
        out.append((label, sub, edge))
    return out


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_map_png(
    project: Project,
    zones: gpd.GeoDataFrame,
    centroids: gpd.GeoDataFrame,
    model_area_bbox: Any,
    output_dir: Path,
    filename: str = "zones_map.png",
    title_suffix: str = "",
    dpi: Optional[int] = None,
    debug_corridors: Optional[gpd.GeoDataFrame] = None,
    debug_points: Optional[gpd.GeoDataFrame] = None,
    *,
    crs_epsg: Optional[int] = None,
) -> None:
    """Render a diagnostics PNG with zones, centroids, network, and optional gateway debug layers."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        logger.warning("matplotlib not installed, skipping map export")
        return

    out_dpi = int(dpi) if dpi is not None else NETWORK_MAP_EXPORT_DPI
    fig_w, fig_h = NETWORK_MAP_EXPORT_FIGSIZE
    title_pt = max(14.0, min(24.0, float(fig_w) * 1.05))

    if zones.empty:
        logger.warning("no zones, skipping map export")
        return

    if zones.crs is not None and zones.crs.to_epsg() is not None:
        target_epsg = int(zones.crs.to_epsg())
    elif crs_epsg is not None:
        target_epsg = int(crs_epsg)
    else:
        raise ValueError("export_map_png: zones have no CRS and crs_epsg was not provided")

    links = project.network.links.data
    if "geometry" not in links.columns or len(links) == 0:
        logger.warning("links have no geometry, skipping map export")
        return

    links_gdf = gpd.GeoDataFrame(links, geometry="geometry", crs=getattr(links, "crs", None))
    if "link_type" in links_gdf.columns:
        links_gdf = links_gdf[links_gdf["link_type"].astype(str) != "centroid_connector"].copy()

    links_gdf = force_to_target_crs(links_gdf, target_epsg, name="network.links(for map)")

    z = zones.copy()
    c = centroids.copy()

    if "is_external" not in z.columns:
        z["is_external"] = 0
    z["is_external"] = pd.to_numeric(z["is_external"], errors="coerce").fillna(0).astype(int)

    if "is_external" not in c.columns:
        c["is_external"] = 0
    c["is_external"] = pd.to_numeric(c["is_external"], errors="coerce").fillna(0).astype(int)

    internal_zones = z[z["is_external"] == 0].copy()
    external_zones = z[z["is_external"] == 1].copy()

    internal_centroids = c[c["is_external"] == 0].copy()
    external_centroids = c[c["is_external"] == 1].copy()

    sx, sy, sx2, sy2 = links_gdf.total_bounds
    zx, zy, zx2, zy2 = z.total_bounds

    minx0 = min(float(sx), float(zx))
    miny0 = min(float(sy), float(zy))
    maxx0 = max(float(sx2), float(zx2))
    maxy0 = max(float(sy2), float(zy2))

    margin_ratio = 0.02
    w = maxx0 - minx0
    h = maxy0 - miny0
    margin_x = max(w * margin_ratio, 50.0)
    margin_y = max(h * margin_ratio, 50.0)

    minx = minx0 - margin_x
    maxx = maxx0 + margin_x
    miny = miny0 - margin_y
    maxy = maxy0 + margin_y

    aoi_gdf = gpd.GeoDataFrame({"geometry": [model_area_bbox]}, crs=f"EPSG:{target_epsg}")

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), facecolor=NETWORK_MAP_PALETTE["figure"])
    ax.set_facecolor(NETWORK_MAP_PALETTE["figure"])

    aoi_gdf.plot(ax=ax, color="#DFDCD3", alpha=0.22, zorder=1)

    district_groups = _prepare_internal_district_groups(internal_zones)
    legend_internal_patches: List[Any] = []
    legend_centroid_lines: List[Any] = []

    _link_overlay_color = "#141414"
    _link_overlay_lw = 0.72

    for label, sub_z, edge in district_groups:
        if sub_z.empty:
            continue
        sub_z.plot(
            ax=ax,
            facecolor="none",
            edgecolor=edge,
            linewidth=2.85,
            alpha=1.0,
            zorder=3,
        )
        legend_internal_patches.append(
            Patch(
                facecolor="none",
                edgecolor=edge,
                linewidth=2.85,
                label=f"{label} ({len(sub_z)})",
            ),
        )

    if not external_zones.empty:
        external_zones.plot(
            ax=ax,
            facecolor="none",
            edgecolor=_EXIT_PURPLE,
            linewidth=3.25,
            alpha=1.0,
            zorder=5,
        )

    links_gdf.plot(
        ax=ax,
        color=_link_overlay_color,
        linewidth=_link_overlay_lw,
        alpha=0.94,
        zorder=12,
    )

    for label, sub_z, edge in district_groups:
        if sub_z.empty or internal_centroids.empty or "zone_id" not in internal_centroids.columns:
            continue
        zids = sub_z["zone_id"]
        cent_sub = internal_centroids[internal_centroids["zone_id"].isin(zids)]
        if cent_sub.empty:
            continue
        cent_sub.plot(
            ax=ax,
            color=edge,
            edgecolors=NETWORK_MAP_PALETTE["figure"],
            linewidths=1.45,
            markersize=30,
            marker="o",
            zorder=14,
        )
        legend_centroid_lines.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=edge,
                linestyle="None",
                markersize=8,
                markeredgecolor=NETWORK_MAP_PALETTE["figure"],
                markeredgewidth=1.2,
                label=f"{label} centroids ({len(cent_sub)})",
            ),
        )

    if not internal_centroids.empty and (
        not district_groups or "zone_id" not in internal_centroids.columns
    ):
        internal_centroids.plot(
            ax=ax,
            color=_NET_FG,
            edgecolors=NETWORK_MAP_PALETTE["figure"],
            linewidths=1.35,
            markersize=28,
            marker="o",
            zorder=14,
        )
        legend_centroid_lines.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=_NET_FG,
                linestyle="None",
                markersize=8,
                markeredgecolor=NETWORK_MAP_PALETTE["figure"],
                markeredgewidth=1.0,
                label=f"Internal centroids ({len(internal_centroids)})",
            ),
        )

    if not external_centroids.empty:
        external_centroids.plot(
            ax=ax,
            color=_EXIT_PURPLE,
            edgecolors=_NET_FG,
            linewidths=1.25,
            markersize=56,
            marker="X",
            zorder=15,
        )

    _gw_corridor_color = "#C0392B"
    _gw_terminal_color = "#F39C12"
    _gw_anchor_color = "#C0392B"

    terminals = gpd.GeoDataFrame()
    anchors = gpd.GeoDataFrame()

    if debug_corridors is not None and not debug_corridors.empty:
        dbg = debug_corridors.copy()
        if dbg.crs is None:
            dbg = dbg.set_crs(epsg=target_epsg, allow_override=True)
        if dbg.crs.to_epsg() != target_epsg:
            dbg = dbg.to_crs(epsg=target_epsg)

        dbg.plot(
            ax=ax,
            color=_gw_corridor_color,
            linewidth=3.0,
            alpha=0.88,
            zorder=18,
        )

    if debug_points is not None and not debug_points.empty:
        dbg_pts = debug_points.copy()
        if dbg_pts.crs is None:
            dbg_pts = dbg_pts.set_crs(epsg=target_epsg, allow_override=True)
        if dbg_pts.crs.to_epsg() != target_epsg:
            dbg_pts = dbg_pts.to_crs(epsg=target_epsg)

        terminals = dbg_pts[dbg_pts["kind"] == "terminal"].copy() if "kind" in dbg_pts.columns else dbg_pts.iloc[0:0]
        anchors = dbg_pts[dbg_pts["kind"] == "anchor"].copy() if "kind" in dbg_pts.columns else dbg_pts.iloc[0:0]

        if not terminals.empty:
            terminals.plot(
                ax=ax,
                color=_gw_terminal_color,
                markersize=40,
                marker="o",
                zorder=19,
            )

        if not anchors.empty:
            anchors.plot(
                ax=ax,
                color=_gw_anchor_color,
                markersize=95,
                marker="X",
                zorder=20,
            )

    aoi_gdf.boundary.plot(
        ax=ax,
        color=NETWORK_MAP_PALETTE["bbox"],
        linewidth=3.5,
        zorder=22,
    )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Network + TAZ zones ({len(zones)}) + AOI{title_suffix}",
        fontsize=title_pt,
        fontweight="bold",
        color=NETWORK_MAP_PALETTE["title"],
        pad=20,
    )

    legend_handles: List[Any] = [
        Line2D(
            [0],
            [0],
            color=_link_overlay_color,
            linewidth=3.0,
            label="Network (overlay)",
        ),
    ]
    legend_handles.extend(legend_internal_patches)
    if not external_zones.empty:
        legend_handles.append(
            Patch(
                facecolor="none",
                edgecolor=_EXIT_PURPLE,
                linewidth=2.8,
                label=f"Exit zones ({len(external_zones)})",
            ),
        )
    legend_handles.extend(legend_centroid_lines)
    if not external_centroids.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="X",
                color=_EXIT_PURPLE,
                linestyle="None",
                markersize=10,
                markeredgecolor=_NET_FG,
                markeredgewidth=0.9,
                label=f"Exit / gateway centroids ({len(external_centroids)})",
            ),
        )
    legend_handles.append(
        Patch(
            facecolor="#DFDCD3",
            edgecolor=NETWORK_MAP_PALETTE["bbox"],
            alpha=0.35,
            linewidth=1.5,
            label="AOI",
        ),
    )
    if debug_corridors is not None and not debug_corridors.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=_gw_corridor_color,
                linewidth=3.0,
                label="Gateway corridors",
            ),
        )
    if debug_points is not None and not debug_points.empty:
        n_term = len(terminals) if not terminals.empty else 0
        n_anch = len(anchors) if not anchors.empty else 0
        if n_term > 0:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=_gw_terminal_color,
                    linestyle="None",
                    markersize=8,
                    label=f"Gateway terminals ({n_term})",
                ),
            )
        if n_anch > 0:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="X",
                    color=_gw_anchor_color,
                    linestyle="None",
                    markersize=10,
                    label=f"Gateway anchors ({n_anch})",
                ),
            )

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.14, 1.02),
        borderaxespad=0.75,
        framealpha=0.96,
        fontsize=max(9.5, title_pt * 0.52),
    )

    ax.set_axis_off()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / filename
    fig.savefig(
        png_path,
        dpi=out_dpi,
        bbox_inches="tight",
        facecolor=NETWORK_MAP_PALETTE["figure"],
        pad_inches=0.55,
    )
    plt.close(fig)
    logger.info("map: %s", png_path)
