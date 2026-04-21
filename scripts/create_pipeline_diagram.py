#!/usr/bin/env python3
"""Create pipeline flowchart diagram for posters / thesis (no caption drawn).

Five grouped stages aligned with ``run.py`` (individual CLI steps are merged
here for layout). Copy is intentionally high-level — see README for detail.
Figure size targets ~A1+ poster inset (~600 mm wide at 300 dpi). Canvas is
transparent; phase bands and step cards remain opaque for readability.
Output: PDF + PNG in excel_materials/diagrams/
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("/home/akankovs/diplomka/simulation/excel_materials/diagrams")

# Print-oriented canvas (~600 mm × ~340 mm @ 300 dpi when saved at SAVE_DPI)
FIGSIZE_IN = (23.62, 13.4)
SAVE_DPI = 300

PHASE_COLORS = {
    "net":   ("#3A4442", "#E8E4D9"),
    "zone":  ("#148F77", "#DFDCD3"),
    "dem":   ("#27AE60", "#E8E4D9"),
    "cal":   ("#CA6F1E", "#DFDCD3"),
    "serve": ("#8E44AD", "#E8E4D9"),
}
DATA_EDGE = "#545454"

# Step card: title vs. body (axes fraction; keep in sync with _narrative_height_axes)
STEP_TITLE_TOP_PAD = 0.019
STEP_TITLE_LINE_H = 0.024
STEP_TITLE_DESC_GAP = 0.014
STEP_DESC_TOP_PAD = 0.008
STEP_DESC_LINE_H = 0.02

# Inner horizontal padding of white step card (matches rbox x/w margins)
def _axes_inner_width_pt(fig: plt.Figure, ax: plt.Axes, width_axes_frac: float) -> float:
    pos = ax.get_position()
    w_in = width_axes_frac * pos.width * fig.get_figwidth()
    return w_in * 72.0


def _chars_per_line(inner_pt: float, fontsize: float, rel_width: float = 0.52) -> int:
    return max(8, int(inner_pt / max(fontsize * rel_width, 1e-6)))


def _wrap_multipara(text: str, cpl: int) -> str:
    chunks: list[str] = []
    for para in text.split("\n"):
        para = " ".join(para.split())
        if not para:
            continue
        chunks.append(
            textwrap.fill(
                para,
                width=cpl,
                break_long_words=True,
                break_on_hyphens=True,
            )
        )
    return "\n".join(chunks)


def _line_count(s: str) -> int:
    s = s.strip()
    return 0 if not s else s.count("\n") + 1


def _wrap_sources(sources: list[str], cpl: int) -> tuple[str, int]:
    if not sources:
        return "", 0
    blocks: list[str] = []
    lines = 0
    for src in sources:
        w = _wrap_multipara(src.replace("\n", " "), cpl)
        blocks.append(w)
        lines += max(1, _line_count(w))
    return "\n".join(blocks), lines


def _source_band_height_axes(n_lines: int) -> float:
    if n_lines <= 0:
        return 0.0
    return 0.014 + n_lines * 0.023


def _narrative_height_axes(n_title_lines: int, n_desc_lines: int) -> float:
    """Axes fraction for title + gap + description (must match draw positions)."""
    nt = max(1, n_title_lines)
    nd = max(1, n_desc_lines)
    title_h = STEP_TITLE_TOP_PAD + nt * STEP_TITLE_LINE_H
    desc_h = STEP_DESC_TOP_PAD + nd * STEP_DESC_LINE_H
    return title_h + STEP_TITLE_DESC_GAP + desc_h


def _setup_montserrat() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Montserrat", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rbox(ax, x, y, w, h, ec, fc, lw=1.2, zorder=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                        facecolor=fc, edgecolor=ec, linewidth=lw,
                        transform=ax.transAxes, zorder=zorder)
    ax.add_patch(p)
    return p


def flow_chevron(
    ax,
    x: float,
    y: float,
    *,
    direction: str,
    color: str,
    fontsize: float = 23.0,
) -> None:
    """Single glyph connector: no shaft, only '>' (rotated for vertical flow)."""
    rot = -90.0 if direction == "down" else 0.0
    ax.text(
        x,
        y,
        ">",
        ha="center",
        va="center",
        rotation=rot,
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
        zorder=8,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _setup_montserrat()

    fig, ax = plt.subplots(figsize=FIGSIZE_IN, dpi=SAVE_DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)

    # ── Layout constants ──
    n_phases = 5
    px0 = 0.056         # first phase left edge
    pw  = 0.152         # phase column width (narrower to free horizontal gap)
    pg  = 0.042         # gap between phases (chevron markers, no stems)
    top = 0.92
    bot = 0.06

    # Data hints mirror config/sim.yaml `datasets` + OSM; keep lines short (wrap fills width).
    phases = [
        ("net", "NETWORK", [
            ("build-network",
             "OSM road graph into\nAequilibraE project",
             ["Geofabrik OSM PBF"]),
            ("normalize-network",
             "Capacities, speeds, VDF prep;\nbaseline closures if configured\n(fetch-data supplies inputs)",
             ["NDIC"]),
        ]),
        ("zone", "ZONES &\nSUPERNETWORK", [
            ("build-zones",
             "TAZ polygons, gateway zones,\ncentroid connectors",
             ["OSM ORP (admin 9)"]),
            ("build-supernetwork",
             "Coarse outer network for\nexternal / through demand\nat gateways",
             ["OSM PBF (CZ)", "ČÚZK RÚIAN"]),
        ]),
        ("dem", "DEMAND", [
            ("build-demand",
             "Seed OD matrix\n(commuting, gateways,\noptional synthetic rows)",
             ["ČSÚ SLDB dojížďka", "ČSÚ SLDB populace"]),
            ("distribute",
             "Gravity + IPF on the seed\nmatrix (network skims when\nassign-warm-skims was used)",
             []),
        ]),
        ("cal", "ASSIGNMENT\n& FIT", [
            ("assign",
             "Traffic assignment on the\nmanaged (urban) network",
             []),
            ("calibrate",
             "Demand vs. observed volumes\n(classic loop or ODME);\noptional tune-supply pass",
             ["Pentlogram"]),
            ("validate",
             "Independent checks vs. CSD;\noptional learn-profile step",
             ["RSD CSD"]),
        ]),
        ("serve", "SERVE", [
            ("strip-closures",
             "Drop baseline closures from\nthe network for clean baseline\nand what-if scenarios",
             []),
            ("serve",
             "Read-only API + maps;\nclosure / lane scenarios,\ndelta views in the UI",
             ["NDIC"]),
        ]),
    ]

    for pi, (key, label, steps) in enumerate(phases):
        ec, fc = PHASE_COLORS[key]
        cx = px0 + pi * (pw + pg)

        # Phase background
        rbox(ax, cx - 0.005, bot, pw + 0.01, top - bot, ec, fc, lw=2.2, zorder=1)

        n = len(steps)
        step_region_top = top - 0.088
        step_region_bot = bot + 0.03
        avail = step_region_top - step_region_bot
        gap = 0.052 if n > 1 else 0.0
        gaps_total = gap * max(n - 1, 0)

        inner_pt = _axes_inner_width_pt(fig, ax, pw - 0.016)
        fs_phase, fs_name, fs_desc, fs_src = 18.5, 17.8, 14.0, 13.2

        base_phase = _chars_per_line(inner_pt, fs_phase, 0.58)
        base_name = _chars_per_line(inner_pt, fs_name, 0.56)
        base_desc = _chars_per_line(inner_pt, fs_desc, 0.50)
        base_src = _chars_per_line(inner_pt, fs_src, 0.48)

        step_layout: list[dict] = []
        label_wrapped = label
        cpl_phase = base_phase
        cpl_name, cpl_desc, cpl_src = base_name, base_desc, base_src

        # Widen CPL (longer lines → fewer rows) until narrative fits the
        # uniform band h_narr that fills the column (like pre-wrap layout).
        for widen in range(55):
            cpl_phase = base_phase + widen // 3
            cpl_name = base_name + widen // 4
            cpl_desc = base_desc + widen // 2
            cpl_src = base_src + widen // 3
            label_wrapped = _wrap_multipara(label, cpl_phase)
            step_layout.clear()
            extras: list[float] = []
            for sname, sdesc, sources in steps:
                wn = _wrap_multipara(sname, cpl_name)
                wd = _wrap_multipara(sdesc, cpl_desc)
                ws, n_src_lines = _wrap_sources(sources, cpl_src)
                src_h = (
                    _source_band_height_axes(n_src_lines) if sources else 0.0
                )
                extras.append(src_h)
                step_layout.append(
                    {
                        "wn": wn,
                        "wd": wd,
                        "ws": ws,
                        "src_h": src_h,
                        "has_src": bool(sources),
                    }
                )
            sum_ex = sum(extras)
            h_narr = (avail - gaps_total - sum_ex) / n
            h_narr = max(0.076, min(0.26, h_narr))
            max_narr_need = max(
                _narrative_height_axes(_line_count(sl["wn"]), _line_count(sl["wd"]))
                for sl in step_layout
            )
            if max_narr_need <= h_narr * 1.02:
                break

        sum_ex = sum(extras)
        h_narr = (avail - gaps_total - sum_ex) / n
        h_narr = max(0.076, min(0.26, h_narr))
        heights_unscaled = [h_narr + ex for ex in extras]
        total_used = sum(heights_unscaled) + gaps_total
        col_scale = 1.0
        if total_used > avail + 1e-9:
            col_scale = (avail - gaps_total) / sum(heights_unscaled)
        heights = [h * col_scale for h in heights_unscaled]

        ax.text(
            cx + pw / 2,
            top - 0.02,
            label_wrapped,
            ha="center",
            va="top",
            fontsize=fs_phase,
            fontweight="bold",
            color=ec,
            transform=ax.transAxes,
            zorder=5,
            linespacing=1.06,
        )

        y_top = step_region_top
        prev_step_sy: float | None = None
        for si, sl in enumerate(step_layout):
            sh_i = heights[si]
            src_h = sl["src_h"] * col_scale
            sy = y_top - sh_i
            y_top = sy - gap

            rbox(ax, cx + 0.008, sy, pw - 0.016, sh_i, ec, "white", lw=1.25, zorder=3)

            line_y = sy + src_h if sl["has_src"] and src_h > 0 else None

            if sl["has_src"] and line_y is not None:
                ax.plot(
                    [cx + 0.014, cx + pw - 0.014],
                    [line_y, line_y],
                    transform=ax.transAxes,
                    color=DATA_EDGE,
                    lw=1.45,
                    linestyle=(0, (1, 3)),
                    clip_on=False,
                    zorder=5,
                )
                band_mid = sy + src_h * 0.5
                ax.text(
                    cx + pw / 2,
                    band_mid,
                    sl["ws"],
                    ha="center",
                    va="center",
                    fontsize=fs_src,
                    color=DATA_EDGE,
                    transform=ax.transAxes,
                    zorder=6,
                    linespacing=1.16,
                )

            upper_top = sy + sh_i
            ax.text(
                cx + pw / 2,
                upper_top - STEP_TITLE_TOP_PAD,
                sl["wn"],
                ha="center",
                va="top",
                fontsize=fs_name,
                fontweight="bold",
                color=ec,
                transform=ax.transAxes,
                zorder=5,
                linespacing=1.06,
            )
            nt = _line_count(sl["wn"])
            desc_y = (
                upper_top
                - STEP_TITLE_TOP_PAD
                - STEP_TITLE_DESC_GAP
                - nt * STEP_TITLE_LINE_H
            )
            ax.text(
                cx + pw / 2,
                desc_y,
                sl["wd"],
                ha="center",
                va="top",
                fontsize=fs_desc,
                color="#545454",
                transform=ax.transAxes,
                zorder=5,
                linespacing=1.1,
            )

            if prev_step_sy is not None:
                y_mid = 0.5 * (prev_step_sy + sy + sh_i)
                flow_chevron(
                    ax,
                    cx + pw / 2,
                    y_mid,
                    direction="down",
                    color=ec,
                    fontsize=24.5,
                )
            prev_step_sy = sy

        if pi < n_phases - 1:
            mid_y = (step_region_top + step_region_bot) / 2
            x_mid = cx + pw + pg / 2
            flow_chevron(
                ax,
                x_mid,
                mid_y,
                direction="right",
                color="#7A7C78",
                fontsize=26.5,
            )

    save_kw = dict(
        bbox_inches="tight",
        dpi=SAVE_DPI,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    fig.savefig(OUT / "pipeline_diagram.pdf", **save_kw)
    fig.savefig(OUT / "pipeline_diagram.png", **save_kw)
    plt.close()
    print(f"Saved: {OUT / 'pipeline_diagram.pdf'}")
    print(f"Saved: {OUT / 'pipeline_diagram.png'}")


if __name__ == "__main__":
    main()
