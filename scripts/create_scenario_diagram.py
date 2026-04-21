#!/usr/bin/env python3
"""Create Baseline vs Scenario comparison diagram for Excel@FIT poster (Figure 4).

Shows shared network+demand → assignment → SPLIT → baseline / scenario → delta.
Output: high-quality PDF + PNG in excel_materials/diagrams/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("/home/akankovs/diplomka/simulation/excel_materials/diagrams")

C = {
    "blue":   ("#3A4442", "#E8E4D9"),
    "green":  ("#27AE60", "#DFDCD3"),
    "gray":   ("#545454", "#DFDCD3"),
    "orange": ("#CA6F1E", "#E8E4D9"),
    "red":    ("#C0392B", "#E8E4D9"),
    "purple": ("#8E44AD", "#E8E4D9"),
}


def rbox(ax, x, y, w, h, ec, fc, lw=1.3, zorder=3):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                        facecolor=fc, edgecolor=ec, linewidth=lw,
                        transform=ax.transAxes, zorder=zorder)
    ax.add_patch(p)


def label(ax, x, y, title, subtitle="", ec="#000000", fs_t=9, fs_s=7):
    if subtitle:
        ax.text(x, y + 0.015, title, ha="center", va="center",
                fontsize=fs_t, fontweight="bold", color=ec,
                transform=ax.transAxes, zorder=5)
        ax.text(x, y - 0.02, subtitle, ha="center", va="center",
                fontsize=fs_s, color="#545454",
                transform=ax.transAxes, zorder=5, linespacing=1.25)
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=fs_t, fontweight="bold", color=ec,
                transform=ax.transAxes, zorder=5)


def arrow(ax, x1, y1, x2, y2, color="#70747D", lw=1.6, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    bw, bh = 0.17, 0.10  # box width, height

    # ── Row 1: Shared inputs (top) ──
    # Road Network
    rbox(ax, 0.04, 0.78, bw, bh, *C["blue"])
    label(ax, 0.04 + bw/2, 0.78 + bh/2, "Road Network", "OSM Brno + 2 km\n52k links, 47k nodes", C["blue"][0])

    # OD Matrix
    rbox(ax, 0.28, 0.78, bw, bh, *C["green"])
    label(ax, 0.28 + bw/2, 0.78 + bh/2, "OD Matrix", "SLDB + supernetwork\n765k daily vehicles", C["green"][0])

    # ── Row 2: BFW Assignment (shared) ──
    rbox(ax, 0.14, 0.58, 0.21, bh, *C["gray"])
    label(ax, 0.14 + 0.105, 0.58 + bh/2, "BFW Assignment", "AequilibraE\nBPR volume-delay", C["gray"][0])

    # Arrows: inputs → assignment
    arrow(ax, 0.04 + bw/2, 0.78, 0.14 + 0.105, 0.58 + bh, C["blue"][0])
    arrow(ax, 0.28 + bw/2, 0.78, 0.14 + 0.105, 0.58 + bh, C["green"][0])

    # ── SPLIT ──
    sx = 0.44
    sy = 0.62
    ax.text(sx, sy, "SPLIT", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#A6A6A6",
            transform=ax.transAxes, zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", fc="#E8E4D9", ec="#B2B4AB", lw=1))

    arrow(ax, 0.14 + 0.21, 0.63, sx - 0.03, sy, "#A6A6A6")

    # ── Baseline (upper branch) ──
    bx_base = 0.54
    by_base = 0.74
    rbox(ax, bx_base, by_base, bw, bh, *C["blue"])
    label(ax, bx_base + bw/2, by_base + bh/2, "Baseline", "Full network\nassignment", C["blue"][0])
    arrow(ax, sx + 0.03, sy + 0.02, bx_base, by_base + bh/2, C["blue"][0])

    # ── Scenario (lower branch) ──
    bx_scen = 0.54
    by_scen = 0.40
    rbox(ax, bx_scen, by_scen, bw, bh + 0.02, *C["orange"])
    label(ax, bx_scen + bw/2, by_scen + (bh+0.02)/2, "Scenario",
          "Modified graph\ncapacity = 0.001\nor lane ratio", C["orange"][0])
    arrow(ax, sx + 0.03, sy - 0.02, bx_scen, by_scen + (bh+0.02)/2, C["orange"][0])

    # ── User Input ──
    bx_user = 0.30
    by_user = 0.28
    rbox(ax, bx_user, by_user, bw, bh, *C["red"])
    label(ax, bx_user + bw/2, by_user + bh/2, "User Input", "Select links on map\nclosure / lane reduction", C["red"][0])
    arrow(ax, bx_user + bw, by_user + bh/2, bx_scen, by_scen + 0.03, C["red"][0])

    # ── Delta Analysis ──
    bx_delta = 0.78
    by_delta = 0.55
    rbox(ax, bx_delta, by_delta, bw, bh + 0.04, *C["purple"])
    label(ax, bx_delta + bw/2, by_delta + (bh+0.04)/2, "Delta\nAnalysis", "", C["purple"][0], fs_t=10)

    # Arrows: baseline → delta, scenario → delta
    arrow(ax, bx_base + bw, by_base + bh/2, bx_delta, by_delta + bh + 0.02, C["blue"][0])
    arrow(ax, bx_scen + bw, by_scen + (bh+0.02)/2, bx_delta, by_delta + 0.02, C["orange"][0])

    # ── Delta outputs ──
    outputs = [
        ("Δ volume", 0.82),
        ("Δ V/C", 0.74),
        ("Δ travel time", 0.66),
    ]
    for olabel, oy in outputs:
        # Output pill to the right
        ox = 0.965
        ax.text(ox, oy + 0.11, olabel, ha="right", va="center",
                fontsize=7.5, fontweight="bold", color=C["purple"][0],
                transform=ax.transAxes, zorder=5,
                bbox=dict(boxstyle="round,pad=0.08", fc=C["purple"][1], ec=C["purple"][0], lw=0.8))

    # Single arrow from delta box to outputs area
    arrow(ax, bx_delta + bw, by_delta + (bh+0.04)/2,
          0.965 - 0.08, by_delta + (bh+0.04)/2, C["purple"][0])

    # ── Annotation ──
    ax.text(0.62, 0.35, "non-destructive:\nproject files unchanged",
            ha="center", va="top", fontsize=6.5, fontstyle="italic",
            color="#A6A6A6", transform=ax.transAxes, linespacing=1.3)

    # ── Caption ──
    ax.text(0.5, 0.02,
            "Figure 4: Scenario Methodology — Shared inputs, diverging at in-memory graph modification, per-link delta output",
            ha="center", va="bottom", fontsize=8, fontstyle="italic",
            color="#70747D", transform=ax.transAxes)

    fig.savefig(OUT / "scenario_diagram.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "scenario_diagram.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {OUT / 'scenario_diagram.pdf'}")
    print(f"Saved: {OUT / 'scenario_diagram.png'}")


if __name__ == "__main__":
    main()
