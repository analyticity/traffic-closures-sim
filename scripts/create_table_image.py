#!/usr/bin/env python3
"""Create Table 1 (Model Metrics) as a standalone high-quality image for the poster."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = Path("/home/akankovs/diplomka/simulation")
OUT = BASE / "excel_materials" / "diagrams"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with open(BASE / "outputs/baseline/demand/od_summary.json") as f:
        od = json.load(f)
    with open(BASE / "outputs/baseline/demand/validation_report.json") as f:
        val = json.load(f)
    with open(BASE / "outputs/baseline/demand/calibration_report.json") as f:
        cal = json.load(f)

    pent = val["pentlogram"]
    sl = val["screenlines"]

    rows = [
        ("Network", "52,382 links · 47,073 nodes"),
        ("Zones", "219 TAZ (191 internal + 28 gateways)"),
        ("Daily demand", f"{od['segments']['combined_daily']:,.0f} vehicles"),
        ("  Commuting (SLDB)", f"{od['segments']['commuting']:,.0f}"),
        ("  Through-traffic", f"{od['segments']['external_through_data']:,.0f}"),
        ("Through-traffic pairs", "68 gateway pairs"),
        ("Count stations matched", f"{pent['matched']:,}"),
        ("Sum modeled / observed", f"{pent['sum_modeled']/1e6:.1f}M / {pent['sum_observed']/1e6:.1f}M"),
        ("Overall bias", f"{pent['bias_pct']:+.2f}%"),
        ("Assignment", "BFW (AequilibraE), BPR VDF"),
        ("ODME iterations", f"{cal['iterations']}"),
        ("Screenline D1 east", f"ratio {sl['D1_east']['ratio']:.3f}"),
        ("Screenline Krenova", f"ratio {sl['Krenova_radial']['ratio']:.3f}"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    n = len(rows)
    rh = 0.065  # row height
    y_start = 0.95
    col1_x = 0.03
    col2_x = 0.55

    # Header
    header_y = y_start
    box = FancyBboxPatch((0.01, header_y - rh), 0.98, rh,
                          boxstyle="round,pad=0.005",
                          facecolor="#3A4442", edgecolor="none",
                          transform=ax.transAxes, zorder=3)
    ax.add_patch(box)
    ax.text(col1_x, header_y - rh / 2, "Metric",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color="white", transform=ax.transAxes, zorder=5)
    ax.text(col2_x, header_y - rh / 2, "Value",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color="white", transform=ax.transAxes, zorder=5)

    for i, (metric, value) in enumerate(rows):
        ry = y_start - (i + 1) * rh
        bg = "#E8E4D9" if i % 2 == 0 else "white"
        is_indent = metric.startswith("  ")

        box = FancyBboxPatch((0.01, ry - rh), 0.98, rh,
                              boxstyle="square,pad=0",
                              facecolor=bg, edgecolor="#B2B4AB", linewidth=0.5,
                              transform=ax.transAxes, zorder=2)
        ax.add_patch(box)

        mx = col1_x + (0.03 if is_indent else 0)
        ax.text(mx, ry - rh / 2, metric.strip(),
                ha="left", va="center", fontsize=8,
                fontweight="normal" if is_indent else "medium",
                color="#70747D" if is_indent else "#000000",
                transform=ax.transAxes, zorder=5)
        ax.text(col2_x, ry - rh / 2, value,
                ha="left", va="center", fontsize=8,
                color="#000000",
                transform=ax.transAxes, zorder=5)

    # Caption
    ax.text(0.5, y_start - (n + 1.5) * rh,
            "Table 1: Model Metrics",
            ha="center", va="top", fontsize=9, fontstyle="italic",
            color="#70747D", transform=ax.transAxes)

    fig.savefig(OUT / "table1_metrics.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "table1_metrics.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {OUT / 'table1_metrics.png'}")
    print(f"Saved: {OUT / 'table1_metrics.pdf'}")


if __name__ == "__main__":
    main()
