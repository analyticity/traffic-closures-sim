#!/usr/bin/env python3
"""Create A1 poster (841×594mm landscape) for Excel@FIT.

Uses matplotlib to produce a high-quality, print-ready PDF with all figures,
diagrams, and table embedded. 3-column layout matching the plan.

Requires: matplotlib, Pillow
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np

BASE = Path("/home/akankovs/diplomka/simulation")
OUT = BASE / "excel_materials"
SCREENSHOTS = OUT / "screenshots"
DIAGRAMS = OUT / "diagrams"
EXISTING_IMAGES = BASE / "outputs" / "baseline"

A1_W_MM = 841
A1_H_MM = 594
A1_W_IN = A1_W_MM / 25.4
A1_H_IN = A1_H_MM / 25.4

C_HEADER_BG = "#3A4442"
C_SECTION_BG = "#E8E4D9"
C_ACCENT = "#CA6F1E"
C_TEXT = "#000000"
C_SUBTLE = "#70747D"
C_WHITE = "#FFFFFF"

COL_LEFT = 0.02
COL_MID = 0.35
COL_RIGHT = 0.68
COL_W = 0.30


def load_img(path: str | Path) -> np.ndarray | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return np.array(Image.open(p))
    except Exception:
        return None


def add_image(ax, img_array, x, y, w, h, label=""):
    """Place an image in axes coordinates."""
    if img_array is None:
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.003",
            facecolor="#DFDCD3", edgecolor="#B2B4AB", linewidth=0.5,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, "[image not found]",
                ha="center", va="center", fontsize=6, color="#A6A6A6",
                transform=ax.transAxes)
    else:
        img_h, img_w = img_array.shape[:2]
        target_aspect = w / h
        img_aspect = img_w / img_h

        if img_aspect > target_aspect:
            crop_w = int(img_h * target_aspect)
            offset = (img_w - crop_w) // 2
            img_array = img_array[:, offset:offset + crop_w]
        else:
            crop_h = int(img_w / target_aspect)
            offset = (img_h - crop_h) // 2
            img_array = img_array[offset:offset + crop_h, :]

        ax.imshow(
            img_array,
            extent=[x, x + w, y, y + h],
            aspect="auto",
            transform=ax.transAxes,
            zorder=2,
            interpolation="lanczos",
        )

    if label:
        ax.text(x + w / 2, y - 0.004, label,
                ha="center", va="top", fontsize=5, fontstyle="italic",
                color=C_SUBTLE, transform=ax.transAxes)


def section_header(ax, x, y, w, title, color=C_HEADER_BG):
    box = FancyBboxPatch(
        (x, y - 0.022), w, 0.022,
        boxstyle="round,pad=0.002",
        facecolor=color, edgecolor="none",
        transform=ax.transAxes, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + 0.005, y - 0.011, title,
            ha="left", va="center", fontsize=7, fontweight="bold",
            color=C_WHITE, transform=ax.transAxes, zorder=4)


def text_block(ax, x, y, w, text, fontsize=5.5, color=C_TEXT, bold=False):
    ax.text(x, y, text,
            ha="left", va="top", fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=color, transform=ax.transAxes,
            wrap=True,
            linespacing=1.35,
            fontfamily="sans-serif",
            bbox=dict(boxstyle="square,pad=0", facecolor="none", edgecolor="none",
                      mutation_aspect=w / 0.01) if False else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with open(BASE / "outputs/baseline/demand/od_summary.json") as f:
        od = json.load(f)
    with open(BASE / "outputs/baseline/demand/validation_report.json") as f:
        val = json.load(f)
    with open(BASE / "outputs/baseline/demand/calibration_report.json") as f:
        cal = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(A1_W_IN, A1_H_IN), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(C_WHITE)

    # ===== HEADER =====
    header_h = 0.065
    header_box = FancyBboxPatch(
        (0, 1 - header_h), 1, header_h,
        boxstyle="square,pad=0",
        facecolor=C_HEADER_BG, edgecolor="none",
        transform=ax.transAxes, zorder=5,
    )
    ax.add_patch(header_box)

    ax.text(0.5, 1 - header_h / 2 + 0.008,
            "Analysis of the Impact of Traffic Closures on Transport Using Simulations",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_WHITE, transform=ax.transAxes, zorder=6)
    ax.text(0.5, 1 - header_h / 2 - 0.012,
            "Bc. Adam Kaňkovský  •  Supervisor: Ing. Magdaléna Ondrušková  •  FIT VUT 2026  •  Excel@FIT",
            ha="center", va="center", fontsize=7,
            color="#A6A6A6", transform=ax.transAxes, zorder=6)

    content_top = 1 - header_h - 0.015
    content_bot = 0.015

    # ========= COLUMN 1 — Problem, Network, Demand =========
    cx = COL_LEFT
    cy = content_top

    # Motivation
    section_header(ax, cx, cy, COL_W, "MOTIVATION")
    cy -= 0.030
    ax.text(cx, cy,
            "Traffic closures redistribute flows network-wide, causing congestion\n"
            "far from the closed area. This open-source pipeline enables What-If\n"
            "analysis of planned closures in Brno using only open data (OSM,\n"
            "Czech census SLDB 2021, CSD 2025 traffic counts).",
            ha="left", va="top", fontsize=5, color=C_TEXT,
            transform=ax.transAxes, linespacing=1.4)
    cy -= 0.055

    # Figure 1: Pipeline diagram
    section_header(ax, cx, cy, COL_W, "Figure 1: Pipeline Overview", color=C_ACCENT)
    cy -= 0.025
    img_pipeline = load_img(DIAGRAMS / "pipeline_diagram.png")
    img_h = 0.22
    add_image(ax, img_pipeline, cx, cy - img_h, COL_W, img_h,
              "16-step deterministic pipeline: 5 color-coded phases")
    cy -= img_h + 0.020

    # Figure 2: Zones + Network
    section_header(ax, cx, cy, COL_W, "Figure 2: Model Area — 219 TAZ + 28 Gateways")
    cy -= 0.025
    img_zones = load_img(EXISTING_IMAGES / "zones/zones_map.png")
    img_h = 0.18
    add_image(ax, img_zones, cx, cy - img_h, COL_W, img_h,
              "191 internal + 28 gateway zones, Brno + 2km")
    cy -= img_h + 0.020

    # Figure 3: Supernetwork
    section_header(ax, cx, cy, COL_W, "Figure 3: National Supernetwork")
    cy -= 0.025
    img_super = load_img(EXISTING_IMAGES / "supernetwork/supernetwork_overview.png")
    img_h = 0.18
    add_image(ax, img_super, cx, cy - img_h, COL_W, img_h,
              "6117 places → 9 gateways, detour ratio ≤ 1.25, 73k veh/day")
    cy -= img_h + 0.010

    # ========= COLUMN 2 — Scenario Engine =========
    cx = COL_MID
    cy = content_top

    # Figure 4: Scenario methodology
    section_header(ax, cx, cy, COL_W, "Figure 4: Scenario Methodology", color="#8E44AD")
    cy -= 0.025
    img_scenario = load_img(DIAGRAMS / "scenario_diagram.png")
    img_h = 0.16
    add_image(ax, img_scenario, cx, cy - img_h, COL_W, img_h,
              "Shared inputs → in-memory graph mod → per-link delta")
    cy -= img_h + 0.020

    # Figure 5: Baseline V/C
    section_header(ax, cx, cy, COL_W, "Figure 5: Baseline Traffic (V/C Ratio)")
    cy -= 0.025
    img_baseline = load_img(SCREENSHOTS / "screenshot_01_baseline_vc.png")
    img_h = 0.15
    add_image(ax, img_baseline, cx, cy - img_h, COL_W, img_h,
              "LOS colored links: green (A) → dark red (F)")
    cy -= img_h + 0.020

    # Figure 6: Scenario closure
    section_header(ax, cx, cy, COL_W, "Figure 6: Scenario — Closure Applied")
    cy -= 0.025
    img_closure = load_img(SCREENSHOTS / "screenshot_02_scenario_panel.png")
    img_h = 0.15
    add_image(ax, img_closure, cx, cy - img_h, COL_W, img_h,
              "Interactive link selection, full/lane closure types")
    cy -= img_h + 0.020

    # Figure 7: Delta
    section_header(ax, cx, cy, COL_W, "Figure 7: Delta — Traffic Redistribution")
    cy -= 0.025
    img_delta = load_img(SCREENSHOTS / "screenshot_03_delta_view.png")
    img_h = 0.15
    add_image(ax, img_delta, cx, cy - img_h, COL_W, img_h,
              "Red = increase, Blue = decrease, width ∝ |Δvol|")
    cy -= img_h + 0.010

    # ========= COLUMN 3 — Calibration, Diagnostics, Results =========
    cx = COL_RIGHT
    cy = content_top

    # Figure 8: Reports / calibration
    section_header(ax, cx, cy, COL_W, "Figure 8: Calibration & Reports")
    cy -= 0.025
    img_reports = load_img(SCREENSHOTS / "screenshot_04_reports.png")
    img_h = 0.13
    add_image(ax, img_reports, cx, cy - img_h, COL_W, img_h,
              "ODME convergence over 6 iterations")
    cy -= img_h + 0.020

    # Table 1
    section_header(ax, cx, cy, COL_W, "Table 1: Model Metrics", color="#27AE60")
    cy -= 0.028

    pent = val["pentlogram"]
    sl = val["screenlines"]
    table_data = [
        ["Metric", "Value"],
        ["Network", "52k links, 47k nodes, 219 zones"],
        ["Daily demand", f"{od['segments']['combined_daily']:,.0f} vehicles"],
        ["  Commuting", f"{od['segments']['commuting']:,.0f}"],
        ["  Through-traffic", f"{od['segments']['external_through_data']:,.0f}"],
        ["Count stations", f"{pent['matched']} matched"],
        ["Overall bias", f"{pent['bias_pct']:.1f}%"],
        ["D1 east ratio", f"{sl['D1_east']['ratio']:.2f}"],
        ["Krenova ratio", f"{sl['Krenova_radial']['ratio']:.2f}"],
        ["ODME iterations", f"{cal['iterations']}"],
    ]

    row_h = 0.014
    for ri, row in enumerate(table_data):
        ry = cy - ri * row_h
        bg = "#E8E4D9" if ri % 2 == 0 else C_WHITE
        if ri == 0:
            bg = C_HEADER_BG
        box = FancyBboxPatch(
            (cx, ry - row_h), COL_W, row_h,
            boxstyle="square,pad=0",
            facecolor=bg, edgecolor="#B2B4AB", linewidth=0.3,
            transform=ax.transAxes, zorder=3,
        )
        ax.add_patch(box)
        tc = C_WHITE if ri == 0 else C_TEXT
        fw = "bold" if ri == 0 else "normal"
        ax.text(cx + 0.005, ry - row_h / 2, row[0],
                ha="left", va="center", fontsize=4.5, fontweight=fw,
                color=tc, transform=ax.transAxes, zorder=4)
        ax.text(cx + COL_W - 0.005, ry - row_h / 2, row[1],
                ha="right", va="center", fontsize=4.5, fontweight=fw,
                color=tc, transform=ax.transAxes, zorder=4)

    cy -= len(table_data) * row_h + 0.020

    # Figure 9: Corridor diagnosis
    section_header(ax, cx, cy, COL_W, "Figure 9: Corridor Diagnosis")
    cy -= 0.025
    img_corr = load_img(SCREENSHOTS / "screenshot_06_corridor_diagnosis.png")
    img_h = 0.13
    add_image(ax, img_corr, cx, cy - img_h, COL_W, img_h,
              "Free-flow vs forced path, time/distance comparison")
    cy -= img_h + 0.020

    # Figure 10: Through traffic
    section_header(ax, cx, cy, COL_W, "Figure 10: Through-Traffic Analysis")
    cy -= 0.025
    img_through = load_img(SCREENSHOTS / "screenshot_07_through_traffic.png")
    img_h = 0.13
    add_image(ax, img_through, cx, cy - img_h, COL_W, img_h,
              "External-through share, gateway markers, screenlines")
    cy -= img_h + 0.020

    # Contributions
    section_header(ax, cx, cy, COL_W, "KEY CONTRIBUTIONS", color="#27AE60")
    cy -= 0.028
    contributions = [
        "• Reproducible 16-step open-source pipeline\n  (OSM + Czech open datasets only)",
        "• National supernetwork for data-driven\n  through-traffic (6117 places, detour-filtered)",
        "• Spiess-style ODME calibration with\n  select-link screenlines + gateway damping",
        "• Non-destructive scenario engine for\n  interactive What-If closure analysis",
        "• Rich diagnostic API: corridor explanation,\n  bias clustering, through-traffic overlay",
    ]
    for ci, contrib in enumerate(contributions):
        ax.text(cx, cy - ci * 0.025, contrib,
                ha="left", va="top", fontsize=4.5, color=C_TEXT,
                transform=ax.transAxes, linespacing=1.3)

    # Save
    poster_path = OUT / "poster_A1.pdf"
    fig.savefig(poster_path, bbox_inches="tight", dpi=200, facecolor=C_WHITE)
    fig.savefig(OUT / "poster_A1.png", bbox_inches="tight", dpi=100, facecolor=C_WHITE)
    plt.close()
    print(f"Poster saved: {poster_path}")
    print(f"Poster preview: {OUT / 'poster_A1.png'}")


if __name__ == "__main__":
    main()
