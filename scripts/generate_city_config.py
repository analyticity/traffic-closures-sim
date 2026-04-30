#!/usr/bin/env python3
"""Generate a minimal city configuration for the simulation pipeline.

Usage:
    python scripts/generate_city_config.py
    python scripts/generate_city_config.py --city "Olomouc" --okres "Olomouc" --kraj "Olomoucký kraj"

Creates config/<slug>/sim.yaml, locale.yaml, and screenlines.yaml
ready for:
    python run.py --config config/<slug>/sim.yaml build-network
"""
from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

# Czech regions (kraje) for CSD area_filter
_KRAJE = [
    "Hlavní město Praha",
    "Středočeský kraj",
    "Jihočeský kraj",
    "Plzeňský kraj",
    "Karlovarský kraj",
    "Ústecký kraj",
    "Liberecký kraj",
    "Královéhradecký kraj",
    "Pardubický kraj",
    "Kraj Vysočina",
    "Jihomoravský kraj",
    "Olomoucký kraj",
    "Zlínský kraj",
    "Moravskoslezský kraj",
]


def _slug(name: str) -> str:
    s = name.split(",")[0].strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{msg}{suffix}: ").strip()
    return val or default


def _prompt_int(msg: str, default: int) -> int:
    raw = _prompt(msg, str(default))
    try:
        return int(raw)
    except ValueError:
        print(f"  Invalid number, using default: {default}")
        return default


def _prompt_choice(msg: str, choices: list[str], default: str = "") -> str:
    print(f"\n{msg}")
    for i, c in enumerate(choices, 1):
        marker = " *" if c == default else ""
        print(f"  {i:2d}) {c}{marker}")
    raw = _prompt("Enter number or name", default)
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    for c in choices:
        if raw.lower() in c.lower():
            return c
    return raw


def generate(
    *,
    city: str,
    okres: str,
    kraj: str,
    total_daily_trips: int | str = "auto",
    gateways: list[str] | None = None,
    output_dir: Path,
) -> Path:
    slug = _slug(city)
    config_dir = output_dir / slug
    config_dir.mkdir(parents=True, exist_ok=True)

    city_name = city if "," in city else f"{city}, Czechia"
    okres_name = okres if "," in okres else f"Okres {okres}, Czechia"
    gw_list = gateways or []
    gw_yaml = str(gw_list) if gw_list else "[]"

    # --- sim.yaml ---
    sim_yaml = textwrap.dedent(f"""\
        project_path: "project/{slug}_aeq"

        osm:
          place_name: "{city_name}"
          # buffer_km: 2  # optional: km buffer around geocoded place (default 0). Omit or use 0 for full-place import.

        zoning:
          sources:
            - type: osm
              place: "{okres_name}"
              admin_level: "9"
          external_gateways:
            enabled: true
            whitelist: {gw_yaml}

        demand:
          segments:
            external_local:
              total_daily_trips: {total_daily_trips if isinstance(total_daily_trips, int) else '"auto"'}

        calibration:
          count_source: "csd_split"
          auto_screenlines:
            enabled: true
            gateway_screenlines: true

        datasets:
          enabled: true
          sources:
            commuting_sldb2021:
              filter:
                enabled: true
                origin:
                  keep_if_any_matches:
                    - field: "op_okres"
                      values: ["{okres}"]
                    - field: "op_orp"
                      values: ["{city.split(',')[0].strip()}"]
                destination:
                  keep_all: true
            calibration_brno_pentlogram_2024:
              enabled: false
            validation_csd2025_v2:
              usage:
                area_filter: {{ enabled: true, region_hint: "{kraj}" }}
    """)

    # --- locale.yaml ---
    locale_yaml = textwrap.dedent(f"""\
        # Locale for {city.split(',')[0].strip()}.
        # National defaults (holidays, road_classification, reference_speeds)
        # are built into the code.

        municipal_parts_to_cadastral: {{}}
    """)

    # --- screenlines.yaml ---
    screenlines_yaml = textwrap.dedent(f"""\
        # Screenlines for {city.split(',')[0].strip()}.
        # Auto-generated from gateways + CSD when calibration.auto_screenlines.enabled
        # is true. Add manual overrides below as needed.

        screenlines: []
    """)

    (config_dir / "sim.yaml").write_text(sim_yaml, encoding="utf-8")
    (config_dir / "locale.yaml").write_text(locale_yaml, encoding="utf-8")
    (config_dir / "screenlines.yaml").write_text(screenlines_yaml, encoding="utf-8")

    return config_dir


_PIPELINE_STEPS = [
    "build-network",
    "fetch-data",
    "normalize-network",
    "build-zones",
    "build-supernetwork",
    "build-demand",
    "assign",
    "calibrate-odme",
    "validate",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate city configuration for the simulation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--city", help='City name, e.g. "Olomouc" or "Olomouc, Czechia"')
    ap.add_argument("--okres", help='District (okres) name for SLDB filter, e.g. "Olomouc"')
    ap.add_argument("--kraj", help='Region (kraj) for CSD filter, e.g. "Olomoucký kraj"')
    ap.add_argument("--trips", type=int, default=None,
                        help="External-local daily trips (omit for auto-estimation from CSD)")
    ap.add_argument("--gateways", help='Comma-separated major gateway road refs, e.g. "D1,D2,I/43" (empty for auto-detect)')
    ap.add_argument("--output-dir", default="config", help="Parent dir for city configs")
    args = ap.parse_args()

    interactive = not args.city

    if interactive:
        print("=== City config generator ===\n")
        city = _prompt("City name (e.g. Olomouc)", "")
        if not city:
            print("City name is required.")
            return
    else:
        city = args.city

    city_short = city.split(",")[0].strip()
    slug = _slug(city)

    gateways: list[str] = []
    if interactive:
        okres = _prompt("Okres (district) for SLDB filter", city_short)
        kraj = _prompt_choice("Select kraj (region) for CSD:", _KRAJE)
        trips_str = _prompt(
            "External-local daily trips (number, or 'auto' for CSD estimation)",
            "auto",
        )
        trips: int | str = "auto"
        if trips_str.strip().lower() != "auto":
            try:
                trips = int(trips_str)
            except ValueError:
                print(f"  Invalid number, using 'auto'")
        gw_str = _prompt(
            "Major gateway road refs, comma-separated (e.g. D1,D2,I/43; empty for auto-detect)",
            "",
        )
        if gw_str.strip():
            gateways = [g.strip() for g in gw_str.split(",") if g.strip()]
    else:
        okres = args.okres or city_short
        kraj = args.kraj or ""
        trips = args.trips if args.trips is not None else "auto"
        if args.gateways:
            gateways = [g.strip() for g in args.gateways.split(",") if g.strip()]

    output_dir = Path(args.output_dir)
    config_dir = generate(
        city=city,
        okres=okres,
        kraj=kraj,
        total_daily_trips=trips,
        gateways=gateways or None,
        output_dir=output_dir,
    )

    print(f"\nGenerated config in: {config_dir}/")
    print(f"  sim.yaml")
    print(f"  locale.yaml")
    print(f"  screenlines.yaml")
    print(f"\nRun the full pipeline:")
    cfg = f"{config_dir}/sim.yaml"
    for step in _PIPELINE_STEPS:
        print(f"  python run.py --config {cfg} {step}")
    print(f"\nOr run all steps at once:")
    steps_str = " ".join(_PIPELINE_STEPS)
    print(f'  for step in {steps_str}; do python run.py --config {cfg} $step; done')


if __name__ == "__main__":
    main()
