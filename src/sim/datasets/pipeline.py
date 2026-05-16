"""Provider dispatch handlers and main dataset-fetch orchestrator."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from sim.io_project import load_config
from sim.datasets.utils import (
    download_atom_latest_file,
    download_file,
    ensure_dir,
    now_iso,
    slug,
)
from sim.datasets.registry import merge_dataset_sources
from sim.datasets.csd import preprocess_csd_xlsx, preprocess_xlsx_table
from sim.datasets.commuting import preprocess_commuting_sldb2021
from sim.datasets.population import preprocess_population_sldb2021
from sim.datasets.centroids import (
    preprocess_grouped_points_to_centroids,
    preprocess_grouped_points_zip_to_centroids,
)
from sim.datasets.closures_fetch import fetch_postgres_closures
from sim.datasets.jams_fetch import fetch_postgres_jams
from sim.datasets.segments_fetch import fetch_postgres_segments
from sim.datasets.event_links_fetch import fetch_postgres_event_links
from sim.datasets.employment import derive_zone_employment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider handlers
# ---------------------------------------------------------------------------

def _handle_http_file(
    cfg: Dict[str, Any],
    scfg: Dict[str, Any],
    *,
    cache_dir: Path,
    timeout_s: int,
    retries: int,
    headers: Dict[str, str],
    force: bool,
) -> Dict[str, Any]:
    out_path = Path(scfg["out_path"])
    info: Dict[str, Any] = {
        "download": download_file(scfg["url"], out_path, timeout_s=timeout_s, retries=retries, headers=headers, force=force)
    }

    fmt_cfg = scfg.get("format") or {}
    usage = scfg.get("usage") or {}
    fmt_type = fmt_cfg.get("type")
    if fmt_type is None:
        ext = out_path.suffix.lstrip(".").lower()
        if ext in ("xlsx", "xls"):
            fmt_type = "xlsx"
        elif ext == "csv":
            fmt_type = "csv"

    if fmt_type == "xlsx":
        out_parquet = Path(scfg.get("out_parquet") or str(out_path.parent / f"{slug(out_path.stem)}.parquet"))
        if usage.get("validation_target") == "aadt_screenlines" or usage.get("calibration_target") == "aadt_screenlines":
            info["preprocess"] = preprocess_csd_xlsx(out_path, out_parquet)
        else:
            info["preprocess"] = preprocess_xlsx_table(out_path, out_parquet)
        return info

    if fmt_type == "csv" and usage.get("socioeconomic") == "population_per_zone":
        zones_path = Path(cfg.get("zoning", {}).get("output_dir", "outputs/baseline/zones")) / "zones.geojson"
        out_parquet = cache_dir / "zone_population.parquet"
        info["preprocess"] = preprocess_population_sldb2021(
            out_path,
            out_parquet,
            zones_geojson=zones_path if zones_path.exists() else None,
            delimiter=str(fmt_cfg.get("delimiter", "auto")),
            encoding=str(fmt_cfg.get("encoding", "auto")),
            cfg=cfg,
        )
    return info


def _handle_csu_open_data_csv(
    scfg: Dict[str, Any],
    *,
    cache_dir: Path,
    timeout_s: int,
    retries: int,
    headers: Dict[str, str],
    force: bool,
) -> Dict[str, Any]:
    out_path = Path(scfg["out_path"])
    fmt = scfg.get("format") or {}
    preprocess_cfg = scfg.get("preprocess") or {}
    info: Dict[str, Any] = {
        "download": download_file(scfg["url"], out_path, timeout_s=timeout_s, retries=retries, headers=headers, force=force),
        "preprocess": {},
    }

    delimiter = str(fmt.get("delimiter", "auto"))
    encoding = str(fmt.get("encoding", "auto"))

    if bool(preprocess_cfg.get("write_filtered", True)):
        filtered_out = Path(scfg.get("filtered_out_parquet") or (cache_dir / f"{slug(out_path.stem)}.parquet"))
        info["preprocess"]["filtered"] = preprocess_commuting_sldb2021(
            out_path,
            filtered_out,
            delimiter=delimiter,
            encoding=encoding,
            filter_cfg=scfg.get("filter") or {},
        )

    if bool(preprocess_cfg.get("write_full_cr", False)):
        full_out = Path(scfg.get("full_cr_out_parquet") or (out_path.parent / f"{out_path.stem}_full_cr.parquet"))
        info["preprocess"]["full_cr"] = preprocess_commuting_sldb2021(
            out_path,
            full_out,
            delimiter=delimiter,
            encoding=encoding,
            filter_cfg={"enabled": False},
        )

    return info


def _handle_atom_file(
    scfg: Dict[str, Any],
    *,
    cache_dir: Path,
    timeout_s: int,
    retries: int,
    headers: Dict[str, str],
    force: bool,
) -> Dict[str, Any]:
    out_path = Path(scfg["out_path"])
    fmt_cfg = scfg.get("format") or {}
    usage = scfg.get("usage") or {}

    info: Dict[str, Any] = {
        "download": download_atom_latest_file(
            scfg["url"],
            out_path,
            timeout_s=timeout_s,
            retries=retries,
            headers=headers,
            force=force,
            asset_pattern=fmt_cfg.get("asset_pattern"),
            feed_cache_path=Path(scfg["feed_cache_path"]) if scfg.get("feed_cache_path") else None,
        )
    }

    if usage.get("supernetwork_places") == "grouped_point_centroids":
        out_parquet = Path(scfg.get("out_parquet") or str(out_path.parent / "cz_place_centroids.parquet"))
        if fmt_cfg.get("type") == "zip_csv":
            info["preprocess"] = preprocess_grouped_points_zip_to_centroids(
                out_path,
                out_parquet,
                member_pattern=fmt_cfg.get("member_pattern"),
                delimiter=str(fmt_cfg.get("delimiter", "auto")),
                encoding=str(fmt_cfg.get("encoding", "auto")),
                source_crs_epsg=int(fmt_cfg.get("source_crs_epsg", 5514)),
                output_crs_epsg=int(fmt_cfg.get("output_crs_epsg", 4326)),
                columns_cfg=scfg.get("columns") or {},
            )
        else:
            extracted_path = Path(scfg.get("extracted_path") or out_path.with_suffix(".csv"))
            ensure_dir(extracted_path.parent)
            with zipfile.ZipFile(out_path, "r") as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                if not names:
                    raise RuntimeError(f"No files found in ZIP archive: {out_path}")
                selected = (
                    next((n for n in names if re.search(str(fmt_cfg.get("member_pattern")), n)), names[0])
                    if fmt_cfg.get("member_pattern")
                    else names[0]
                )
                extracted_path.write_bytes(zf.read(selected))
            info["extract"] = {
                "zip_path": str(out_path),
                "member_name": selected,
                "path": str(extracted_path),
                "bytes": extracted_path.stat().st_size,
            }
            info["preprocess"] = preprocess_grouped_points_to_centroids(
                extracted_path,
                out_parquet,
                delimiter=str(fmt_cfg.get("delimiter", "auto")),
                encoding=str(fmt_cfg.get("encoding", "auto")),
                source_crs_epsg=int(fmt_cfg.get("source_crs_epsg", 5514)),
                output_crs_epsg=int(fmt_cfg.get("output_crs_epsg", 4326)),
                columns_cfg=scfg.get("columns") or {},
            )

    return info


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def run_fetch_datasets(
    config_path: str | Path = "config/brno/sim.yaml",
    force: bool = False,
    only: Optional[List[str]] = None,
) -> None:
    cfg = load_config(config_path)
    ds = cfg.get("datasets") or {}
    if ds.get("enabled") is False:
        logger.info("datasets.enabled=false, nothing to do")
        return

    timeout_s = int((ds.get("http") or {}).get("timeout_s", 60))
    retries = int((ds.get("http") or {}).get("retries", 3))
    user_agent = str((ds.get("http") or {}).get("user_agent", "simulation-pipeline/1.0"))
    headers = {"User-Agent": user_agent}

    cache_dir = Path(ds.get("cache_dir", "data/cache"))
    ensure_dir(cache_dir)

    manifest: Dict[str, Any] = {
        "generated_at": now_iso(),
        "config": str(config_path),
        "sources": {},
    }

    errors: list[tuple[str, str]] = []
    selected = set(only) if only else None
    all_sources = merge_dataset_sources(ds.get("sources"))
    for key, scfg in all_sources.items():
        if selected is not None and key not in selected:
            continue
        if not (scfg or {}).get("enabled", False):
            continue

        provider = str(scfg.get("provider", "")).strip()
        logger.info("%s: provider=%s", key, provider)

        try:
            if provider == "http_file":
                info = _handle_http_file(cfg, scfg, cache_dir=cache_dir, timeout_s=timeout_s, retries=retries, headers=headers, force=force)
            elif provider == "csu_open_data_csv":
                info = _handle_csu_open_data_csv(scfg, cache_dir=cache_dir, timeout_s=timeout_s, retries=retries, headers=headers, force=force)
            elif provider == "atom_file":
                info = _handle_atom_file(scfg, cache_dir=cache_dir, timeout_s=timeout_s, retries=retries, headers=headers, force=force)
            elif provider == "postgres_closures":
                info = fetch_postgres_closures(cfg, scfg, force=force)
            elif provider == "postgres_jams":
                info = fetch_postgres_jams(cfg, scfg, force=force)
            elif provider == "postgres_segments":
                info = fetch_postgres_segments(cfg, scfg, force=force)
            elif provider == "postgres_event_links":
                info = fetch_postgres_event_links(cfg, scfg, force=force)
            else:
                raise ValueError(f"Unknown provider '{provider}' for source '{key}'")
        except Exception as exc:
            logger.error("%s: %s", key, exc)
            errors.append((key, str(exc)))
            manifest["sources"][key] = {"error": str(exc)}
            continue

        manifest["sources"][key] = info

    # --- Derived products: employment from commuting destinations ---
    emp_cfg = (ds.get("employment") or {})
    if emp_cfg.get("enabled", True):
        try:
            commuting_src = all_sources.get("commuting_sldb2021") or {}
            comm_path = Path(commuting_src.get("out_path", "data/sources/csu/sldb2021/dojizdka_obce.csv"))
            full_cr_parquet = comm_path.parent / f"{comm_path.stem}_full_cr.parquet"
            if full_cr_parquet.exists():
                src_file = full_cr_parquet
            elif comm_path.exists():
                src_file = comm_path
            else:
                src_file = None

            if src_file:
                zones_path = Path(cfg.get("zoning", {}).get("output_dir", "outputs/baseline/zones")) / "zones.geojson"
                emp_out = cache_dir / "zone_employment.parquet"
                emp_info = derive_zone_employment(
                    src_file,
                    emp_out,
                    zones_geojson=zones_path if zones_path.exists() else None,
                    cfg=cfg,
                )
                manifest["sources"]["_derived_employment"] = emp_info
                logger.info("Employment derived: %s", emp_info.get("parquet"))
            else:
                logger.info("Employment derivation skipped: commuting source not available yet")
        except Exception as exc:
            logger.warning("Employment derivation failed: %s", exc)

    manifest_path = cache_dir / "datasets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote manifest: %s", manifest_path)

    if errors:
        names = ", ".join(k for k, _ in errors)
        logger.warning("%d source(s) failed: %s", len(errors), names)
        for k, msg in errors:
            logger.warning("  %s: %s", k, msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/brno/sim.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    run_fetch_datasets(config_path=args.config, force=args.force, only=args.only)


if __name__ == "__main__":
    _src_dir = Path(__file__).resolve().parent.parent.parent
    if _src_dir.exists() and str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    main()
