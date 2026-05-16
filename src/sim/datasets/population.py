"""SLDB 2021 population data preprocessing."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from sim._text import strip_diacritics as _strip_diacritics
from sim.datasets.utils import (
    coerce_object_columns_for_parquet,
    ensure_dir,
    read_csv_flexible,
    resolve_input_col,
)

logger = logging.getLogger(__name__)

_DEFAULT_CZ_SUFFIXES = [
    "-mesto", "-ves",
    " u brna", " u prahy", " u mostu", " u olomouce", " u ostravy",
    " nad labem", " nad svitavou", " nad sazavou", " nad vltavou",
    " nad orlici", " nad moravou", " nad jihlavou", " nad luznici",
    " v cechach", " na morave", " pod rizem", " pod radhostem",
]


def _load_mc_to_ku(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """Load municipal-part -> cadastral-unit mapping from locale config."""
    if cfg:
        from sim.io_project import load_locale
        locale = load_locale(cfg)
        mapping = locale.get("municipal_parts_to_cadastral")
        if mapping and isinstance(mapping, dict):
            return mapping
    return {}


def _load_place_name_suffixes(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Load place-name suffixes to strip during zone<->obec matching.

    Returns the list from ``locale.yaml`` -> ``place_name_suffixes_to_strip``
    if present; otherwise falls back to common Czech municipality suffixes.
    """
    if cfg:
        from sim.io_project import load_locale
        locale = load_locale(cfg)
        custom = locale.get("place_name_suffixes_to_strip")
        if custom and isinstance(custom, list):
            return [str(s).lower() for s in custom]
    return list(_DEFAULT_CZ_SUFFIXES)


def preprocess_population_sldb2021(
    csv_path: Path,
    out_parquet: Path,
    zones_geojson: Optional[Path] = None,
    *,
    delimiter: str = "auto",
    encoding: str = "auto",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import difflib

    def norm(text: Any) -> str:
        return _strip_diacritics(text).lower().strip()

    df = read_csv_flexible(csv_path, delimiter=delimiter, encoding=encoding)

    pohlavi_col = resolve_input_col(df, {}, "pohlavi_kod", ["pohlavi_kod", "kod_pohlavi", "pohlavi"])
    hodnota_col = resolve_input_col(df, {}, "hodnota", ["hodnota", "value", "pocet"])
    nazev_col = resolve_input_col(df, {}, "uzemi_txt", ["uzemi_txt", "nazev", "nazev_uzemi", "uzemi"])
    uzemi_cis_col = resolve_input_col(df, {}, "uzemi_cis", ["uzemi_cis", "typ_uzemi", "uzemni_uroven", "uzemi_typ"])
    uzemi_kod_col = resolve_input_col(df, {}, "uzemi_kod", ["uzemi_kod", "kod_uzemi", "kod"])

    missing = [
        key for key, value in {
            "pohlavi_kod": pohlavi_col,
            "hodnota": hodnota_col,
            "uzemi_txt": nazev_col,
            "uzemi_cis": uzemi_cis_col,
            "uzemi_kod": uzemi_kod_col,
        }.items() if value is None
    ]
    if missing:
        raise RuntimeError(f"Population CSV is missing required columns {missing}. Available columns: {list(df.columns)}")

    tmp = pd.DataFrame({
        "pohlavi_kod": df[pohlavi_col],
        "hodnota": df[hodnota_col],
        "uzemi_txt": df[nazev_col],
        "uzemi_cis": df[uzemi_cis_col],
        "uzemi_kod": df[uzemi_kod_col],
    })

    total = tmp[tmp["pohlavi_kod"].isna()].copy()
    total["hodnota"] = pd.to_numeric(total["hodnota"], errors="coerce").fillna(0).astype(int)
    total["nazev"] = total["uzemi_txt"].astype(str).str.strip()

    place_name = (cfg or {}).get("osm", {}).get("place_name", "")
    city_prefix = place_name.split(",")[0].strip() if place_name else ""
    mc_filter = (
        (pd.to_numeric(total["uzemi_cis"], errors="coerce") == 44)
        & (total["nazev"].str.startswith(city_prefix, na=False) if city_prefix else False)
    )
    city_mc = total[mc_filter]
    mc_pop = {str(r["nazev"]).strip(): int(r["hodnota"]) for _, r in city_mc.iterrows()}

    obce = total[pd.to_numeric(total["uzemi_cis"], errors="coerce") == 43][["uzemi_kod", "nazev", "hodnota"]].copy()
    obce = obce.sort_values("hodnota", ascending=False).drop_duplicates(subset="nazev", keep="first")
    obec_pop = {str(r["nazev"]).strip(): int(r["hodnota"]) for _, r in obce.iterrows()}
    obec_norm = {norm(k): k for k in obec_pop}

    if not (zones_geojson and zones_geojson.exists()):
        result = pd.DataFrame([
            {"zone_id": 0, "zone_name": name, "population": pop, "match": "no_zones"}
            for name, pop in {**mc_pop, **obec_pop}.items()
        ])
        ensure_dir(out_parquet.parent)
        result.to_parquet(out_parquet, index=False)
        return {
            "parquet": str(out_parquet),
            "zones_total": len(result),
            "matched": 0,
            "total_population": int(result["population"].sum()),
        }

    from sim.io_project import get_metric_epsg

    zones = gpd.read_file(zones_geojson)
    mc_to_ku = _load_mc_to_ku(cfg)
    ku_to_mc = {norm(ku): mc for mc, ku_list in mc_to_ku.items() for ku in ku_list}

    zones_m = zones.to_crs(epsg=get_metric_epsg(cfg or {}))
    zone_area = {int(r["zone_id"]): r.geometry.area for _, r in zones_m.iterrows()}

    mc_zone_ids: Dict[str, List[int]] = {}
    zone_mc_map: Dict[int, str] = {}
    if ku_to_mc:
        for _, zrow in zones.iterrows():
            zid = int(zrow["zone_id"])
            znorm = norm(zrow["name"])
            if znorm in ku_to_mc:
                mc = ku_to_mc[znorm]
                mc_zone_ids.setdefault(mc, []).append(zid)
                zone_mc_map[zid] = mc

    suffixes_to_strip = _load_place_name_suffixes(cfg)

    max_zone_mc_share = 0.70
    result_rows: List[Dict[str, Any]] = []

    for _, zrow in zones.iterrows():
        zid = int(zrow["zone_id"])
        zname = str(zrow["name"])
        znorm = norm(zname)

        if zid in zone_mc_map:
            mc = zone_mc_map[zid]
            zone_ids = mc_zone_ids.get(mc, [])
            total_area = sum(zone_area.get(z, 0.0) for z in zone_ids) or 1.0
            share = min(zone_area.get(zid, 0.0) / total_area, max_zone_mc_share)
            pop = int(round(mc_pop.get(mc, 0) * share))
            result_rows.append({"zone_id": zid, "zone_name": zname, "population": pop, "match": f"mc_area:{mc}"})
            continue

        if znorm in obec_norm:
            original = obec_norm[znorm]
            result_rows.append({"zone_id": zid, "zone_name": zname, "population": obec_pop[original], "match": "obec_exact"})
            continue

        matched = False
        for suffix in suffixes_to_strip:
            stripped = znorm.replace(suffix, "")
            if stripped != znorm and stripped in obec_norm:
                original = obec_norm[stripped]
                result_rows.append({"zone_id": zid, "zone_name": zname, "population": obec_pop[original], "match": f"obec_strip:{original}"})
                matched = True
                break
        if matched:
            continue

        close = difflib.get_close_matches(znorm, list(obec_norm.keys()), n=1, cutoff=0.82)
        if close:
            original = obec_norm[close[0]]
            candidate_pop = obec_pop[original]
            ratio = difflib.SequenceMatcher(None, znorm, close[0]).ratio()
            if ratio < 0.95 and candidate_pop > 5000:
                logger.warning(
                    "Fuzzy match rejected for zone '%s' -> '%s' (pop=%d, ratio=%.3f): "
                    "population too large for imprecise match",
                    zname, original, candidate_pop, ratio,
                )
            else:
                result_rows.append({"zone_id": zid, "zone_name": zname, "population": candidate_pop, "match": f"fuzzy:{original}"})
                continue

        all_pops = [v for v in {**mc_pop, **obec_pop}.values() if v > 0]
        avg = int(sorted(all_pops)[len(all_pops) // 2]) if all_pops else 1000
        result_rows.append({"zone_id": zid, "zone_name": zname, "population": avg, "match": "default_median"})

    result = coerce_object_columns_for_parquet(pd.DataFrame(result_rows))
    ensure_dir(out_parquet.parent)
    result.to_parquet(out_parquet, index=False)

    matched_count = len([r for r in result_rows if r["match"] != "default_median"])
    total_pop = int(result["population"].sum())
    return {
        "parquet": str(out_parquet),
        "zones_total": len(result_rows),
        "matched": matched_count,
        "total_population": total_pop,
    }
