"""Gateway screenline evaluation (car + corridor, pentlogram ref supplement)."""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from sim.calibration.screenlines import (
    ScreenlineDef,
    evaluate_screenline,
    resolve_gateway_screenline_eval_cfg,
)


def _cfg_gateway_eval() -> dict:
    return {
        "calibration": {
            "auto_screenlines": {
                "gateway_count_target": "car_only",
                "gateway_use_corridor_volume": True,
                "gateway_prefer_pentlogram": True,
                "gateway_supplement_from_ref_match": True,
            },
        },
    }


def test_resolve_gateway_screenline_eval_cfg():
    gw = resolve_gateway_screenline_eval_cfg(_cfg_gateway_eval())
    assert gw["count_target"] == "car_only"
    assert gw["obs_col"] == "observed_car"
    assert gw["use_corridor_volume"] is True


def test_gateway_d2_style_car_and_corridor():
    """One count post + twin link: cars + corridor, not motor sum of both PCE."""
    vol_df = pd.DataFrame({
        "link_id": [59255, 59256],
        "PCE_tot": [14011.0, 41186.0],
    })
    matched = gpd.GeoDataFrame({
        "link_id": [59255],
        "osm_ref": ["D2"],
        "observed_car": [20910.0],
        "observed_motor_total": [32864.0],
        "_corridor_volume": [27259.0],
        "geometry": [Point(0, 0)],
    }, crs="EPSG:4326")
    sl = ScreenlineDef(
        name="auto_gw_D2_S",
        links=[(59255, 0), (59256, 0)],
        has_explicit_links=True,
        attr_filter={"osm_ref_norm": "D2"},
        observed_aadt_all=32864.0,
        observed_aadt_cars=20910.0,
    )
    r = evaluate_screenline(
        sl, vol_df, matched, "PCE_tot", "observed_motor_total", cfg=_cfg_gateway_eval(),
    )
    assert r.observed_total == 20910.0
    assert r.modeled_total == 27259.0
    assert abs(r.ratio - 1.303) < 0.02


def test_gateway_ref_from_name_without_attr_filter():
    from sim.calibration.screenlines import _gateway_ref_from_screenline, ScreenlineDef

    sl = ScreenlineDef(name="auto_gw_I52_S", links=[(1, 0)])
    assert _gateway_ref_from_screenline(sl) == "52"
    sl2 = ScreenlineDef(name="auto_gw_D2_S", attr_filter={"osm_ref_norm": "D2"})
    assert _gateway_ref_from_screenline(sl2) == "D2"


def test_gateway_i52_supplement_from_videnka_ref():
    """Boundary connectors without counts → pentlogram on same sil (52)."""
    vol_df = pd.DataFrame({
        "link_id": [8600, 22128, 23502],
        "PCE_tot": [4310.0, 9606.0, 7727.0],
    })
    matched = gpd.GeoDataFrame({
        "link_id": [23502],
        "osm_ref": ["52"],
        "observed_car": [20943.0],
        "observed_motor_total": [26997.0],
        "_corridor_volume": [12956.0],
        "geometry": [Point(0, 0)],
    }, crs="EPSG:4326")
    sl = ScreenlineDef(
        name="auto_gw_I52_S",
        links=[(8600, 0), (22128, 0)],
        has_explicit_links=True,
        observed_aadt_all=25588.0,
        observed_aadt_cars=20000.0,
    )
    r = evaluate_screenline(
        sl, vol_df, matched, "PCE_tot", "observed_motor_total", cfg=_cfg_gateway_eval(),
    )
    assert r.observed_total == 20943.0
    assert r.modeled_total == 12956.0
    assert abs(r.ratio - 0.619) < 0.02
    assert r.obs_source == "pentlogram_ref_match"
