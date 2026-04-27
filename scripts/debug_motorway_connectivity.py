"""Motorway connectivity debug: directed SCC, ramp audit, reachability, bearing mismatch.

Usage:
    python scripts/debug_motorway_connectivity.py --config config/brno/sim.yaml
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, mapping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOTORWAY_TYPES = {"motorway", "motorway_link"}
TRUNK_TYPES = {"trunk", "trunk_link"}
MAJOR_TYPES = MOTORWAY_TYPES | TRUNK_TYPES | {"primary", "primary_link"}

_COARSE = {
    "motorway": "motorway",
    "motorway_link": "motorway",
    "trunk": "trunk",
    "trunk_link": "trunk",
    "primary": "primary",
    "primary_link": "primary",
    "secondary": "secondary",
    "secondary_link": "secondary",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary",
}


def _load_links(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT link_id, a_node, b_node, direction, link_type, osm_id, "
        "osm_highway, osm_ref, speed_ab, speed_ba, capacity_ab, capacity_ba, "
        "lanes_ab, lanes_ba, distance, name FROM links",
        conn,
    )
    conn.close()
    return df


def _load_links_geo(db_path: str) -> gpd.GeoDataFrame:
    conn = sqlite3.connect(db_path)
    gdf = gpd.read_file(db_path, layer="links")
    conn.close()
    return gdf


def _load_nodes(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT node_id, is_centroid FROM nodes", conn)
    conn.close()
    return df


def _build_digraph(links: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in links.iterrows():
        a, b = int(row["a_node"]), int(row["b_node"])
        d = int(row.get("direction", 0))
        attrs = {
            "link_id": int(row["link_id"]),
            "link_type": str(row.get("link_type", "")),
            "osm_highway": str(row.get("osm_highway", "")),
        }
        if d == 1:
            G.add_edge(a, b, **attrs)
        elif d == -1:
            G.add_edge(b, a, **attrs)
        else:
            G.add_edge(a, b, **attrs)
            G.add_edge(b, a, **attrs)
    return G


# ---------------------------------------------------------------------------
# 1. Directed SCC analysis
# ---------------------------------------------------------------------------

def directed_scc_analysis(
    G: nx.DiGraph, links: pd.DataFrame
) -> Dict[str, Any]:
    sccs = list(nx.strongly_connected_components(G))
    sccs_sorted = sorted(sccs, key=len, reverse=True)
    largest_scc = sccs_sorted[0] if sccs_sorted else set()

    node_to_scc: Dict[int, int] = {}
    for idx, comp in enumerate(sccs_sorted):
        for n in comp:
            node_to_scc[n] = idx

    outside_largest: List[Dict[str, Any]] = []
    for _, row in links.iterrows():
        lt = str(row.get("link_type", ""))
        osm_hw = str(row.get("osm_highway", ""))
        if lt not in ("motorway", "trunk") and osm_hw not in MOTORWAY_TYPES | TRUNK_TYPES:
            continue
        a, b = int(row["a_node"]), int(row["b_node"])
        d = int(row.get("direction", 0))

        nodes_of_link = set()
        if d == 1:
            nodes_of_link = {a, b}
        elif d == -1:
            nodes_of_link = {a, b}
        else:
            nodes_of_link = {a, b}

        in_largest = all(n in largest_scc for n in nodes_of_link)
        if not in_largest:
            in_deg_a = G.in_degree(a) if a in G else 0
            out_deg_a = G.out_degree(a) if a in G else 0
            in_deg_b = G.in_degree(b) if b in G else 0
            out_deg_b = G.out_degree(b) if b in G else 0
            outside_largest.append({
                "link_id": int(row["link_id"]),
                "link_type": lt,
                "osm_highway": osm_hw,
                "a_node": a,
                "b_node": b,
                "direction": d,
                "osm_id": row.get("osm_id"),
                "osm_ref": str(row.get("osm_ref", "")),
                "a_node_scc": node_to_scc.get(a, -1),
                "b_node_scc": node_to_scc.get(b, -1),
                "a_in_deg": in_deg_a,
                "a_out_deg": out_deg_a,
                "b_in_deg": in_deg_b,
                "b_out_deg": out_deg_b,
            })

    result = {
        "total_scc": len(sccs_sorted),
        "largest_scc_size": len(largest_scc),
        "top5_scc_sizes": [len(c) for c in sccs_sorted[:5]],
        "motorway_trunk_outside_largest_count": len(outside_largest),
        "motorway_trunk_outside_largest": outside_largest,
    }
    print(f"\n=== DIRECTED SCC ANALYSIS ===")
    print(f"  Total strongly connected components: {result['total_scc']}")
    print(f"  Largest SCC: {result['largest_scc_size']} nodes")
    print(f"  Top-5 SCC sizes: {result['top5_scc_sizes']}")
    print(f"  Motorway/trunk links outside largest SCC: {len(outside_largest)}")
    if outside_largest:
        by_type = defaultdict(int)
        for item in outside_largest:
            by_type[item["osm_highway"]] += 1
        for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"    {t}: {c}")
    return result


# ---------------------------------------------------------------------------
# 2. Motorway ramp audit
# ---------------------------------------------------------------------------

def ramp_audit(links: pd.DataFrame, G: nx.DiGraph) -> Dict[str, Any]:
    """Check that every mainline motorway link has at least one ramp connection."""
    ramp_osm = links[links["osm_highway"].isin({"motorway_link", "trunk_link"})].copy()
    mainline_motorway = links[links["osm_highway"] == "motorway"].copy()
    mainline_trunk = links[links["osm_highway"] == "trunk"].copy()

    ramp_nodes: Set[int] = set()
    for _, row in ramp_osm.iterrows():
        ramp_nodes.add(int(row["a_node"]))
        ramp_nodes.add(int(row["b_node"]))

    isolated_motorway: List[Dict[str, Any]] = []
    for _, row in mainline_motorway.iterrows():
        a, b = int(row["a_node"]), int(row["b_node"])
        a_has_ramp = a in ramp_nodes
        b_has_ramp = b in ramp_nodes

        if not a_has_ramp and not b_has_ramp:
            in_types = set()
            out_types = set()
            for pred in G.predecessors(a):
                for _, ed in G[pred][a].items() if isinstance(G[pred][a], dict) else [(None, G[pred][a])]:
                    in_types.add(ed.get("osm_highway", "?") if isinstance(ed, dict) else "?")
            for succ in G.successors(b):
                for _, ed in G[b][succ].items() if isinstance(G[b][succ], dict) else [(None, G[b][succ])]:
                    out_types.add(ed.get("osm_highway", "?") if isinstance(ed, dict) else "?")

            isolated_motorway.append({
                "link_id": int(row["link_id"]),
                "a_node": a,
                "b_node": b,
                "osm_id": row.get("osm_id"),
                "osm_ref": str(row.get("osm_ref", "")),
                "a_neighbor_types": sorted(in_types),
                "b_neighbor_types": sorted(out_types),
            })

    result = {
        "total_ramp_links": len(ramp_osm),
        "motorway_link_count": int((ramp_osm["osm_highway"] == "motorway_link").sum()),
        "trunk_link_count": int((ramp_osm["osm_highway"] == "trunk_link").sum()),
        "total_mainline_motorway": len(mainline_motorway),
        "total_mainline_trunk": len(mainline_trunk),
        "isolated_motorway_segments": isolated_motorway,
        "isolated_motorway_count": len(isolated_motorway),
    }

    print(f"\n=== RAMP AUDIT ===")
    print(f"  Ramp links (osm_highway=*_link): {len(ramp_osm)}")
    print(f"    motorway_link: {result['motorway_link_count']}")
    print(f"    trunk_link: {result['trunk_link_count']}")
    print(f"  Mainline motorway links: {len(mainline_motorway)}")
    print(f"  Mainline trunk links: {len(mainline_trunk)}")
    print(f"  Motorway segments with no ramp at either endpoint: {len(isolated_motorway)}")
    for seg in isolated_motorway[:10]:
        print(f"    link_id={seg['link_id']}  ref={seg['osm_ref']}  "
              f"a_neighbors={seg['a_neighbor_types']}  b_neighbors={seg['b_neighbor_types']}")
    return result


# ---------------------------------------------------------------------------
# 3. Link-type classification mismatch
# ---------------------------------------------------------------------------

def link_type_mismatch_audit(links: pd.DataFrame) -> Dict[str, Any]:
    """Detect links where osm_highway differs from link_type (ramps merged into parent)."""
    mismatch = links[
        links["osm_highway"].notna()
        & (links["osm_highway"] != "None")
        & (links["osm_highway"] != links["link_type"])
    ].copy()

    summary: Dict[str, Dict[str, int]] = {}
    for _, row in mismatch.iterrows():
        key = f"{row['osm_highway']} -> {row['link_type']}"
        if key not in summary:
            summary[key] = {"count": 0}
        summary[key]["count"] += 1

    mismatched_ramps = mismatch[
        mismatch["osm_highway"].str.endswith("_link", na=False)
    ]

    speed_issues: List[Dict[str, Any]] = []
    for _, row in mismatched_ramps.iterrows():
        speed_issues.append({
            "link_id": int(row["link_id"]),
            "osm_highway": str(row["osm_highway"]),
            "link_type": str(row["link_type"]),
            "speed_ab": row.get("speed_ab"),
            "speed_ba": row.get("speed_ba"),
            "capacity_ab": row.get("capacity_ab"),
            "lanes_ab": row.get("lanes_ab"),
        })

    result = {
        "total_mismatched": len(mismatch),
        "mismatched_ramps": len(mismatched_ramps),
        "mapping_summary": summary,
        "ramp_speed_samples": speed_issues[:20],
    }

    print(f"\n=== LINK TYPE MISMATCH AUDIT ===")
    print(f"  Total links with osm_highway != link_type: {len(mismatch)}")
    print(f"  Of which are ramps (*_link): {len(mismatched_ramps)}")
    for key, val in sorted(summary.items(), key=lambda x: -x[1]["count"]):
        print(f"    {key}: {val['count']}")

    if speed_issues:
        print(f"  Sample ramp speed/capacity (should differ from mainline):")
        for s in speed_issues[:5]:
            print(f"    link_id={s['link_id']}  osm_hw={s['osm_highway']}  "
                  f"link_type={s['link_type']}  speed_ab={s['speed_ab']}  "
                  f"cap_ab={s['capacity_ab']}  lanes_ab={s['lanes_ab']}")
    return result


# ---------------------------------------------------------------------------
# 4. Reachability trace for worst links
# ---------------------------------------------------------------------------

def reachability_trace(
    G: nx.DiGraph,
    links: pd.DataFrame,
    nodes: pd.DataFrame,
    worst_link_ids: List[int],
) -> Dict[str, Any]:
    """BFS from random centroids to check reachability of under-loaded motorway links."""
    centroids = nodes[nodes["is_centroid"] == 1]["node_id"].tolist()
    if not centroids:
        return {"error": "no centroids found"}

    rng = np.random.default_rng(42)
    sample_centroids = rng.choice(centroids, size=min(10, len(centroids)), replace=False).tolist()

    link_id_to_nodes: Dict[int, Tuple[int, int, int]] = {}
    for _, row in links.iterrows():
        lid = int(row["link_id"])
        if lid in worst_link_ids:
            link_id_to_nodes[lid] = (int(row["a_node"]), int(row["b_node"]), int(row.get("direction", 0)))

    results: Dict[str, Any] = {}
    for lid in worst_link_ids:
        if lid not in link_id_to_nodes:
            results[str(lid)] = {"error": "link_id not found"}
            continue

        a, b, d = link_id_to_nodes[lid]
        target_node = a if d in (1, 0) else b

        reachable_from = []
        unreachable_from = []
        for c in sample_centroids:
            c = int(c)
            if c not in G:
                unreachable_from.append(c)
                continue
            try:
                _path = nx.shortest_path(G, c, target_node)
                reachable_from.append(c)
            except nx.NetworkXNoPath:
                unreachable_from.append(c)
            except nx.NodeNotFound:
                unreachable_from.append(c)

        target_is_origin = target_node in G
        origin_from = []
        dest_unreachable = []
        for c in sample_centroids:
            c = int(c)
            if c not in G:
                dest_unreachable.append(c)
                continue
            end_node = b if d in (1, 0) else a
            try:
                _path = nx.shortest_path(G, end_node, c)
                origin_from.append(c)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                dest_unreachable.append(c)

        row_data = links[links["link_id"] == lid].iloc[0]
        results[str(lid)] = {
            "link_type": str(row_data.get("link_type", "")),
            "osm_highway": str(row_data.get("osm_highway", "")),
            "osm_ref": str(row_data.get("osm_ref", "")),
            "direction": d,
            "a_node": a,
            "b_node": b,
            "target_node_in_graph": target_is_origin,
            "reachable_from_centroids": len(reachable_from),
            "unreachable_from_centroids": len(unreachable_from),
            "can_reach_centroids_from_link": len(origin_from),
            "cannot_reach_centroids_from_link": len(dest_unreachable),
            "sample_centroids_tested": len(sample_centroids),
        }

    print(f"\n=== REACHABILITY TRACE (top under-loaded motorway links) ===")
    for lid, info in results.items():
        if "error" in info:
            print(f"  link_id={lid}: {info['error']}")
            continue
        pct_reach = info["reachable_from_centroids"] / max(info["sample_centroids_tested"], 1) * 100
        pct_exit = info["can_reach_centroids_from_link"] / max(info["sample_centroids_tested"], 1) * 100
        print(f"  link_id={lid}  ref={info['osm_ref']}  osm_hw={info['osm_highway']}  "
              f"reachable={info['reachable_from_centroids']}/{info['sample_centroids_tested']} ({pct_reach:.0f}%)  "
              f"can_exit={info['can_reach_centroids_from_link']}/{info['sample_centroids_tested']} ({pct_exit:.0f}%)")
    return results


# ---------------------------------------------------------------------------
# 5. Bearing mismatch audit
# ---------------------------------------------------------------------------

def bearing_mismatch_audit(diag_path: Path) -> Dict[str, Any]:
    """Flag motorway/trunk matches where bearing_diff > 90 deg."""
    if not diag_path.exists():
        return {"error": f"file not found: {diag_path}"}

    df = pd.read_csv(diag_path)
    if "_bearing_diff" not in df.columns or "link_type" not in df.columns:
        return {"error": "required columns missing"}

    major = df[df["link_type"].isin(["motorway", "trunk"])].copy()
    bad_bearing = major[major["_bearing_diff"].abs() > 90].copy()

    items = []
    for _, row in bad_bearing.iterrows():
        items.append({
            "objectid": int(row["objectid"]) if pd.notna(row.get("objectid")) else None,
            "link_id": int(row["link_id"]) if pd.notna(row.get("link_id")) else None,
            "link_type": str(row["link_type"]),
            "bearing_diff": round(float(row["_bearing_diff"]), 1),
            "observed_total": float(row.get("observed_total", 0)),
            "modeled": float(row.get("total_vehicles_tot", 0)) if pd.notna(row.get("total_vehicles_tot")) else 0,
            "GEH": round(float(row.get("GEH", 0)), 1) if pd.notna(row.get("GEH")) else None,
        })

    result = {
        "total_major_matches": len(major),
        "bearing_mismatch_count": len(bad_bearing),
        "bearing_mismatch_pct": round(len(bad_bearing) / max(len(major), 1) * 100, 1),
        "mismatched_items": items,
    }

    print(f"\n=== BEARING MISMATCH AUDIT (motorway/trunk, |bearing_diff| > 90°) ===")
    print(f"  Total motorway/trunk matched: {len(major)}")
    print(f"  Bearing mismatch (>90°): {len(bad_bearing)} ({result['bearing_mismatch_pct']}%)")
    for item in items[:10]:
        print(f"    objectid={item['objectid']}  link_id={item['link_id']}  "
              f"type={item['link_type']}  bearing_diff={item['bearing_diff']}°  "
              f"obs={item['observed_total']:.0f}  mod={item['modeled']:.0f}  GEH={item['GEH']}")
    return result


# ---------------------------------------------------------------------------
# 6. Neighborhood analysis for worst links
# ---------------------------------------------------------------------------

def neighborhood_analysis(
    G: nx.DiGraph, links: pd.DataFrame, worst_link_ids: List[int]
) -> Dict[str, Any]:
    """For each worst link, show all links connected to its nodes in the directed graph."""
    link_lookup = links.set_index("link_id")

    results: Dict[str, Any] = {}
    for lid in worst_link_ids:
        if lid not in link_lookup.index:
            continue
        row = link_lookup.loc[lid]
        a, b = int(row["a_node"]), int(row["b_node"])
        d = int(row.get("direction", 0))

        incoming_to_a: List[Dict] = []
        outgoing_from_b: List[Dict] = []

        if a in G:
            for pred in G.predecessors(a):
                ed = G[pred][a]
                incoming_to_a.append({
                    "from_node": pred,
                    "link_id": ed.get("link_id"),
                    "link_type": ed.get("link_type"),
                    "osm_highway": ed.get("osm_highway"),
                })

        if b in G:
            for succ in G.successors(b):
                ed = G[b][succ]
                outgoing_from_b.append({
                    "to_node": succ,
                    "link_id": ed.get("link_id"),
                    "link_type": ed.get("link_type"),
                    "osm_highway": ed.get("osm_highway"),
                })

        a_out_to_non_motorway = sum(
            1 for succ in G.successors(a) if G[a][succ].get("osm_highway") not in MOTORWAY_TYPES
        ) if a in G else 0
        b_in_from_non_motorway = sum(
            1 for pred in G.predecessors(b) if G[pred][b].get("osm_highway") not in MOTORWAY_TYPES
        ) if b in G else 0

        results[str(lid)] = {
            "link_type": str(row.get("link_type", "")),
            "osm_highway": str(row.get("osm_highway", "")),
            "osm_ref": str(row.get("osm_ref", "")),
            "direction": d,
            "a_node": a,
            "b_node": b,
            "incoming_to_a": incoming_to_a,
            "outgoing_from_b": outgoing_from_b,
            "a_has_non_motorway_exit": a_out_to_non_motorway > 0,
            "b_has_non_motorway_entry": b_in_from_non_motorway > 0,
        }

    print(f"\n=== NEIGHBORHOOD ANALYSIS (worst links) ===")
    for lid_s, info in results.items():
        print(f"  link_id={lid_s}  ref={info['osm_ref']}  osm_hw={info['osm_highway']}  dir={info['direction']}")
        print(f"    a_node={info['a_node']}:  {len(info['incoming_to_a'])} incoming edges  "
              f"non-motorway exit={info['a_has_non_motorway_exit']}")
        for e in info["incoming_to_a"]:
            print(f"      <- node {e['from_node']}  link_id={e['link_id']}  osm_hw={e['osm_highway']}")
        print(f"    b_node={info['b_node']}:  {len(info['outgoing_from_b'])} outgoing edges  "
              f"non-motorway entry={info['b_has_non_motorway_entry']}")
        for e in info["outgoing_from_b"]:
            print(f"      -> node {e['to_node']}  link_id={e['link_id']}  osm_hw={e['osm_highway']}")
    return results


# ---------------------------------------------------------------------------
# 7. GeoJSON export of problem links
# ---------------------------------------------------------------------------

def export_issues_geojson(
    links_geo: gpd.GeoDataFrame,
    scc_result: Dict,
    ramp_result: Dict,
    bearing_result: Dict,
    output_path: Path,
) -> None:
    """Write a GeoJSON with all flagged motorway/trunk links for QGIS inspection."""
    problem_link_ids: Set[int] = set()
    issue_map: Dict[int, List[str]] = defaultdict(list)

    for item in scc_result.get("motorway_trunk_outside_largest", []):
        lid = item["link_id"]
        problem_link_ids.add(lid)
        issue_map[lid].append("outside_largest_SCC")

    for item in ramp_result.get("isolated_motorway_segments", []):
        lid = item["link_id"]
        problem_link_ids.add(lid)
        issue_map[lid].append("no_ramp_at_endpoints")

    for item in bearing_result.get("mismatched_items", []):
        lid = item.get("link_id")
        if lid:
            problem_link_ids.add(lid)
            issue_map[lid].append(f"bearing_mismatch_{item['bearing_diff']}deg")

    if not problem_link_ids:
        print(f"\n  No problem links to export to GeoJSON.")
        return

    subset = links_geo[links_geo["link_id"].isin(problem_link_ids)].copy()
    subset["issues"] = subset["link_id"].map(lambda x: "; ".join(issue_map.get(int(x), [])))

    keep_cols = [c for c in ["link_id", "link_type", "osm_highway", "osm_ref", "name",
                              "direction", "a_node", "b_node", "issues", "geometry"]
                 if c in subset.columns]
    subset = subset[keep_cols]

    if hasattr(subset, "to_crs"):
        try:
            subset = subset.to_crs(epsg=4326)
        except Exception:
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_file(str(output_path), driver="GeoJSON")
    print(f"\n  Exported {len(subset)} problem links to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Motorway connectivity debug")
    parser.add_argument("--config", default="config/brno/sim.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_path = Path(cfg.get("project_path", "project/brno_aeq"))
    db_path = str(project_path / "project_database.sqlite")
    diag_csv = Path(cfg.get("demand", {}).get("output_dir", "outputs/baseline/demand")) / "matching_diagnostics.csv"
    net_output = Path(cfg.get("network", {}).get("output_dir", "outputs/baseline/network"))

    print(f"Project: {project_path}")
    print(f"Database: {db_path}")

    links = _load_links(db_path)
    nodes = _load_nodes(db_path)
    print(f"Loaded {len(links)} links, {len(nodes)} nodes ({nodes['is_centroid'].sum()} centroids)")

    G = _build_digraph(links)
    print(f"DiGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 1. Directed SCC
    scc_result = directed_scc_analysis(G, links)

    # 2. Ramp audit
    ramp_result = ramp_audit(links, G)

    # 3. Link type mismatch
    mismatch_result = link_type_mismatch_audit(links)

    # 4. Reachability trace for worst motorway links (from matching_diagnostics)
    worst_link_ids = [5480, 60807, 43563, 1752, 43559, 1446, 80128, 60908, 80090, 80093]
    reach_result = reachability_trace(G, links, nodes, worst_link_ids)

    # 5. Bearing mismatch
    bearing_result = bearing_mismatch_audit(diag_csv)

    # 6. Neighborhood analysis
    neighborhood_result = neighborhood_analysis(G, links, worst_link_ids)

    # 7. Export GeoJSON
    try:
        links_geo = _load_links_geo(db_path)
        export_issues_geojson(links_geo, scc_result, ramp_result, bearing_result,
                              net_output / "motorway_issues.geojson")
    except Exception as e:
        print(f"\n  GeoJSON export failed: {e}")

    # Write JSON report
    report = {
        "directed_scc": scc_result,
        "ramp_audit": ramp_result,
        "link_type_mismatch": mismatch_result,
        "reachability": reach_result,
        "bearing_mismatch": bearing_result,
        "neighborhood": neighborhood_result,
    }

    out_json = net_output / "motorway_connectivity_debug.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=_default)
    print(f"\n=== Report written to {out_json} ===")


if __name__ == "__main__":
    main()
