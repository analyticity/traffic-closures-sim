#!/usr/bin/env python3
"""Investigate Brno network links for roads 383 and 384."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
from aequilibrae import Project

ROOT = Path(__file__).resolve().parents[1]  # repo root (scripts/..)
PROJECT_PATH = ROOT / "project" / "brno_aeq"
ASSIGNMENT_PARQUET = ROOT / "outputs" / "brno" / "baseline" / "demand" / "assignment_results.parquet"


def ref_has_road(ref: str, targets: frozenset[str]) -> bool:
    if not ref or not isinstance(ref, str):
        return False
    parts = re.split(r"[;,]", str(ref).strip())
    for p in parts:
        t = re.sub(r"\s+", "", p.strip().upper()).replace("II/", "").replace("I/", "")
        t = re.sub(r"^II", "", t).replace("/", "")
        if t in targets:
            return True
    return False


def split_ref_components(ref: str) -> list[str]:
    if not ref or (isinstance(ref, float) and pd.isna(ref)):
        return []
    parts = re.split(r"[;,]", str(ref).strip())
    out = []
    for p in parts:
        t = re.sub(r"\s+", "", p.strip().upper()).replace("II/", "").replace("I/", "")
        t = re.sub(r"^II", "", t).replace("/", "")
        out.append(t)
    return out


def car_mode(m: str) -> bool:
    return "c" in str(m or "")


def link_edges_for_routing(row) -> list[tuple[int, int]]:
    """Directed edges for car routing (same direction convention as connectivity.check_connectivity)."""
    if not car_mode(row.get("modes")):
        return []
    a, b = int(row["a_node"]), int(row["b_node"])
    d = int(row.get("direction", 0) or 0)
    edges = []
    if d == 1:
        edges.append((a, b))
    elif d == -1:
        edges.append((b, a))
    elif d == 0:
        edges.extend([(a, b), (b, a)])
    return edges


def build_car_digraph(links: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in links.iterrows():
        for u, v in link_edges_for_routing(row):
            G.add_edge(u, v, link_id=int(row["link_id"]))
    return G


def largest_scc_nodes(G: nx.DiGraph) -> set:
    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        return set()
    return max(sccs, key=len)


def subgraph_terminals(sub_und: nx.Graph) -> list[int]:
    deg1 = [n for n in sub_und.nodes() if sub_und.degree(n) == 1]
    if len(deg1) >= 2:
        return deg1
    # Fallback: two highest-degree nodes if no clear leaves (ring)
    nodes = list(sub_und.nodes())
    if len(nodes) < 2:
        return nodes
    by_deg = sorted(nodes, key=lambda n: sub_und.degree(n), reverse=True)
    return by_deg[:2]


def main() -> int:
    targets = frozenset({"383", "384"})

    print("=== AequilibraE project ===")
    print(f"path: {PROJECT_PATH}")
    project = Project()
    project.open(str(PROJECT_PATH))
    links_all = project.network.links.data.copy()
    print(f"total links in project: {len(links_all)}")

    def ref_match_col(series: pd.Series) -> pd.Series:
        return series.apply(lambda r: ref_has_road(str(r) if pd.notna(r) else "", targets))

    mask = ref_match_col(links_all["osm_ref_norm"])
    mask_o = ref_match_col(links_all["osm_ref"]) if "osm_ref" in links_all.columns else pd.Series(False, index=links_all.index)
    hl = links_all[mask | mask_o].copy()
    print(f"links matching roads 383/384 (component match on osm_ref_norm or osm_ref): {len(hl)}")

    # Build car graph for SCC
    car_links = links_all[links_all["modes"].astype(str).str.contains("c", na=False, regex=False)]
    G_car = build_car_digraph(car_links)
    main_scc = largest_scc_nodes(G_car)
    print(f"car-mode digraph: |V|={G_car.number_of_nodes()} |E|={G_car.number_of_edges()} largest_SCC={len(main_scc)}")

    asg = pd.read_parquet(ASSIGNMENT_PARQUET)
    asg = asg.set_index("link_id")

    rows = []
    for _, row in hl.iterrows():
        lid = int(row["link_id"])
        a, b = int(row["a_node"]), int(row["b_node"])
        in_scc = (a in main_scc) and (b in main_scc)
        ser = asg.loc[lid] if lid in asg.index else None
        if ser is not None:
            tot = float(
                ser["wd_daily_local_tot"]
                + ser["wd_daily_external_through_tot"]
                + ser["Preload_tot"]
            )
        else:
            tot = None
        ph = (float(ser["peak_hour_vol_AB"]) + float(ser["peak_hour_vol_BA"])) if ser is not None else None
        cap_ab = float(row.get("capacity_ab") or 0)
        cap_ba = float(row.get("capacity_ba") or 0)
        vmax = max(cap_ab, cap_ba, 1e-6)
        voc = (tot / vmax) if tot is not None else None
        rows.append(
            {
                "link_id": lid,
                "osm_ref": row.get("osm_ref"),
                "osm_ref_norm": row.get("osm_ref_norm"),
                "link_type": row.get("link_type"),
                "direction": int(row.get("direction") or 0),
                "modes": row.get("modes"),
                "car_mode": car_mode(row.get("modes")),
                "speed_ab": row.get("speed_ab"),
                "speed_ba": row.get("speed_ba"),
                "capacity_ab": cap_ab,
                "capacity_ba": cap_ba,
                "lanes_ab": row.get("lanes_ab"),
                "lanes_ba": row.get("lanes_ba"),
                "a_node": a,
                "b_node": b,
                "both_ends_in_main_SCC_car": in_scc,
                "wd_daily_local_tot": float(ser["wd_daily_local_tot"]) if ser is not None else None,
                "wd_daily_ext_through_tot": float(ser["wd_daily_external_through_tot"]) if ser is not None else None,
                "preload_tot": float(ser["Preload_tot"]) if ser is not None else None,
                "modeled_total_approx": tot,
                "peak_hour_vol_sum": ph,
                "v_over_cap_daily_equiv": voc,
                "name": row.get("name"),
            }
        )

    rep = pd.DataFrame(rows).sort_values(["osm_ref_norm", "link_id"])
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print("\n=== Links on roads 383 / 384 ===")
    print(rep.to_string(index=False))

    # Low volume vs capacity for exact osm_ref components 383/384
    print("\n=== Low volume links (exact ref 383/384), total modeled < 5% of max direction capacity ===")
    low = rep[rep["modes"].astype(str).str.contains("c", na=False)].copy()
    low = low[low["capacity_ab"].notna() | low["capacity_ba"].notna()]
    low = low[low["modeled_total_approx"].notna()]
    def low_vol(r):
        tot = r["modeled_total_approx"]
        capm = max(float(r["capacity_ab"] or 0), float(r["capacity_ba"] or 0), 1)
        return tot < 0.05 * capm

    lv = low[low.apply(low_vol, axis=1)]
    print(f"count: {len(lv)}")
    if len(lv):
        print(lv[["link_id", "osm_ref_norm", "link_type", "modeled_total_approx", "capacity_ab", "capacity_ba"]].to_string(index=False))

    # Connectivity traces per road ref
    print("\n=== Connectivity along corridor (car-mode DiGraph restricted to matching links) ===")
    for road in sorted(targets):
        sub = hl[hl["osm_ref_norm"].apply(lambda r: road in split_ref_components(str(r) if pd.notna(r) else ""))].copy()
        if sub.empty:
            sub = hl[hl["osm_ref"].apply(lambda r: road in split_ref_components(str(r) if pd.notna(r) else ""))].copy()
        print(f"\n--- Road {road}: {len(sub)} link(s) in filter ---")
        if sub.empty:
            continue
        G_sub = nx.DiGraph()
        Und = nx.Graph()
        for _, row in sub.iterrows():
            for u, v in link_edges_for_routing(row):
                G_sub.add_edge(u, v, link_id=int(row["link_id"]))
                Und.add_edge(u, v)
        print(f"car-routable subgraph: |V|={G_sub.number_of_nodes()} |E|={G_sub.number_of_edges()} weakly_connected={nx.number_weakly_connected_components(G_sub)}")
        term = subgraph_terminals(Und)
        print(f"heuristic corridor endpoints (deg-1 or top degree): {term[:10]}{'...' if len(term)>10 else ''}")
        if len(term) >= 2:
            s, t = term[0], term[1]
            ok_s_t = nx.has_path(G_sub, s, t)
            ok_t_s = nx.has_path(G_sub, t, s)
            print(f"path {s}->{t}: {ok_s_t}, path {t}->{s}: {ok_t_s}")
            if ok_s_t:
                try:
                    print(f"sample shortest path len (edges): {nx.shortest_path_length(G_sub, s, t)}")
                except Exception as e:
                    print(f"path length error: {e}")
            # Both ends in global car SCC?
            s_scc = s in main_scc
            t_scc = t in main_scc
            print(f"endpoints in global main car SCC: {s}->{s_scc}, {t}->{t_scc}")
        comps = sorted(nx.weakly_connected_components(G_sub), key=len, reverse=True)
        if len(comps) > 1:
            sizes = [len(c) for c in comps[:8]]
            print(f"WARNING: multiple weak components, sizes={sizes}")

    project.close()

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
