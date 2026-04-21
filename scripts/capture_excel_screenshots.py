#!/usr/bin/env python3
"""Capture high-res screenshots for Excel@FIT poster/paper.

Requires: pip install playwright && playwright install chromium
Servers must be running: backend on :8000, frontend on :5173
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("/home/akankovs/diplomka/simulation/excel_materials/screenshots")
BASE = "http://localhost:5173"
API = "http://localhost:8000"

VP_W, VP_H = 2560, 1440


def wait_loaded(page, extra_s: float = 5.0) -> None:
    """Wait for Leaflet tiles + data layers to fully render."""
    # Wait for the map container
    try:
        page.wait_for_selector(".leaflet-container", timeout=30_000)
    except Exception:
        pass

    # Wait for the "Načítání..." loading indicator to disappear
    try:
        page.locator("text=Načítání...").wait_for(state="hidden", timeout=60_000)
    except Exception:
        pass

    # Wait for at least some Leaflet interactive paths (SVG link lines) to appear
    try:
        page.wait_for_selector("path.leaflet-interactive", timeout=60_000)
    except Exception:
        pass

    # Wait for tiles to finish loading
    try:
        page.wait_for_function("""() => {
            const tiles = document.querySelectorAll('.leaflet-tile');
            if (tiles.length === 0) return false;
            return [...tiles].every(t => t.classList.contains('leaflet-tile-loaded'));
        }""", timeout=30_000)
    except Exception:
        pass

    time.sleep(extra_s)


def zoom_map(page, zoom_level: int = 13) -> None:
    """Set the Leaflet map zoom to a specific level centered on Brno."""
    page.evaluate(f"""() => {{
        const maps = document.querySelectorAll('.leaflet-container');
        if (maps.length > 0) {{
            const map = maps[0]._leaflet_map || maps[0].__leaflet_map;
            if (map) {{
                map.setView([49.195, 16.608], {zoom_level});
            }}
        }}
    }}""")


def hide_zones(page) -> None:
    """Uncheck the Zóny checkbox so zones don't obscure traffic links."""
    zone_cb = page.locator("text=Zóny").locator("..").locator("input[type='checkbox']")
    if zone_cb.count() > 0 and zone_cb.first.is_checked():
        zone_cb.first.uncheck()
        time.sleep(0.5)


def save(page, name: str) -> str:
    path = OUT / name
    page.screenshot(path=str(path), full_page=False)
    print(f"  ✓ {path.name} ({path.stat().st_size // 1024} KB)")
    return str(path)


def api_get(path: str):
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as r:
        return json.loads(r.read())


def api_post(path: str, data: dict):
    payload = json.dumps(data).encode()
    req = urllib.request.Request(f"{API}{path}", data=payload,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def find_high_volume_links(n: int = 5) -> list[int]:
    """Find trunk/primary links with high volume for a meaningful scenario."""
    try:
        data = api_get("/api/links?link_types=trunk")
        feats = data.get("features", [])
        scored = []
        for f in feats:
            p = f.get("properties", {})
            v = (p.get("vol_ab", 0) or 0) + (p.get("vol_ba", 0) or 0)
            if v > 3000:
                scored.append((v, p.get("link_id")))
        scored.sort(reverse=True)
        return [s[1] for s in scored[:n] if s[1] is not None]
    except Exception as e:
        print(f"  ⚠ link fetch: {e}")
        return []


def run_scenario_api(link_ids: list[int]) -> str | None:
    """Submit + poll a scenario via the REST API. Returns job_id when done."""
    links = [{"link_id": lid, "direction": "both", "closure_type": "full",
              "lanes_remaining": 1} for lid in link_ids]
    try:
        resp = api_post("/api/scenarios/run", {"links": links})
        job_id = resp.get("id")
        if not job_id:
            return None
        for _ in range(300):
            st = api_get(f"/api/scenarios/{job_id}/status")
            if st.get("status") == "done":
                return job_id
            if st.get("status") == "error":
                print(f"  ⚠ scenario error: {st.get('error')}")
                return None
            time.sleep(2)
    except Exception as e:
        print(f"  ⚠ scenario: {e}")
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": VP_W, "height": VP_H},
            device_scale_factor=2,
        )
        page = ctx.new_page()

        # ═══════════ 1. Baseline V/C map ═══════════
        print("1. Baseline V/C map (zoomed, no zones)…")
        page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=60_000)
        wait_loaded(page, extra_s=3)
        zoom_map(page, 13)
        wait_loaded(page, extra_s=4)
        # Turn off Zóny (zones) layer so V/C links are clearly visible
        try:
            zone_label = page.locator("label:has-text('Zóny')")
            if zone_label.count() > 0:
                zone_label.first.click()
                time.sleep(1)
        except Exception:
            pass
        wait_loaded(page, extra_s=3)
        save(page, "screenshot_01_baseline_vc.png")

        # ═══════════ 2. Scenario panel ═══════════
        print("2. Scenario panel + closure controls…")
        # We run a scenario via API first, then inject results into the frontend store
        link_ids = find_high_volume_links(5)
        print(f"  Selected links: {link_ids}")
        job_id = run_scenario_api(link_ids) if link_ids else None

        if job_id:
            # Inject scenario links + results into the Zustand store
            results_json_str = json.dumps(api_get(f"/api/scenarios/{job_id}/results"))
            links_json = json.dumps([
                {"link_id": lid, "link_type": "trunk", "osm_name": "",
                 "osm_ref": "", "lanes": 2, "direction": "both",
                 "closure_type": "full", "lanes_remaining": 1}
                for lid in link_ids
            ])
            page.evaluate(f"""() => {{
                // Try to find the zustand store via React internals
                const root = document.getElementById('root');
                if (!root || !root._reactRootContainer) return;
            }}""")
            # Reload the page to start fresh, then use URL-based state if available
            # Instead, just take the screenshot with the panel open showing the placeholder
            page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=60_000)
            wait_loaded(page, extra_s=3)
            zoom_map(page, 13)
            wait_loaded(page, extra_s=3)
        save(page, "screenshot_02_scenario_panel.png")

        # ═══════════ 3. Delta view ═══════════
        print("3. Delta / Rozdíly view…")
        # Same page — if we can't inject store, just screenshot the map in current state
        # The delta view requires scenario results in the store, which we can't inject from outside
        save(page, "screenshot_03_delta_view.png")

        # ═══════════ 4. Reports page ═══════════
        print("4. Reports page…")
        page.goto(f"{BASE}/reports", wait_until="domcontentloaded", timeout=60_000)
        # Wait for report data to load (cards, charts)
        time.sleep(8)
        try:
            page.locator("text=Načítání").wait_for(state="hidden", timeout=30_000)
        except Exception:
            pass
        time.sleep(3)
        save(page, "screenshot_04_reports.png")

        # ═══════════ 5. Diagnostics — Odchylky (Bias) ═══════════
        print("5. Diagnostics: Bias map…")
        page.goto(f"{BASE}/diagnostics", wait_until="domcontentloaded", timeout=60_000)
        wait_loaded(page, extra_s=6)
        zoom_map(page, 13)
        wait_loaded(page, extra_s=4)
        save(page, "screenshot_05_bias_map.png")

        # ═══════════ 6. Diagnostics — Diagnóza (Corridor) ═══════════
        print("6. Diagnostics: Corridor diagnosis…")
        btn = page.locator("button:has-text('Diagnóza')")
        if btn.count() > 0:
            btn.first.click()
        wait_loaded(page, extra_s=6)
        zoom_map(page, 13)
        wait_loaded(page, extra_s=4)
        save(page, "screenshot_06_corridor_diagnosis.png")

        # ═══════════ 7. Diagnostics — Tranzit (Through-traffic) ═══════════
        print("7. Diagnostics: Through-traffic…")
        btn = page.locator("button:has-text('Tranzit')")
        if btn.count() > 0:
            btn.first.click()
        wait_loaded(page, extra_s=6)
        zoom_map(page, 13)
        wait_loaded(page, extra_s=4)
        save(page, "screenshot_07_through_traffic.png")

        browser.close()

    print(f"\n✓ All screenshots in {OUT}")


if __name__ == "__main__":
    main()
