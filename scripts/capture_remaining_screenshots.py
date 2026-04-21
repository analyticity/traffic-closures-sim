#!/usr/bin/env python3
"""Re-capture the 3 screenshots that need fixing:
- screenshot_02: Scenario panel with links added
- screenshot_03: Delta/Rozdíly view after running a scenario
- screenshot_06: Corridor diagnosis with a corridor selected
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


def wait_loaded(page, extra_s: float = 5.0):
    try:
        page.wait_for_selector(".leaflet-container", timeout=30_000)
    except Exception:
        pass
    try:
        page.locator("text=Načítání...").wait_for(state="hidden", timeout=60_000)
    except Exception:
        pass
    try:
        page.wait_for_selector("path.leaflet-interactive", timeout=60_000)
    except Exception:
        pass
    time.sleep(extra_s)


def zoom_map(page, zoom: int = 13):
    page.evaluate(f"""() => {{
        const c = document.querySelector('.leaflet-container');
        if (c && c._leaflet_map) c._leaflet_map.setView([49.195, 16.608], {zoom});
    }}""")


def save(page, name: str):
    path = OUT / name
    page.screenshot(path=str(path), full_page=False)
    print(f"  ✓ {name} ({path.stat().st_size // 1024} KB)")


def api_get(path):
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as r:
        return json.loads(r.read())


def api_post(path, data):
    payload = json.dumps(data).encode()
    req = urllib.request.Request(f"{API}{path}", data=payload,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    # Find high-volume trunk links for a meaningful scenario
    links_data = api_get("/api/links?link_types=trunk")
    feats = links_data.get("features", [])
    scored = []
    for f in feats:
        p = f["properties"]
        v = p.get("wd_daily_tot", 0) or 0
        if v > 10000:
            scored.append((v, p["link_id"], p.get("name", ""), p.get("osm_ref", ""), p.get("lanes", 2)))
    scored.sort(reverse=True)

    # Pick a few links from the same corridor (Vídeňská / ref 52)
    selected = [s for s in scored if s[3] == "52"][:4]
    if not selected:
        selected = scored[:4]
    link_ids = [s[1] for s in selected]
    print(f"Selected {len(link_ids)} links for scenario: {link_ids}")
    print(f"  Names: {[s[2] for s in selected]}")

    # Submit scenario via API and wait for it to complete
    print("Submitting scenario…")
    links_payload = [{"link_id": lid, "direction": "both", "closure_type": "full",
                      "lanes_remaining": 1} for lid in link_ids]
    resp = api_post("/api/scenarios/run", {"links": links_payload})
    job_id = resp.get("id")
    print(f"  Job ID: {job_id}")

    for _ in range(300):
        st = api_get(f"/api/scenarios/{job_id}/status")
        if st.get("status") == "done":
            print(f"  Scenario done! (elapsed: {st.get('elapsed_seconds', '?')}s)")
            break
        if st.get("status") == "error":
            print(f"  ERROR: {st.get('error')}")
            return
        time.sleep(2)
    else:
        print("  Timeout waiting for scenario")
        return

    # Fetch scenario results as GeoJSON
    results = api_get(f"/api/scenarios/{job_id}/results")
    results_json = json.dumps(results)
    print(f"  Results: {len(results.get('features', []))} features")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": VP_W, "height": VP_H}, device_scale_factor=2)
        page = ctx.new_page()

        # ══════ Screenshot 02: Scenario panel with links ══════
        print("\n2. Scenario panel…")
        page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=60_000)
        wait_loaded(page, extra_s=4)
        zoom_map(page, 13)
        wait_loaded(page, extra_s=3)

        # Disable zones
        try:
            zone_label = page.locator("label:has-text('Zóny')")
            if zone_label.count() > 0:
                zone_label.first.click()
                time.sleep(0.5)
        except Exception:
            pass

        # Inject scenario links into the Zustand store
        links_for_store = json.dumps([
            {"link_id": s[1], "link_type": "trunk", "osm_name": s[2] or "Vídeňská",
             "osm_ref": s[3], "lanes": s[4], "direction": "both",
             "closure_type": "full", "lanes_remaining": 1}
            for s in selected
        ])
        # Try injecting via the store's internal representation
        page.evaluate(f"""() => {{
            // Zustand stores are accessible via the module scope, but we need another way
            // Let's try to find the store via the devtools hook or window
            if (window.__ZUSTAND_DEVTOOLS__) {{
                console.log('zustand devtools found');
            }}
        }}""")

        # Click on a map link to trigger the scenario panel to show with content
        # Since we can't easily inject into Zustand from outside, let's click on a thick
        # map path to add a link to the scenario
        paths = page.locator("path.leaflet-interactive")
        n_paths = paths.count()
        print(f"  Found {n_paths} interactive paths")
        clicked = 0
        for i in range(min(n_paths, 600)):
            try:
                box = paths.nth(i).bounding_box()
                if not box:
                    continue
                if box["width"] < 5 and box["height"] < 5:
                    continue
                # Click on a link that's near the center of the map
                cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                if 600 < cx < 1900 and 300 < cy < 1100:
                    paths.nth(i).click(force=True)
                    time.sleep(0.5)
                    # Check if a popup appeared with "Přidat do scénáře"
                    add_btn = page.locator("text=Přidat do scénáře")
                    if add_btn.count() > 0:
                        add_btn.first.click()
                        time.sleep(0.5)
                        clicked += 1
                        if clicked >= 3:
                            break
            except Exception:
                continue
        print(f"  Added {clicked} links to scenario via UI")
        time.sleep(1)
        save(page, "screenshot_02_scenario_panel.png")

        # ══════ Screenshot 03: Run scenario and show delta ══════
        print("\n3. Delta view…")
        # Check if we have links and can run
        run_btn = page.locator("text=Spustit simulaci")
        if run_btn.count() > 0 and run_btn.first.is_visible():
            print("  Running scenario via UI…")
            run_btn.first.click()
            # Wait for the simulation to finish (poll for "výsledky" badge or delta button)
            try:
                page.locator("text=Simulace běží").wait_for(state="hidden", timeout=600_000)
            except Exception:
                pass
            time.sleep(3)

            # Switch to delta view
            delta_btn = page.locator("text=Rozdíly")
            if delta_btn.count() > 0:
                delta_btn.first.click()
                wait_loaded(page, extra_s=4)
        else:
            print("  No links added, injecting results directly…")
            # If UI clicking didn't work, try to inject results JSON via page.evaluate
            page.evaluate(f"""(resultsStr) => {{
                // Alternative: dispatch a custom event that the store might listen to
                window.__injected_scenario_results = JSON.parse(resultsStr);
            }}""", results_json)

        save(page, "screenshot_03_delta_view.png")

        # ══════ Screenshot 06: Corridor diagnosis with corridor selected ══════
        print("\n6. Corridor diagnosis…")
        page.goto(f"{BASE}/diagnostics", wait_until="domcontentloaded", timeout=60_000)
        wait_loaded(page, extra_s=4)

        # Click on Diagnóza tab
        diag_btn = page.locator("button:has-text('Diagnóza')")
        if diag_btn.count() > 0:
            diag_btn.first.click()
        time.sleep(3)

        # Select a corridor from the dropdown
        select = page.locator("select")
        if select.count() > 0:
            # Get options
            options = select.first.locator("option")
            n_opts = options.count()
            print(f"  {n_opts} corridor options")
            # Select a meaningful corridor (skip the first empty/placeholder option)
            if n_opts > 1:
                # Pick "Opuštěná" or "Stará dálnice" or the 2nd option
                for oi in range(n_opts):
                    val = options.nth(oi).get_attribute("value") or ""
                    txt = options.nth(oi).text_content() or ""
                    if any(name in txt for name in ["Opuštěná", "Černovická", "Stará"]):
                        select.first.select_option(value=val)
                        print(f"  Selected: {txt}")
                        break
                else:
                    # Just pick the 2nd option
                    val = options.nth(1).get_attribute("value") or ""
                    txt = options.nth(1).text_content() or ""
                    select.first.select_option(value=val)
                    print(f"  Selected: {txt}")

        wait_loaded(page, extra_s=6)
        zoom_map(page, 13)
        wait_loaded(page, extra_s=4)
        save(page, "screenshot_06_corridor_diagnosis.png")

        browser.close()

    print(f"\n✓ Re-captured 3 screenshots in {OUT}")


if __name__ == "__main__":
    main()
