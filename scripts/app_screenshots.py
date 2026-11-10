"""Drive a RUNNING Kormat app and save screenshots:
   uv run --with playwright python scripts/app_screenshots.py URL [prefix]
Chromium comes from ~/.cache/ms-playwright. Samples come from the app's /api/samples (train/val only)."""
import glob
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "app_real"
OUT = Path(__file__).resolve().parents[1] / "report" / "figures"
CHROME = sorted(glob.glob(str(Path.home() / ".cache/ms-playwright/chromium-*/chrome-linux/chrome")))[-1]
MAX_CRIT = 4

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1400, "height": 900})
    pg.goto(URL)
    pg.wait_for_selector("#health.ready", timeout=600_000)
    pg.wait_for_function("document.querySelectorAll('#sample option').length > 2")
    pg.select_option("#sample", index=2)  # option 0 = placeholder, 1 = built-in demo, 2 = first val sample
    while pg.locator("#crits .crit").count() > MAX_CRIT:
        pg.locator("#crits [data-del]").last.click()
    pg.select_option("#mode", "kv_cached")
    pg.screenshot(path=str(OUT / f"{PREFIX}_main.png"))
    pg.click("#run")
    pg.wait_for_selector("#rows tr", timeout=900_000)
    n = pg.locator("#rows tr").count()
    pick = 0
    for i in range(n):  # prefer a row that cites evidence
        if "none" not in pg.locator("#rows tr").nth(i).locator("td").nth(4).inner_text():
            pick = i
            break
    pg.locator("#rows tr").nth(pick).click(); pg.wait_for_timeout(400)
    pg.screenshot(path=str(OUT / f"{PREFIX}_results.png"), full_page=True)
    # lab: kv_cached + speculative (gamma 4) + no_cache, 24 new tokens
    pg.click("[data-tab=lab]")
    pg.uncheck("#labmodes input[value=hf_generate]")
    pg.fill("#labmax", "24"); pg.fill("#labgamma", "4")
    pg.click("#runlab")
    pg.wait_for_selector("#ch1", timeout=900_000); pg.wait_for_timeout(1200)
    pg.screenshot(path=str(OUT / f"{PREFIX}_lab.png"), full_page=True)
    # exported report
    rid = pg.evaluate("fetch('/api/history').then(r=>r.json()).then(d=>d.reports[0].id)")
    rp = b.new_page(viewport={"width": 1000, "height": 1200})
    rp.goto(f"{URL}/api/report/{rid}.html"); rp.screenshot(path=str(OUT / f"{PREFIX}_report.png"), full_page=True)
    # validation error (empty note)
    pg.click("[data-tab=screen]"); pg.fill("#note", ""); pg.click("#run")
    pg.wait_for_selector("#screenerr .err"); pg.screenshot(path=str(OUT / f"{PREFIX}_error.png"))
    b.close()
print("saved", sorted(x.name for x in OUT.glob(f"{PREFIX}_*.png")))
