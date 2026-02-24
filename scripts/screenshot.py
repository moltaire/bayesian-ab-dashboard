#!/usr/bin/env python3
"""Take a screenshot of the running Streamlit app and save it to docs/screenshot.png."""

import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

STREAMLIT_URL = "http://localhost:8501"
SCREENSHOT_PATH = Path("docs/screenshot.png")
VIEWPORT = {"width": 1400, "height": 900}


def wait_for_streamlit(timeout: int = 60) -> None:
    health_url = f"{STREAMLIT_URL}/_stcore/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as r:
                if r.status == 200:
                    print("Streamlit is ready.")
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"Streamlit did not respond within {timeout}s")


def take_screenshot() -> None:
    SCREENSHOT_PATH.parent.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(STREAMLIT_URL)
        # Wait for the main heading to appear, then let charts finish rendering.
        page.wait_for_selector("h1", timeout=30_000)
        page.wait_for_timeout(3_000)
        page.screenshot(path=str(SCREENSHOT_PATH))
        browser.close()
    print(f"Screenshot saved to {SCREENSHOT_PATH}")


if __name__ == "__main__":
    wait_for_streamlit()
    take_screenshot()
