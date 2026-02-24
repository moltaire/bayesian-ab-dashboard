#!/usr/bin/env python3
"""Take a dark-mode screenshot of the Streamlit app and add a subtle drop shadow."""

import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from playwright.sync_api import sync_playwright

STREAMLIT_URL = "http://localhost:8501"
SCREENSHOT_PATH = Path("docs/screenshot.png")
VIEWPORT = {"width": 1600, "height": 1200}


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


def take_screenshot(path: Path) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport=VIEWPORT, color_scheme="dark")
        page = ctx.new_page()
        page.goto(STREAMLIT_URL)
        page.wait_for_selector("h1", timeout=30_000)
        page.wait_for_timeout(3_000)

        # Add two batches so the screenshot shows live posteriors.
        btn = page.get_by_text("Add batch and update posteriors")
        for _ in range(2):
            btn.click()
            page.wait_for_timeout(2_000)  # let Streamlit re-render between batches

        page.wait_for_timeout(1_000)  # let charts settle
        page.screenshot(path=str(path))
        browser.close()
    print(f"Raw screenshot saved to {path}")


def add_shadow(src: Path, dst: Path) -> None:
    shot = Image.open(src).convert("RGBA")
    sw, sh = shot.size

    SHADOW_BLUR = 8
    SHADOW_OFFSET_Y = 6
    MARGIN = SHADOW_BLUR + SHADOW_OFFSET_Y + 6

    canvas_w = sw + MARGIN * 2
    canvas_h = sh + MARGIN * 2
    fx, fy = MARGIN, MARGIN

    # ── Drop shadow ───────────────────────────────────────────────────────────
    shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rectangle(
        [fx, fy + SHADOW_OFFSET_Y, fx + sw, fy + sh + SHADOW_OFFSET_Y],
        fill=(0, 0, 0, 160),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))

    # ── Composite: transparent canvas → shadow → screenshot ───────────────────
    result = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    result = Image.alpha_composite(result, shadow)
    result.paste(shot, (fx, fy))

    result.save(dst)
    print(f"Screenshot saved to {dst}")


if __name__ == "__main__":
    SCREENSHOT_PATH.parent.mkdir(exist_ok=True)
    raw = SCREENSHOT_PATH.parent / "_raw.png"
    wait_for_streamlit()
    take_screenshot(raw)
    add_shadow(raw, SCREENSHOT_PATH)
    raw.unlink()
