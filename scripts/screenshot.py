#!/usr/bin/env python3
"""Take a dark-mode screenshot of the Streamlit app with rounded corners and a drop shadow."""

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
        page.screenshot(path=str(path))
        browser.close()
    print(f"Raw screenshot saved to {path}")


def add_rounded_shadow(src: Path, dst: Path) -> None:
    shot = Image.open(src).convert("RGBA")
    sw, sh = shot.size

    CORNER_R = 12
    SHADOW_BLUR = 28
    SHADOW_OFFSET_Y = 14
    MARGIN = SHADOW_BLUR + SHADOW_OFFSET_Y + 10

    canvas_w = sw + MARGIN * 2
    canvas_h = sh + MARGIN * 2
    fx, fy = MARGIN, MARGIN

    # ── Drop shadow ───────────────────────────────────────────────────────────
    shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [fx, fy + SHADOW_OFFSET_Y, fx + sw, fy + sh + SHADOW_OFFSET_Y],
        radius=CORNER_R,
        fill=(0, 0, 0, 190),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))

    # ── Screenshot clipped to rounded corners ─────────────────────────────────
    mask = Image.new("L", (sw, sh), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, sw, sh], radius=CORNER_R, fill=255)
    shot.putalpha(mask)

    # ── Composite: transparent canvas → shadow → screenshot ───────────────────
    result = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    result = Image.alpha_composite(result, shadow)
    result.paste(shot, (fx, fy), mask=shot)

    result.save(dst)
    print(f"Screenshot saved to {dst}")


if __name__ == "__main__":
    SCREENSHOT_PATH.parent.mkdir(exist_ok=True)
    raw = SCREENSHOT_PATH.parent / "_raw.png"
    wait_for_streamlit()
    take_screenshot(raw)
    add_rounded_shadow(raw, SCREENSHOT_PATH)
    raw.unlink()
