#!/usr/bin/env python3
"""Take a dark-mode screenshot of the Streamlit app and wrap it in a browser frame."""

import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
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


def add_browser_frame(src: Path, dst: Path) -> None:
    shot = Image.open(src).convert("RGBA")
    sw, sh = shot.size

    TITLE_H = 44        # height of the title bar (traffic lights)
    ADDR_H = 36         # height of the address bar row
    CHROME_H = TITLE_H + ADDR_H
    CORNER_R = 12       # border radius of the window
    FRAME_BG = (32, 33, 36)
    ADDR_BG = (55, 56, 60)
    SHADOW_BLUR = 28
    SHADOW_OFFSET_Y = 14
    MARGIN = SHADOW_BLUR + SHADOW_OFFSET_Y + 10

    frame_w, frame_h = sw, sh + CHROME_H
    canvas_w = frame_w + MARGIN * 2
    canvas_h = frame_h + MARGIN * 2
    fx, fy = MARGIN, MARGIN

    # ── Drop shadow ───────────────────────────────────────────────────────────
    shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [fx, fy + SHADOW_OFFSET_Y, fx + frame_w, fy + frame_h + SHADOW_OFFSET_Y],
        radius=CORNER_R,
        fill=(0, 0, 0, 190),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))

    # ── Browser window ────────────────────────────────────────────────────────
    frame = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(frame)

    # Fill entire window shape (provides rounded bottom corners for screenshot)
    fdraw.rounded_rectangle([0, 0, frame_w, frame_h], radius=CORNER_R, fill=FRAME_BG)

    # Paste screenshot into the content area
    frame.paste(shot, (0, CHROME_H))

    # Redraw chrome bar on top (covers the screenshot's top edge)
    fdraw.rectangle([0, 0, frame_w, CHROME_H], fill=FRAME_BG)

    # Traffic light dots
    for i, colour in enumerate(["#FF5F57", "#FEBC2E", "#28C840"]):
        dx, dy = 20 + i * 22, TITLE_H // 2
        fdraw.ellipse([dx - 6, dy - 6, dx + 6, dy + 6], fill=colour)

    # Address bar (centered pill)
    cx = frame_w // 2
    fdraw.rounded_rectangle(
        [cx - 150, TITLE_H + 6, cx + 150, TITLE_H + ADDR_H - 6],
        radius=4,
        fill=ADDR_BG,
    )

    # Clip the whole window to the rounded rectangle shape
    mask = Image.new("L", (frame_w, frame_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, frame_w, frame_h], radius=CORNER_R, fill=255)
    frame.putalpha(mask)

    # ── Composite: transparent canvas → shadow → frame ────────────────────────
    result = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    result = Image.alpha_composite(result, shadow)
    result.paste(frame, (fx, fy), mask=frame)

    result.save(dst)  # RGBA PNG — transparent outside the frame
    print(f"Framed screenshot saved to {dst}")


if __name__ == "__main__":
    SCREENSHOT_PATH.parent.mkdir(exist_ok=True)
    raw = SCREENSHOT_PATH.parent / "_raw.png"
    wait_for_streamlit()
    take_screenshot(raw)
    add_browser_frame(raw, SCREENSHOT_PATH)
    raw.unlink()
