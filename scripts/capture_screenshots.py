"""Capture screenshots of the running app for the README.

    python scripts/capture_screenshots.py

Starts the Streamlit app on a spare port with the mock providers forced, drives
it with a headless browser, and writes PNGs to ``docs/screenshots/``.

Mock providers are forced deliberately: the screenshots should be reproducible
and must not depend on a reviewer's API keys or spend the author's quota.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTPUT_DIR = REPO_ROOT / "docs" / "screenshots"
PORT = 8556
BASE_URL = f"http://localhost:{PORT}"
VIEWPORT = {"width": 1600, "height": 1200}

# Streamlit renders progressively, so each step waits for the network to settle
# rather than for a fixed delay.
SETTLE_MS = 3500  # gallery images load after the page settles


def start_app(use_mock: bool) -> subprocess.Popen[bytes]:
    """Launch Streamlit with settings pinned for reproducible screenshots.

    Args:
        use_mock: Force the deterministic model driver. When False the configured
            provider is used, so the screenshots show real model prose.

    Returns:
        The running process.
    """
    environment = {
        **os.environ,
        "IMAGE_FALLBACK_MODE": "remote",
        # Pinned so the screenshots show the calibrated value regardless of what
        # the developer happens to have in .env.
        "ROUTER_SIMILARITY_THRESHOLD": "0.07",
        "MOCK_WEATHER_LATENCY_MS": "500",
        "MOCK_IMAGE_LATENCY_MS": "700",
        "MOCK_SEARCH_LATENCY_MS": "400",
        "TOOL_MAX_ATTEMPTS": "1",
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    if use_mock:
        environment["LLM_PROVIDER"] = "mock"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(REPO_ROOT / "src" / "travel_agent" / "ui" / "app.py"),
            "--server.port",
            str(PORT),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(10)
    return process


def downsample(max_width: int = 1000, colours: int = 128) -> None:
    """Shrink the captured PNGs so the repository stays small.

    Captured at a 2x device scale for sharp text, each screenshot is around a
    megabyte - roughly 6 MB for the set, which looks careless in a submitted
    repository. Resizing to a sensible width and quantising to a 256-colour
    palette keeps the text legible while cutting the total by an order of
    magnitude; screenshots of a mostly-white interface quantise very well.

    Args:
        max_width: Width to scale down to, preserving aspect ratio.
        colours: Palette size. Screenshots of a mostly-flat interface stay
            legible well below the full 256.
    """
    from PIL import Image

    print()
    print("downsampling:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        before = path.stat().st_size
        with Image.open(path) as image:
            if image.width > max_width:
                height = round(image.height * max_width / image.width)
                image = image.resize((max_width, height), Image.LANCZOS)
            image.convert("RGB").quantize(colors=colours, method=Image.MEDIANCUT).save(
                path, "PNG", optimize=True
            )
        print(f"  {path.name:<28} {before / 1024:7.0f} KB -> {path.stat().st_size / 1024:6.0f} KB")


def main() -> int:
    """Capture every demo state.

    Returns:
        A process exit code.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. Run: pip install playwright && playwright install chromium"
        )
        return 1

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Capture screenshots of the running app.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force the mock model driver instead of the configured provider.",
    )
    args = parser.parse_args()

    print(f"starting the app ({'mock model' if args.mock else 'configured provider'}) ...")
    process = start_app(use_mock=args.mock)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
            page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(SETTLE_MS)

            def shoot(name: str, description: str) -> None:
                path = OUTPUT_DIR / f"{name}.png"
                page.screenshot(path=str(path), full_page=True)
                print(f"  {path.name:<28} {path.stat().st_size / 1024:6.0f} KB   {description}")

            def ask(question: str) -> None:
                box = page.get_by_placeholder("Tell me about Tokyo")
                box.fill(question)
                page.get_by_role("button", name="Send", exact=True).click()
                page.wait_for_timeout(SETTLE_MS + 6000)

            shoot("01-empty-state", "first load, before any question")

            ask("Tell me about Tokyo")
            shoot("02-in-store-city", "Tokyo: routed to the vector store")

            # The trace panel tabs, which are the highest-value view.
            page.get_by_role("tab", name="Parallelism").click()
            page.wait_for_timeout(1200)
            shoot("03-trace-parallelism", "measured fan-out speed-up")

            page.get_by_role("tab", name="Routing").click()
            page.wait_for_timeout(1200)
            shoot("04-trace-routing", "routing decision with scores")

            ask("what about next week?")
            page.get_by_role("tab", name="Memory").click()
            page.wait_for_timeout(1200)
            shoot("05-follow-up-skipped", "follow-up: skipped branches and work avoided")

            ask("Tell me about Kyoto")
            shoot("06-out-of-store-city", "Kyoto: routed to web search")

            # Break the weather API from the sidebar and re-ask.
            page.get_by_text("Break the weather API").click()
            page.wait_for_timeout(2000)
            ask("Tell me about Paris")
            shoot("07-weather-api-broken", "graceful degradation: summary and images survive")

            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    downsample()

    total = sum(path.stat().st_size for path in OUTPUT_DIR.glob("*.png"))
    count = len(list(OUTPUT_DIR.glob("*.png")))
    print()
    print(f"wrote {count} screenshots to {OUTPUT_DIR} ({total / 1024:.0f} KB total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
