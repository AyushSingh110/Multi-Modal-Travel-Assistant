"""Regenerate the bundled offline fallback images, and optionally ATTRIBUTION.md."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from html import unescape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "images"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

WIDTH, HEIGHT = 800, 500

# slug, display name, gradient top, gradient bottom, the four labels
CITIES: list[tuple[str, str, tuple[int, int, int], tuple[int, int, int], list[str]]] = [
    (
        "paris",
        "Paris",
        (54, 68, 112),
        (188, 152, 118),
        ["Eiffel Tower", "The Louvre", "Notre-Dame", "Arc de Triomphe"],
    ),
    (
        "tokyo",
        "Tokyo",
        (34, 44, 78),
        (196, 96, 104),
        ["Shinjuku", "Shibuya Crossing", "Senso-ji", "Tokyo Tower"],
    ),
    (
        "new-york",
        "New York",
        (38, 52, 76),
        (150, 158, 176),
        ["Lower Manhattan", "Times Square", "Brooklyn Bridge", "Central Park"],
    ),
    (
        "generic",
        "Travel",
        (48, 62, 70),
        (140, 156, 148),
        ["Cityscape", "Old town", "Waterfront", "Market"],
    ),
]

# Commons filename -> (city, caption) for the attribution table.
COMMONS_FILES: dict[str, tuple[str, str]] = {
    "Tour_Eiffel_Wikimedia_Commons.jpg": ("Paris", "The Eiffel Tower from the Champ de Mars"),
    "Louvre_Museum_Wikimedia_Commons.jpg": ("Paris", "The Louvre and its glass pyramid"),
    "Notre-Dame_de_Paris_2013-07-24.jpg": ("Paris", "Notre-Dame de Paris on the Ile de la Cite"),
    "Arc_de_Triomphe,_Paris_21_October_2010.jpg": ("Paris", "The Arc de Triomphe"),
    "Skyscrapers_of_Shinjuku_2009_January.jpg": ("Tokyo", "The skyscrapers of Nishi-Shinjuku"),
    "Shibuya_Crossing.jpg": ("Tokyo", "Shibuya scramble crossing"),
    "Asakusa_Sensoji.jpg": ("Tokyo", "Senso-ji temple in Asakusa"),
    "Tokyo_Tower_and_around_Skyscrapers.jpg": ("Tokyo", "Tokyo Tower above the Minato skyline"),
    "Lower_Manhattan_skyline_-_June_2017.jpg": ("New York", "The Lower Manhattan skyline"),
    "Times_Square,_New_York_City_(HDR).jpg": ("New York", "Times Square after dark"),
    "Brooklyn_Bridge_Postdlf.jpg": ("New York", "Brooklyn Bridge across the East River"),
    "Central_Park_-_The_Pond_(48377220157).jpg": ("New York", "The Pond in Central Park"),
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a truetype font, falling back to the bitmap default.

    Args:
        size: Point size.

    Returns:
        A usable font object.
    """
    for candidate in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_placeholder(
    path: Path,
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
    city: str,
    label: str,
) -> None:
    """Render one gradient placeholder image with a suggested skyline.

    Args:
        path: Destination PNG path.
        top: Gradient colour at the top of the frame.
        bottom: Gradient colour at the horizon.
        city: City name drawn large.
        label: Landmark name drawn beneath it.
    """
    image = Image.new("RGB", (WIDTH, HEIGHT), top)
    draw = ImageDraw.Draw(image)

    for y in range(HEIGHT):
        blend = y / HEIGHT
        draw.line(
            [(0, y), (WIDTH, y)],
            fill=tuple(int(top[i] * (1 - blend) + bottom[i] * blend) for i in range(3)),
        )

    horizon = int(HEIGHT * 0.72)
    x = 0
    for index, factor in enumerate(
        [0.30, 0.46, 0.22, 0.55, 0.34, 0.62, 0.28, 0.44, 0.36, 0.52, 0.26]
    ):
        width = 40 + (index * 17) % 55
        shade = 26 + (index * 9) % 30
        draw.rectangle(
            [x, horizon - int(HEIGHT * factor), x + width, horizon],
            fill=(shade, shade + 6, shade + 14),
        )
        x += width + 12
        if x > WIDTH:
            break
    draw.rectangle([0, horizon, WIDTH, HEIGHT], fill=(22, 26, 34))

    draw.text((36, 34), city, font=_font(46), fill=(245, 245, 248))
    draw.text((36, 92), label, font=_font(24), fill=(214, 218, 228))
    draw.text(
        (36, HEIGHT - 46),
        "offline fallback image - bundled with this repository",
        font=_font(16),
        fill=(150, 158, 172),
    )
    image.save(path, "PNG", optimize=True)


def generate_images() -> int:
    """Write every placeholder image.

    Returns:
        Total bytes written.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for slug, name, top, bottom, labels in CITIES:
        for index, label in enumerate(labels, start=1):
            path = OUTPUT_DIR / f"{slug}-{index}.png"
            render_placeholder(path, top, bottom, name, label)
            total += path.stat().st_size
            print(f"  {path.name:<18} {path.stat().st_size / 1024:6.1f} KB   {label}")
    return total


def _strip_html(value: str) -> str:
    """Reduce an HTML metadata fragment to plain text.

    Args:
        value: Raw metadata value from the Commons API.

    Returns:
        Plain text with collapsed whitespace.
    """
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def fetch_licences() -> dict[str, dict[str, str]]:
    """Read photographer and licence data for every Commons photograph.

    Returns:
        A mapping of filename to metadata.
    """
    import httpx

    metadata: dict[str, dict[str, str]] = {}
    params_for = lambda name: {  # noqa: E731 - a local shorthand reads better here
        "action": "query",
        "titles": f"File:{name}",
        "prop": "imageinfo",
        "iiprop": "extmetadata|url",
        "format": "json",
    }

    with httpx.Client(timeout=30.0, headers={"User-Agent": "travel-agent/1.0"}) as client:
        for filename in COMMONS_FILES:
            response = client.get(COMMONS_API, params=params_for(filename))
            if response.status_code == 429:
                wait = float(response.headers.get("retry-after", 5))
                print(f"  rate limited, waiting {wait:.0f}s ...")
                time.sleep(wait)
                response = client.get(COMMONS_API, params=params_for(filename))
            response.raise_for_status()

            page = next(iter(response.json()["query"]["pages"].values()))
            info = page.get("imageinfo", [{}])[0]
            extra = info.get("extmetadata", {})
            metadata[filename] = {
                "artist": _strip_html(extra.get("Artist", {}).get("value", "")) or "Unknown",
                "licence": _strip_html(extra.get("LicenseShortName", {}).get("value", ""))
                or "Unknown",
                "licence_url": _strip_html(extra.get("LicenseUrl", {}).get("value", "")),
                "description_url": info.get("descriptionurl", ""),
            }
            print(f"  {filename[:44]:<46} {metadata[filename]['licence']}")
            time.sleep(2.0)
    return metadata


def write_attribution(metadata: dict[str, dict[str, str]]) -> None:
    """Write ATTRIBUTION.md from fetched licence data.

    Args:
        metadata: Output of :func:`fetch_licences`.
    """
    lines = [
        "# Image attribution",
        "",
        "The gallery shows photographs hosted on Wikimedia Commons. They are used",
        "under the licences below, with the photographer credited in the interface",
        "alongside each image. Licence and author data was read from the Commons",
        "API rather than transcribed by hand.",
        "",
        "## Photographs shown in the gallery (loaded from Wikimedia Commons)",
        "",
        "| City | Photograph | Photographer | Licence |",
        "|---|---|---|---|",
    ]
    for filename, meta in metadata.items():
        city, caption = COMMONS_FILES[filename]
        licence = meta["licence"]
        licence_cell = f"[{licence}]({meta['licence_url']})" if meta["licence_url"] else licence
        page = meta["description_url"]
        lines.append(
            f"| {city} | {f'[{caption}]({page})' if page else caption} "
            f"| {meta['artist']} | {licence_cell} |"
        )

    lines += [
        "",
        "## Bundled fallback images",
        "",
        "The `*.png` files in this directory are **not** the photographs above.",
        "They are simple generated placeholders, around 12 KB each, committed so the",
        "gallery still renders a correct layout when Wikimedia Commons is unreachable",
        "- a blocked network, an offline demo, or captive-portal wifi. They are",
        "original output of `scripts/make_fallback_images.py` and carry no",
        "third-party rights.",
        "",
        "## Placeholder photography for uncurated cities",
        "",
        "Cities with no curated set (Kyoto, Snohomish, anything reached through web",
        "search) are illustrated with seeded images from picsum.photos, captioned",
        "explicitly as representative imagery rather than photographs of that city.",
        "",
    ]
    path = OUTPUT_DIR / "ATTRIBUTION.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"  written: {path.relative_to(REPO_ROOT)}")


def main() -> int:
    """Regenerate the fallback assets.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description="Regenerate offline fallback images.")
    parser.add_argument(
        "--with-attribution",
        action="store_true",
        help="Also re-read licence data from the Commons API and rewrite ATTRIBUTION.md.",
    )
    args = parser.parse_args()

    print("GENERATING FALLBACK IMAGES")
    total = generate_images()
    print(f"\n  total bundled size: {total / 1024:.1f} KB")

    if args.with_attribution:
        print("\nFETCHING LICENCE METADATA (rate limited, this takes a minute)")
        # Fetch once and reuse. The Commons API rate limits aggressively - this
        # script hit a 429 on its first run - so calling it twice is a real bug,
        # not just waste.
        metadata = fetch_licences()
        write_attribution(metadata)
        licences_path = OUTPUT_DIR / "licences.json"
        licences_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"  written: {licences_path.relative_to(REPO_ROOT)}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
