"""Export the compiled graph to graph.png and graph.mmd."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from travel_agent.config.settings import Settings  # noqa: E402
from travel_agent.graph.builder import build_dependencies, build_graph  # noqa: E402
from travel_agent.graph.diagram import annotate_mermaid, render_png  # noqa: E402
from travel_agent.logging_setup import configure_logging, get_logger  # noqa: E402

logger = get_logger("export_graph")

PNG_PATH = REPO_ROOT / "graph.png"
MMD_PATH = REPO_ROOT / "graph.mmd"
ASCII_PATH = REPO_ROOT / "graph.txt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Export the LangGraph topology.")
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if the files already exist."
    )
    parser.add_argument("--print", action="store_true", help="Print the mermaid source.")
    return parser.parse_args()


def main() -> int:
    """Export the diagram.

    Returns:
        A process exit code. Zero even when only the mermaid fallback was written,
        because the topology is still recorded.
    """
    args = parse_args()
    configure_logging("WARNING")

    if PNG_PATH.exists() and MMD_PATH.exists() and not args.force:
        print(f"graph.png and graph.mmd already exist ({PNG_PATH.stat().st_size / 1024:.0f} KB).")
        print("Nothing to do. Use --force to regenerate.")
        if args.print:
            print()
            print(MMD_PATH.read_text(encoding="utf-8"))
        return 0

    # The diagram describes the topology, which does not depend on live providers,
    # so it is built with mocks and no network.
    settings = Settings(_env_file=None, image_fallback_mode="local")
    app = build_graph(build_dependencies(settings))
    graph = app.get_graph()

    # Generated from the compiled graph, then labelled so a reader can tell the
    # XOR knowledge branches from the concurrent tool branches.
    mermaid = annotate_mermaid(graph.draw_mermaid())
    MMD_PATH.write_text(mermaid, encoding="utf-8", newline="\n")
    print(f"wrote {MMD_PATH.relative_to(REPO_ROOT)} ({len(mermaid)} chars)")

    try:
        ascii_art = graph.draw_ascii()
        ASCII_PATH.write_text(ascii_art, encoding="utf-8", newline="\n")
        print(f"wrote {ASCII_PATH.relative_to(REPO_ROOT)}")
    except Exception as exc:  # noqa: BLE001 - needs the optional grandalf package
        logger.debug("ASCII rendering unavailable: %s", exc)

    try:
        png = render_png(mermaid)
        PNG_PATH.write_bytes(png)
        print(f"wrote {PNG_PATH.relative_to(REPO_ROOT)} ({len(png) / 1024:.0f} KB)")
    except Exception as exc:  # noqa: BLE001 - the renderer is a remote service
        print()
        print(f"PNG rendering failed: {type(exc).__name__}: {exc}")
        print("graph.mmd was still written. Render it at https://mermaid.live, or")
        print("keep the committed graph.png that ships with this repository.")

    if args.print:
        print()
        print(mermaid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
