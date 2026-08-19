"""Annotate the generated topology diagram so the parallelism is readable."""

from __future__ import annotations

import base64
import re

import httpx

from travel_agent.logging_setup import get_logger

logger = get_logger(__name__)

MERMAID_RENDER_URL = "https://mermaid.ink/img/{payload}?type=png&bgColor=ffffff"

# (source, target) -> label. Anything not listed here is left exactly as it was.
EDGE_LABELS: dict[tuple[str, str], str] = {
    ("classify_intent", "plan_tools"): "work needed",
    ("classify_intent", "synthesize"): "cached / clarify",
    ("plan_tools", "retrieve_vector"): "knowledge: in store",
    ("plan_tools", "web_search"): "knowledge: not in store",
    ("plan_tools", "execute_weather"): "concurrent",
    ("plan_tools", "execute_images"): "concurrent",
}

_EDGE_PATTERN = re.compile(
    r"^(?P<indent>\s*)(?P<source>\w+)\s*(?P<arrow>-\.->|-->)\s*(?P<target>\w+);\s*$"
)


def annotate_mermaid(source: str) -> str:
    """Add explanatory labels to known edges in a generated mermaid diagram.

    Args:
        source: Mermaid source produced by ``graph.draw_mermaid()``.

    Returns:
        The same diagram with labels applied to recognised edges.
    """
    lines: list[str] = []

    for line in source.splitlines():
        match = _EDGE_PATTERN.match(line)
        if match is None:
            lines.append(line)
            continue

        label = EDGE_LABELS.get((match.group("source"), match.group("target")))
        if label is None:
            lines.append(line)
            continue

        lines.append(
            f"{match.group('indent')}{match.group('source')} "
            f"{match.group('arrow')}|{label}| {match.group('target')};"
        )

    return "\n".join(lines) + "\n"


def parse_edges(source: str) -> set[tuple[str, str]]:
    """Extract the edge set from mermaid source, labelled or not.

    Used by the test that proves the annotation did not change the topology.

    Args:
        source: Mermaid source.

    Returns:
        A set of ``(source, target)`` pairs.
    """
    pattern = re.compile(r"^\s*(\w+)\s*(?:-\.->|-->)(?:\|[^|]*\|)?\s*(\w+);\s*$")
    return {
        (match.group(1), match.group(2))
        for line in source.splitlines()
        if (match := pattern.match(line))
    }


def render_png(source: str, timeout: float = 30.0) -> bytes:
    """Render mermaid source to PNG through the mermaid.ink service.

    LangGraph's own ``draw_mermaid_png`` renders the graph object directly, which
    cannot carry the annotations, so the annotated source is sent here instead.

    Args:
        source: Mermaid source to render.
        timeout: Seconds to wait for the service.

    Returns:
        PNG bytes.

    Raises:
        httpx.HTTPError: If the service is unreachable or refuses the request.
    """
    payload = base64.urlsafe_b64encode(source.encode("utf-8")).decode("ascii")
    response = httpx.get(
        MERMAID_RENDER_URL.format(payload=payload), timeout=timeout, follow_redirects=True
    )
    response.raise_for_status()
    return response.content


__all__ = ["EDGE_LABELS", "annotate_mermaid", "parse_edges", "render_png"]
