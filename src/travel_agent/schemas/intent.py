"""Models describing what the user asked for on this turn.

The follow-up distinction ("Tokyo" then "what about next week?") is really a
*slot tracking* problem: the city slot carries over from the previous turn while
the date slot changes. Making that explicit - rather than hiding it inside a
prompt - is what lets the graph skip work it has already done.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Intent = Literal[
    "new_city",  # a fresh city request: run the whole graph
    "weather_only",  # same city, different dates: run only the weather branch
    "refine",  # same city, asking about the existing summary
    "clarify",  # no city could be resolved: ask the user
]

RouteName = Literal["vector", "web", "clarify"]

# How the router arrived at its answer. Displayed in the trace panel so the
# decision is explainable rather than a bare verdict.
MatchReason = Literal[
    "exact",  # the city name matched the gazetteer outright
    "similarity",  # the centroid cosine score decided it
    "none",  # no city could be resolved from the turn
]


class DateRange(BaseModel):
    """The forecast window a turn is asking about.

    Attributes:
        start: First day of the window.
        days: Number of days requested; the assignment asks for five to seven.
        label: Human-readable description used in the UI and the trace.
    """

    model_config = ConfigDict(extra="forbid")

    start: date = Field(default_factory=date.today)
    days: int = Field(default=7, ge=1, le=14)
    label: str = "next 7 days"

    @property
    def end(self) -> date:
        """Last day of the window (inclusive)."""
        return self.start + timedelta(days=self.days - 1)

    def shifted(self, *, weeks: int = 0, days: int = 0) -> DateRange:
        """Return a copy of this range moved forward in time.

        Used when a follow-up says "next week": the width of the window stays the
        same, only its start moves.

        Args:
            weeks: Whole weeks to shift by.
            days: Additional days to shift by.

        Returns:
            A new :class:`DateRange`.
        """
        offset = timedelta(weeks=weeks, days=days)
        label = (
            "next 7 days"
            if offset == timedelta(0)
            else f"{self.days} days from {self.start + offset}"
        )
        return DateRange(start=self.start + offset, days=self.days, label=label)


class IntentDecision(BaseModel):
    """Output of the intent classifier node.

    Attributes:
        intent: What kind of turn this is.
        city: City resolved for this turn, carried over from memory when the user
            did not name one.
        city_changed: Whether the city slot differs from the previous turn.
        date_changed: Whether the date slot differs from the previous turn.
        date_range: The forecast window to use.
        reason: Plain-English justification, shown in the trace panel.
    """

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    city: str | None = None
    city_changed: bool = True
    date_changed: bool = False
    date_range: DateRange = Field(default_factory=DateRange)
    reason: str = ""

    @model_validator(mode="after")
    def _clarify_has_no_city(self) -> IntentDecision:
        """Keep the decision internally consistent.

        Returns:
            The validated decision.

        Raises:
            ValueError: If a non-clarify intent has no city to act on.
        """
        if self.intent != "clarify" and not self.city:
            raise ValueError(f"intent {self.intent!r} requires a resolved city")
        return self


class RouteDecision(BaseModel):
    """Output of the knowledge router.

    Attributes:
        route: Which knowledge source won.
        match_reason: Which layer of the router decided - the gazetteer or the
            similarity score.
        score: Best centroid cosine similarity for the resolved city.
        threshold: The value ``score`` was compared against.
        matched_city: Closest city in the vector store, whatever the outcome.
        all_scores: Score for every known city, so the UI can show the runners-up.
        reason: Plain-English justification, shown in the trace panel.
    """

    model_config = ConfigDict(extra="forbid")

    route: RouteName
    match_reason: MatchReason = "similarity"
    score: float = 0.0
    threshold: float = 0.0
    matched_city: str | None = None
    all_scores: dict[str, float] = Field(default_factory=dict)
    reason: str = ""


__all__ = [
    "DateRange",
    "Intent",
    "IntentDecision",
    "MatchReason",
    "RouteDecision",
    "RouteName",
]
