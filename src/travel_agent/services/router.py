"""The knowledge router: decide where a city's facts should come from.

WHY THIS IS LAYERED RATHER THAN A SINGLE THRESHOLD
    The measured separation between seeded and unseeded cities is real but thin -
    seeded cities score 0.10 to 0.21, unseeded ones 0.00 to 0.04. A bare
    threshold at 0.07 works on the probes I measured, but it is a single fragile
    gate: anything that lands in the 0.04-0.10 band is decided by a hair, and
    "New York" typed as "NYC" would fail it outright.

    So the router asks a cheaper, more certain question first. Whether the
    knowledge base has documents about a city is, in the ordinary case, a
    question about *names*, and a name lookup answers it exactly. Only when the
    name is unrecognised does the similarity score have to make a judgement call.

        1. gazetteer  - does this name (or an alias of it) match a city I hold
                        documents for?          -> vector, reason "exact"
        2. similarity - is the centroid cosine above the threshold?
                                                -> vector or web, reason "similarity"
        3. nothing    - no city resolved at all -> clarify

    The trade-off, stated plainly: the gazetteer needs maintenance as the corpus
    grows, because every new city and nickname is another entry. The similarity
    path is the one that generalises, and it is still what decides every case the
    gazetteer has not been taught. The score is recorded and displayed either
    way, so layering makes the router robust without hiding the measurement.
"""

from __future__ import annotations

from travel_agent.config.settings import Settings, get_settings
from travel_agent.logging_setup import get_logger
from travel_agent.schemas.intent import RouteDecision
from travel_agent.schemas.trace import ThresholdDiagnostics
from travel_agent.services.retriever import KnowledgeRetriever

logger = get_logger(__name__)

# Cities used to measure where unseeded scores actually land. They are not in the
# corpus; "Kyoto" is deliberately included because the Tokyo day-trips passage
# mentions it once, which makes it the hardest negative the corpus contains.
CONTROL_UNKNOWN_CITIES: tuple[str, ...] = (
    "Kyoto",
    "Snohomish",
    "Reykjavik",
    "Bogota",
    "Ulaanbaatar",
)


class KnowledgeRouter:
    """Decides between the vector store and web search for a given city.

    Attributes:
        retriever: Retrieval service used for scoring.
        threshold: Cosine score a city must beat to be considered known.
    """

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        *,
        settings: Settings | None = None,
        validate: bool = True,
    ) -> None:
        """Initialise the router.

        Args:
            retriever: A loaded retrieval service.
            settings: Settings to read from. Defaults to the process singleton.
            validate: Run the threshold sanity check and log its verdict. Only
                turned off in tests that deliberately use odd thresholds.
        """
        self._settings = settings or get_settings()
        self.retriever = retriever
        self.threshold = self._settings.router_similarity_threshold
        self._diagnostics: ThresholdDiagnostics | None = None

        if validate:
            self.check_threshold()

    # ----------------------------------------------------------- the decision --
    def decide(self, city: str | None) -> RouteDecision:
        """Choose a knowledge source for a city.

        Args:
            city: The city slot resolved by the intent node, or ``None`` when the
                turn named no city at all.

        Returns:
            A fully populated decision: the route, which layer decided it, the
            score, the threshold it was compared against, and every city's score.
        """
        if not city or not city.strip():
            return RouteDecision(
                route="clarify",
                match_reason="none",
                threshold=self.threshold,
                reason="No city could be resolved from the request.",
            )

        scores = self.retriever.city_scores(city)

        # Layer 1: the gazetteer. An exact or alias match is a certainty, not an
        # estimate, so it does not need to clear a similarity bar.
        exact = self.retriever.find_city_by_name(city)
        if exact is not None:
            score = scores.get(exact, 0.0)
            return RouteDecision(
                route="vector",
                match_reason="exact",
                score=score,
                threshold=self.threshold,
                matched_city=exact,
                all_scores=scores,
                reason=(
                    f"'{city}' matches known city '{exact}' by name; "
                    f"similarity {score:.3f} (threshold {self.threshold:.2f})"
                ),
            )

        # Layer 2: the similarity score decides.
        matched_city, score = self.retriever.best_match(city)
        if score >= self.threshold:
            return RouteDecision(
                route="vector",
                match_reason="similarity",
                score=score,
                threshold=self.threshold,
                matched_city=matched_city,
                all_scores=scores,
                reason=(
                    f"No exact name match, but similarity {score:.3f} to "
                    f"'{matched_city}' clears the {self.threshold:.2f} threshold"
                ),
            )

        return RouteDecision(
            route="web",
            match_reason="similarity",
            score=score,
            threshold=self.threshold,
            matched_city=matched_city,
            all_scores=scores,
            reason=(
                f"'{city}' is not a known city and its best similarity "
                f"{score:.3f} is below the {self.threshold:.2f} threshold"
            ),
        )

    # ------------------------------------------------------ the start-up guard --
    def check_threshold(self) -> ThresholdDiagnostics:
        """Measure the score separation and warn if the threshold is unusable.

        A misconfigured threshold fails *silently*: set it too high and every city
        routes to web search, the vector store never runs, and the app looks like
        it is working. That happened during development - the planned value of
        0.55 was above the highest score any seeded city can reach - and it went
        unnoticed until a test caught it. This turns that failure into a loud,
        specific warning at start-up.

        Returns:
            The measured diagnostics, cached after the first call.
        """
        if self._diagnostics is not None:
            return self._diagnostics

        known_scores = {
            city: self.retriever.best_match(city)[1] for city in self.retriever.known_cities
        }
        control_scores = {
            city: self.retriever.best_match(city)[1] for city in CONTROL_UNKNOWN_CITIES
        }

        if not known_scores:
            diagnostics = ThresholdDiagnostics(
                threshold=self.threshold,
                status="unknown",
                message="Vector store is empty; threshold cannot be validated.",
            )
            logger.warning(diagnostics.message)
            self._diagnostics = diagnostics
            return diagnostics

        weakest_city, lowest_known = min(known_scores.items(), key=lambda item: item[1])
        strongest_unknown_city, highest_unknown = max(
            control_scores.items(), key=lambda item: item[1]
        )

        if self.threshold >= lowest_known:
            status = "too_high"
            message = (
                f"ROUTER THRESHOLD TOO HIGH: {self.threshold:.3f} is at or above the "
                f"lowest seeded-city score ({lowest_known:.3f} for '{weakest_city}'). "
                f"Every city will route to WEB SEARCH and the vector-store path will "
                f"never run. Measured separation is {highest_unknown:.3f}..{lowest_known:.3f}; "
                f"a value near {(lowest_known + highest_unknown) / 2:.2f} is correct. "
                f"Check ROUTER_SIMILARITY_THRESHOLD in your .env."
            )
        elif self.threshold <= highest_unknown:
            status = "too_low"
            message = (
                f"ROUTER THRESHOLD TOO LOW: {self.threshold:.3f} is at or below the "
                f"highest unseeded-city score ({highest_unknown:.3f} for "
                f"'{strongest_unknown_city}'). Cities the knowledge base does not cover "
                f"may be answered from the vector store. Measured separation is "
                f"{highest_unknown:.3f}..{lowest_known:.3f}."
            )
        else:
            status = "ok"
            message = (
                f"Router threshold {self.threshold:.3f} sits inside the measured "
                f"separation band {highest_unknown:.3f}..{lowest_known:.3f} "
                f"(margin {lowest_known - highest_unknown:.3f})."
            )

        diagnostics = ThresholdDiagnostics(
            threshold=self.threshold,
            status=status,  # type: ignore[arg-type]
            lowest_known_score=lowest_known,
            highest_unknown_score=highest_unknown,
            weakest_known_city=weakest_city,
            strongest_unknown_city=strongest_unknown_city,
            message=message,
        )

        if status == "ok":
            logger.info(message)
        else:
            # Loud on purpose: this is the failure that hides itself.
            logger.warning("=" * 78)
            logger.warning(message)
            logger.warning("=" * 78)

        self._diagnostics = diagnostics
        return diagnostics

    @property
    def diagnostics(self) -> ThresholdDiagnostics:
        """Threshold diagnostics, measured on first access."""
        return self.check_threshold()


__all__ = ["CONTROL_UNKNOWN_CITIES", "KnowledgeRouter"]
