"""src/decisions — decision-model (JEV) integration.

Provides the Ch2 style-substitution seam: one JEV ``choice`` question over the
35 ``STYLE_AFFINITY`` keys plus ``none`` replaces the LLM's prose-sampled
``brief.architecture_style`` behind a default-off config switch.

Modules:
    client: DecisionModelClient (cache, request, dual-shape parse, degrade).
"""

from src.decisions.client import (
    STYLE_QUESTION_ID,
    DecisionModelClient,
    StyleDecision,
    build_style_criteria,
)

__all__ = [
    "STYLE_QUESTION_ID",
    "DecisionModelClient",
    "StyleDecision",
    "build_style_criteria",
]
