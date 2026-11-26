"""src/decisions — the decision-model layer (linear style head, one seam).

One serving stack sits behind the seam: the **linear style head** (``style-head-serve``) —
TypeSafe-compatible ``POST /v1/systemone``. It answers the Ch2 architecture-style substitution
(:mod:`src.decisions.client`), one unsharded 36-criterion ``choice`` question.

``decisions.style.provider`` is pinned to ``linear``.

Modules:
    client: DecisionModelClient (linear-head choice, cache, dual-shape parse, degrade).
    seam_prompts: the prompt vocabulary shared by the server and the trainers.
"""

from src.decisions.client import (
    STYLE_QUESTION_ID,
    DecisionModelClient,
    StyleDecision,
    build_style_criteria,
)
from src.decisions.seam_prompts import (
    STYLE_GLOSSES,
    STYLE_INSTRUCTIONS,
    STYLE_NONE_GLOSS,
    STYLE_NONE_KEY,
    STYLE_VOCABULARY,
)

__all__ = [
    "STYLE_GLOSSES",
    "STYLE_INSTRUCTIONS",
    "STYLE_NONE_GLOSS",
    "STYLE_NONE_KEY",
    "STYLE_QUESTION_ID",
    "STYLE_VOCABULARY",
    "DecisionModelClient",
    "StyleDecision",
    "build_style_criteria",
]
