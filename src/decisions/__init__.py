"""src/decisions — the decision-model layer (two self-hosted providers, four seams).

Two serving stacks sit behind the seams, each with its own wire:

* **CLM-8B** (``clm-serve``) — TypeSafe-compatible ``/v1/systemone`` and ``/v1/rank``.
  Answers the Ch2 style substitution (:mod:`src.decisions.client`) and the J1 rerank
  leg (:mod:`src.patterns.clm_rerank`).
* **AnyJev Tacit-4B** (``anyjev.serve`` gateway) — ``/v1/decide``. Answers the bounded
  closed-set judgments: J4 quality weights, J2 category narrowing, J5 reasoning gate
  (:mod:`src.decisions.tacit`).

Modules:
    client: DecisionModelClient (cache, request, dual-shape parse, degrade).
    tacit: TacitClient (choice/yes_no/score, batch, degrade) + the category glosses
        and the weight-level scale.
"""

from src.decisions.client import (
    STYLE_QUESTION_ID,
    DecisionModelClient,
    StyleDecision,
    build_style_criteria,
)
from src.decisions.seam_prompts import (
    CATEGORY_GLOSSES,
    CATEGORY_SLUGS,
    WEIGHT_LEVELS,
    axis_question,
    category_question,
    gate_question,
)
from src.decisions.tacit import TacitClient, TacitDecision, category_options

__all__ = [
    "CATEGORY_GLOSSES",
    "CATEGORY_SLUGS",
    "STYLE_QUESTION_ID",
    "WEIGHT_LEVELS",
    "DecisionModelClient",
    "StyleDecision",
    "TacitClient",
    "TacitDecision",
    "axis_question",
    "build_style_criteria",
    "category_options",
    "category_question",
    "gate_question",
]
