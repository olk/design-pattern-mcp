"""src/decisions — the decision-model layer (linear style head plus Tacit, five seams).

Two serving stacks sit behind the seams, each with its own wire:

* **Linear style head** (``style-head-serve``) — TypeSafe-compatible ``POST /v1/systemone``.
  Answers the Ch2 architecture-style substitution (:mod:`src.decisions.client`), one
  unsharded 36-criterion ``choice`` question. The seam's only provider.
* **AnyJev Tacit-4B** (``anyjev.serve`` gateway) — ``POST /v1/decide``. Answers the
  bounded closed-set judgments — J4 quality weights, J2 category narrowing, J5
  reasoning gate (:mod:`src.decisions.tacit`); the J1 rerank leg is Tacit's
  (:mod:`src.patterns.tacit_rerank`).

``decisions.style.provider`` is pinned to ``linear``; every other seam is pinned to ``tacit``.

Modules:
    client: DecisionModelClient (linear-head choice, cache, dual-shape parse, degrade).
    seam_prompts: the prompt vocabulary shared by the server and the trainers.
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
    STYLE_GLOSSES,
    STYLE_INSTRUCTIONS,
    STYLE_NONE_GLOSS,
    STYLE_NONE_KEY,
    STYLE_VOCABULARY,
    WEIGHT_LEVELS,
    axis_question,
    category_question,
    gate_question,
)
from src.decisions.tacit import TacitClient, TacitDecision, category_options

__all__ = [
    "CATEGORY_GLOSSES",
    "CATEGORY_SLUGS",
    "STYLE_GLOSSES",
    "STYLE_INSTRUCTIONS",
    "STYLE_NONE_GLOSS",
    "STYLE_NONE_KEY",
    "STYLE_QUESTION_ID",
    "STYLE_VOCABULARY",
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
