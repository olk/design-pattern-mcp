"""src/decisions — the decision-model layer (two self-hosted providers, five seams).

Two serving stacks sit behind the seams, each with its own wire:

* **CLM-8B** (``clm-serve``) — TypeSafe-compatible ``POST /v1/systemone``. Answers the
  Ch2 architecture-style substitution (:mod:`src.decisions.client`), one unsharded
  36-criterion ``choice`` question.
* **AnyJev Tacit-4B** (``anyjev.serve`` gateway) — ``POST /v1/decide``. Answers the
  bounded closed-set judgments — J4 quality weights, J2 category narrowing, J5
  reasoning gate (:mod:`src.decisions.tacit`) — and the Ch2 style substitution as a
  two-stage choice (category → ≤26-style shortlist), which is the default style
  provider; the J1 rerank leg is Tacit's (:mod:`src.patterns.tacit_rerank`).

Which stack answers the style seam is ``decisions.style.provider``.

Modules:
    client: DecisionModelClient (CLM choice, cache, dual-shape parse, degrade).
    seam_prompts: the prompt vocabulary shared by the server and both trainers.
    tacit: TacitClient (choice/yes_no/score, batch, degrade) + the category glosses,
        the weight-level scale, and the two-stage style choice.
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
    style_question,
    style_shards,
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
    "style_question",
    "style_shards",
]
