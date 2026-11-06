"""src/decisions — the decision-model layer (one self-hosted provider, five seams).

**AnyJev Tacit-4B** (``anyjev.serve`` gateway) speaks ``POST /v1/decide`` and answers
every seam:

* the bounded closed-set judgments — J4 quality weights, J2 category narrowing, J5
  reasoning gate (:mod:`src.decisions.tacit`);
* the Ch2 architecture-style substitution, served as a two-stage choice (category →
  ≤26-style shortlist) because the gateway caps ``choice`` options at 26;
* the J1 rerank leg, one ``yes_no`` relevance question per candidate
  (:mod:`src.patterns.tacit_rerank`).

Modules:
    seam_prompts: the prompt vocabulary shared by the server and the Tacit trainer.
    tacit: TacitClient (choice/yes_no/score, batch, degrade) + the category glosses,
        the weight-level scale, and the two-stage style choice.
"""

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
from src.decisions.tacit import StyleDecision, TacitClient, TacitDecision, category_options

__all__ = [
    "CATEGORY_GLOSSES",
    "CATEGORY_SLUGS",
    "STYLE_GLOSSES",
    "STYLE_INSTRUCTIONS",
    "STYLE_NONE_GLOSS",
    "STYLE_NONE_KEY",
    "STYLE_VOCABULARY",
    "WEIGHT_LEVELS",
    "StyleDecision",
    "TacitClient",
    "TacitDecision",
    "axis_question",
    "category_options",
    "category_question",
    "gate_question",
    "style_question",
    "style_shards",
]
