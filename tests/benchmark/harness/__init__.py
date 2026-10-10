"""Stage-0 selection benchmark harness (plan: docs/JEV_DECISION_MODEL_ANALYSIS_PLAN.md §6).

Every module here is a *runner*, never a pytest test: no ``test_`` prefix and no
``conftest.py`` anywhere in the tree, so ``make test-unit`` / ``make test-oracles`` never
collect it. Modes:

- ``offline``  scripted retrieval + scripted LLM over the real catalog: deterministic plumbing
  and metric-integrity run (<60 s, no network);
- ``live``     in-process real pipeline (TEI embedder/reranker + LiteLLM generator + reasoning
  client) with wrapped seams: full stage attribution, real latency;
- ``e2e``      black-box fastmcp client against the running compose stack: deploy-relevant
  totals, no stage attribution.

``src/`` is untouched: the pipeline is constructor-injected (``src/pipeline.py:1294``), so the
harness only wraps the instances it passes in.
"""

HARNESS_PACKAGE = "tests.benchmark.harness"
