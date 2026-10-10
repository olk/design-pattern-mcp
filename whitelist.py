# Vulture whitelist: symbol references invisible to static analysis.

# MCP tools are dispatched dynamically via DesignPatternServer._TOOL_METHOD_MAP
# (src/server.py): getattr(tool_instance, method_name) -> server.add_tool().
tool.select
tool.analyze
tool.generate
tool.evaluate
tool.list_patterns
tool.get_pattern
tool.submit_job
tool.get_status
tool.cancel

# MCP resource handlers registered through @server.resource inside the lifespan.
resource_hook
variants_resource

# CLI entry point declared in pyproject.toml [project.scripts].
cli

# Prompt closures are registered through @server.prompt inside register_prompts.
select_design_patterns_workflow
explore_pattern_catalog
evaluate_my_design
compare_pattern_candidates

# Protocol methods implemented by the retrieval and LLM clients (called via structural typing).
embed_documents
embed_query
rerank
extract_brief_and_parts
extract_weights_batch
generate
evaluate
generate_structured

# Reasoning client surface called through the pipeline's optional seam.
run_pre_llm
health_check
