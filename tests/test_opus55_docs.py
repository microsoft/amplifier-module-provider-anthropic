"""Keep the Opus 5.5 migration surface documented."""

from pathlib import Path


def test_readme_documents_opus55_migration_and_config():
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    for required in (
        "### Claude Opus 5.5",
        "claude-opus-5-5",
        "computer_toolset_20260801",
        "thinking-display-updates-2026-08-18",
        "thinking-binding-controls-2026-08-01",
        "thinking_prefix_mismatch_behavior",
        "computer_use_tool_type",
        "computer_batch_actions",
        "inference_geo",
        "provider:thinking_binding_retry",
        "provider:thinking_blocks_dropped",
        "InvalidRequestError",
        "bedrock-runtime",
        "bedrock-mantle",
        "Microsoft Foundry",
        "Claude Platform on AWS",
        "ANTHROPIC_BASE_URL",
        "reasoning_extraction",
    ):
        assert required in readme
