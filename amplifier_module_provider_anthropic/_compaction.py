"""Lossless Messages on-demand compaction, qualified for the direct Claude API."""

import copy
import hmac

KEY = "anthropic:compaction"
BETA = "compact-2026-09-04"
MODELS = frozenset({"claude-sonnet-5-5", "claude-fable-5-2"})


def validate_block(block):
    if (not isinstance(block, dict) or block.get("type") != "compaction"
            or not isinstance(block.get("signature"), str) or not block["signature"]
            or not isinstance(block.get("content"), str) or not block["content"]):
        raise ValueError("Expected one signed Claude compaction block")


def scope_for(client):
    endpoint = str(getattr(client, "base_url", "")).rstrip("/")
    key = getattr(client, "api_key", None)
    if endpoint != "https://api.anthropic.com" or not isinstance(key, str) or not key:
        return None
    return hmac.digest(key.encode(), b"anthropic:compaction:api.anthropic.com:v1", "sha256").hex()


def saved_message(model, scope, block):
    validate_block(block)
    return {
        # Managed Context persists this derived row separately from real turns.
        # The Messages adapter expands it to the signed assistant block on the wire.
        "role": "user",
        "content": "Summary of earlier conversation (historical reference):\n" + block["content"],
        "metadata": {
            "ephemeral": True, "persisted": True, "source": "context-managed",
            KEY: {"version": 1, "model": model, "scope": scope, "block": copy.deepcopy(block)},
        },
    }


def load_block(message, model, scope):
    state = (message.get("metadata") or {}).get(KEY)
    if not isinstance(state, dict) or state.get("version") != 1 or state.get("model") != model:
        return None
    if not scope or not isinstance(state.get("scope"), str) or not hmac.compare_digest(scope, state["scope"]):
        return None
    block = state.get("block")
    validate_block(block)
    return copy.deepcopy(block)


def add_header(params):
    headers = dict(params.get("extra_headers") or {})
    betas = [v for v in headers.get("anthropic-beta", "").split(",") if v]
    if BETA not in betas:
        betas.append(BETA)
    headers["anthropic-beta"] = ",".join(betas)
    params["extra_headers"] = headers


def compaction_usage(usage):
    """A standalone summary reports billed usage in iterations, with zero top-level usage."""
    totals = {}
    keys = {"input_tokens": "input_tokens", "output_tokens": "output_tokens",
            "cache_creation_input_tokens": "cache_write_tokens", "cache_read_input_tokens": "cache_read_tokens"}
    for iteration in usage.get("iterations") or []:
        if iteration.get("type") == "compaction":
            for source, dest in keys.items():
                count = iteration.get(source)
                if type(count) is int and count >= 0:
                    totals[dest] = totals.get(dest, 0) + count
    if totals:
        totals["input_tokens"] = totals.get("input_tokens", 0) + totals.get("cache_read_tokens", 0)
        totals["total_tokens"] = totals["input_tokens"] + totals.get("output_tokens", 0)
    return totals
