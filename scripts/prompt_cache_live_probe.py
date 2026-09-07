"""Live prompt-cache probe: drive a 12-call tool-using loop and report hits.

Reproduces the deployment shape that measured a 6.8% cache hit ratio over
164 provider calls (session c2e6940c-4582-4569-9f51-d8d90ff44c48):

  * NO message carries a `Message.metadata` dict -- the orchestrator never
    populates the ephemeral contract this provider's conversation-region
    breakpoint placement depends on.
  * A `<system-reminder>` block carrying a live clock is regenerated on
    every request and appended to the tail. It is never persisted into
    history, so any cached prefix ending in it can never be reproduced.
  * The conversation grows through tool rounds, not user turns -- one real
    user instruction, then assistant tool_use / tool_result pairs.

Each run mints a fresh nonce into the system prompt, so two runs can never
share a cache entry on Anthropic's side and a before/after comparison is
honest.

Usage:
    ANTHROPIC_API_KEY=... uv run python scripts/prompt_cache_live_probe.py \
        --label after
    ANTHROPIC_API_KEY=... uv run python scripts/prompt_cache_live_probe.py \
        --label before --config cache_infer_stability_from_history=false
    # Against a snapshot of another revision's provider:
    #   --provider-module prov_main --sys-path /tmp/beforepkg
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from amplifier_core.message_models import (  # noqa: E402
    ChatRequest,
    Message,
    ToolSpec,
)

CALLS = 12

# ~11k tokens. Sized ABOVE claude-haiku-4-5's minimum cacheable prefix,
# which was measured here at ~4096 tokens (NOT the 2048 the docs list for
# Haiku): a system+tools prefix of 4,369 tokens produced no cache write on
# two consecutive identical requests, while 8,659 cached immediately.
# Sizing under that threshold would silently disable the system-block
# breakpoint in BOTH arms and flatter the "after" number by making the
# baseline 0% instead of the frozen-constant signature actually measured in
# production.
SYSTEM_PROMPT_BODY = (
    "You are a meticulous engineering assistant operating inside an "
    "automated harness. Work one step at a time. Always call the `lookup` "
    "tool with the next chunk number before answering. Keep prose short. "
) * 260

TOOL = ToolSpec(
    name="lookup",
    description="Fetch one chunk of the reference corpus by number.",
    parameters={
        "type": "object",
        "properties": {"chunk": {"type": "integer"}},
        "required": ["chunk"],
    },
)


def _tool_output(chunk: int) -> str:
    """~500 tokens of deterministic filler, so history grows realistically."""
    return (
        f"chunk {chunk}: "
        + f"reference line for chunk {chunk} with stable deterministic body. " * 55
    )


def _reminder(call_index: int) -> Message:
    """The regenerated tail. No metadata -- that is the whole point."""
    return Message(
        role="user",
        content=(
            f"<system-reminder>turn={call_index} "
            f"clock={time.time():.6f} git=clean</system-reminder>"
        ),
    )


def _assistant_message(response) -> Message:
    """Replay the model's turn back into history.

    Tool calls ride the `tool_calls` extra field, which is the shape
    `_convert_messages` translates into Anthropic `tool_use` blocks (it reads
    `tc["tool"]` for the name). The kernel's own `ToolCallBlock` is NOT that
    shape -- `_clean_content_block` passes its `type: "tool_call"` straight
    through and the API rejects it -- so this harness uses the path the
    provider actually supports.
    """
    text = ""
    for block in response.content or []:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "") or ""
    tool_calls = [
        {"id": call.id, "tool": call.name, "arguments": call.arguments or {}}
        for call in (response.tool_calls or [])
    ]
    if tool_calls:
        return Message(
            role="assistant",
            content=text or "",
            tool_calls=tool_calls,
        )
    return Message(role="assistant", content=text or "(continuing)")


async def run(provider_cls, config: dict, model: str) -> list[dict]:
    provider = provider_cls(api_key=os.environ["ANTHROPIC_API_KEY"], config=config)
    nonce = uuid.uuid4().hex
    system = Message(
        role="system",
        content=f"[run:{nonce}]\n{SYSTEM_PROMPT_BODY}",
    )
    history: list[Message] = [
        Message(
            role="user",
            content=(
                "Task: walk the reference corpus. Call `lookup` for chunk 1, "
                "then chunk 2, and so on, one chunk per turn. After each "
                "result, briefly note one fact from it, then continue."
            ),
        )
    ]

    rows: list[dict] = []
    try:
        for i in range(CALLS):
            view = [system] + history + [_reminder(i)]
            request = ChatRequest(messages=view, tools=[TOOL])
            started = time.perf_counter()
            response = await provider.complete(request, model=model)
            latency = time.perf_counter() - started

            usage = response.usage
            rows.append(
                {
                    "call": i + 1,
                    "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                    "cache_read_input_tokens": getattr(usage, "cache_read_tokens", 0)
                    or 0,
                    "cache_creation_input_tokens": getattr(
                        usage, "cache_write_tokens", 0
                    )
                    or 0,
                    "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                    "latency_s": round(latency, 3),
                }
            )

            history.append(_assistant_message(response))
            if response.tool_calls:
                for call in response.tool_calls:
                    chunk = (call.arguments or {}).get("chunk", i + 1)
                    history.append(
                        Message(
                            role="tool",
                            tool_call_id=call.id,
                            content=_tool_output(int(chunk or i + 1)),
                        )
                    )
            else:
                history.append(
                    Message(role="user", content=f"Continue with step {i + 2}.")
                )
    finally:
        await provider.close()
    return rows


def _print_table(label: str, rows: list[dict]) -> None:
    """Report the hit ratio against BILLED input.

    `ChatResponse.usage.input_tokens` from this provider is
    `raw_input + cache_read` (see `_convert_to_chat_response`), so the
    denominator for "what did this call cost to send" is
    `input_tokens + cache_creation` -- NOT `input_tokens + cache_read`,
    which double-counts the cached portion.
    """
    print(f"\n### {label}\n")
    print(
        "| call | billed input | cache_read | cache_write | uncached | "
        "hit % | latency s |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        billed = r["input_tokens"] + r["cache_creation_input_tokens"]
        uncached = r["input_tokens"] - r["cache_read_input_tokens"]
        pct = (r["cache_read_input_tokens"] / billed * 100) if billed else 0.0
        print(
            f"| {r['call']} | {billed} | {r['cache_read_input_tokens']} | "
            f"{r['cache_creation_input_tokens']} | {uncached} | "
            f"{pct:.1f}% | {r['latency_s']} |"
        )

    def _ratio(subset: list[dict]) -> tuple[int, int, float]:
        read = sum(x["cache_read_input_tokens"] for x in subset)
        billed = sum(
            x["input_tokens"] + x["cache_creation_input_tokens"] for x in subset
        )
        return read, billed, (read / billed * 100) if billed else 0.0

    read, billed, pct = _ratio(rows)
    print(f"\n- overall hit ratio: **{pct:.1f}%** ({read} cache_read / {billed} billed)")
    for first in (2, 3):
        r2, b2, p2 = _ratio(rows[first - 1 :])
        print(f"- calls {first}-{len(rows)} hit ratio: **{p2:.1f}%**")
    mean_latency = sum(r["latency_s"] for r in rows) / len(rows)
    print(f"- mean latency per call: **{mean_latency:.2f}s**")


def _parse_config(pairs: list[str]) -> dict:
    config: dict = {"use_streaming": False, "enable_prompt_caching": True}
    for pair in pairs:
        key, _, value = pair.partition("=")
        config[key] = value
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument(
        "--provider-module", default="amplifier_module_provider_anthropic"
    )
    parser.add_argument("--sys-path", default=None)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.sys_path:
        sys.path.insert(0, args.sys_path)

    module = importlib.import_module(args.provider_module)
    provider_cls = module.AnthropicProvider

    rows = asyncio.run(run(provider_cls, _parse_config(args.config), args.model))
    _print_table(args.label, rows)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"label": args.label, "rows": rows}, handle, indent=2)


if __name__ == "__main__":
    main()
