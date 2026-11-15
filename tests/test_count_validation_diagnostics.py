"""Closed, payload-free count-endpoint diagnostics through real SDK transport."""
import asyncio
import json
from typing import cast

import anthropic
import httpx2
import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import FakeCoordinator

EVENT = "provider:token_count_unavailable"
CANARY = "sk-private-prompt-thinking-toolname-config-DO-NOT-EXPORT"


async def exercise(bodies):
    paths, events = [], []
    provider = AnthropicProvider(api_key=CANARY, config={"default_model": "claude-fable-5-2",
        "max_retries": 0, "extra_request_params": {"metadata": {"user_id": CANARY}}})
    coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, coordinator)

    def handle(request):
        paths.append(request.url.path)
        assert request.url.path.endswith("/count_tokens")
        return httpx2.Response(400, headers={"request-id": f"req_test_{len(paths)}", "x-private": CANARY},
                               json=bodies[len(paths) - 1])

    async with anthropic.AsyncAnthropic(api_key=CANARY, max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle), trust_env=False)) as sdk:
        provider._client = sdk
        request = ChatRequest(messages=[Message(role="user", content=CANARY)], max_output_tokens=32768)
        before = request.model_dump()
        for _ in bodies:
            assert await provider.request_budget(request, context_estimate=10000) is None
        assert request.model_dump() == before
    events = coordinator.hooks.events
    return provider, paths, events


@pytest.mark.parametrize("outer", [True, False])
@pytest.mark.parametrize("message,path,reason", [
    ("thinking.type: Field required", "thinking.type", "required_field"),
    ("thinking: Unable to extract tag using discriminator 'type'", "thinking", "missing_discriminator"),
    ("thinking.block_binding: Extra inputs are not permitted", "thinking.block_binding", "extra_field"),
    ("messages.987.content.12.signature: Input should be a valid string", "messages.[].content.[].signature", "invalid_type"),
    ("output_config.effort: Input should be a valid string", "output_config.effort", "invalid_type"),
])
def test_real_sdk_recognizes_only_closed_reason_and_structural_path(outer, message, path, reason, caplog):
    inner = {"type": "invalid_request_error", "message": message, "private": CANARY}
    body = {"type": "error", "error": inner} if outer else inner
    _, paths, events = asyncio.run(exercise([body]))
    assert paths == ["/v1/messages/count_tokens"]
    payload = events[0][1]
    assert payload["diagnostic_version"] == 1
    assert payload["api_error_type"] == "invalid_request_error"
    assert payload["reason_code"] == reason and payload["schema_path"] == path
    assert payload["detail_status"] == "recognized"
    serialized = json.dumps(events)
    assert CANARY not in serialized + caplog.text
    assert message not in serialized + caplog.text


@pytest.mark.parametrize("message,status", [
    (CANARY, "unrecognized"),
    ("thinking.type: Field required " + CANARY, "unrecognized"),
    ("tools.0.input_schema.properties." + CANARY + ": Field required", "unrecognized"),
    ("messages.0.content.0." + CANARY + ": Field required", "unrecognized"),
    ("https://secret.invalid/: Field required", "unrecognized"),
    ("x" * 1025, "oversize"),
    ({"private": CANARY}, "malformed"),
    (None, "malformed"),
])
def test_unknown_hostile_and_oversize_details_never_export_text(message, status, caplog):
    body = {"type": "error", "error": {"type": CANARY, "message": message}}
    _, _, events = asyncio.run(exercise([body]))
    payload = events[0][1]
    assert payload["detail_status"] == status
    assert payload["api_error_type"] == "unknown"
    assert payload["reason_code"] == "unknown" and payload["schema_path"] == "unknown"
    assert CANARY not in json.dumps(events) + caplog.text


def test_different_400_reasons_survive_but_request_ids_and_indices_do_not_bypass_dedupe():
    messages = ["thinking.type: Field required", "thinking.block_binding: Extra inputs are not permitted",
                "messages.5.content.3.signature: Field required",
                "messages.999.content.88.signature: Field required"]
    provider, paths, events = asyncio.run(exercise([
        {"type": "invalid_request_error", "message": text} for text in messages]))
    assert len(paths) == 4 and len(events) == 3
    assert len(provider._count_unavailable_warning_categories) == 3
    assert [data["request_id"] for _, data in events] == ["req_test_1", "req_test_2", "req_test_3"]


def test_body_inspection_is_nonrecursive_and_does_not_stringify_arbitrary_objects():
    from amplifier_module_provider_anthropic._count_diagnostics import validation_detail

    class Poison:
        def __str__(self):
            raise AssertionError("Do not stringify private objects")

    assert validation_detail({"message": Poison(), "type": "invalid_request_error"})["detail_status"] == "malformed"
    recursive = {}; recursive["error"] = recursive
    assert validation_detail(recursive)["detail_status"] == "malformed"


@pytest.mark.parametrize("outer", [True, False])
def test_parser_recognizes_both_exact_dictionary_envelopes(outer):
    from amplifier_module_provider_anthropic._count_diagnostics import validation_detail
    body = {"type": "invalid_request_error", "message": "thinking.type: Field required"}
    if outer:
        body = {"error": body}
    assert validation_detail(body) == {"diagnostic_version": 1, "api_error_type": "invalid_request_error",
        "reason_code": "required_field", "schema_path": "thinking.type", "detail_status": "recognized"}


def test_diagnostic_key_and_output_budget_is_bounded():
    provider = AnthropicProvider(api_key="synthetic", config={"default_model": "claude-fable-5-2"})
    coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, coordinator)

    async def run():
        for index in range(140):
            await provider._warn_count_unavailable(f"model-{index}", "request_error", RuntimeError(CANARY))

    asyncio.run(run())
    assert len(provider._count_unavailable_warning_categories) == 128
    assert len(coordinator.hooks.events) == 128
    assert CANARY not in json.dumps(coordinator.hooks.events)


@pytest.mark.parametrize("value,maximum,expected", [
    ("req_ok-123", 128, "req_ok-123"), ("x" * 128, 128, "x" * 128),
    ("x" * 129, 128, None), ("not safe\n", 128, None), ("nonascii-é", 128, None),
    (True, 128, None), (None, 128, None), ({"secret": CANARY}, 128, None),
])
def test_existing_identifier_fields_are_bounded_not_coerced(value, maximum, expected):
    from amplifier_module_provider_anthropic._count_diagnostics import bounded_identifier
    assert bounded_identifier(value, maximum) == expected


def test_throwing_status_attributes_cannot_change_count_failure_result(caplog):
    class BrokenStatus(anthropic.APIStatusError):
        def __getattribute__(self, name):
            if name == "status_code":
                raise RuntimeError(CANARY)
            return super().__getattribute__(name)

    error = BrokenStatus("private", response=httpx2.Response(400,
        request=httpx2.Request("POST", "https://synthetic.invalid/")), body={})
    provider = AnthropicProvider(api_key="synthetic")
    coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, coordinator)
    asyncio.run(provider._warn_count_unavailable("claude-fable-5-2", "request_error", error))
    assert coordinator.hooks.events == [] and CANARY not in caplog.text
