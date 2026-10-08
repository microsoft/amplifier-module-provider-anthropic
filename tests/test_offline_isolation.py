"""Regressions for synthetic fixture scope and trusted-test network denial."""

import asyncio
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from anthropic import AsyncAnthropic
from anthropic._base_client import AsyncHttpxClientWrapper
from httpx2._utils import URLPattern

from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import DummyResponse, FakeCoordinator


MODEL = "claude-haiku-5-5"
RUNNER = Path(__file__).with_name("run_offline.py")


def test_messages_only_cold_metadata_is_explicit_and_cached(
    offline_messages_metadata,
):
    async def run():
        provider = AnthropicProvider(
            "synthetic-offline-credential",
            {"use_streaming": False, "default_model": MODEL},
        )
        provider.coordinator = FakeCoordinator()
        assert provider._runtime_model_info_cache == {}
        raw = MagicMock()
        raw.parse = AsyncMock(return_value=DummyResponse())
        raw.headers = {}
        provider.client.messages.with_raw_response.create = AsyncMock(return_value=raw)
        try:
            metadata = await provider.client.models.retrieve(MODEL)
            assert metadata.id == MODEL
            assert "Synthetic" in metadata.display_name
            assert metadata.capabilities is None
            assert metadata.max_input_tokens is None
            assert metadata.max_tokens is None
            request = ChatRequest(
                messages=[Message(role="user", content="Synthetic fixture prompt")],
            )
            await provider.complete(request)
            first = provider._runtime_model_info_cache[MODEL]
            assert first is not None
            assert first.max_input_tokens is None
            assert first.max_tokens is None
            await provider.complete(request)
            assert provider._runtime_model_info_cache[MODEL] is first
            assert offline_messages_metadata == [MODEL, MODEL]
            assert provider.client.messages.with_raw_response.create.await_count == 2
        finally:
            await provider.client.close()

    asyncio.run(run())


@pytest.mark.parametrize("status", [200, 404, 500])
@pytest.mark.parametrize("mounted", [False, True])
def test_explicit_transport_metadata_and_errors_remain_authoritative(
    status, mounted, offline_messages_metadata
):
    async def run():
        calls = []

        def handler(request):
            calls.append((request.method, request.url.path))
            assert request.method == "GET"
            assert request.url.path == f"/v1/models/{MODEL}"
            if status != 200:
                return httpx2.Response(
                    status,
                    json={
                        "type": "error",
                        "error": {"type": "not_found_error", "message": "Synthetic"},
                    },
                )
            return httpx2.Response(
                200,
                json={
                    "id": MODEL,
                    "type": "model",
                    "display_name": "Synthetic explicit metadata",
                    "created_at": "1970-01-01T00:00:00Z",
                    "max_input_tokens": 654321,
                    "max_tokens": 4321,
                    "capabilities": {
                        "thinking": {
                            "supported": True,
                            "types": {"adaptive": {"supported": True}},
                        }
                    },
                },
            )

        provider = AnthropicProvider("synthetic-offline-credential", {})
        await provider.client.close()
        transport = httpx2.MockTransport(handler)
        routing = (
            {"mounts": {"https://api.anthropic.com": transport}}
            if mounted
            else {"transport": transport}
        )
        provider._client = AsyncAnthropic(
            api_key="synthetic-offline-credential",
            max_retries=0,
            http_client=httpx2.AsyncClient(**routing),
        )
        provider.client.messages.with_raw_response.create = AsyncMock()
        try:
            cold = await provider._get_request_capabilities(MODEL)
            warm = await provider._get_request_capabilities(MODEL)
            assert cold == warm
            assert calls == [("GET", f"/v1/models/{MODEL}")]
            assert offline_messages_metadata == []
            if status == 200:
                assert cold.base_context_window == 654321
                assert cold.max_output_tokens == 4321
                assert cold.supports_adaptive_thinking
            else:
                assert cold == provider._get_capabilities(MODEL)
                assert provider._runtime_model_info_cache[MODEL] is None
        finally:
            await provider.client.close()

    asyncio.run(run())


@pytest.mark.parametrize("status", [200, 404, 500])
@pytest.mark.parametrize(
    "override", ["supplied_wrapper", "replaced_client", "replaced_transport", "mount"]
)
def test_same_type_http_overrides_remain_authoritative(
    status, override, monkeypatch, offline_messages_metadata
):
    async def run():
        provider = AnthropicProvider("synthetic-offline-credential", {})
        transport = httpx2.AsyncHTTPTransport()
        send = AsyncMock(
            return_value=httpx2.Response(
                status,
                json=(
                    {
                        "id": MODEL,
                        "type": "model",
                        "display_name": "Synthetic caller-owned metadata",
                        "created_at": "1970-01-01T00:00:00Z",
                        "lifecycle": "active",
                        "max_input_tokens": 654321,
                        "max_tokens": 4321,
                    }
                    if status == 200
                    else {
                        "type": "error",
                        "error": {"type": "not_found_error", "message": "Synthetic"},
                    }
                ),
            )
        )
        # Exercise the actual SDK metadata request without Python networking.
        monkeypatch.setattr(transport, "handle_async_request", send)
        if override == "supplied_wrapper":
            await provider.client.close()
            provider._client = AsyncAnthropic(
                api_key="synthetic-offline-credential",
                max_retries=0,
                http_client=AsyncHttpxClientWrapper(
                    transport=transport, trust_env=False
                ),
            )
        elif override == "replaced_client":
            await provider.client._client.aclose()
            provider.client._client = AsyncHttpxClientWrapper(
                transport=transport, trust_env=False
            )
        elif override == "replaced_transport":
            await provider.client._client._transport.aclose()
            provider.client._client._transport = transport
        else:
            provider.client._client._mounts[URLPattern("https://api.anthropic.com")] = (
                transport
            )
        provider.client.messages.with_raw_response.create = AsyncMock()
        try:
            assert type(provider.client._client) is AsyncHttpxClientWrapper
            selected_transport = provider.client._client._transport_for_url(
                provider.client.base_url.join(f"models/{MODEL}")
            )
            assert selected_transport is transport
            assert type(selected_transport) is httpx2.AsyncHTTPTransport
            cold = await provider._get_request_capabilities(MODEL)
            warm = await provider._get_request_capabilities(MODEL)
            assert cold == warm
            send.assert_awaited_once()
            request = send.call_args.args[0]
            assert (request.method, request.url.path) == (
                "GET",
                f"/v1/models/{MODEL}",
            )
            assert offline_messages_metadata == []
            if status == 200:
                assert cold.base_context_window == 654321
                assert cold.max_output_tokens == 4321
                assert provider._runtime_model_info_cache[MODEL] is not None
            else:
                assert cold == provider._get_capabilities(MODEL)
                assert provider._runtime_model_info_cache[MODEL] is None
        finally:
            await provider.client.close()

    asyncio.run(run())


def test_explicit_unavailable_metadata_mock_is_not_replaced(
    offline_messages_metadata,
):
    async def run():
        provider = AnthropicProvider("synthetic-offline-credential", {})
        provider.client.messages.with_raw_response.create = AsyncMock()
        unavailable = AsyncMock(side_effect=RuntimeError("Synthetic unavailable"))
        provider.client.models.retrieve = unavailable
        try:
            assert await provider._get_runtime_model_info(MODEL) is None
            assert await provider._get_runtime_model_info(MODEL) is None
            unavailable.assert_awaited_once_with(MODEL)
            assert offline_messages_metadata == []
        finally:
            await provider.client.close()

    asyncio.run(run())


def _child(tmp_path, source, *args):
    case = tmp_path / "test_guard_case.py"
    case.write_text(source)
    receipt = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(RUNNER),
            "--receipt",
            str(receipt),
            str(case),
            "-q",
            "--basetemp",
            str(tmp_path / "child-temp"),
            *args,
        ],
        # No credentials, user configs, proxies, PYTHONPATH or pytest flags.
        env={},
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, json.loads(receipt.read_text())


@pytest.mark.parametrize("phase", ["collection", "runtest"])
def test_guard_fails_independently_after_every_swallowed_dns_attempt(tmp_path, phase):
    attempts = """
import socket
for _ in range(2):
    try:
        socket.getaddrinfo("synthetic.invalid", 443)
    except PermissionError:
        pass
"""
    source = (
        attempts + "\ndef test_swallowed(): pass\n"
        if phase == "collection"
        else "def test_swallowed():\n"
        + "\n".join("    " + line for line in attempts.splitlines())
    )
    result, receipt = _child(tmp_path, source)
    assert result.returncode == receipt["exit_code"] == 1
    assert len(receipt["attempts"]) == 2
    assert {attempt["event"] for attempt in receipt["attempts"]} == {
        "socket.getaddrinfo"
    }
    assert all(attempt["stack"] for attempt in receipt["attempts"])
    assert any(
        report["when"] == "call" and report["outcome"] == "passed"
        for report in receipt["reports"]
    )
    if phase == "collection":
        assert {a["phase_or_nodeid"] for a in receipt["attempts"]} == {"collection"}
    else:
        assert all(
            "test_swallowed" in a["phase_or_nodeid"] for a in receipt["attempts"]
        )


def test_guard_blocks_literal_ip_connect_before_syscall(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
def test_literal_ip():
    import socket
    with socket.socket() as sock:
        try:
            sock.connect(("192.0.2.1", 443))
        except PermissionError:
            pass
""",
    )
    assert result.returncode == 1
    assert [a["event"] for a in receipt["attempts"]] == ["socket.connect"]


def test_guard_denies_unguarded_child_before_it_can_run(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
def test_child_denied():
    import subprocess, sys
    try:
        subprocess.run([sys.executable, "-I", "-c", "raise SystemExit(99)"])
    except PermissionError:
        pass
    else:
        raise AssertionError("Unguarded child escaped")
""",
    )
    assert result.returncode == 1
    assert [a["event"] for a in receipt["attempts"]] == ["subprocess.Popen"]


def test_guard_blocks_reverse_dns_before_resolution(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
def test_reverse_dns():
    import socket
    try:
        socket.getnameinfo(("192.0.2.1", 443), 0)
    except PermissionError:
        pass
""",
    )
    assert result.returncode == 1
    assert [a["event"] for a in receipt["attempts"]] == ["socket.getnameinfo"]


def test_guard_retains_swallowed_shutdown_attempt_and_fails_exit(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
def test_shutdown():
    import atexit, socket
    def late_request():
        try:
            socket.getaddrinfo("synthetic.invalid", 443)
        except PermissionError:
            pass
    atexit.register(late_request)
""",
    )
    assert result.returncode == receipt["exit_code"] == 1
    assert receipt["shutdown_accounted"]
    assert [a["event"] for a in receipt["attempts"]] == ["socket.getaddrinfo"]
    assert receipt["attempts"][0]["phase_or_nodeid"] == "shutdown"


def test_guard_still_records_after_public_audit_functions_are_mocked(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
def test_audit_patch(monkeypatch):
    import socket, sys
    monkeypatch.setattr(sys, "audit", lambda *args: None)
    monkeypatch.setattr(sys, "addaudithook", lambda *args: None)
    for _ in range(2):
        try:
            socket.getaddrinfo("synthetic.invalid", 443)
        except PermissionError:
            pass
""",
    )
    assert result.returncode == 1
    assert len(receipt["attempts"]) == 2
    assert receipt["shutdown_accounted"]


def test_runner_sanitizes_polluted_placeholder_environment(tmp_path):
    case = tmp_path / "test_sanitized.py"
    case.write_text(
        """
def test_sanitized():
    import os
    assert all(name not in os.environ for name in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "HTTP_PROXY", "PYTHONPATH",
        "PYTEST_ADDOPTS"
    ))
"""
    )
    receipt_path = tmp_path / "sanitized.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(RUNNER),
            "--receipt",
            str(receipt_path),
            str(case),
            "-q",
            "--basetemp",
            str(tmp_path / "sanitized-temp"),
        ],
        env={
            "ANTHROPIC_API_KEY": "synthetic-offline-credential",
            "ANTHROPIC_BASE_URL": "https://synthetic.invalid",
            "HTTP_PROXY": "http://synthetic.invalid",
            "PYTHONPATH": "synthetic-nonexistent-path",
            "PYTEST_ADDOPTS": "--required-live",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    receipt = json.loads(receipt_path.read_text())
    assert result.returncode == 0, result.stdout + result.stderr
    assert receipt["attempts"] == []
    assert receipt["shutdown_accounted"]


def test_real_sdk_constructor_alone_makes_zero_dns_attempts(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
def test_constructor():
    import asyncio
    from amplifier_module_provider_anthropic import AnthropicProvider
    provider = AnthropicProvider("synthetic-offline-credential", {})
    assert provider.client.max_retries == 0
    assert provider._runtime_model_info_cache == {}
    asyncio.run(provider.client.close())
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert receipt["attempts"] == []
    assert receipt["preimports"] == receipt["postimports"]
    assert receipt["isolated"]


def test_guard_catches_unmocked_cold_models_after_provider_swallow(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
def test_cold_unmocked():
    import asyncio
    from amplifier_module_provider_anthropic import AnthropicProvider
    async def run():
        provider = AnthropicProvider("synthetic-offline-credential", {})
        try:
            assert await provider._get_runtime_model_info("claude-haiku-5-5") is None
            assert provider._runtime_model_info_cache == {"claude-haiku-5-5": None}
        finally:
            await provider.client.close()
    asyncio.run(run())
""",
    )
    assert result.returncode == 1
    assert receipt["attempts"]
    assert all(a["event"] == "socket.getaddrinfo" for a in receipt["attempts"])
    assert any(
        r["when"] == "call" and r["outcome"] == "passed" for r in receipt["reports"]
    )


def test_required_live_missing_key_fails_instead_of_skipping(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(RUNNER),
            "--receipt",
            str(tmp_path / "live-gate.json"),
            "tests/test_image_support.py",
            "--required-live",
            "-q",
            "--basetemp",
            str(tmp_path / "live-gate-temp"),
        ],
        env={},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "UNMET required real vision" in result.stderr
    assert "skipped" not in result.stdout
    receipt = json.loads((tmp_path / "live-gate.json").read_text())
    assert receipt["attempts"] == []
    assert receipt["reports"] == []


def test_required_live_collect_only_cannot_qualify_with_placeholder(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
def test_collect_only_is_unmet():
    import os, pytest
    os.environ["ANTHROPIC_API_KEY"] = "synthetic-offline-credential"
    code = pytest.main([
        "tests/test_image_support.py", "--required-live", "--collect-only", "-q"
    ])
    assert int(code) != 0
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert receipt["attempts"] == []
