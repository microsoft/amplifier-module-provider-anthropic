"""Tests for venv-swap resilience.

`amplifier update` can replace the uv tool venv while a session is running.
The running process then holds a certifi module whose cacert.pem path no
longer exists, so every NEW httpx2 client (one per sub-agent delegation)
dies at SSL-context creation with a bare FileNotFoundError
"[Errno 2] No such file or directory" that used to surface as an opaque,
retryable LLMError.

Two defenses under test:
1. The provider builds one process-wide SSL context and reuses it for every
   client, so clients created after the venv swap still work.
2. If an ENOENT still escapes (e.g. the very first context build happens
   after the swap), the translator raises a non-retryable LLMError whose
   message names the venv swap and tells the user to restart the session.
"""

import asyncio
import os
from typing import cast
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import LLMError as KernelLLMError
from amplifier_core.message_models import ChatRequest, Message

import amplifier_module_provider_anthropic as provider_module
from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import FakeCoordinator


def _make_provider() -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="[REDACTED:SECRET]",
        config={"use_streaming": False, "max_retries": 0},
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


@pytest.fixture(autouse=True)
def _reset_ssl_context_cache():
    """Each test starts and ends with an empty process-wide context cache."""
    provider_module._SSL_CONTEXT = None
    yield
    provider_module._SSL_CONTEXT = None


class TestEnoentTranslation:
    def test_direct_enoent_is_loud_and_nonretryable(self):
        provider = _make_provider()
        enoent = FileNotFoundError(2, "No such file or directory")
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=enoent
        )

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.provider == "anthropic"
        assert e.retryable is False
        assert e.__cause__ is enoent
        # Message must diagnose the venv swap and say how to recover.
        assert "amplifier update" in str(e)
        assert "restart" in str(e).lower()
        # Original errno detail preserved for correlation with old logs.
        assert "[Errno 2]" in str(e)

    def test_enoent_in_cause_chain_is_detected(self):
        provider = _make_provider()
        cause = FileNotFoundError(2, "No such file or directory")
        wrapper = RuntimeError("client construction failed")
        wrapper.__cause__ = cause
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=wrapper
        )

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.retryable is False
        assert "amplifier update" in str(e)

    def test_enoent_in_context_branch_is_detected(self):
        """ENOENT hiding in __context__ while __cause__ points elsewhere.

        Raising `X from Y` inside an `except Z:` block sets __cause__=Y and
        __context__=Z; the walk must search both branches, not just cause.
        """
        provider = _make_provider()
        try:
            raise FileNotFoundError(2, "No such file or directory")
        except FileNotFoundError:
            wrapper = RuntimeError("wrapped")
            try:
                raise wrapper from ValueError("unrelated cause")
            except RuntimeError as caught:
                chained = caught
        assert chained.__cause__ is not None
        assert isinstance(chained.__context__, FileNotFoundError)

        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=chained
        )

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.retryable is False
        assert "amplifier update" in str(e)

    def test_shallow_cause_not_starved_by_long_context_chain(self):
        """A root with an immediate ENOENT cause must be detected even when
        the sibling context branch is a long chain that could exhaust a
        node-budgeted traversal first (branch starvation)."""
        provider = _make_provider()
        root = RuntimeError("root")
        root.__cause__ = FileNotFoundError(2, "No such file or directory")
        chain = None
        for i in range(11):
            e = ValueError(f"ctx{i}")
            e.__context__ = chain
            chain = e
        root.__context__ = chain

        provider.client.messages.with_raw_response.create = AsyncMock(side_effect=root)

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.retryable is False
        assert "amplifier update" in str(e)

    def test_enoent_found_in_wide_exception_graph(self):
        """ENOENT sitting deep in a WIDE cause/context graph (full binary
        graph above it) must still be found -- the walk is complete, not
        budget-bounded."""
        provider = _make_provider()

        # Full binary graph to depth 2 (7 nodes), ENOENT as the final
        # context-branch leaf at depth 3 (the 8th and last depth-3 node in
        # BFS order -- well past a ten-node budget).
        def leaf(enoent: bool) -> BaseException:
            if enoent:
                return FileNotFoundError(2, "No such file or directory")
            return ValueError("leaf")

        d2 = []
        for i in range(4):
            e = ValueError(f"d2-{i}")
            e.__cause__ = leaf(False)
            e.__context__ = leaf(i == 3)  # ENOENT: last depth-3 node visited
            d2.append(e)
        d1a, d1b = ValueError("d1a"), ValueError("d1b")
        d1a.__cause__, d1a.__context__ = d2[0], d2[1]
        d1b.__cause__, d1b.__context__ = d2[2], d2[3]
        root = RuntimeError("root")
        root.__cause__, root.__context__ = d1a, d1b

        provider.client.messages.with_raw_response.create = AsyncMock(side_effect=root)

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.retryable is False
        assert "amplifier update" in str(e)

    def test_other_oserror_stays_generic_and_retryable(self):
        provider = _make_provider()
        eperm = PermissionError(13, "Permission denied")
        provider.client.messages.with_raw_response.create = AsyncMock(side_effect=eperm)

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.retryable is True
        assert "amplifier update" not in str(e)


class TestSharedSslContext:
    def test_context_built_once_and_shared_across_clients(self, monkeypatch):
        calls = []
        real_builder = provider_module.httpx2.create_ssl_context

        def counting_builder(*args, **kwargs):
            calls.append(1)
            return real_builder(*args, **kwargs)

        monkeypatch.setattr(
            provider_module.httpx2, "create_ssl_context", counting_builder
        )

        p1 = _make_provider()
        p2 = _make_provider()
        c1 = p1.client
        c2 = p2.client
        assert c1 is not c2
        assert len(calls) == 1, "SSL context must be built once per process"

    def test_client_survives_trust_store_deletion(self, monkeypatch, tmp_path):
        """Once the context is cached, a vanished CA bundle must not break
        NEW client creation -- the venv-swap scenario.

        httpx2's default SSL context normally comes from `truststore` (the
        OS-native trust store, no file involved), so it isn't exposed to a
        vanishing venv path. But `SSL_CERT_FILE` routes it through
        `ssl.create_default_context(cafile=...)` instead -- a real
        file-based path, structurally identical to the certifi path that
        broke in production. Copy the system CA bundle to a temp file to
        exercise that path.
        """
        import shutil
        import ssl

        source_cafile = ssl.get_default_verify_paths().cafile
        if not source_cafile or not os.path.exists(source_cafile):
            pytest.skip("no system CA bundle file available to copy")

        cacert = tmp_path / "cacert.pem"
        shutil.copyfile(source_cafile, cacert)
        monkeypatch.setenv("SSL_CERT_FILE", str(cacert))

        p1 = _make_provider()
        _ = p1.client  # builds and caches the context while the file exists

        cacert.unlink()  # the "amplifier update" moment

        p2 = _make_provider()
        _ = p2.client  # must not raise FileNotFoundError

    def test_builder_failure_leaves_client_creation_to_sdk_default(self, monkeypatch):
        """If OUR cached builder fails, client creation must not crash there.

        This only covers the _shared_ssl_context -> None fallback wiring;
        httpx2's internal builder (a separate import reference this patch
        does not touch) still runs with an intact trust store here. The true
        swap-before-first-client case is covered by
        test_swap_before_first_client_is_loud below.
        """

        def broken_builder(*args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory")

        monkeypatch.setattr(
            provider_module.httpx2, "create_ssl_context", broken_builder
        )

        p = _make_provider()
        _ = p.client

    # The SDK's AsyncHttpxClientWrapper is left half-constructed when httpx2
    # raises during its __init__; its __del__ then raises at GC time. That
    # is a side effect of the failure being simulated, not a defect.
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_swap_before_first_client_is_loud(self, monkeypatch):
        """Swap-before-first-client, the real failure sequence end-to-end:
        the trust store is already gone when the FIRST client is built, so
        (1) our builder fails -> _shared_ssl_context() returns None,
        (2) the provider passes http_client=None and the SDK builds its own
            default client, whose httpx2 transport reads the same missing CA
            bundle and raises,
        (3) complete() surfaces the loud non-retryable diagnostic.
        Only the two on-disk trust-store reads are stubbed; the provider,
        SDK, and httpx2 construction code all run for real."""
        import httpx2._transports.default as httpx2_transport

        # Distinct sentinel instances so the final error provably came from
        # the SDK/httpx2 fallback path, not from our own builder leaking.
        our_enoent = FileNotFoundError(2, "No such file or directory")
        transport_enoent = FileNotFoundError(2, "No such file or directory")
        calls: list[str] = []

        def our_broken_builder(*args, **kwargs):
            calls.append("ours")
            raise our_enoent

        def transport_broken_builder(*args, **kwargs):
            calls.append("transport")
            raise transport_enoent

        # Our builder's reference (module attribute lookup on httpx2)...
        monkeypatch.setattr(
            provider_module.httpx2, "create_ssl_context", our_broken_builder
        )
        # ...and the reference httpx2's transport construction actually uses.
        monkeypatch.setattr(
            httpx2_transport, "create_ssl_context", transport_broken_builder
        )

        provider = _make_provider()
        assert provider._client is None  # construction deferred until the call

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.retryable is False
        assert "amplifier update" in str(e)
        assert "restart" in str(e).lower()

        # Prove the REAL sequence ran: our builder failed first (so
        # _shared_ssl_context returned None instead of propagating), then
        # the SDK built its default client and hit the transport builder --
        # whose exact exception instance is the translated error's cause.
        assert calls[0] == "ours"
        assert "transport" in calls
        assert e.__cause__ is transport_enoent

        # Collect the half-constructed wrapper NOW, while this test's
        # unraisable-warning filter is still active.
        import gc

        del e, exc_info, provider
        gc.collect()
