"""Regressions for synthetic fixture scope and trusted-test network denial."""

import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from anthropic import AsyncAnthropic
from anthropic._base_client import AsyncHttpxClientWrapper
from httpx2._utils import URLPattern

from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import DummyResponse, FakeCoordinator
from tests.run_offline import (
    _guarded_runner_command,
    _internal_socketpair_connect,
    _socketpair_connect_verdict,
    _stdlib_socketpair_codes,
)


MODEL = "claude-haiku-5-5"
RUNNER = Path(__file__).with_name("run_offline.py")


@pytest.fixture
def simulated_socket():
    """Pure Windows socket facts, with no dependence on native SO_ACCEPTCONN."""
    ports = iter(range(31000, 32000))
    acceptconn = object()

    class Socket:
        def __init__(self, family=socket.AF_INET, kind=socket.SOCK_STREAM):
            self.family, self.type = family, kind
            self.open = True
            self.listening = False
            self.address = ("0.0.0.0", 0)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def bind(self, address):
            self.address = (address[0], next(ports))

        def listen(self):
            self.listening = True

        def fileno(self):
            return 7 if self.open else -1

        def close(self):
            self.open = False

        def getsockopt(self, level, option):
            assert level == socket.SOL_SOCKET and option is acceptconn
            return int(self.listening)

        def getsockname(self):
            return self.address

    return SimpleNamespace(
        socket=Socket,
        AF_INET=socket.AF_INET,
        AF_INET6=socket.AF_INET6,
        SOCK_STREAM=socket.SOCK_STREAM,
        SOCK_DGRAM=socket.SOCK_DGRAM,
        SOL_SOCKET=socket.SOL_SOCKET,
        SO_ACCEPTCONN=acceptconn,
    )


@pytest.mark.parametrize(
    "platform, expected",
    [("win32", True), ("linux", False), ("darwin", False), ("cygwin", False)],
)
def test_socketpair_predicate_is_windows_only(platform, expected, simulated_socket):
    # Pure predicate with explicit socket facts; not native Windows proof.
    codes = _stdlib_socketpair_codes()
    assert codes
    for name in ("socketpair", "_fallback_socketpair"):
        function = getattr(socket, name, None)
        code = getattr(function, "__code__", None)
        if code is not None:
            assert any(code is captured for captured in codes)
    socket_module = simulated_socket
    with socket_module.socket() as lsock, socket_module.socket() as csock:
        lsock.bind(("127.0.0.1", 0))
        lsock.listen()
        for code in codes:
            caller = SimpleNamespace(
                f_code=code, f_locals={"lsock": lsock, "csock": csock}
            )
            assert (
                _internal_socketpair_connect(
                    (csock, lsock.getsockname()),
                    caller,
                    platform=platform,
                    codes=codes,
                    socket_module=socket_module,
                )
                is expected
            )


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_code",
        "ancestor_only",
        "missing_caller",
        "equal_but_not_identical_code",
        "filename_impostor",
        "no_codes",
        "wrong_socket",
        "missing_csock",
        "same_socket",
        "missing_listener",
        "non_socket_listener",
        "closed_client",
        "closed_listener",
        "not_listening",
        "wildcard_listener",
        "unrelated_listener",
        "udp_client",
        "udp_listener",
        "ipv6_client",
        "ipv6_listener",
        "wrong_host",
        "wrong_port",
        "wrong_address_shape",
    ],
)
def test_socketpair_predicate_fails_closed(fault, simulated_socket):
    socket = simulated_socket
    codes = _stdlib_socketpair_codes()
    with ExitStack() as cleanup:

        def make_socket(family=socket.AF_INET, kind=socket.SOCK_STREAM):
            return cleanup.enter_context(socket.socket(family, kind))

        lsock, csock = make_socket(), make_socket()
        lsock.bind(("0.0.0.0" if fault == "wildcard_listener" else "127.0.0.1", 0))
        if fault != "not_listening":
            lsock.listen()
        address = lsock.getsockname()
        caller = SimpleNamespace(
            f_code=codes[0], f_locals={"lsock": lsock, "csock": csock}
        )
        if fault == "wrong_code":
            caller.f_code = test_socketpair_predicate_fails_closed.__code__
        elif fault == "ancestor_only":
            caller.f_back = SimpleNamespace(f_code=codes[0], f_locals=caller.f_locals)
            caller.f_code = test_socketpair_predicate_fails_closed.__code__
        elif fault == "missing_caller":
            caller = None
        elif fault == "equal_but_not_identical_code":
            caller.f_code = codes[0].replace()
            assert caller.f_code == codes[0] and caller.f_code is not codes[0]
        elif fault == "filename_impostor":
            caller.f_code = test_socketpair_predicate_fails_closed.__code__.replace(
                co_filename=codes[0].co_filename, co_name=codes[0].co_name
            )
        elif fault == "no_codes":
            codes = ()
        elif fault == "wrong_socket":
            csock = make_socket()
        elif fault == "missing_csock":
            caller.f_locals.pop("csock")
        elif fault == "same_socket":
            csock = lsock
            caller.f_locals["csock"] = csock
        elif fault == "missing_listener":
            caller.f_locals.pop("lsock")
        elif fault == "non_socket_listener":
            caller.f_locals["lsock"] = object()
        elif fault == "closed_client":
            csock.close()
        elif fault == "closed_listener":
            lsock.close()
        elif fault == "unrelated_listener":
            other = make_socket()
            other.bind(("127.0.0.1", 0))
            other.listen()
            caller.f_locals["lsock"] = other
        elif fault in {"udp_client", "ipv6_client"}:
            csock = make_socket(
                socket.AF_INET6 if fault == "ipv6_client" else socket.AF_INET,
                socket.SOCK_DGRAM if fault == "udp_client" else socket.SOCK_STREAM,
            )
            caller.f_locals["csock"] = csock
        elif fault in {"udp_listener", "ipv6_listener"}:
            caller.f_locals["lsock"] = make_socket(
                socket.AF_INET6 if fault == "ipv6_listener" else socket.AF_INET,
                socket.SOCK_DGRAM if fault == "udp_listener" else socket.SOCK_STREAM,
            )
        elif fault == "wrong_host":
            address = ("127.0.0.2", address[1])
        elif fault == "wrong_port":
            address = (address[0], 0)
        elif fault == "wrong_address_shape":
            address = (*address, 0, 0)
        assert not _internal_socketpair_connect(
            (csock, address),
            caller,
            platform="win32",
            codes=codes,
            socket_module=socket,
        )
        expected_clause = {
            "wrong_code": "direct_code_identity",
            "ancestor_only": "direct_code_identity",
            "missing_caller": "direct_code_identity",
            "equal_but_not_identical_code": "direct_code_identity",
            "filename_impostor": "direct_code_identity",
            "no_codes": "direct_code_identity",
            "wrong_socket": "client_identity",
            "missing_csock": "client_identity",
            "same_socket": "distinct_sockets",
            "missing_listener": "listener_type",
            "non_socket_listener": "listener_type",
            "closed_client": "client_open",
            "closed_listener": "listener_open",
            "not_listening": "listener_listening",
            "wildcard_listener": "listener_address",
            "unrelated_listener": "exact_address",
            "udp_client": "client_stream",
            "udp_listener": "listener_stream",
            "ipv6_client": "client_family",
            "ipv6_listener": "listener_family",
            "wrong_host": "exact_address",
            "wrong_port": "exact_address",
            "wrong_address_shape": "exact_address",
        }[fault]
        assert _socketpair_connect_verdict(
            (csock, address),
            caller,
            platform="win32",
            codes=codes,
            socket_module=socket,
        ) == {
            "allowed": False,
            "clause": expected_clause,
            "exception": "AttributeError" if fault == "missing_caller" else None,
        }


@pytest.mark.parametrize("capability", ["absent", "raises", "false"])
def test_socketpair_missing_listener_capability_fails_closed(
    capability, simulated_socket
):
    lsock, csock = simulated_socket.socket(), simulated_socket.socket()
    lsock.bind(("127.0.0.1", 0))
    lsock.listen()
    if capability == "absent":
        del simulated_socket.SO_ACCEPTCONN
    elif capability == "raises":

        def unavailable(*args):
            raise OSError("Synthetic private exception text must not be retained")

        lsock.getsockopt = unavailable
    else:
        lsock.listening = False
    codes = _stdlib_socketpair_codes()
    caller = SimpleNamespace(f_code=codes[0], f_locals={"lsock": lsock, "csock": csock})
    verdict = _socketpair_connect_verdict(
        (csock, lsock.getsockname()),
        caller,
        platform="win32",
        codes=codes,
        socket_module=simulated_socket,
    )
    assert verdict == {
        "allowed": False,
        "clause": "listener_listening",
        "exception": {"absent": "AttributeError", "raises": "OSError", "false": None}[
            capability
        ],
    }


def _native_listener_capability(lsock):
    # Independent native observation, not a relaxed predicate or inferred cause.
    try:
        return bool(lsock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)), None
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        return False, type(exc).__name__


def test_stdlib_fallback_direct_caller_predicate_before_connect(record_property):
    # Observe the actual stdlib frame at its C connect boundary, then abort
    # BEFORE audit/syscall. No stdlib patches or network policy changes.
    # Linux 3.11 has only native socketpair; later Linux has a callable fallback.
    fallback = getattr(socket, "_fallback_socketpair", None)
    if fallback is None and sys.platform != "win32":
        assert any(
            socket.socketpair.__code__ is code for code in _stdlib_socketpair_codes()
        )
        with ExitStack() as cleanup:
            for sock in socket.socketpair():
                cleanup.enter_context(sock)
                assert sock.family == socket.AF_UNIX
        record_property(
            "native_socketpair_predicate",
            {"platform": sys.platform, "fallback": "absent", "native_pair": "AF_UNIX"},
        )
        return
    function = fallback or socket.socketpair
    observed = []

    class ProbeComplete(Exception):
        pass

    def profile(frame, event, arg):
        if event == "c_call" and getattr(arg, "__name__", None) == "connect":
            csock = arg.__self__
            lsock = frame.f_locals["lsock"]
            args = (csock, lsock.getsockname())
            capability, exception = _native_listener_capability(lsock)
            verdict = _socketpair_connect_verdict(
                args, frame, platform="win32", codes=_stdlib_socketpair_codes()
            )
            observed.append(
                (
                    frame.f_code is function.__code__,
                    capability,
                    exception,
                    verdict,
                    _internal_socketpair_connect(
                        args, frame, platform="linux", codes=_stdlib_socketpair_codes()
                    ),
                )
            )
            raise ProbeComplete

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        with pytest.raises(ProbeComplete):
            function(socket.AF_INET)
    finally:
        sys.setprofile(previous)
    assert len(observed) == 1, observed
    identity, capability, exception, verdict, ordinary = observed[0]
    record_property(
        "native_socketpair_predicate",
        {
            "platform": sys.platform,
            "code_identity": identity,
            "listener_capability": capability,
            "capability_exception": exception,
            "verdict": verdict,
            "ordinary_allowed": ordinary,
        },
    )
    assert identity and not ordinary, observed
    if sys.platform == "win32":
        assert capability and exception is None, observed
    assert verdict == (
        {"allowed": True, "clause": "accepted", "exception": None}
        if capability
        else {"allowed": False, "clause": "listener_listening", "exception": exception}
    ), observed


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
@pytest.mark.parametrize("shape", ["list", "tuple", "serialized"])
@pytest.mark.parametrize("paths", ["native", "spaces"])
def test_guarded_runner_audit_representation(platform, shape, paths):
    expected = (
        [sys.executable, "-I", "-S", str(RUNNER.resolve())]
        if paths == "native"
        else [
            r"C:\Program Files\Python\python.exe",
            "-I",
            "-S",
            r"C:\Owned tests\run_offline.py",
        ]
    )
    argv = [*expected, "--receipt", "space and tab\tpath\\", '-k=literal"quote', ""]
    command = (
        tuple(argv)
        if shape == "tuple"
        else subprocess.list2cmdline(argv)
        if shape == "serialized"
        else argv
    )
    assert _guarded_runner_command(
        expected[0], command, platform=platform, expected=expected
    ) is (shape != "serialized" or platform == "win32")


@pytest.mark.parametrize(
    "fault",
    [
        "none_executable",
        "different_executable",
        "basename_executable",
        "wrong_argv_executable",
        "missing_isolated",
        "missing_no_site",
        "wrong_runner",
        "runner_suffix",
        "runner_quote_boundary",
        "shell_prefix",
        "python_c",
        "python_m",
        "unterminated_quote",
        "extra_space",
        "tab_boundary",
        "redundant_quotes",
        "nul",
        "bytes",
    ],
)
def test_guarded_runner_serialized_audit_fails_closed(fault):
    expected = [
        r"C:\Program Files\Python\python.exe",
        "-I",
        "-S",
        r"C:\Owned tests\run_offline.py",
    ]
    executable = expected[0]
    argv = [*expected, "-q"]
    command = subprocess.list2cmdline(argv)
    if fault == "none_executable":
        executable = None
    elif fault == "different_executable":
        executable += ".other"
    elif fault == "basename_executable":
        executable = "python.exe"
    elif fault == "wrong_argv_executable":
        argv[0] = "python.exe"
        command = subprocess.list2cmdline(argv)
    elif fault in {"missing_isolated", "missing_no_site"}:
        argv.remove("-I" if fault == "missing_isolated" else "-S")
        command = subprocess.list2cmdline(argv)
    elif fault in {"wrong_runner", "runner_suffix"}:
        argv[3] += ".unguarded"
        command = subprocess.list2cmdline(argv)
    elif fault == "runner_quote_boundary":
        command = subprocess.list2cmdline(expected)[:-1] + ' extra.py" -q'
    elif fault == "shell_prefix":
        command = "cmd.exe /c " + command
    elif fault in {"python_c", "python_m"}:
        argv[3:4] = (
            ["-c", "raise SystemExit(99)"] if fault == "python_c" else ["-m", "pytest"]
        )
        command = subprocess.list2cmdline(argv)
    elif fault == "unterminated_quote":
        command += ' "unfinished'
    elif fault == "extra_space":
        command = subprocess.list2cmdline(expected) + "  -q"
    elif fault == "tab_boundary":
        command = subprocess.list2cmdline(expected) + "\t-q"
    elif fault == "redundant_quotes":
        command = subprocess.list2cmdline(expected) + ' "-q"'
    elif fault == "nul":
        command += "\0"
    elif fault == "bytes":
        command = command.encode()
    assert not _guarded_runner_command(
        executable, command, platform="win32", expected=expected
    )


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


def _child_environment(env=None):
    isolated = {} if env is None else dict(env)
    if sys.platform == "win32":
        # Windows needs its OS bootstrap directory even for isolated Python.
        # Do not inherit credentials, proxies, PATH or the parent's private HOME.
        isolated = {
            name: value for name, value in isolated.items()
            if name.upper() != "SYSTEMROOT"
        }
        system_root = next(
            (value for name, value in os.environ.items()
             if name.upper() == "SYSTEMROOT" and value),
            None,
        )
        if system_root is not None:
            isolated["SYSTEMROOT"] = system_root
    return isolated


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
@pytest.mark.parametrize("explicit", [False, True])
def test_owned_child_environment_keeps_only_windows_bootstrap(monkeypatch, platform, explicit):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os, "environ", {
        "SystemRoot": r"C:\Windows",
        "ANTHROPIC_API_KEY": "synthetic-parent-only",
        "HTTP_PROXY": "http://synthetic.invalid",
        "PATH": "synthetic-parent-path",
        "HOME": "synthetic-parent-home",
    })
    supplied = {"CONTROLLED": "synthetic", "systemroot": "synthetic-wrong"} if explicit else None
    original = None if supplied is None else dict(supplied)
    expected = {} if supplied is None else dict(supplied)
    if platform == "win32":
        expected.pop("systemroot", None)
        expected["SYSTEMROOT"] = r"C:\Windows"
    assert _child_environment(supplied) == expected
    assert supplied == original


@pytest.mark.parametrize("system_root", [None, ""])
def test_owned_child_environment_does_not_invent_windows_bootstrap(monkeypatch, system_root):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "environ", {} if system_root is None else {"SYSTEMROOT": system_root})
    assert _child_environment() == {}
    assert _child_environment({"systemroot": "synthetic-wrong", "CONTROLLED": "synthetic"}) == {
        "CONTROLLED": "synthetic"
    }


def _child(
    tmp_path,
    source,
    *args,
    cwd=None,
    env=None,
    default_basetemp=False,
    receipt_path=None,
):
    case = tmp_path / "test_guard_case.py"
    case.write_text(source)
    receipt = receipt_path or tmp_path / "receipt.json"
    temp_args = [] if default_basetemp else ["--basetemp", str(tmp_path / "child-temp")]
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
            *temp_args,
            *args,
        ],
        executable=sys.executable,
        # No credentials, user configs, proxies, PYTHONPATH or pytest flags.
        env=_child_environment(env),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, json.loads(receipt.read_text())


def test_runner_default_basetemp_works_without_ai_working(tmp_path):
    assert not (tmp_path / "ai_working").exists()
    result, receipt = _child(
        tmp_path,
        """
def test_first_run(tmp_path):
    from pathlib import Path
    assert tmp_path.is_dir()
    assert tmp_path.parent == Path("ai_working/tmp/offline-pytest").resolve()
""",
        cwd=tmp_path,
        default_basetemp=True,
    )
    assert result.returncode == receipt["exit_code"] == 0, result.stdout + result.stderr
    assert receipt["attempts"] == []
    assert receipt["shutdown_accounted"]
    assert len(receipt["selected"]) == 1
    assert all(report["outcome"] == "passed" for report in receipt["reports"])


@pytest.mark.parametrize("polluted", [False, True])
def test_runner_provides_fresh_home_for_posix_and_windows_resolution(
    tmp_path, polluted
):
    private_home = tmp_path / "synthetic-private-home"
    settings = private_home / ".amplifier" / "settings.yaml"
    settings.parent.mkdir(parents=True)
    settings.write_text("synthetic_private_setting: must-not-be-loaded\n")
    result, receipt = _child(
        tmp_path,
        """
def test_owned_home():
    import ntpath, os
    from pathlib import Path
    home = Path.home()
    assert home.is_absolute() and home.is_dir()
    assert str(home) == os.environ["HOME"] == os.environ["USERPROFILE"]
    assert home.parent == Path.cwd() and home.name.startswith("offline-home-")
    assert list(home.iterdir()) == []
    # Execute the stdlib Windows resolver even on POSIX. This is not Windows
    # execution proof; native Path.home() is also checked on each hosted OS.
    assert ntpath.expanduser("~") == str(home)
    assert os.environ["HOMEDRIVE"] + os.environ["HOMEPATH"] == str(home)
    assert not (home / ".amplifier" / "settings.yaml").exists()
    assert all(name not in os.environ for name in (
        "AMPLIFIER_HOME", "XDG_CONFIG_HOME", "APPDATA", "LOCALAPPDATA",
        "ANTHROPIC_API_KEY", "HTTP_PROXY", "PYTHONPATH", "PYTEST_ADDOPTS"
    ))
    # Windows also supports drive/path resolution when USERPROFILE is absent.
    del os.environ["USERPROFILE"]
    assert ntpath.expanduser("~") == str(home)
""",
        cwd=tmp_path,
        env=(
            {
                "HOME": str(private_home),
                "USERPROFILE": str(private_home),
                "HOMEDRIVE": "Z:",
                "HOMEPATH": "\\synthetic-private-home",
                "AMPLIFIER_HOME": str(settings.parent),
                "XDG_CONFIG_HOME": str(private_home),
                "APPDATA": str(private_home),
                "LOCALAPPDATA": str(private_home),
                "ANTHROPIC_API_KEY": "synthetic-offline-credential",
                "HTTP_PROXY": "http://synthetic.invalid",
                "PYTHONPATH": str(private_home),
                "PYTEST_ADDOPTS": "--required-live",
            }
            if polluted
            else {}
        ),
    )
    assert result.returncode == receipt["exit_code"] == 0, result.stdout + result.stderr
    assert receipt["attempts"] == []
    assert receipt["preimports"] == receipt["postimports"]
    assert settings.read_text() == "synthetic_private_setting: must-not-be-loaded\n"
    assert not list(tmp_path.glob("offline-home-*"))
    assert str(private_home) not in json.dumps(receipt)


def test_runner_home_survives_receipt_inside_pytest_deletion_tree(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
def test_home_survives(tmp_path):
    from pathlib import Path
    assert tmp_path.is_dir()
    assert Path.home().is_dir()
    assert Path.home().parent == Path.cwd()
    assert list(Path.home().iterdir()) == []
""",
        cwd=tmp_path,
        receipt_path=tmp_path / "child-temp" / "receipt.json",
    )
    assert result.returncode == receipt["exit_code"] == 0, result.stdout + result.stderr
    assert receipt["attempts"] == []


@pytest.mark.parametrize("deny", [False, True])
def test_runner_home_lives_through_retained_finalizers_then_is_removed(tmp_path, deny):
    result, receipt = _child(
        tmp_path,
        f"""
_retained = []

def test_retained_finalizers():
    import json, socket, weakref
    from pathlib import Path
    class Retained:
        pass
    home = Path.home()
    witness = Path.cwd() / "finalizer-home.json"
    assert home.is_dir()
    assert not witness.exists()
    def write_cache():
        existed = home.is_dir()
        cache = home / ".cache" / "synthetic-dependency" / "entry"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("synthetic finalizer cache")
        witness.write_text(json.dumps({{
            "home": str(home), "existed": existed, "cache": cache.read_text()
        }}))
    writer = Retained()
    _retained.append(writer)
    weakref.finalize(writer, write_cache)
    if {deny!r}:
        def denied_request():
            try:
                socket.getaddrinfo("synthetic.invalid", 443)
            except PermissionError:
                pass
        requester = Retained()
        _retained.append(requester)
        weakref.finalize(requester, denied_request)
""",
        cwd=tmp_path,
    )
    assert result.returncode == receipt["exit_code"] == int(deny), (
        result.stdout + result.stderr
    )
    assert receipt["shutdown_accounted"]
    assert all(report["outcome"] == "passed" for report in receipt["reports"])
    witness = json.loads((tmp_path / "finalizer-home.json").read_text())
    assert witness["existed"], "HOME was removed before the retained finalizer ran"
    assert witness["cache"] == "synthetic finalizer cache"
    assert not Path(witness["home"]).exists(), "Finalizer cache escaped HOME cleanup"
    assert not list(tmp_path.glob("offline-home-*"))
    if deny:
        assert [a["event"] for a in receipt["attempts"]] == ["socket.getaddrinfo"]
        assert receipt["attempts"][0]["phase_or_nodeid"] == "shutdown"
    else:
        assert receipt["attempts"] == []


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


def test_runner_socketpair_and_event_loops_are_internal_not_vendor_requests(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
_retained = []

def test_internal_plumbing():
    import asyncio, socket, weakref
    def pair():
        left, right = socket.socketpair()
        try:
            left.send(b"internal")
            assert right.recv(8) == b"internal"
        finally:
            left.close()
            right.close()
    async def ready():
        return True
    pair()
    assert asyncio.run(ready())
    assert asyncio.run(ready())
    class Retained:
        pass
    obj = Retained()
    _retained.append(obj)
    weakref.finalize(obj, pair)
""",
    )
    assert result.returncode == receipt["exit_code"] == 0, result.stdout + result.stderr
    assert receipt["attempts"] == []
    assert receipt["shutdown_accounted"]
    count = receipt["permitted_operations"]["internal_socketpair_connect"]
    if sys.platform == "win32":
        assert count >= 4  # one pair, two loops, and the shutdown pair
    else:
        assert count == 0


@pytest.mark.parametrize("operation", ["connect", "connect_ex", "sendto", "sendmsg"])
def test_guard_still_denies_ordinary_loopback_and_datagrams(tmp_path, operation):
    result, receipt = _child(
        tmp_path,
        f"""
def test_denied_operation():
    import socket
    with socket.socket() as lsock, socket.socket() as csock:
        lsock.bind(("127.0.0.1", 0))
        lsock.listen()
        address = lsock.getsockname()
        try:
            operation = {operation!r}
            if operation == "connect":
                csock.connect(address)
            elif operation == "connect_ex":
                csock.connect_ex(address)
            else:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
                    if operation == "sendto":
                        udp.sendto(b"denied", address)
                    elif hasattr(udp, "sendmsg"):
                        udp.sendmsg([b"denied"], [], 0, address)
                    else:
                        # Windows has no sendmsg method; exercise its audit
                        # event without claiming a native sendmsg syscall.
                        import sys
                        sys.audit("socket.sendmsg", udp, address)
        except PermissionError:
            pass
        else:
            raise AssertionError("Ordinary loopback/datagram escaped")
""",
    )
    assert result.returncode == receipt["exit_code"] == 1
    event = "socket.connect" if operation == "connect_ex" else f"socket.{operation}"
    assert [a["event"] for a in receipt["attempts"]] == [event]
    assert receipt["shutdown_accounted"]


@pytest.mark.parametrize("operation", ["gethostbyname", "gethostbyaddr"])
def test_guard_still_denies_other_dns_operations(tmp_path, operation):
    result, receipt = _child(
        tmp_path,
        f"""
def test_dns_denied():
    import socket
    try:
        socket.{operation}("127.0.0.1")
    except PermissionError:
        pass
    else:
        raise AssertionError("DNS escaped")
""",
    )
    assert result.returncode == receipt["exit_code"] == 1
    assert [a["event"] for a in receipt["attempts"]] == [f"socket.{operation}"]


def test_guard_denies_socketpair_filename_impostor(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
def test_impostor():
    import socket
    source = '''
def socketpair():
    with socket.socket() as lsock, socket.socket() as csock:
        lsock.bind(("127.0.0.1", 0))
        lsock.listen()
        try:
            csock.connect(lsock.getsockname())
        except PermissionError:
            pass
        else:
            raise AssertionError("Filename impostor escaped")
'''
    namespace = {"socket": socket}
    exec(compile(source, socket.__file__, "exec"), namespace)
    namespace["socketpair"]()
""",
    )
    assert result.returncode == receipt["exit_code"] == 1
    assert [a["event"] for a in receipt["attempts"]] == ["socket.connect"]
    assert any(f["function"] == "socketpair" for f in receipt["attempts"][0]["stack"])


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


@pytest.mark.parametrize(
    "fault", ["missing_isolated", "wrong_runner", "shell", "malformed"]
)
def test_guard_denies_malformed_child_before_execution(tmp_path, fault):
    # Isolated temp-directory collection does not guarantee the tests namespace.
    # Pass the already-known runner path, without changing import or guard rules.
    result, receipt = _child(
        tmp_path,
        f"""
def test_no_execution():
    import subprocess, sys
    from pathlib import Path
    runner = {str(RUNNER.resolve())!r}
    witness = Path.cwd() / "unguarded-executed"
    payload = "from pathlib import Path; Path(" + repr(str(witness)) + ").touch()"
    command = [sys.executable, "-I", "-S", str(Path(runner).resolve()), "-q"]
    shell = False
    fault = {fault!r}
    if fault == "missing_isolated":
        command.remove("-I")
    elif fault == "wrong_runner":
        command[3:] = ["-c", payload]
    elif fault == "shell":
        command = subprocess.list2cmdline([sys.executable, "-c", payload])
        shell = True
    else:
        command = subprocess.list2cmdline(command) + ' "unfinished'
    try:
        subprocess.run(command, executable=sys.executable, shell=shell, timeout=10)
    except PermissionError:
        pass
    else:
        raise AssertionError("Malformed child was not refused before execution")
    assert not witness.exists()
""",
        cwd=tmp_path,
    )
    assert result.returncode == receipt["exit_code"] == 1, result.stdout + result.stderr
    assert [a["event"] for a in receipt["attempts"]] == ["subprocess.Popen"]
    assert all(r["outcome"] == "passed" for r in receipt["reports"])
    assert not (tmp_path / "unguarded-executed").exists()


def test_owned_launch_uses_exact_native_audit_representation(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
def test_native_audit():
    import json, subprocess, sys
    from pathlib import Path
    runner = __RUNNER_PATH__
    runner = str(Path(runner).resolve())
    case = Path.cwd() / "test_nested.py"
    case.write_text("def test_nested(): pass\\n")
    destination = Path.cwd() / "nested-receipt.json"
    observed = []
    def observer(event, args):
        if event == "subprocess.Popen":
            observed.append({
                "executable_is_exact": args[0] == sys.executable,
                "command_type": type(args[1]).__name__,
                "canonical_command": args[1] == subprocess.list2cmdline(command)
                    if sys.platform == "win32" else args[1] == command,
            })
    sys.addaudithook(observer)
    command = [sys.executable, "-I", "-S", runner, "--receipt", str(destination),
               str(case), "-q", "--basetemp", str(Path.cwd() / "nested-temp")]
    result = subprocess.run(command, executable=sys.executable, env=__BOOTSTRAP_ENV__,
                            capture_output=True, text=True, timeout=60)
    Path("native-audit.json").write_text(json.dumps(observed))
    assert result.returncode == 0, result.stdout + result.stderr
    assert observed == [{
        "executable_is_exact": True,
        "command_type": "str" if sys.platform == "win32" else "list",
        "canonical_command": True,
    }], observed
    receipt = json.loads(destination.read_text())
    assert receipt["attempts"] == [] and receipt["shutdown_accounted"]
""".replace("__RUNNER_PATH__", repr(str(RUNNER.resolve()))).replace(
            "__BOOTSTRAP_ENV__", repr(_child_environment())
        ),
        cwd=tmp_path,
    )
    assert result.returncode == receipt["exit_code"] == 0, result.stdout + result.stderr
    assert receipt["attempts"] == []
    observed = json.loads((tmp_path / "native-audit.json").read_text())
    assert observed[0]["executable_is_exact"] and observed[0]["canonical_command"]


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


def test_guard_retains_swallowed_weakref_finalizer_attempt_and_fails_exit(tmp_path):
    result, receipt = _child(
        tmp_path,
        """
_retained = []

def test_finalizer():
    import socket, weakref
    class Retained:
        pass
    obj = Retained()
    _retained.append(obj)
    def late_request():
        try:
            socket.getaddrinfo("synthetic.invalid", 443)
        except PermissionError:
            pass
    weakref.finalize(obj, late_request)
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
        executable=sys.executable,
        env=_child_environment({
            "ANTHROPIC_API_KEY": "synthetic-offline-credential",
            "ANTHROPIC_BASE_URL": "https://synthetic.invalid",
            "HTTP_PROXY": "http://synthetic.invalid",
            "PYTHONPATH": "synthetic-nonexistent-path",
            "PYTEST_ADDOPTS": "--required-live",
        }),
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
    assert result.returncode == 0, (
        result.stdout + result.stderr,
        [(attempt["event"], attempt["stack"]) for attempt in receipt["attempts"]],
    )
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
    assert all(a["event"] == "socket.getaddrinfo" for a in receipt["attempts"]), [
        (attempt["event"], attempt["stack"]) for attempt in receipt["attempts"]
    ]
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
        executable=sys.executable,
        env=_child_environment(),
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
