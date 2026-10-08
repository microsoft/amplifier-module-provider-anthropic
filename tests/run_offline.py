"""Run trusted offline tests with a pre-import Python network guard.

Provision dev dependencies plus amplifier-foundation, then:
    uv run python -I -S tests/run_offline.py -q

Pytest arguments follow this script's optional --receipt PATH. The receipt
contains the complete selected/deselected inventory, skips, warnings, imported
source identities and denied audited Python network attempts. This is test
isolation, not a tamper-proof security sandbox for hostile Python or native code.
Offline success never meets the separate required real-vision gate.
"""

import argparse
import atexit
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import traceback
from types import CodeType


def _stdlib_socketpair_codes():
    """Capture normal stdlib Python implementations, not filenames or wrappers."""
    return tuple(
        code
        for name in ("socketpair", "_fallback_socketpair")
        if isinstance(
            code := getattr(getattr(socket, name, None), "__code__", None), CodeType
        )
    )


def _socketpair_connect_verdict(args, caller, *, platform, codes, socket_module=socket):
    """Safe clause/type diagnostics; never include addresses or exception text.

    socket_module is a pure-test seam. The installed guard always uses stdlib.
    """
    clause = "platform"
    try:
        if platform != "win32":
            return {"allowed": False, "clause": clause, "exception": None}
        clause = "direct_code_identity"
        if not any(caller.f_code is code for code in codes):
            return {"allowed": False, "clause": clause, "exception": None}
        clause = "audit_arguments"
        csock, address = args
        lsock = caller.f_locals.get("lsock")
        # Evaluate sequentially: the first rejecting check or exception is known.
        checks = (
            ("client_identity", lambda: caller.f_locals.get("csock") is csock),
            ("client_type", lambda: type(csock) is socket_module.socket),
            ("listener_type", lambda: type(lsock) is socket_module.socket),
            ("distinct_sockets", lambda: csock is not lsock),
            ("client_family", lambda: csock.family == socket_module.AF_INET),
            ("listener_family", lambda: lsock.family == socket_module.AF_INET),
            ("client_stream", lambda: csock.type == socket_module.SOCK_STREAM),
            ("listener_stream", lambda: lsock.type == socket_module.SOCK_STREAM),
            ("client_open", lambda: csock.fileno() >= 0),
            ("listener_open", lambda: lsock.fileno() >= 0),
            (
                "listener_listening",
                lambda: bool(
                    lsock.getsockopt(
                        socket_module.SOL_SOCKET, socket_module.SO_ACCEPTCONN
                    )
                ),
            ),
        )
        for clause, check in checks:
            if not check():
                return {"allowed": False, "clause": clause, "exception": None}
        clause = "listener_address"
        own_address = lsock.getsockname()
        if own_address[0] != "127.0.0.1":
            return {"allowed": False, "clause": clause, "exception": None}
        clause = "exact_address"
        if address != own_address:
            return {"allowed": False, "clause": clause, "exception": None}
        return {"allowed": True, "clause": "accepted", "exception": None}
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        return {"allowed": False, "clause": clause, "exception": type(exc).__name__}


def _internal_socketpair_connect(
    args, caller, *, platform, codes, socket_module=socket
):
    """Allow only Windows' stdlib IPv4 self-pipe connect to its own listener."""
    return _socketpair_connect_verdict(
        args, caller, platform=platform, codes=codes, socket_module=socket_module
    )["allowed"]


def _canonical_windows_argv(command):
    """Decode only round-trippable subprocess.list2cmdline output, not a shell."""
    if "\0" in command:
        return None
    argv = []
    index = 0
    while index < len(command):
        while index < len(command) and command[index] in " \t":
            index += 1
        if index == len(command):
            break
        argument = []
        quoted = False
        while index < len(command):
            if command[index] in " \t" and not quoted:
                break
            slashes = 0
            while index < len(command) and command[index] == "\\":
                slashes += 1
                index += 1
            if index < len(command) and command[index] == '"':
                argument.append("\\" * (slashes // 2))
                if slashes % 2:
                    argument.append('"')
                else:
                    quoted = not quoted
                index += 1
            else:
                argument.append("\\" * slashes)
                if index < len(command):
                    if command[index] in " \t" and not quoted:
                        break
                    argument.append(command[index])
                    index += 1
        if quoted:
            return None
        argv.append("".join(argument))
    return argv if subprocess.list2cmdline(argv) == command else None


def _guarded_runner_command(executable, command, *, platform, expected):
    """Exact executable/isolated-runner argv; Windows also audits serialized argv."""
    if executable != expected[0]:
        # Owned launch sites pass executable explicitly, including on Windows.
        return False
    if platform == "win32" and isinstance(command, str):
        prefix = subprocess.list2cmdline(expected)
        if command != prefix and not command.startswith(prefix + " "):
            return False
        command = _canonical_windows_argv(command)
    return isinstance(command, (list, tuple)) and list(command[:4]) == expected


def install_network_guard():
    """Retain audited attempts separately from provider/test exception handling."""
    attempts = []
    phase = ["preimports"]
    # main calls us after isolated no-site re-exec and normal stdlib import,
    # before site/dependencies. Keep those exact identities for this guard.
    platform = sys.platform
    socketpair_codes = _stdlib_socketpair_codes()
    permitted = {"internal_socketpair_connect": 0}

    def audit(event, args):
        socketpair_verdict = None
        if event == "socket.connect":
            socketpair_verdict = _socketpair_connect_verdict(
                args, sys._getframe(1), platform=platform, codes=socketpair_codes
            )
            if socketpair_verdict["allowed"]:
                # No target, payload or vendor attribution: this is stdlib plumbing.
                permitted["internal_socketpair_connect"] += 1
                return
        dns = event in {
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.gethostbyaddr",
            "socket.getnameinfo",
        }
        inet = event in {"socket.connect", "socket.sendto", "socket.sendmsg"} and args[
            0
        ].family in {socket.AF_INET, socket.AF_INET6}
        unguarded_child = event in {"os.system", "os.exec", "os.posix_spawn"}
        if event == "subprocess.Popen":
            command = args[1]
            # Permit only this guarded runner in an isolated no-site child.
            # Other subprocesses do not inherit this Python audit hook.
            expected = [sys.executable, "-I", "-S", str(Path(__file__).resolve())]
            unguarded_child = not _guarded_runner_command(
                args[0], command, platform=platform, expected=expected
            )
        if dns or inet or unguarded_child:
            attempts.append(
                {
                    "event": event,
                    "phase_or_nodeid": phase[0],
                    # No request bodies, headers, environment values, or targets.
                    "stack": [
                        {
                            "file": Path(frame.filename).name,
                            "line": frame.lineno,
                            "function": frame.name,
                        }
                        for frame in traceback.extract_stack(limit=16)[:-1]
                    ],
                }
            )
            if socketpair_verdict is not None:
                attempts[-1]["socketpair_predicate"] = socketpair_verdict
            raise PermissionError("Offline suite denied audited Python operation")

    sys.addaudithook(audit)

    def snapshot():
        return json.loads(json.dumps(attempts))

    def set_phase(value):
        phase[0] = value

    def permitted_snapshot():
        return dict(permitted)

    return snapshot, set_phase, permitted_snapshot


def imported_identity():
    names = (
        "amplifier_core",
        "amplifier_core.models",
        "amplifier_foundation",
        "amplifier_foundation.spawn_utils",
        "amplifier_module_provider_anthropic",
        "anthropic",
    )
    return {
        name: {
            "path": str(Path(sys.modules[name].__file__).resolve()),
            "sha256": hashlib.sha256(
                Path(sys.modules[name].__file__).read_bytes()
            ).hexdigest(),
        }
        for name in names
    }


def bootstrap_stdlib_platform():
    """Populate genuine Windows OS metadata before site/dependency execution."""
    result = {"windows_os_metadata": False, "process_count": 0}
    if sys.platform != "win32":
        return result
    import platform

    active = True

    def observe(event, args):
        if active and event == "subprocess.Popen":
            result["process_count"] += 1

    sys.addaudithook(observe)
    try:
        # CPython3.11's real win32_ver may query `ver`; no fabricated cache/value.
        platform.uname()
        result["windows_os_metadata"] = True
    finally:
        active = False
    return result


def main():
    # Re-exec in isolated mode before any third-party imports. Allow only OS
    # execution/locale essentials, not inherited keys, proxies or pytest flags.
    clean_env = {
        name: os.environ[name]
        for name in ("PATH", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL")
        if name in os.environ
    }
    clean_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if not (sys.flags.isolated and sys.flags.no_site):
        os.execve(
            sys.executable,
            [sys.executable, "-I", "-S", str(Path(__file__).resolve()), *sys.argv[1:]],
            clean_env,
        )
    os.environ.clear()
    os.environ.update(clean_env)

    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", default="ai_working/tmp/offline-receipt.json")
    options, pytest_args = parser.parse_known_args()
    # --required-live is useful as a negative gate probe here: sanitation leaves
    # it unmet. It does not disable the guard or authorize a vendor request.

    bootstrap = bootstrap_stdlib_platform()
    snapshot, set_phase, permitted_snapshot = install_network_guard()
    destination = Path(options.receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "bootstrap": bootstrap,
        "network_guard": "Python audit hook, installed before site and third-party imports",
        "executable": sys.executable,
        "isolated": bool(sys.flags.isolated),
        "credentials": "sanitized before imports; fixture placeholders only",
        "required_live": {
            "status": "UNMET",
            "nodeid": (
                "tests/test_image_support.py::"
                "test_image_vision_integration_with_real_api"
            ),
            "reason": "Offline inventory is not real vendor vision qualification.",
        },
        "selected": [],
        "deselected": [],
        "reports": [],
        "warnings": [],
    }
    code = 1
    home = None

    def shutdown_accounting():
        # Registered BEFORE site/dependencies, so their later atexit callbacks
        # run first. A swallowed shutdown attempt must not escape the receipt
        # or the process's exit status.
        try:
            # Retain home through main's return and later dependency finalizers;
            # its earlier weakref finalizer normally already cleaned it here.
            if home is not None:
                home.cleanup()
            receipt["attempts"] = snapshot()
            receipt["permitted_operations"] = permitted_snapshot()
            receipt["shutdown_accounted"] = True
            receipt["exit_code"] = 1 if receipt["attempts"] else code
            destination = Path(options.receipt)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(receipt, indent=2) + "\n")
            if receipt["attempts"]:
                print(
                    "FAIL: denied audited Python attempts retained through shutdown.",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(1)
        except Exception:
            # Ignored atexit exceptions otherwise leave a successful exit code.
            os._exit(1)

    atexit.register(shutdown_accounting)
    try:
        # Register accounting first: TemporaryDirectory initializes weakref's
        # shared atexit dispatcher, including later dependency/test finalizers.
        # Its earlier finalizer runs AFTER theirs. A separate atexit cleanup
        # would run before that dispatcher and remove HOME too soon.
        # Cwd is outside pytest's deletion tree (pytest rejects cwd/ancestors as
        # basetemp), even when the caller puts the receipt inside that tree.
        home = tempfile.TemporaryDirectory(prefix="offline-home-", dir=Path.cwd())
        home_path = str(Path(home.name).resolve())
        drive, path = os.path.splitdrive(home_path)
        os.environ.update(
            HOME=home_path, USERPROFILE=home_path, HOMEDRIVE=drive, HOMEPATH=path
        )
        # -S defers .pth/sitecustomize execution until sanitation and denial are
        # live. site.main() performs normal venv/site setup; no sys.path hacks.
        import site

        site.main()
        # Real installed packages in this SAME process, before pytest collection.
        for name in (
            "amplifier_core",
            "amplifier_core.models",
            "amplifier_foundation",
            "amplifier_foundation.spawn_utils",
            "amplifier_module_provider_anthropic",
            "anthropic",
        ):
            importlib.import_module(name)
        receipt["preimports"] = imported_identity()
        receipt["versions"] = {
            name: importlib.metadata.version(name)
            for name in (
                "amplifier-core",
                "amplifier-foundation",
                "anthropic",
                "pytest",
                "pytest-asyncio",
            )
        }
        import pytest

        class Evidence:
            def pytest_collection_finish(self, session):
                receipt["selected"] = [item.nodeid for item in session.items]

            def pytest_deselected(self, items):
                receipt["deselected"].extend(item.nodeid for item in items)

            def pytest_runtest_setup(self, item):
                set_phase(item.nodeid)

            def pytest_runtest_logreport(self, report):
                receipt["reports"].append(
                    {
                        "nodeid": report.nodeid,
                        "when": report.when,
                        "outcome": report.outcome,
                        "detail": str(report.longrepr) if report.longrepr else None,
                    }
                )
                if report.when == "call":
                    for name, value in report.user_properties:
                        if name == "native_socketpair_predicate":
                            receipt["reports"][-1]["native_socketpair_predicate"] = (
                                value
                            )

            def pytest_warning_recorded(self, warning_message, when, nodeid):
                receipt["warnings"].append(
                    {
                        "category": warning_message.category.__name__,
                        "message": str(warning_message.message),
                        "when": when,
                        "nodeid": nodeid,
                    }
                )

            def pytest_sessionfinish(self, session, exitstatus):
                if snapshot():
                    session.exitstatus = pytest.ExitCode.TESTS_FAILED

        set_phase("collection")
        if not any(arg.startswith("--basetemp") for arg in pytest_args):
            basetemp = Path("ai_working/tmp/offline-pytest")
            # Pytest creates basetemp itself, but not missing parent directories.
            basetemp.parent.mkdir(parents=True, exist_ok=True)
            pytest_args.append(f"--basetemp={basetemp}")
        code = int(
            pytest.main(
                [
                    "-p",
                    "pytest_asyncio.plugin",
                    "-p",
                    "amplifier_core.pytest_plugin",
                    *pytest_args,
                ],
                plugins=[Evidence()],
            )
        )
        receipt["postimports"] = imported_identity()
        print("UNMET: required real vision; offline success does not qualify it.")
    except Exception as exc:
        code = 1
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        print(receipt["error"], file=sys.stderr)
    finally:
        atexit.register(set_phase, "shutdown")
        receipt["attempts"] = snapshot()
        receipt["permitted_operations"] = permitted_snapshot()
        if receipt["attempts"]:
            code = 1
            print(
                f"FAIL: {len(receipt['attempts'])} denied audited Python attempts "
                "(including swallowed exceptions).",
                file=sys.stderr,
            )
        receipt["exit_code"] = code
        destination = Path(options.receipt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(receipt, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
