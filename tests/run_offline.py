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
import sys
import traceback


def install_network_guard():
    """Retain audited attempts separately from provider/test exception handling."""
    attempts = []
    phase = ["preimports"]

    def audit(event, args):
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
            unguarded_child = not (
                isinstance(command, (list, tuple))
                and list(command[:4]) == expected
                and args[0] == sys.executable
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
            raise PermissionError("Offline suite denied audited Python operation")

    sys.addaudithook(audit)

    def snapshot():
        return json.loads(json.dumps(attempts))

    def set_phase(value):
        phase[0] = value

    return snapshot, set_phase


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

    snapshot, set_phase = install_network_guard()
    receipt = {
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

    def shutdown_accounting():
        # Registered BEFORE site/dependencies, so their later atexit callbacks
        # run first. A swallowed shutdown attempt must not escape the receipt
        # or the process's exit status.
        try:
            receipt["attempts"] = snapshot()
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
            pytest_args.append("--basetemp=ai_working/tmp/offline-pytest")
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
