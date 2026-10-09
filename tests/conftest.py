"""Pytest configuration for module tests.

Behavioral tests use inheritance from amplifier-core base classes.
See tests/test_behavioral.py for the inherited tests.

The amplifier-core pytest plugin provides fixtures automatically:
- module_path: Detected path to this module
- module_type: Detected type (provider, tool, hook, etc.)
- provider_module, tool_module, etc.: Mounted module instances
"""

import os

import pytest

from tests._offline_metadata import install_metadata_stub


LIVE_VISION_NODE = (
    "tests/test_image_support.py::test_image_vision_integration_with_real_api"
)
LIVE_PASSED = pytest.StashKey[bool]()


def pytest_addoption(parser):
    parser.addoption(
        "--required-live",
        action="store_true",
        help="Run required real vision only; missing credentials are an unmet gate.",
    )


def pytest_collection_modifyitems(config, items):
    required_live = config.getoption("--required-live")
    live = [item for item in items if item.nodeid == LIVE_VISION_NODE]
    offline = [item for item in items if item.nodeid != LIVE_VISION_NODE]
    if required_live:
        if not live:
            raise pytest.UsageError("Required real vision case was not collected.")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise pytest.UsageError(
                "UNMET required real vision: ANTHROPIC_API_KEY is not set. "
                "No synthetic response or automatic skip satisfies this gate."
            )
        selected, excluded = live, offline
    else:
        selected, excluded = offline, live
    items[:] = selected
    config.hook.pytest_deselected(items=excluded)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    if item.nodeid == LIVE_VISION_NODE and report.when == "call" and report.passed:
        item.config.stash[LIVE_PASSED] = True


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("--required-live") and not session.config.stash.get(
        LIVE_PASSED, False
    ):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(autouse=True)
def offline_messages_metadata(request, monkeypatch):
    """Do not interfere with the real vision test or explicit MockTransport."""
    if request.node.nodeid != LIVE_VISION_NODE:
        return install_metadata_stub(monkeypatch)


@pytest.fixture(autouse=True)
def offline_contract_credentials(request, monkeypatch):
    """Mount real providers for offline inherited contracts, without secrets."""
    if request.node.path.name in {"test_behavioral.py", "test_validation.py"}:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-contract-placeholder")
