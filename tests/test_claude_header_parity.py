"""Drift checks against an installed Claude Code executable.

Claude Code does not expose its final request headers, and OAuth traffic cannot
safely be redirected to a plaintext recorder. This test checks the observable
identity contract in the installed executable. The live integration tests then
verify that Anthropic accepts the resulting provider request.
"""

from pathlib import Path
import re
import shutil
import subprocess

import pytest

from amplifier_anthropic_oauth.auth import (
    AUTHORIZE_URL,
    CLIENT_ID,
    OAUTH_BETAS,
    TOKEN_URL,
    installed_claude_code_version,
    oauth_request_headers,
)


def test_installed_claude_code_oauth_header_contract():
    executable = shutil.which("claude")
    if not executable:
        pytest.skip("Claude Code is not installed")

    resolved = Path(executable).resolve()
    binary = resolved.read_bytes()
    headers = oauth_request_headers()
    version = installed_claude_code_version()

    reported = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout
    assert re.search(rf"\b{re.escape(version)}\b", reported)
    expected_user_agent = f"claude-cli/{version} (external, sdk-cli)"
    assert headers["User-Agent"] == expected_user_agent
    assert b"claude-cli/" in binary
    assert b"(external, " in binary
    assert b"x-app" in binary
    assert AUTHORIZE_URL.encode() in binary
    assert TOKEN_URL.encode() in binary
    assert CLIENT_ID.encode() in binary

    for beta in OAUTH_BETAS:
        assert beta.encode() in binary, (
            f"Installed Claude Code no longer contains {beta}; inspect its "
            "request contract before updating OAUTH_BETAS"
        )
