"""Tests for expanded rate limit header parsing.

Verifies that _extract_rate_limit_headers() reads all Anthropic headers:
- Existing: requests-remaining/limit, tokens-remaining/limit, retry-after
- New: requests-reset, tokens-reset, input-tokens-*, output-tokens-*
"""

from amplifier_module_provider_anthropic import AnthropicProvider


def _make_provider() -> AnthropicProvider:
    """Create a provider instance for testing header parsing."""
    return AnthropicProvider(
        api_key="test-key",
        config={"use_streaming": False, "max_retries": 0},
    )


class TestExistingHeadersParsing:
    """Verify existing header parsing still works (regression guard)."""

    def test_requests_remaining_and_limit(self):
        provider = _make_provider()
        headers = {
            "anthropic-ratelimit-requests-remaining": "95",
            "anthropic-ratelimit-requests-limit": "100",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info["requests_remaining"] == 95
        assert info["requests_limit"] == 100

    def test_tokens_remaining_and_limit(self):
        provider = _make_provider()
        headers = {
            "anthropic-ratelimit-tokens-remaining": "450000",
            "anthropic-ratelimit-tokens-limit": "500000",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info["tokens_remaining"] == 450000
        assert info["tokens_limit"] == 500000

    def test_retry_after(self):
        provider = _make_provider()
        headers = {"retry-after": "58.5"}
        info = provider._extract_rate_limit_headers(headers)
        assert info["retry_after_seconds"] == 58.5

    def test_empty_headers_returns_empty_dict(self):
        provider = _make_provider()
        assert provider._extract_rate_limit_headers({}) == {}
        assert provider._extract_rate_limit_headers(None) == {}


class TestNewResetHeaders:
    """Verify new *-reset timestamp headers are parsed as strings."""

    def test_requests_reset(self):
        provider = _make_provider()
        headers = {
            "anthropic-ratelimit-requests-reset": "2026-02-24T10:30:00Z",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info["requests_reset"] == "2026-02-24T10:30:00Z"

    def test_tokens_reset(self):
        provider = _make_provider()
        headers = {
            "anthropic-ratelimit-tokens-reset": "2026-02-24T10:30:00Z",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info["tokens_reset"] == "2026-02-24T10:30:00Z"


class TestNewInputTokenHeaders:
    """Verify new input-tokens dimension headers are parsed."""

    def test_input_tokens_remaining_and_limit(self):
        provider = _make_provider()
        headers = {
            "anthropic-ratelimit-input-tokens-remaining": "800000",
            "anthropic-ratelimit-input-tokens-limit": "1000000",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info["input_tokens_remaining"] == 800000
        assert info["input_tokens_limit"] == 1000000

    def test_input_tokens_reset(self):
        provider = _make_provider()
        headers = {
            "anthropic-ratelimit-input-tokens-reset": "2026-02-24T10:31:00Z",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info["input_tokens_reset"] == "2026-02-24T10:31:00Z"


class TestNewOutputTokenHeaders:
    """Verify new output-tokens dimension headers are parsed."""

    def test_output_tokens_remaining_and_limit(self):
        provider = _make_provider()
        headers = {
            "anthropic-ratelimit-output-tokens-remaining": "90000",
            "anthropic-ratelimit-output-tokens-limit": "100000",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info["output_tokens_remaining"] == 90000
        assert info["output_tokens_limit"] == 100000

    def test_output_tokens_reset(self):
        provider = _make_provider()
        headers = {
            "anthropic-ratelimit-output-tokens-reset": "2026-02-24T10:31:00Z",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info["output_tokens_reset"] == "2026-02-24T10:31:00Z"


class TestAllHeadersTogether:
    """Verify all ~15 keys are returned when all headers are present."""

    def test_full_header_set(self):
        provider = _make_provider()
        headers = {
            # Existing
            "anthropic-ratelimit-requests-remaining": "95",
            "anthropic-ratelimit-requests-limit": "100",
            "anthropic-ratelimit-tokens-remaining": "450000",
            "anthropic-ratelimit-tokens-limit": "500000",
            "retry-after": "5.0",
            # New: reset timestamps
            "anthropic-ratelimit-requests-reset": "2026-02-24T10:30:00Z",
            "anthropic-ratelimit-tokens-reset": "2026-02-24T10:30:00Z",
            # New: input tokens dimension
            "anthropic-ratelimit-input-tokens-remaining": "800000",
            "anthropic-ratelimit-input-tokens-limit": "1000000",
            "anthropic-ratelimit-input-tokens-reset": "2026-02-24T10:31:00Z",
            # New: output tokens dimension
            "anthropic-ratelimit-output-tokens-remaining": "90000",
            "anthropic-ratelimit-output-tokens-limit": "100000",
            "anthropic-ratelimit-output-tokens-reset": "2026-02-24T10:31:00Z",
        }
        info = provider._extract_rate_limit_headers(headers)

        # All 13 keys should be present
        expected_keys = {
            "requests_remaining",
            "requests_limit",
            "requests_reset",
            "tokens_remaining",
            "tokens_limit",
            "tokens_reset",
            "input_tokens_remaining",
            "input_tokens_limit",
            "input_tokens_reset",
            "output_tokens_remaining",
            "output_tokens_limit",
            "output_tokens_reset",
            "retry_after_seconds",
        }
        assert set(info.keys()) == expected_keys

    def test_partial_headers_only_includes_present_keys(self):
        """Only headers that are actually present should appear in the dict."""
        provider = _make_provider()
        headers = {
            "anthropic-ratelimit-input-tokens-remaining": "800000",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info == {"input_tokens_remaining": 800000}


class TestFastModeInputTokenHeaders:
    """Verify fast-mode input token headers are parsed into fast_input_tokens_* keys."""

    def test_fast_input_tokens_remaining_and_limit(self):
        provider = _make_provider()
        headers = {
            "anthropic-fast-input-tokens-remaining": "500000",
            "anthropic-fast-input-tokens-limit": "1000000",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info["fast_input_tokens_remaining"] == 500000
        assert info["fast_input_tokens_limit"] == 1000000

    def test_fast_input_tokens_reset(self):
        provider = _make_provider()
        headers = {
            "anthropic-fast-input-tokens-reset": "2026-05-29T12:00:00Z",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info["fast_input_tokens_reset"] == "2026-05-29T12:00:00Z"


class TestFastModeOutputTokenHeaders:
    """Verify fast-mode output token headers are parsed into fast_output_tokens_* keys."""

    def test_fast_output_tokens_remaining_and_limit(self):
        provider = _make_provider()
        headers = {
            "anthropic-fast-output-tokens-remaining": "40000",
            "anthropic-fast-output-tokens-limit": "50000",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info["fast_output_tokens_remaining"] == 40000
        assert info["fast_output_tokens_limit"] == 50000

    def test_fast_output_tokens_reset(self):
        provider = _make_provider()
        headers = {
            "anthropic-fast-output-tokens-reset": "2026-05-29T12:00:00Z",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert info["fast_output_tokens_reset"] == "2026-05-29T12:00:00Z"

    def test_fast_headers_absent_when_not_present(self):
        """Fast-mode fields should not appear in dict when headers are absent."""
        provider = _make_provider()
        headers = {
            "anthropic-ratelimit-requests-remaining": "95",
        }
        info = provider._extract_rate_limit_headers(headers)
        assert "fast_input_tokens_remaining" not in info
        assert "fast_input_tokens_limit" not in info
        assert "fast_output_tokens_remaining" not in info
        assert "fast_output_tokens_limit" not in info


class TestHeaderParsingEdgeCases:
    """Edge cases for header value parsing."""

    def test_non_numeric_int_header_ignored(self):
        provider = _make_provider()
        headers = {"anthropic-ratelimit-requests-remaining": "not-a-number"}
        info = provider._extract_rate_limit_headers(headers)
        assert "requests_remaining" not in info

    def test_non_numeric_retry_after_ignored(self):
        provider = _make_provider()
        headers = {"retry-after": "not-a-number"}
        info = provider._extract_rate_limit_headers(headers)
        assert "retry_after_seconds" not in info

    def test_empty_reset_string_ignored(self):
        """Empty string reset values should not be included."""
        provider = _make_provider()
        headers = {"anthropic-ratelimit-tokens-reset": ""}
        info = provider._extract_rate_limit_headers(headers)
        assert "tokens_reset" not in info
