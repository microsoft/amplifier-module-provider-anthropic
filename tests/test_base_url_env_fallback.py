"""Tests for base_url environment variable fallback.

These tests verify backward compatibility when adding ANTHROPIC_BASE_URL
environment variable support for ollama launch integration.

Critical invariant: config["base_url"] MUST take precedence over env var.
"""

from amplifier_module_provider_anthropic import AnthropicProvider


class TestBackwardCompatibility:
    """Tests that prove existing behavior is preserved."""

    def test_config_base_url_used_when_provided(self, monkeypatch):
        """Config base_url must be used - this is existing user behavior."""
        # Ensure env var is NOT set (clean environment)
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        config = {"base_url": "https://my-proxy.example.com"}
        provider = AnthropicProvider("test-key", config)

        assert provider._base_url == "https://my-proxy.example.com"

    def test_config_takes_precedence_over_env_var(self, monkeypatch):
        """Config MUST win when both config and env var are set.

        This is CRITICAL for backward compatibility - users with explicit
        config should not be affected by system-wide env vars.
        """
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://from-env.example.com")

        config = {"base_url": "https://from-config.example.com"}
        provider = AnthropicProvider("test-key", config)

        assert provider._base_url == "https://from-config.example.com"

    def test_none_base_url_when_neither_set(self, monkeypatch):
        """When neither config nor env var set, base_url should be None.

        This preserves default SDK behavior (uses api.anthropic.com).
        """
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        provider = AnthropicProvider("test-key", {})

        assert provider._base_url is None


class TestEnvVarFallback:
    """Tests for new environment variable fallback feature."""

    def test_env_var_used_when_config_missing(self, monkeypatch):
        """Env var should be used when config key is not present."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:11434")

        provider = AnthropicProvider("test-key", {})

        assert provider._base_url == "http://localhost:11434"

    def test_env_var_used_when_config_is_none(self, monkeypatch):
        """Env var should be used when config explicitly sets None."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:11434")

        config = {"base_url": None}
        provider = AnthropicProvider("test-key", config)

        assert provider._base_url == "http://localhost:11434"
