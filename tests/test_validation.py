"""Structural validation tests for anthropic provider.

Inherits authoritative tests from amplifier-core.
"""

from amplifier_core.validation.structural import ProviderStructuralTests
from amplifier_module_provider_anthropic import AnthropicProvider


class TestAnthropicProviderStructural(ProviderStructuralTests):
    """Run standard provider structural tests for anthropic.

    All tests from ProviderStructuralTests run automatically.
    Add module-specific structural tests below if needed.
    """


class TestBaseUrlConfigField:
    """Tests for base_url ConfigField declaration."""

    def test_base_url_config_field_declared(self):
        """Test that base_url ConfigField is properly declared in get_info()."""
        provider = AnthropicProvider("test-api-key", {})
        info = provider.get_info()

        # Find the base_url config field
        base_url_field = next(
            (f for f in info.config_fields if f.id == "base_url"),
            None,
        )

        assert base_url_field is not None, "base_url ConfigField should be declared"
        assert base_url_field.display_name == "API Base URL"
        assert base_url_field.field_type == "text"
        assert base_url_field.required is False

    def test_base_url_config_field_has_env_var(self):
        """Test that base_url ConfigField declares ANTHROPIC_BASE_URL env var."""
        provider = AnthropicProvider("test-api-key", {})
        info = provider.get_info()

        base_url_field = next(
            (f for f in info.config_fields if f.id == "base_url"),
            None,
        )

        assert base_url_field is not None
        assert base_url_field.env_var == "ANTHROPIC_BASE_URL"

    def test_base_url_config_field_has_default(self):
        """Test that base_url ConfigField has default value."""
        provider = AnthropicProvider("test-api-key", {})
        info = provider.get_info()

        base_url_field = next(
            (f for f in info.config_fields if f.id == "base_url"),
            None,
        )

        assert base_url_field is not None
        assert base_url_field.default == "https://api.anthropic.com"


class TestFallbackConfigFields:
    """Tests for overload fallback ConfigField declarations."""

    def test_fallback_toggle_is_declared(self):
        provider = AnthropicProvider("test-api-key", {})
        info = provider.get_info()

        fallback_field = next(
            (f for f in info.config_fields if f.id == "fallback_on_overload"),
            None,
        )

        assert fallback_field is not None
        assert fallback_field.display_name == "Temporary Overload Downgrade"
        assert fallback_field.field_type == "boolean"
        assert fallback_field.default == "false"
        assert fallback_field.requires_model is True

    def test_fallback_models_have_expected_defaults(self):
        provider = AnthropicProvider("test-api-key", {})
        info = provider.get_info()

        sonnet_field = next(
            (f for f in info.config_fields if f.id == "fallback_sonnet_model"),
            None,
        )
        haiku_field = next(
            (f for f in info.config_fields if f.id == "fallback_haiku_model"),
            None,
        )

        assert sonnet_field is not None
        assert sonnet_field.default == "claude-sonnet-4-6"

        assert haiku_field is not None
        assert haiku_field.default == "claude-haiku-4-5"

    def test_persist_fallback_state_toggle_is_declared(self):
        provider = AnthropicProvider("test-api-key", {})
        info = provider.get_info()

        persist_field = next(
            (f for f in info.config_fields if f.id == "persist_fallback_state"),
            None,
        )

        assert persist_field is not None
        assert persist_field.display_name == "Share Downgrade State"
        assert persist_field.field_type == "boolean"
        assert persist_field.default == "false"
        assert persist_field.requires_model is True
