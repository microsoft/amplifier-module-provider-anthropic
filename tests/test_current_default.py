"""A default refresh must not silently change an operator's explicit pin."""

from amplifier_module_provider_anthropic import AnthropicProvider


def test_default_is_current_sonnet():
    assert AnthropicProvider(api_key="test").default_model == "claude-sonnet-5-5"


def test_explicit_pin_is_preserved():
    provider = AnthropicProvider(api_key="test", config={"default_model": "claude-sonnet-5"})
    assert provider.default_model == "claude-sonnet-5"