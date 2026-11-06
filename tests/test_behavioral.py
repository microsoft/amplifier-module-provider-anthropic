"""Behavioral tests for anthropic provider.

Inherits authoritative tests from amplifier-core.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from amplifier_core.validation.behavioral import ProviderBehaviorTests
from tests._helpers import models_page

class TestAnthropicProviderBehavior(ProviderBehaviorTests):
    """Run standard provider behavioral tests for anthropic.

    All tests from ProviderBehaviorTests run automatically.
    Add module-specific tests below if needed.
    """

    @pytest.mark.asyncio
    async def test_list_models_returns_list(self, provider_module):
        """Exercise actual family mapping with an offline SDK page."""
        provider_module.client.models.with_raw_response.list = AsyncMock(return_value=models_page(
            [SimpleNamespace(id="claude-sonnet-5-5", display_name="Sonnet", created_at="")]
        ))
        await super().test_list_models_returns_list(provider_module)
