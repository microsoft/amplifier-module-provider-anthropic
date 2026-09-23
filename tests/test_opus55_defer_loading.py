"""computer_toolset_20260801: defer_loading must be valid per-member config.

The toolset schema (anthropic-sdk-python `ComputerToolsetConfigsParam` /
`Computer<Member>ConfigParam`, 2026-09-22) has no top-level or blanket
`defer_loading` field, and no `_defer_loading` key -- `configs` accepts only
the documented member names (screenshot, left_click, ..., zoom), each with
exactly two optional fields: `enabled` and `defer_loading`. A legacy
`computer_20251124` tool's whole-tool `defer_loading: true` must therefore be
translated into `defer_loading: true` on every enabled member's own config
entry, never onto a fictitious `configs["_defer_loading"]` or
`translated["defer_loading"]` key -- both are rejected outright ("Extra
inputs are not permitted") by the real schema.
"""

from __future__ import annotations

from amplifier_module_provider_anthropic import _computer_toolset


class TestDeferLoadingWireShape:
    def test_defer_loading_applied_to_every_enabled_member(self):
        tools = [
            {
                "type": "computer_20251124",
                "name": "computer",
                "display_width_px": 1024,
                "display_height_px": 768,
                "defer_loading": True,
            }
        ]
        wire, _ = _computer_toolset.translate_tools(
            tools, _computer_toolset.TOOLSET_TYPE
        )
        assert len(wire) == 1
        configs = wire[0]["configs"]

        # No invalid keys anywhere in the emitted configs.
        assert "_defer_loading" not in configs
        assert "defer_loading" not in wire[0]  # never a top-level key either

        # Zoom defaults to disabled (legacy tool had no zoom capability) and
        # is skipped for defer_loading -- a disabled member carries no
        # defer_loading value, only `enabled: False`.
        assert configs["zoom"] == {"enabled": False}

        # Every OTHER documented member gets defer_loading: true.
        expected_members = _computer_toolset.TOOLSET_MEMBERS - {"zoom"}
        assert set(configs.keys()) == _computer_toolset.TOOLSET_MEMBERS
        for member in expected_members:
            assert configs[member] == {"defer_loading": True}, member

    def test_defer_loading_with_zoom_enabled_includes_zoom(self):
        tools = [
            {
                "type": "computer_20251124",
                "name": "computer",
                "enable_zoom": True,
                "defer_loading": True,
            }
        ]
        wire, _ = _computer_toolset.translate_tools(
            tools, _computer_toolset.TOOLSET_TYPE
        )
        configs = wire[0]["configs"]
        assert set(configs.keys()) == _computer_toolset.TOOLSET_MEMBERS
        for member in _computer_toolset.TOOLSET_MEMBERS:
            assert configs[member] == {"defer_loading": True}, member

    def test_no_defer_loading_requested_only_zoom_config_present(self):
        """Baseline (no defer_loading): only the zoom-disable override is
        emitted, exactly as before this fix -- no member configs at all."""
        tools = [{"type": "computer_20251124", "name": "computer"}]
        wire, _ = _computer_toolset.translate_tools(
            tools, _computer_toolset.TOOLSET_TYPE
        )
        assert wire == [
            {
                "type": "computer_toolset_20260801",
                "configs": {"zoom": {"enabled": False}},
            }
        ]

    def test_every_emitted_config_key_is_a_documented_member(self):
        tools = [
            {"type": "computer_20251124", "name": "computer", "defer_loading": True}
        ]
        wire, _ = _computer_toolset.translate_tools(
            tools, _computer_toolset.TOOLSET_TYPE
        )
        configs = wire[0]["configs"]
        assert set(configs.keys()) <= _computer_toolset.TOOLSET_MEMBERS
        for member_config in configs.values():
            assert set(member_config.keys()) <= {"enabled", "defer_loading"}


class TestToolsetMemberSetDriftGuard:
    """If practical under the locked SDK pin: compare our documented member
    set against the real SDK's typed schema, so an SDK bump that adds or
    removes a member is caught immediately instead of silently emitting an
    invalid (or incomplete) configs dict."""

    def test_toolset_members_match_installed_sdk_typed_schema(self):
        try:
            from anthropic.types.computer_toolset_configs_param import (
                ComputerToolsetConfigsParam,
            )
        except ImportError:
            import pytest

            pytest.skip(
                "installed anthropic SDK has no typed "
                "ComputerToolsetConfigsParam -- nothing to drift-check"
            )
        sdk_members = set(ComputerToolsetConfigsParam.__annotations__.keys())
        assert _computer_toolset.TOOLSET_MEMBERS == sdk_members

    def test_each_member_config_param_has_exactly_enabled_and_defer_loading(self):
        try:
            import anthropic.types as sdk_types
        except ImportError:
            import pytest

            pytest.skip("installed anthropic SDK has no typed member config params")

        checked = 0
        for member in _computer_toolset.TOOLSET_MEMBERS:
            type_name = (
                "Computer"
                + "".join(part.capitalize() for part in member.split("_"))
                + "ConfigParam"
            )
            param_cls = getattr(sdk_types, type_name, None)
            if param_cls is None:
                continue
            checked += 1
            assert set(param_cls.__annotations__.keys()) == {
                "enabled",
                "defer_loading",
            }, member
        assert checked > 0, "no member config param types found to drift-check"
