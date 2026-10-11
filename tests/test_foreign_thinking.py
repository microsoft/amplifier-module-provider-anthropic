"""Cross-provider history projects unsigned summaries without rewriting history."""
import copy

import pytest

from amplifier_module_provider_anthropic import AnthropicProvider


@pytest.mark.parametrize('signature', [None, 12, False])
def test_unsigned_summary_replays_as_text_without_mutating_history(signature):
    provider = AnthropicProvider.__new__(AnthropicProvider)
    messages = [{'role': 'assistant', 'content': [
        {'type': 'thinking', 'thinking': 'Previous model summary', 'signature': signature},
        {'type': 'text', 'text': 'Answer'},
        {'type': 'tool_use', 'id': 'call1', 'name': 'lookup', 'input': {}},
    ]}, {'role': 'tool', 'tool_call_id': 'call1', 'content': 'Found'}]
    original = copy.deepcopy(messages)
    wire = provider._convert_messages(messages)
    assert wire[0]['content'] == [
        {'type': 'text', 'text': 'Previous model summary'},
        {'type': 'text', 'text': 'Answer'},
        {'type': 'tool_use', 'id': 'call1', 'name': 'lookup', 'input': {}},
    ]
    assert wire[1]['content'][0]['tool_use_id'] == 'call1'
    assert messages == original


@pytest.mark.parametrize('signature', ['signed-native', ''])
def test_signed_and_redacted_native_blocks_keep_exact_order_and_bytes(signature):
    provider = AnthropicProvider.__new__(AnthropicProvider)
    blocks = [{'type': 'thinking', 'thinking': 'Native thought', 'signature': signature},
              {'type': 'redacted_thinking', 'data': 'opaque-data'},
              {'type': 'text', 'text': 'Reply'}]
    assert provider._convert_messages([{'role': 'assistant', 'content': blocks}])[0]['content'] == blocks


@pytest.mark.parametrize('content', ['Answer', [{'type': 'text', 'text': 'Answer'}]])
def test_legacy_unsigned_thinking_precedes_answer(content):
    provider = AnthropicProvider.__new__(AnthropicProvider)
    row = {'role': 'assistant', 'content': content, 'thinking_block': {'type': 'thinking', 'thinking': 'Summary'}}
    original = copy.deepcopy(row)
    assert provider._convert_messages([row])[0]['content'] == [
        {'type': 'text', 'text': 'Summary'}, {'type': 'text', 'text': 'Answer'}]
    assert row == original


def test_empty_unsigned_only_turn_does_not_create_an_invalid_empty_message():
    provider = AnthropicProvider.__new__(AnthropicProvider)
    messages = [{'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': '', 'signature': None}]},
                {'role': 'user', 'content': 'Continue'}]
    assert provider._convert_messages(messages) == [messages[0], messages[2]]


def test_canonical_unsigned_summary_does_not_duplicate_legacy_copy():
    provider = AnthropicProvider.__new__(AnthropicProvider)
    thinking = {'type': 'thinking', 'thinking': 'Summary', 'signature': None}
    messages = [{'role': 'assistant', 'content': [thinking, {'type': 'text', 'text': 'Answer'}], 'thinking_block': thinking}]
    assert provider._convert_messages(messages)[0]['content'] == [
        {'type': 'text', 'text': 'Summary'}, {'type': 'text', 'text': 'Answer'}]


def test_empty_legacy_thinking_does_not_move_answer_after_tool_use():
    provider = AnthropicProvider.__new__(AnthropicProvider)
    messages = [{'role': 'assistant', 'content': 'Answer',
        'thinking_block': {'type': 'thinking', 'thinking': '', 'signature': None},
        'tool_calls': [{'id': 'call1', 'name': 'lookup', 'arguments': {}}]}]
    assert [block['type'] for block in provider._convert_messages(messages)[0]['content']] == ['text', 'tool_use']
