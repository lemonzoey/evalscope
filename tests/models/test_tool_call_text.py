"""Unit tests for parse_tool_calls_from_text (modelscope/evalscope#1908).

Pure-function tests with no torch/modelscope dependency, so they run in a
minimal environment. The Qwen/Hermes tool-call delimiters are built via chr()
so this file source does not contain the literal tags.
"""

import json

from evalscope.api.tool import ToolInfo, ToolParams
from evalscope.models.utils.tool_call_text import parse_tool_calls_from_text

S = chr(60) + 'tool_call' + chr(62)
E = chr(60) + '/' + 'tool_call' + chr(62)


def _tool(name: str = 'get_weather') -> ToolInfo:
    return ToolInfo(name=name, description='desc', parameters=ToolParams())


def test_no_block_returns_none_and_original_text():
    calls, remaining = parse_tool_calls_from_text('today is sunny', [_tool()])
    assert calls is None
    assert remaining == 'today is sunny'


def test_single_block_parses_name_and_arguments():
    text = S + chr(10) + json.dumps({'name': 'get_weather', 'arguments': {'city': 'Beijing'}}) + chr(10) + E
    calls, remaining = parse_tool_calls_from_text(text, [_tool()])
    assert calls is not None and len(calls) == 1
    assert calls[0].function.name == 'get_weather'
    assert calls[0].function.arguments == {'city': 'Beijing'}
    assert calls[0].parse_error is None
    assert remaining == ''


def test_multiple_blocks_preserve_order():
    def block(name):
        return S + chr(10) + json.dumps({'name': name, 'arguments': {}}) + chr(10) + E
    text = block('a') + chr(10) + block('b')
    calls, remaining = parse_tool_calls_from_text(text, [_tool()])
    assert [c.function.name for c in calls] == ['a', 'b']
    assert remaining == ''


def test_block_with_surrounding_prose_kept_in_remaining():
    payload = json.dumps({'name': 'get_weather', 'arguments': {'city': 'Boston'}})
    text = 'Let me check.' + chr(10) + S + chr(10) + payload + chr(10) + E + chr(10) + 'Done.'
    calls, remaining = parse_tool_calls_from_text(text, [_tool()])
    assert calls is not None and calls[0].function.name == 'get_weather'
    assert 'Let me check.' in remaining
    assert 'Done.' in remaining


def test_string_form_arguments_not_double_encoded():
    # Qwen2.5 style: arguments is a JSON string, not a dict.
    inner = json.dumps({'city': 'Boston'})
    payload = json.dumps({'name': 'get_weather', 'arguments': inner})
    text = S + chr(10) + payload + chr(10) + E
    calls, _ = parse_tool_calls_from_text(text, [_tool()])
    assert calls[0].function.arguments == {'city': 'Boston'}
    assert calls[0].parse_error is None


def test_dict_form_arguments_qwen3():
    # Qwen3 style: arguments is a nested JSON object.
    payload = json.dumps({'name': 'get_weather', 'arguments': {'city': 'Beijing', 'unit': 'c'}})
    text = S + chr(10) + payload + chr(10) + E
    calls, _ = parse_tool_calls_from_text(text, [_tool()])
    assert calls[0].function.arguments == {'city': 'Beijing', 'unit': 'c'}
    assert calls[0].parse_error is None


def test_malformed_json_yields_parse_error_tool_call():
    text = S + chr(10) + '{name: bad json}' + chr(10) + E
    calls, _ = parse_tool_calls_from_text(text, [_tool()])
    assert calls is not None and len(calls) == 1
    assert calls[0].parse_error is not None
    assert calls[0].function.arguments == {}


def test_missing_name_yields_parse_error():
    payload = json.dumps({'arguments': {'city': 'Boston'}})
    text = S + chr(10) + payload + chr(10) + E
    calls, _ = parse_tool_calls_from_text(text, [_tool()])
    assert calls is not None and len(calls) == 1
    assert calls[0].parse_error is not None


def test_call_id_is_unique_per_block():
    def block(name):
        return S + chr(10) + json.dumps({'name': name, 'arguments': {}}) + chr(10) + E
    text = block('a') + chr(10) + block('b')
    calls, _ = parse_tool_calls_from_text(text, [_tool()])
    ids = [c.id for c in calls]
    assert len(ids) == len(set(ids)), 'call ids must be unique'
