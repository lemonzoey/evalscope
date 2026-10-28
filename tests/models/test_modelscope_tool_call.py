"""Integration tests for ModelScopeAPI.generate tool-call parsing (#1908).

These import evalscope.models.modelscope (which needs torch), so they are
skipped when torch is unavailable. The pure-function parser is covered by
test_tool_call_text.py. Delimiters via chr() to avoid literal tags in source.
"""

import json
from typing import Any, List

import pytest

torch = pytest.importorskip('torch')

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig
from evalscope.api.tool import ToolInfo, ToolParams
from evalscope.models import modelscope as modelscope_module
from evalscope.models.modelscope import GenerateOutput, ModelScopeAPI

S = chr(60) + 'tool_call' + chr(62)
E = chr(60) + '/' + 'tool_call' + chr(62)


def _tool(name: str = 'get_weather') -> ToolInfo:
    return ToolInfo(name=name, description='desc', parameters=ToolParams())


def _resp(text: str, stop_reason: str = 'stop') -> GenerateOutput:
    return GenerateOutput(
        output=text,
        input_tokens=5,
        output_tokens=3,
        total_tokens=8,
        logprobs=None,
        time=0.1,
        stop_reason=stop_reason,
    )


class _FakeTokenizer:
    chat_template = None

    def __call__(self, *a, **k):
        raise AssertionError('tokenizer must not be called when batched_generate is mocked')

    def batch_decode(self, *a, **k):
        raise AssertionError('batch_decode must not be called when batched_generate is mocked')


class _FakeModel:
    device = 'cpu'

    def generate(self, *a, **k):
        raise AssertionError('model.generate must not be called when batched_generate is mocked')


def _make_api() -> ModelScopeAPI:
    api = object.__new__(ModelScopeAPI)
    api.model_name = 'test-model'
    api.tokenizer = _FakeTokenizer()
    api.model = _FakeModel()
    api.chat_template = None
    api.tokenizer_call_args = {}
    api.enable_thinking = None
    return api


def _generate(api, responses, tools, monkeypatch):
    monkeypatch.setattr(modelscope_module, 'batched_generate', lambda _input: responses)
    return api.generate(
        input=[ChatMessageUser(content='hi')],
        tools=tools,
        tool_choice='none' if not tools else 'auto',
        config=GenerateConfig(),
    )


def test_generate_parses_tool_call_and_sets_stop_reason(monkeypatch: Any):
    payload = json.dumps({'name': 'get_weather', 'arguments': {'city': 'Beijing'}})
    text = S + chr(10) + payload + chr(10) + E
    output = _generate(_make_api(), [_resp(text)], [_tool()], monkeypatch)
    assert output.message.tool_calls is not None
    assert output.message.tool_calls[0].function.name == 'get_weather'
    assert output.message.tool_calls[0].function.arguments == {'city': 'Beijing'}
    assert output.stop_reason == 'tool_calls'


def test_generate_no_block_keeps_content_and_stop_reason(monkeypatch: Any):
    text = 'today is sunny'
    output = _generate(_make_api(), [_resp(text, 'stop')], [_tool()], monkeypatch)
    assert output.message.tool_calls is None
    assert output.message.content == 'today is sunny'
    assert output.stop_reason == 'stop'


def test_generate_empty_tools_does_not_parse(monkeypatch: Any):
    payload = json.dumps({'name': 'get_weather', 'arguments': {}})
    text = S + chr(10) + payload + chr(10) + E
    output = _generate(_make_api(), [_resp(text)], [], monkeypatch)
    assert output.message.tool_calls is None
    assert output.message.content == text
    assert output.stop_reason == 'stop'


def test_generate_malformed_block_still_counts_as_tool_call(monkeypatch: Any):
    text = S + chr(10) + '{bad json}' + chr(10) + E
    output = _generate(_make_api(), [_resp(text)], [_tool()], monkeypatch)
    assert output.stop_reason == 'tool_calls'
    assert output.message.tool_calls is not None
    assert output.message.tool_calls[0].parse_error is not None
