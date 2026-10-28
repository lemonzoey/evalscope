"""Parse Hermes/Qwen tool_call blocks from a local model text generation.

Local models served through ``ModelScopeAPI`` (``eval_type=llm_ckpt``) render
tools into the chat template via ``apply_chat_template`` but decode the
generated text with ``skip_special_tokens=True``. Qwen2.5/Qwen3 register their
tool-call delimiters as ``special=False``, so those delimiters survive decoding
and the assistant turn carries literal tool_call blocks of the form::

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Beijing"}}
    </tool_call>

This module extracts those blocks into structured ``ToolCall`` objects so that
downstream function-calling benchmarks (``general_fc``) and agent strategies
(``function_calling``, ``react``) - which read ``message.tool_calls`` and
``stop_reason == "tool_calls"`` - see a local model tool calls the same way
they see an OpenAI-compatible API. See modelscope/evalscope#1908.

The contract mirrors the OpenAI-API parsing path
(``evalscope.models.utils.openai.chat_tool_calls_from_openai``): tool calls are
built via ``parse_tool_call`` so that malformed arguments surface as a
``ToolCall.parse_error`` rather than raising, and the surrounding prose is
returned separately to use as ``message.content``.
"""

import json
import re
import uuid
from typing import List, Optional, Tuple

from evalscope.api.tool import ToolCall, ToolFunction, ToolInfo, parse_tool_call
from evalscope.utils import get_logger

logger = get_logger()

# Qwen2.5/Qwen3 (Hermes convention) wrap a tool call in these literal tokens.
# They are registered ``special=False`` in the Qwen tokenizer, so they survive
# ``skip_special_tokens=True`` decoding and appear verbatim in the generation.
_TOOL_CALL_START = '<' + 'tool_call' + '>'
_TOOL_CALL_END = '<' + '/tool_call' + '>'
# Match a single block lazily across newlines; capture the JSON payload.
_TOOL_CALL_RE = re.compile(
    re.escape(_TOOL_CALL_START) + r'(.*?)' + re.escape(_TOOL_CALL_END),
    re.DOTALL,
)


def parse_tool_calls_from_text(
    text: str,
    tools: Optional[List[ToolInfo]] = None,
) -> Tuple[Optional[List[ToolCall]], str]:
    """Parse Hermes/Qwen tool_call blocks from a local model text generation.

    Args:
        text: The decoded assistant generation.
        tools: The tools advertised to the model, forwarded to ``parse_tool_call``
            so that non-JSON argument payloads can fall back to the first
            declared parameter (same behaviour as the OpenAI parsing path).

    Returns:
        ``(tool_calls, remaining_text)``:

        - ``tool_calls`` is ``None`` when no tool_call block is present
          (the caller treats the whole text as plain content and does not flip
          ``stop_reason``). When at least one block is present it is a list with
          one ``ToolCall`` per block, including blocks whose JSON failed to
          parse (those carry ``parse_error`` and empty arguments).
        - ``remaining_text`` is the text outside any block, with surrounding
          whitespace trimmed, to use as ``message.content``.
    """
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        return None, text

    tool_calls: List[ToolCall] = []
    # Reconstruct the prose outside the blocks by stitching the gaps between
    # matches. Using per-match spans (rather than str.replace) keeps the prose
    # on both sides of every block and tolerates multiple blocks.
    remaining_parts: List[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        remaining_parts.append(text[cursor : match.start()])
        cursor = match.end()
        payload = match.group(1).strip()
        call_id = f'call_{uuid.uuid4().hex[:8]}'
        tool_calls.append(_build_tool_call(call_id, payload, index, tools or []))
    remaining_parts.append(text[cursor:])
    remaining_text = ''.join(remaining_parts).strip()

    return tool_calls, remaining_text


def _loads_or_empty(arguments: str) -> dict:
    """Best-effort parse of a JSON-string arguments payload; empty dict on failure."""
    arguments = arguments.strip()
    if arguments.startswith('{'):
        try:
            value = json.loads(arguments)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _build_tool_call(
    call_id: str,
    payload: str,
    index: int,
    tools: List[ToolInfo],
) -> ToolCall:
    """Build one ``ToolCall`` from a single block payload.

    ``parse_tool_call`` takes ``arguments`` as a string, so a dict ``arguments``
    (Qwen3 style) is re-serialized while a string ``arguments`` (Qwen2.5 style)
    is passed through. Blindly ``json.dumps``-ing the latter would double-
    encode it and push it into ``parse_tool_call`` yaml fallback, corrupting
    the arguments. A block whose JSON is entirely unparseable still yields a
    ``ToolCall`` carrying ``parse_error`` so the attempt is surfaced to the
    downstream benchmark rather than silently dropped.
    """
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as ex:
        logger.info('Failed to parse tool-call block #%d as JSON: %s', index, ex)
        return parse_tool_call(call_id, '', payload, tools)

    if not isinstance(parsed, dict) or 'name' not in parsed:
        # Not the Hermes shape: a block without a "name" is malformed, so surface
        # it as a parse_error rather than letting parse_tool_call succeed on the
        # (otherwise valid) JSON payload and silently swallow the missing name.
        arguments = parsed.get('arguments', {}) if isinstance(parsed, dict) else {}
        if isinstance(arguments, str):
            arguments = _loads_or_empty(arguments)
        elif not isinstance(arguments, dict):
            arguments = {}
        return ToolCall(
            id=call_id,
            function=ToolFunction(name='', arguments=arguments),
            parse_error=f'tool-call block missing "name" field: {payload}',
        )

    name = str(parsed.get('name', '') or '')
    arguments = parsed.get('arguments', {})
    # arguments may already be a dict (Qwen3) or a JSON string (Qwen2.5).
    args_str = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return parse_tool_call(call_id, name, args_str, tools)
