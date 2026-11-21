"""Map a tau-bench episode's flat chat trajectory into the normalized trace model.

tau-bench (sierra-research/tau-bench, MIT licensed) ships each episode as
``{"task_id", "reward", "info": {...}, "traj": [...]}``. ``traj`` is a flat
OpenAI-chat-format message list: ``role`` in ``system``/``user``/
``assistant``/``tool``, an assistant message either carrying ``tool_calls``
(``[{"id", "function": {"name", "arguments"}}]``) or plain text ``content``,
and a ``tool`` message pairing back to its call via ``tool_call_id``.

Unlike Langfuse/OTel/Patronus, there is no explicit span tree here: one
flat message list for one single support agent. This parser reconstructs
the shape the rest of the toolkit expects: one top-level ``"agent"`` span
(its ``input`` is every user-turn text joined, its ``output`` is the final
text-only assistant reply) plus one ``"tool"`` span per tool call and one
``"llm"`` span per text-only assistant reply, both parented to the agent
span. The agent span is returned **last**, not first: ``trajectory_optimality``
reads span order as step order (see ``format_trajectory``), and the agent's
own summary line is not a step that happened before the tool calls and
replies that produced it.

A harness-dependent serialization quirk, live-confirmed across tau-bench's
own shipped ``historical_trajectories``: GPT-4o's ``tool_calls[].function.
arguments`` is a JSON *string*; Sonnet's is already a native *dict*. Both
are handled explicitly below (``_args_text``): no blind ``json.loads``,
no silent ``except Exception`` swallowing a real ``TypeError`` (that broad
except masked this exact bug once already, in the scratch pilot this
parser was promoted from).
"""

from __future__ import annotations

import json
from typing import Any

from aixeval.ingest.trace import Span, text_or_none, validate_spans


AGENT_SPAN_ID = "agent"
DEFAULT_AGENT_NAME = "agent"


def parse_taubench_trace(payload: dict[str, Any], *, agent_name: str = DEFAULT_AGENT_NAME) -> list[Span]:
    """Return the normalized spans for one tau-bench episode, validated.

    Parameters
    ----------
    payload
        A tau-bench episode dict (or just its ``{"traj": [...]}`` shape,
        callers typically pass ``episode`` itself, since ``task_id``/
        ``reward``/``info`` are ignored here; declared ground truth like
        ``gold_tool_calls`` belongs in ``ingest_any``'s ``declared_fields``,
        not auto-extracted by this parser).
    agent_name
        tau-bench's own episodes don't declare a name for the single
        support agent; defaulted rather than guessed from the system
        prompt's free text.
    """
    traj = payload.get("traj")
    if not isinstance(traj, list):
        msg = "tau-bench payload has no traj list"
        raise ValueError(msg)
    tool_results = _tool_results_by_call_id(traj)
    spans: list[Span] = []
    user_texts: list[str] = []
    last_reply: str | None = None
    for index, message in enumerate(traj):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            text = text_or_none(message.get("content"))
            if text:
                user_texts.append(text)
            continue
        if role != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            spans.extend(_tool_spans(index, tool_calls, tool_results))
            continue
        text = text_or_none(message.get("content"))
        if not text:
            continue
        spans.append(Span(id=f"llm:{index}", kind="llm", parent_id=AGENT_SPAN_ID, output=text))
        last_reply = text
    spans.append(
        Span(
            id=AGENT_SPAN_ID,
            kind="agent",
            agent=agent_name,
            input="\n\n".join(user_texts) or None,
            output=last_reply,
        )
    )
    validate_spans(spans)
    return spans


def _tool_spans(message_index: int, tool_calls: list[Any], tool_results: dict[str, str]) -> list[Span]:
    spans: list[Span] = []
    for call_index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        call_id = call.get("id")
        output = tool_results.get(call_id) if isinstance(call_id, str) else None
        spans.append(
            Span(
                id=f"tool:{message_index}:{call_index}",
                kind="tool",
                parent_id=AGENT_SPAN_ID,
                source_type=name,
                input=_args_text(function.get("arguments")),
                output=output,
            )
        )
    return spans


def _args_text(arguments: object) -> str | None:
    """Render a tool call's arguments as text, handling both shipped shapes.

    GPT-4o's harness serializes ``arguments`` as a JSON string already;
    Sonnet's serializes it as a native dict. Each is handled by name --
    no blind ``json.loads`` on a value that might already be text, and
    no broad ``except`` that would mask a real type mismatch as "no args".
    """
    if isinstance(arguments, str):
        return arguments or None
    if isinstance(arguments, dict):
        return json.dumps(arguments, sort_keys=True, default=str)
    return None


def _tool_results_by_call_id(traj: list[Any]) -> dict[str, str]:
    results: dict[str, str] = {}
    for message in traj:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        content = message.get("content")
        if isinstance(content, str):
            results[call_id] = content
        elif content is not None:
            results[call_id] = json.dumps(content, default=str)
    return results


def extract_tool_calls_taubench(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Pull the actual tool calls a tau-bench episode made, as structured data.

    Mirrors ``extract_tool_calls_langfuse``: walks the raw payload (not the
    parsed ``Span`` list) and reshapes to ``{"tool": <name>, "args": <dict>}``,
    ready to compare against a declared ``gold_tool_calls`` list with no LLM
    involved, via ``run_tool_call_accuracy_check``.

    Returns
    -------
    list[dict] | None
        One entry per tool call, in order, or ``None`` if there were none.
    """
    traj = payload.get("traj")
    if not isinstance(traj, list):
        return None
    calls: list[dict[str, Any]] = []
    for message in traj:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            calls.append({"tool": name, "args": _args_dict(function.get("arguments"))})
    return calls or None


def _args_dict(arguments: object) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def print_taubench_example() -> None:
    """Print the spans parsed from one small, hand-built tau-bench episode."""
    episode = {
        "traj": [
            {"role": "system", "content": "You are a retail support agent."},
            {"role": "user", "content": "Please cancel order #100 and refund to gift card."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "cancel_pending_order", "arguments": '{"order_id": "#100"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "cancelled"}'},
            {"role": "assistant", "content": "Order #100 has been cancelled and refunded to your gift card."},
        ]
    }
    for span in parse_taubench_trace(episode):
        print(f"  {span.id}: kind={span.kind} source_type={span.source_type} input={span.input} output={span.output}")
    print(f"  tool calls: {extract_tool_calls_taubench(episode)}")


if __name__ == "__main__":
    print_taubench_example()
