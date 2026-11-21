"""tau-bench's flat chat traj -> normalized spans, plus the deterministic extractor.

Hand-written payloads matching tau-bench's real shape (sierra-research/tau-bench):
a flat OpenAI-chat-format ``traj`` list, with the two real harness-dependent
``tool_calls[].function.arguments`` shapes (GPT-4o's JSON string, Sonnet's
native dict) both covered, since a pilot script once confused them.
"""

import pytest

from aixeval.ingest.trace import collect_trace_input_texts, detect_and_parse, format_trajectory
from aixeval.ingest.trace_taubench import extract_tool_calls_taubench, parse_taubench_trace


def _episode(arguments: object) -> dict:
    return {
        "traj": [
            {"role": "system", "content": "You are a retail support agent."},
            {"role": "user", "content": "Please cancel order #100 and refund to gift card."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "cancel_pending_order", "arguments": arguments}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "cancelled"}'},
            {"role": "assistant", "content": "Order #100 has been cancelled and refunded to your gift card."},
        ]
    }


def test_no_traj_list_raises() -> None:
    with pytest.raises(ValueError, match="no traj list"):
        parse_taubench_trace({"not_traj": []})


def test_agent_span_is_last_and_carries_user_input_and_final_reply() -> None:
    spans = parse_taubench_trace(_episode('{"order_id": "#100"}'))
    assert spans[-1].kind == "agent"
    assert spans[-1].input == "Please cancel order #100 and refund to gift card."
    assert spans[-1].output == "Order #100 has been cancelled and refunded to your gift card."
    assert spans[-1].parent_id is None


def test_tool_call_becomes_a_tool_span_parented_to_the_agent() -> None:
    spans = {s.id: s for s in parse_taubench_trace(_episode('{"order_id": "#100"}'))}
    tool_span = spans["tool:2:0"]
    assert tool_span.kind == "tool"
    assert tool_span.parent_id == "agent"
    assert tool_span.source_type == "cancel_pending_order"
    assert tool_span.output == '{"status": "cancelled"}'


def test_string_arguments_gpt4o_style_kept_as_is() -> None:
    spans = {s.id: s for s in parse_taubench_trace(_episode('{"order_id": "#100"}'))}
    assert spans["tool:2:0"].input == '{"order_id": "#100"}'


def test_dict_arguments_sonnet_style_rendered_to_text() -> None:
    spans = {s.id: s for s in parse_taubench_trace(_episode({"order_id": "#100"}))}
    assert spans["tool:2:0"].input == '{"order_id": "#100"}'


def test_text_only_assistant_reply_without_tool_calls_becomes_an_llm_span() -> None:
    episode = {
        "traj": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello, how can I help?"},
        ]
    }
    spans = parse_taubench_trace(episode)
    llm_spans = [s for s in spans if s.kind == "llm"]
    assert len(llm_spans) == 1
    assert llm_spans[0].output == "Hello, how can I help?"
    assert llm_spans[0].parent_id == "agent"


def test_multiple_user_turns_are_joined_for_the_agent_spans_input() -> None:
    episode = {
        "traj": [
            {"role": "user", "content": "I want to cancel my order."},
            {"role": "assistant", "content": "Which order?"},
            {"role": "user", "content": "Order #100."},
            {"role": "assistant", "content": "Done."},
        ]
    }
    spans = parse_taubench_trace(episode)
    agent_span = next(s for s in spans if s.kind == "agent")
    assert agent_span.input == "I want to cancel my order.\n\nOrder #100."


def test_detect_and_parse_recognizes_the_traj_key() -> None:
    spans = detect_and_parse(_episode('{"order_id": "#100"}'))
    assert any(s.kind == "agent" for s in spans)


def test_format_trajectory_shows_the_tool_name_as_the_step_label() -> None:
    spans = parse_taubench_trace(_episode('{"order_id": "#100"}'))
    trajectory = format_trajectory(spans)
    assert trajectory is not None
    assert "[cancel_pending_order]" in trajectory


def test_collect_trace_input_texts_includes_the_user_turn() -> None:
    spans = parse_taubench_trace(_episode('{"order_id": "#100"}'))
    assert "Please cancel order #100 and refund to gift card." in collect_trace_input_texts(spans)


def test_extract_tool_calls_taubench_reshapes_string_arguments() -> None:
    calls = extract_tool_calls_taubench(_episode('{"order_id": "#100"}'))
    assert calls == [{"tool": "cancel_pending_order", "args": {"order_id": "#100"}}]


def test_extract_tool_calls_taubench_reshapes_dict_arguments() -> None:
    calls = extract_tool_calls_taubench(_episode({"order_id": "#100"}))
    assert calls == [{"tool": "cancel_pending_order", "args": {"order_id": "#100"}}]


def test_extract_tool_calls_taubench_returns_none_with_no_tool_calls() -> None:
    episode = {"traj": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
    assert extract_tool_calls_taubench(episode) is None


def test_extract_tool_calls_taubench_returns_none_without_a_traj_list() -> None:
    assert extract_tool_calls_taubench({"not_traj": []}) is None


def test_agent_name_defaults_but_can_be_overridden() -> None:
    episode = {"traj": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
    assert parse_taubench_trace(episode)[-1].agent == "agent"
    assert parse_taubench_trace(episode, agent_name="retail_agent")[-1].agent == "retail_agent"
