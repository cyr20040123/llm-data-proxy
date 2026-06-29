"""Unit tests for SessionManager."""

import json
import os
import tempfile

import pytest

from llmdataproxy.session import (
    SessionManager,
    _canonical,
    _normalize_msg_arguments,
    _normalize_tool_calls,
    _now_iso,
)


class TestNormalizeToolCalls:
    def test_empty_tool_calls(self):
        assert _normalize_tool_calls(None) is None
        assert _normalize_tool_calls([]) is None
        # [{}] is processed — returns normalized empty tool call, not None
        result = _normalize_tool_calls([{}])
        assert result is not None
        assert len(result) == 1

    def test_parse_arguments_json(self):
        tc = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
        }]
        result = _normalize_tool_calls(tc)
        assert result[0]["function"]["arguments"] == {"city": "Beijing"}

    def test_already_parsed_arguments(self):
        tc = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Beijing"}},
        }]
        result = _normalize_tool_calls(tc)
        assert result[0]["function"]["arguments"] == {"city": "Beijing"}

    def test_invalid_json_passthrough(self):
        tc = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "not json"},
        }]
        result = _normalize_tool_calls(tc)
        assert result[0]["function"]["arguments"] == "not json"


class TestNormalizeMsgArguments:
    def test_mutates_tool_call_arguments(self):
        msg = {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": '{"a": 1}'}}
        ]}
        _normalize_msg_arguments(msg)
        assert msg["tool_calls"][0]["function"]["arguments"] == {"a": 1}

    def test_no_tool_calls(self):
        msg = {"role": "user", "content": "hello"}
        result = _normalize_msg_arguments(msg)
        assert result is msg
        assert result["content"] == "hello"

    def test_invalid_json_tool_args_passthrough(self):
        msg = {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": "bad json"}}
        ]}
        _normalize_msg_arguments(msg)
        assert msg["tool_calls"][0]["function"]["arguments"] == "bad json"


class TestCanonical:
    def test_strips_timestamp(self):
        msg = {"role": "user", "content": "hi", "timestamp": "2026-01-01T00:00:00Z"}
        c = _canonical(msg)
        assert "timestamp" not in json.loads(c)

    def test_strips_non_match_fields(self):
        msg = {"role": "user", "content": "hi", "reasoning": "think", "refusal": None}
        c = _canonical(msg)
        d = json.loads(c)
        assert "reasoning" not in d
        assert "refusal" not in d

    def test_none_content_equals_empty_string(self):
        msg_none = {"role": "user", "content": None}
        msg_empty = {"role": "user", "content": ""}
        assert _canonical(msg_none) == _canonical(msg_empty)

    def test_equivalent_messages_match(self):
        a = {"role": "assistant", "content": "hi"}
        b = {"role": "assistant", "content": "hi", "timestamp": "2026-01-01T00:00:00Z"}
        assert _canonical(a) == _canonical(b)

    def test_sort_keys_stable(self):
        a = {"content": "hi", "role": "user"}
        b = {"role": "user", "content": "hi"}
        assert _canonical(a) == _canonical(b)


class TestSessionManager:
    @pytest.fixture
    def mgr(self):
        d = tempfile.mkdtemp()
        mgr = SessionManager(d, "test_sess", "multi", session_path=d)
        yield mgr
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    def test_disabled_when_mode_none(self):
        mgr = SessionManager("/tmp", "s", "none")
        assert not mgr.enabled
        assert mgr.find_matching_session([]) == (None, 0)

    def test_create_session(self, mgr):
        msgs = [{"role": "user", "content": "hello"}]
        sess = mgr.create_session(msgs, "2026-01-01T00:00:00Z")
        assert len(sess["messages"]) == 1
        assert sess["messages"][0]["timestamp"] == "2026-01-01T00:00:00Z"
        assert len(mgr.sessions) == 1

    def test_create_session_with_tools(self, mgr):
        msgs = [{"role": "user", "content": "weather?"}]
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        sess = mgr.create_session(msgs, "ts", tools=tools)
        assert len(sess["tools"]) == 1
        assert sess["tools"][0]["function"]["name"] == "get_weather"

    def test_create_session_single_mode(self):
        mgr = SessionManager("/tmp", "s", "single")
        msgs = [{"role": "user", "content": "hello"}]
        sess = mgr.create_session(msgs, "2026-01-01T00:00:00Z")
        assert sess["messages"][0]["timestamp"] == ""

    def test_find_matching_session_exact_prefix(self, mgr):
        msgs1 = [{"role": "user", "content": "Q1"}]
        mgr.create_session(msgs1, "ts")

        sess, match_len = mgr.find_matching_session(msgs1)
        assert sess is not None
        assert match_len == 1

    def test_find_matching_session_system_mismatch(self, mgr):
        mgr.create_session(
            [{"role": "system", "content": "sysA"},
             {"role": "user", "content": "Q1"}],
            "ts",
        )
        sess, match_len = mgr.find_matching_session(
            [{"role": "system", "content": "sysB"},
             {"role": "user", "content": "Q1"}],
        )
        assert sess is not None
        assert match_len == 2  # system mismatch tolerated

    def test_find_matching_session_partial_match(self, mgr):
        mgr.create_session(
            [{"role": "user", "content": "Q1"},
             {"role": "assistant", "content": "A1"}],
            "ts",
        )
        sess, match_len = mgr.find_matching_session(
            [{"role": "user", "content": "Q1"},
             {"role": "assistant", "content": "A1"},
             {"role": "user", "content": "Q2"}],
        )
        assert sess is not None
        assert match_len == 2

    def test_find_matching_session_no_match(self, mgr):
        mgr.create_session(
            [{"role": "user", "content": "different"}],
            "ts",
        )
        sess, match_len = mgr.find_matching_session(
            [{"role": "user", "content": "hello"}],
        )
        assert sess is None
        assert match_len == 0

    def test_find_matching_session_not_in_multi_mode(self):
        mgr = SessionManager("/tmp", "s", "single")
        mgr.create_session([{"role": "user", "content": "hi"}], "ts")
        sess, match_len = mgr.find_matching_session(
            [{"role": "user", "content": "hi"}]
        )
        assert sess is None

    def test_append_request_messages(self, mgr):
        sess = mgr.create_session(
            [{"role": "user", "content": "Q1"}], "ts1"
        )
        mgr.append_request_messages(
            sess,
            [{"role": "user", "content": "Q1"},
             {"role": "assistant", "content": "A1"}],
            match_len=1,
            timestamp="ts2",
        )
        assert len(sess["messages"]) == 2
        assert sess["messages"][1]["content"] == "A1"
        assert sess["messages"][1]["timestamp"] == "ts2"

    def test_append_request_messages_with_tools(self, mgr):
        sess = mgr.create_session(
            [{"role": "user", "content": "Q1"}], "ts"
        )
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        mgr.append_request_messages(
            sess,
            [{"role": "user", "content": "Q1"},
             {"role": "user", "content": "weather?"}],
            match_len=1,
            timestamp="ts2",
            tools=tools,
        )
        assert sess["tools"][0]["function"]["name"] == "get_weather"

    def test_append_response(self, mgr):
        sess = mgr.create_session(
            [{"role": "user", "content": "Q1"}], "ts1"
        )
        mgr.append_response(
            sess, {"role": "assistant", "content": "A1"}, "ts2"
        )
        assert len(sess["messages"]) == 2
        assert sess["messages"][1]["role"] == "assistant"
        assert sess["messages"][1]["timestamp"] == "ts2"

    def test_dump_multi_mode(self, mgr):
        sess = mgr.create_session(
            [{"role": "user", "content": "Q1"}], "2026-01-01T00:00:00Z"
        )
        mgr.append_response(
            sess, {"role": "assistant", "content": "A1"}, "2026-01-01T00:00:01Z"
        )
        output_dir = mgr.dump_all()
        files = [f for f in os.listdir(output_dir) if f.endswith(".chatml.json")]
        assert len(files) == 1
        with open(os.path.join(output_dir, files[0])) as f:
            data = json.load(f)
        assert "messages" in data
        assert "remarks" in data
        assert data["remarks"]["incomplete"] is False
        assert len(data["messages"]) == 2

    def test_dump_single_mode(self):
        d = tempfile.mkdtemp()
        try:
            mgr = SessionManager(d, "test_sess", "single", session_path=d)
            mgr.create_session(
                [{"role": "user", "content": "Q1"}], "ts"
            )
            mgr.append_response(
                mgr.sessions[0],
                {"role": "assistant", "content": "A1"},
                "ts2",
            )
            output_dir = mgr.dump_all()
            files = [f for f in os.listdir(output_dir) if f.endswith(".json") and not f.endswith(".chatml.json")]
            assert len(files) == 1
            with open(os.path.join(output_dir, files[0])) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert "timestamp" not in data[0]["messages"][0]
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_dump_incomplete_session(self, mgr):
        mgr.create_session(
            [{"role": "user", "content": "Q1"}], "ts"
            # No assistant response — incomplete
        )
        output_dir = mgr.dump_all()
        files = [f for f in os.listdir(output_dir) if f.endswith(".chatml.json")]
        assert len(files) == 0  # No assistant msg → not dumped

    def test_get_session_chats(self, mgr):
        sess = mgr.create_session(
            [{"role": "user", "content": "Q1"}], "2026-01-01T00:00:00Z"
        )
        mgr.append_response(
            sess, {"role": "assistant", "content": "A1"}, "2026-01-01T00:00:01Z"
        )
        chats = mgr.get_session_chats()
        assert len(chats) == 1
        assert chats[0]["remarks"]["incomplete"] is False

    def test_append_response_single_mode_no_timestamp(self):
        mgr = SessionManager("/tmp", "s", "single")
        sess = mgr.create_session([{"role": "user", "content": "Q1"}], "ts")
        mgr.append_response(sess, {"role": "assistant", "content": "A1"}, "ts2")
        assert sess["messages"][1]["timestamp"] == ""


class TestNowIso:
    def test_returns_iso_format(self):
        ts = _now_iso()
        assert "T" in ts
        assert "+" in ts or "Z" in ts
