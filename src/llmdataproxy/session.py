import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("llm_proxy")


def _normalize_tool_calls(tool_calls):
    """Normalize tool_calls for stable comparison — parse arguments JSON."""
    if not tool_calls:
        return None
    result = []
    for tc in tool_calls:
        func = tc.get("function", {})
        args = func.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        result.append({
            "id": tc.get("id", ""),
            "type": tc.get("type", "function"),
            "function": {"name": func.get("name", ""), "arguments": args},
        })
    return result


def _normalize_msg_arguments(msg: dict) -> dict:
    """Mutate *msg* in-place: convert tool_calls arguments from JSON string to dict."""
    tcs = msg.get("tool_calls")
    if not tcs:
        return msg
    for tc in tcs:
        func = tc.get("function", {})
        args = func.get("arguments")
        if isinstance(args, str):
            try:
                func["arguments"] = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                pass
    return msg


# Fields used for matching — only standard OpenAI message fields.
# Model-specific extras (reasoning, refusal, annotations, audio, etc.)
# are excluded so that the client's echoed messages match responses.
_MATCH_FIELDS = {"role", "content", "tool_calls", "tool_call_id", "name"}


def _canonical(msg):
    """Serialize a message dict to a canonical JSON string for comparison.
    Strips 'timestamp' and non-standard fields, normalizes tool_calls.
    Null values, empty strings, and empty tool_calls are omitted.
    None and '' are treated as equivalent so that stored messages (which
    may have content=None after reconstruction) match client messages
    (which may send content='')."""
    d = {}
    for k, v in msg.items():
        if k == "timestamp":
            continue
        if k not in _MATCH_FIELDS:
            continue
        if k == "tool_calls":
            n = _normalize_tool_calls(v)
            if n:
                d[k] = n
        elif v is not None and v != "":
            d[k] = v
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


class SessionManager:
    def __init__(self, log_folder, session_name, mode, session_path="",
                 rl_enabled=False):
        self.log_folder = log_folder
        self.session_name = session_name
        self.mode = mode  # "none", "multi", "single"
        self.session_path = session_path or log_folder
        self.rl_enabled = rl_enabled
        self.sessions = []

    @property
    def enabled(self):
        return self.mode != "none"

    # ------------------------------------------------------------------
    # Prefix matching
    # ------------------------------------------------------------------
    def find_matching_session(self, request_messages):
        """Return (session, match_len) where session.messages[:match_len]
        is a prefix of request_messages.  Only active in 'multi' mode.
        Comparison ignores timestamps and normalises tool-call arguments.
        A mismatch on system messages alone is tolerated (logged as a warning)."""
        if self.mode != "multi":
            return None, 0
        for sess in self.sessions:
            sess_msgs = sess["messages"]
            if len(sess_msgs) > len(request_messages):
                continue
            ok = True
            system_mismatch = False
            for i, sm in enumerate(sess_msgs):
                if _canonical(sm) != _canonical(request_messages[i]):
                    if (sm.get("role") == "system" and
                            request_messages[i].get("role") == "system"):
                        system_mismatch = True
                        continue
                    ok = False
                    break
            if ok:
                if system_mismatch:
                    logger.warning("session matched with different system prompt")
                return sess, len(sess_msgs)
        return None, 0

    # ------------------------------------------------------------------
    # Session creation / update
    # ------------------------------------------------------------------
    def create_session(self, request_messages, timestamp, tools=None):
        """Create a new session.  In 'multi' mode only the *last* message
        gets a real timestamp.  In 'single' mode timestamps are omitted."""
        session = {"messages": [], "tools": []}
        if tools:
            session["tools"] = list(tools)
        for i, msg in enumerate(request_messages):
            if self.mode == "single":
                ts = ""
            else:
                ts = timestamp if i == len(request_messages) - 1 else ""
            _normalize_msg_arguments(msg)
            session["messages"].append({**msg, "timestamp": ts})
        self.sessions.append(session)
        return session

    def append_request_messages(self, session, request_messages, match_len, timestamp,
                                tools=None):
        """Append suffix messages (those beyond match_len) to the session.
        New tool definitions are merged in (deduplicated by function name)."""
        new_msgs = request_messages[match_len:]
        ts = "" if self.mode == "single" else timestamp
        for msg in new_msgs:
            _normalize_msg_arguments(msg)
            session["messages"].append({**msg, "timestamp": ts})
        if tools:
            known = {t.get("function", {}).get("name") for t in session["tools"]}
            for t in tools:
                name = t.get("function", {}).get("name")
                if name and name not in known:
                    session["tools"].append(t)
                    known.add(name)

    def append_response(self, session, response_message, timestamp):
        """Append an assistant response message.  Timestamp omitted in 'single' mode."""
        ts = "" if self.mode == "single" else timestamp
        _normalize_msg_arguments(response_message)
        session["messages"].append({**response_message, "timestamp": ts})

    # ------------------------------------------------------------------
    # Dump to ChatML JSON
    # ------------------------------------------------------------------
    def get_session_chats(self):
        """Return all current sessions in ChatML format, sorted by most
        recently active (descending by last message timestamp)."""
        results = []
        for sess in self.sessions:
            msgs = sess["messages"]
            chatml_msgs, incomplete = self._build_chatml(msgs)
            if chatml_msgs is None:
                continue
            entry = {
                "messages": chatml_msgs,
                "remarks": {"incomplete": incomplete},
            }
            if sess.get("tools"):
                entry["tools"] = sess["tools"]
            results.append(entry)

        # Sort by most recent activity (last message timestamp, descending)
        def _last_ts(entry):
            msgs = entry["messages"]
            for m in reversed(msgs):
                ts = m.get("timestamp", "")
                if ts:
                    return ts
            return ""

        results.sort(key=_last_ts, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Dump to ChatML JSON
    # ------------------------------------------------------------------
    def dump_all(self):
        if not self.enabled:
            self.sessions = []
            return
        output_dir = self.session_path or self.log_folder
        os.makedirs(output_dir, exist_ok=True)

        if self.mode == "single":
            self._dump_single(output_dir)
        else:
            for i, sess in enumerate(self.sessions):
                suffix = f"_{i}" if len(self.sessions) > 1 else ""
                self._dump_session(sess, suffix, output_dir)
        self.sessions = []
        return output_dir

    def _dump_single(self, output_dir):
        """Single mode: all sessions in one file, no timestamps, no remarks."""
        entries = []
        for sess in self.sessions:
            msgs = sess["messages"]
            # Remove timestamp from each message
            clean = [{k: v for k, v in m.items() if k != "timestamp"}
                     for m in msgs]
            entries.append({"messages": clean})
        filepath = os.path.join(output_dir, f"{self.session_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def _dump_session(self, sess, suffix="", output_dir=None):
        msgs = sess["messages"]
        chatml_msgs, incomplete = self._build_chatml(msgs)
        if chatml_msgs is None:
            return

        output = {
            "messages": chatml_msgs,
            "remarks": {"incomplete": incomplete},
        }
        if sess.get("tools"):
            output["tools"] = sess["tools"]

        out = output_dir or self.session_path or self.log_folder
        filepath = os.path.join(out, f"{self.session_name}{suffix}.chatml.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def _build_chatml(self, messages):
        """If the session ends with an assistant message it is complete.
        Otherwise truncate to the longest prefix ending with an assistant
        message and mark incomplete."""
        if not messages:
            return None, False

        if messages[-1].get("role") == "assistant":
            return messages, False

        # Truncate to last assistant
        trunc = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                trunc = i
                break
        if trunc == -1:
            return None, False  # no assistant message at all — skip

        return messages[: trunc + 1], True
