import base64
import hashlib
import json
import logging
import os
import re
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
                logger.warning(
                    "failed to parse tool_calls arguments JSON, keeping as string: %.200s",
                    args,
                )
    return msg


# Fields used for matching — only standard OpenAI message fields.
# Model-specific extras (reasoning, refusal, annotations, audio, etc.)
# are excluded so that the client's echoed messages match responses.
_MATCH_FIELDS = {"role", "content", "tool_calls", "tool_call_id", "name"}


def _canonical_content(content):
    """Normalize message content for prefix matching.

    Upstream responses use a plain string, while clients often echo the same
    text back as a list of content parts (e.g. ``[{"type": "text", ...}]``).
    Treat both forms as equivalent; image URLs are compared by hash so that
    matching stays cheap for large base64 screenshots.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content != "" else None
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                parts.append({"type": "raw", "raw": str(part)})
                continue
            ptype = part.get("type")
            if ptype == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                parts.append({
                    "type": "image_url",
                    "sha1": hashlib.sha1(url.encode("utf-8")).hexdigest(),
                })
            else:
                parts.append({
                    "type": ptype or "unknown",
                    "raw": json.dumps(part, sort_keys=True, ensure_ascii=False),
                })
        return parts or None
    return [{"type": "raw", "raw": str(content)}]


def _canonical(msg):
    """Serialize a message dict to a canonical JSON string for comparison.
    Strips 'timestamp' and non-standard fields, normalizes content and
    tool_calls.  Null values, empty strings, and empty tool_calls are omitted.
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
        elif k == "content":
            n = _canonical_content(v)
            if n is not None:
                d[k] = n
        elif v is not None and v != "":
            d[k] = v
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _is_hint_message(msg: dict) -> bool:
    """Return True if *msg* is a standalone global-hint user message."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content", "")
    return isinstance(content, str) and content.startswith("(Global hint:")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _split_turns(messages):
    """Group non-system, non-hint messages into assistant-ending turns.

    A turn starts after the previous assistant message and includes every
    user/tool/assistant message up to and including the next assistant.
    Messages after the last assistant form one incomplete turn.

    Returns a list of ``{"items": [(index, message), ...], "complete": bool}``.
    """
    turns = []
    current = []
    for idx, msg in enumerate(messages):
        if msg.get("role") == "system" or _is_hint_message(msg):
            continue
        current.append((idx, msg))
        if msg.get("role") == "assistant":
            turns.append({"items": current, "complete": True})
            current = []
    if current:
        turns.append({"items": current, "complete": False})
    return turns


def _turn_key(turn):
    """Canonical key for a turn, used for merged-mode matching."""
    return tuple(_canonical(msg) for _, msg in turn["items"])


class SessionManager:
    def __init__(self, log_folder, session_name, mode, session_path="",
                 rl_enabled=False, parse_arguments=False,
                 image_mode="inline", image_folder=""):
        self.log_folder = log_folder
        self.session_name = session_name
        self.mode = mode  # "none", "multi", "multi-strict", "merged", "single"
        self.session_path = session_path or log_folder
        self.rl_enabled = rl_enabled
        self.parse_arguments = parse_arguments
        self.image_mode = image_mode  # "inline" or "files"
        self.image_folder = image_folder
        self.sessions = []
        self.hint_content: str | None = None
        self.seen_session_paths = []
        self.remember_session_path(self.session_path)

    def remember_session_path(self, path):
        """Remember a ChatML output directory (for the viewer's directory list)."""
        if not path:
            return
        normalized = os.path.abspath(os.path.expanduser(path))
        if normalized not in self.seen_session_paths:
            self.seen_session_paths.append(normalized)

    @property
    def enabled(self):
        return self.mode != "none"

    @property
    def _is_multi_like(self):
        return self.mode in ("multi", "multi-strict")

    @property
    def _is_merged(self):
        return self.mode == "merged"

    # ------------------------------------------------------------------
    # Prefix matching
    # ------------------------------------------------------------------
    def find_matching_session(self, request_messages):
        """Return (session, match_len) where the non-hint messages in
        session.messages are a prefix of the non-hint messages in
        *request_messages*.  Only active in 'multi' and 'multi-strict' modes.

        Global-hint user messages are skipped during comparison so that
        hints injected by the proxy (recorded in ChatML) do not prevent
        the client (which does not send hints) from matching stored
        sessions.  *match_len* is an unfiltered index into
        *request_messages* — the first message not yet stored."""
        if not self._is_multi_like:
            return None, 0
        for sess in self.sessions:
            sess_msgs = sess["messages"]
            si = 0          # index in session messages
            ri = 0          # index in request messages
            ok = True
            system_mismatch = False

            while si < len(sess_msgs) and ri < len(request_messages):
                # Skip hint messages on both sides
                if _is_hint_message(sess_msgs[si]):
                    si += 1
                    continue
                if _is_hint_message(request_messages[ri]):
                    ri += 1
                    continue

                if _canonical(sess_msgs[si]) != _canonical(request_messages[ri]):
                    if (sess_msgs[si].get("role") == "system" and
                            request_messages[ri].get("role") == "system"):
                        system_mismatch = True
                        si += 1
                        ri += 1
                        continue
                    ok = False
                    break
                si += 1
                ri += 1

            # Skip any trailing hint messages in the stored session
            while si < len(sess_msgs) and _is_hint_message(sess_msgs[si]):
                si += 1

            if ok and si == len(sess_msgs):
                if system_mismatch:
                    logger.warning("session matched with different system prompt")
                return sess, ri

        # Fallback for clients that only resend a sliding window of recent
        # turns (e.g. OSWorld's max_trajectory_length).  The request no longer
        # contains the whole earlier conversation, but its beginning overlaps
        # the tail of a stored session; append only the unseen tail.
        req = [(i, m) for i, m in enumerate(request_messages)
               if not _is_hint_message(m) and m.get("role") != "system"]
        req_msgs = [m for _, m in req]
        if len(req_msgs) >= 2:
            best = None  # (overlap_len, session, match_len)
            for sess in self.sessions:
                sess_msgs = [m for m in sess["messages"]
                             if not _is_hint_message(m) and m.get("role") != "system"]
                if len(sess_msgs) < 2:
                    continue
                max_l = min(len(sess_msgs), len(req_msgs))
                for l in range(max_l, 1, -1):
                    if all(_canonical(sess_msgs[-l + k]) == _canonical(req_msgs[k])
                           for k in range(l)):
                        match_len = req[l][0] if l < len(req_msgs) else len(request_messages)
                        if best is None or l > best[0]:
                            best = (l, sess, match_len)
                        break
            if best is not None:
                logger.info("sliding-window session match: %d overlapping messages",
                            best[0])
                return best[1], best[2]
        return None, 0

    # ------------------------------------------------------------------
    # Merged matching
    # ------------------------------------------------------------------
    def find_merged_session(self, request_messages):
        """Find a session using the lenient 'merged' turn-overlap rule.

        Consider every completed assistant turn in the request.  Together
        with the response about to be appended, these are all turns up to and
        including the penultimate assistant turn.  If at least
        ``max(1, ceil(x / 2))`` of those *x* turns already exist in a session,
        treat it as the same conversation.

        Returns ``(session, append_indices)`` where *append_indices* are the
        indices of request messages that are not already represented in the
        session.  Matched turns are dropped; unmatched turns are appended.
        """
        if not self._is_merged:
            return None, []

        request_turns = _split_turns(request_messages)
        complete_turns = [t for t in request_turns if t["complete"]]
        if not complete_turns:
            return None, []

        # The current response is not in the request yet.  All completed
        # request turns are therefore up to and including the penultimate
        # assistant turn of the combined request+response.
        prefix = complete_turns
        x = len(prefix)
        required = max(1, (x + 1) // 2)

        best = None  # (score, session, append_indices, matched)
        for sess in self.sessions:
            sess_turns = _split_turns(sess["messages"])
            if not sess_turns:
                continue

            available = {}
            for turn in sess_turns:
                key = _turn_key(turn)
                available[key] = available.get(key, 0) + 1

            matched = 0
            for turn in prefix:
                key = _turn_key(turn)
                if available.get(key, 0):
                    available[key] -= 1
                    matched += 1

            if matched < required:
                continue

            score = (matched, len(sess_turns))
            if best is None or score > best[0]:
                append_indices = self._unmatched_turn_indices(
                    request_turns, sess_turns
                )
                best = (score, sess, append_indices, matched)

        if best is None:
            return None, []

        logger.info(
            "merged session match: %d/%d turns (required %d), "
            "appending %d messages",
            best[3], x, required, len(best[2]),
        )
        return best[1], best[2]

    @staticmethod
    def _unmatched_turn_indices(request_turns, session_turns):
        """Return message indices from request turns not present in session."""
        available = {}
        for turn in session_turns:
            key = _turn_key(turn)
            available[key] = available.get(key, 0) + 1

        indices = []
        for turn in request_turns:
            key = _turn_key(turn)
            if available.get(key, 0):
                available[key] -= 1
            else:
                indices.extend(idx for idx, _ in turn["items"])
        return indices

    # ------------------------------------------------------------------
    # Session creation / update
    # ------------------------------------------------------------------
    def create_session(self, request_messages, timestamp, tools=None):
        """Create a new session.  In merged/multi modes only the *last*
        message gets a real timestamp.  In 'single' mode timestamps are omitted."""
        session = {"messages": [], "tools": []}
        if tools:
            session["tools"] = list(tools)
        for i, msg in enumerate(request_messages):
            if self.mode == "single":
                ts = ""
            else:
                ts = timestamp if i == len(request_messages) - 1 else ""
            if self.parse_arguments:
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
            if self.parse_arguments:
                _normalize_msg_arguments(msg)
            session["messages"].append({**msg, "timestamp": ts})
        self._merge_tools(session, tools)

    def append_merged_request_messages(self, session, request_messages, indices,
                                       timestamp, tools=None):
        """Append only the selected request messages to a merged session."""
        ts = "" if self.mode == "single" else timestamp
        for idx in indices:
            msg = request_messages[idx]
            if self.parse_arguments:
                _normalize_msg_arguments(msg)
            session["messages"].append({**msg, "timestamp": ts})
        self._merge_tools(session, tools)

    @staticmethod
    def _merge_tools(session, tools):
        if not tools:
            return
        known = {t.get("function", {}).get("name") for t in session["tools"]}
        for t in tools:
            name = t.get("function", {}).get("name")
            if name and name not in known:
                session["tools"].append(t)
                known.add(name)

    def append_response(self, session, response_message, timestamp):
        """Append an assistant response message.  Timestamp omitted in 'single' mode."""
        ts = "" if self.mode == "single" else timestamp
        if self.parse_arguments:
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
            strict = self.mode == "multi-strict"
            chatml_msgs, incomplete = self._build_chatml(msgs, strict=strict)
            if chatml_msgs is None:
                continue
            entry = {
                "messages": chatml_msgs,
            }
            if strict:
                entry["remarks"] = {"incomplete": incomplete}
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
            entry = {"messages": clean}
            if sess.get("tools"):
                entry["tools"] = sess["tools"]
            entries.append(entry)
        if self.image_mode == "files":
            for i, entry in enumerate(entries):
                entry["messages"] = self._externalize_images(
                    entry["messages"], output_dir, f"{self.session_name}_e{i}")
        filepath = os.path.join(output_dir, f"{self.session_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def _dump_session(self, sess, suffix="", output_dir=None):
        msgs = sess["messages"]
        strict = self.mode == "multi-strict"
        chatml_msgs, incomplete = self._build_chatml(msgs, strict=strict)
        if chatml_msgs is None:
            return

        output = {
            "messages": chatml_msgs,
        }
        if strict:
            output["remarks"] = {"incomplete": incomplete}
        if sess.get("tools"):
            output["tools"] = sess["tools"]

        out = output_dir or self.session_path or self.log_folder
        if self.image_mode == "files":
            output["messages"] = self._externalize_images(
                chatml_msgs, out, f"{self.session_name}{suffix}")
        filepath = os.path.join(out, f"{self.session_name}{suffix}.chatml.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def _externalize_images(self, messages, output_dir, prefix):
        """Return a copy of *messages* with base64 images written to files.

        Each ``image_url`` data URL is decoded into ``image_folder`` (default
        ``<output_dir>/images``) and replaced by a path relative to the JSON
        file.  Non-data URLs are left untouched.
        """
        img_dir = self.image_folder or os.path.join(output_dir, "images")
        if not os.path.isabs(img_dir):
            img_dir = os.path.join(output_dir, img_dir)
        os.makedirs(img_dir, exist_ok=True)
        safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix)
        ext_map = {
            "image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
            "image/webp": "webp", "image/gif": "gif", "image/bmp": "bmp",
            "image/svg+xml": "svg",
        }

        out_msgs = []
        for mi, msg in enumerate(messages):
            content = msg.get("content")
            if not isinstance(content, list):
                out_msgs.append(msg)
                continue
            new_parts = []
            for pi, part in enumerate(content):
                if not (isinstance(part, dict) and part.get("type") == "image_url"):
                    new_parts.append(part)
                    continue
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if not (isinstance(url, str) and url.startswith("data:")):
                    new_parts.append(part)
                    continue
                match = re.match(r"data:(image/[^;]+);base64,(.*)", url, re.DOTALL)
                if not match:
                    new_parts.append(part)
                    continue
                mime, b64 = match.group(1), match.group(2)
                try:
                    raw = base64.b64decode(b64)
                except Exception:
                    new_parts.append(part)
                    continue
                ext = ext_map.get(mime, mime.split("/")[-1].split("+")[0] or "img")
                digest = hashlib.sha1(raw).hexdigest()[:12]
                filename = f"{safe_prefix}_m{mi}_p{pi}_{digest}.{ext}"
                filepath = os.path.join(img_dir, filename)
                if not os.path.exists(filepath):
                    with open(filepath, "wb") as f:
                        f.write(raw)
                rel_path = os.path.relpath(filepath, output_dir)
                if isinstance(image_url, dict):
                    new_parts.append({**part, "image_url": {**image_url, "url": rel_path}})
                else:
                    new_parts.append({**part, "image_url": rel_path})
            out_msgs.append({**msg, "content": new_parts})
        return out_msgs

    def _build_chatml(self, messages, strict=True):
        """Build ChatML output from messages.

        In strict mode (multi-strict): if the session ends with an
        assistant message it is complete.  Otherwise truncate to the
        longest prefix ending with an assistant message and mark
        incomplete.

        In non-strict mode (multi): return all messages as-is, never
        truncate or mark incomplete."""
        if not messages:
            return None, False

        if not strict:
            return messages, False

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
