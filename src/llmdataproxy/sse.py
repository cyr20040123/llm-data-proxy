"""SSE reconstruction helpers — parse streaming response chunks back
into a complete non-streaming response dict."""

import json


def reconstruct_chat_response(chunks: list[bytes]) -> dict | None:
    """Parse SSE chunks from a streaming /v1/chat/completions response
    and reconstruct a non-streaming response dict.  Returns None on parse
    failure."""
    # Join all raw bytes before decoding so that SSE data lines split
    # across aiter_bytes() chunk boundaries are not silently lost.
    full_text = b"".join(chunks).decode("utf-8", errors="replace")
    collected = {}
    for line in full_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if "id" in obj:
            collected.setdefault("id", obj["id"])
        if "object" in obj:
            collected.setdefault("object", obj["object"].replace(".chunk", ""))
        if "model" in obj and "model" not in collected:
            collected["model"] = obj["model"]
        if "usage" in obj and obj["usage"]:
            collected["usage"] = obj["usage"]
        if "prompt_token_ids" in obj:
            collected["prompt_token_ids"] = obj["prompt_token_ids"]
        for choice in obj.get("choices", []):
            idx = choice.get("index", 0)
            if "choices" not in collected:
                collected["choices"] = []
            while len(collected["choices"]) <= idx:
                collected["choices"].append({
                    "index": idx,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                })
            c = collected["choices"][idx]
            delta = choice.get("delta", {})
            if delta.get("role"):
                c["message"]["role"] = delta["role"]
            if delta.get("content"):
                c["message"]["content"] += delta["content"]
            if delta.get("reasoning"):
                c["message"].setdefault("reasoning", "")
                c["message"]["reasoning"] += delta["reasoning"]
            if delta.get("reasoning_content"):
                c["message"].setdefault("reasoning_content", "")
                c["message"]["reasoning_content"] += delta["reasoning_content"]
            if delta.get("tool_calls"):
                tc_map = c["message"].setdefault("tool_calls", [])
                for tc in delta["tool_calls"]:
                    tci = tc.get("index", 0)
                    while len(tc_map) <= tci:
                        tc_map.append({
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                    if tc.get("id"):
                        tc_map[tci]["id"] = tc["id"]
                    if tc.get("function", {}).get("name"):
                        tc_map[tci]["function"]["name"] = tc["function"]["name"]
                    if tc.get("function", {}).get("arguments"):
                        tc_map[tci]["function"]["arguments"] += tc["function"]["arguments"]
            if choice.get("finish_reason"):
                c["finish_reason"] = choice["finish_reason"]
            # --- accumulate RL fields from streaming chunks ---
            if "token_ids" in choice and choice["token_ids"] is not None:
                c.setdefault("token_ids", []).extend(choice["token_ids"])
            if "logprobs" in choice and choice["logprobs"]:
                c.setdefault("_logprobs_content", [])
                c["_logprobs_content"].extend(
                    choice["logprobs"].get("content", []))
    if "choices" not in collected:
        return None
    return collected


def reconstruct_completion_response(chunks: list[bytes]) -> dict | None:
    """Parse SSE chunks from a streaming /v1/completions response."""
    # Join all raw bytes before decoding so that SSE data lines split
    # across aiter_bytes() chunk boundaries are not silently lost.
    full_text = b"".join(chunks).decode("utf-8", errors="replace")
    collected = {}
    for line in full_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if "id" in obj:
            collected.setdefault("id", obj["id"])
        if "object" in obj:
            collected.setdefault("object", obj["object"].replace(".chunk", ""))
        if "model" in obj and "model" not in collected:
            collected["model"] = obj["model"]
        if "usage" in obj and obj["usage"]:
            collected["usage"] = obj["usage"]
        if "prompt_token_ids" in obj:
            collected["prompt_token_ids"] = obj["prompt_token_ids"]
        for choice in obj.get("choices", []):
            idx = choice.get("index", 0)
            if "choices" not in collected:
                collected["choices"] = []
            while len(collected["choices"]) <= idx:
                collected["choices"].append({"text": "", "index": idx, "finish_reason": None})
            collected["choices"][idx]["text"] += choice.get("text", "")
            if choice.get("finish_reason"):
                collected["choices"][idx]["finish_reason"] = choice["finish_reason"]
            # --- accumulate RL fields from streaming chunks ---
            if "token_ids" in choice and choice["token_ids"] is not None:
                collected["choices"][idx].setdefault("token_ids", []).extend(
                    choice["token_ids"])
            if "logprobs" in choice and choice["logprobs"]:
                collected["choices"][idx].setdefault("_logprobs_content", [])
                collected["choices"][idx]["_logprobs_content"].extend(
                    choice["logprobs"].get("content", []))
    if "choices" not in collected:
        return None
    return collected
