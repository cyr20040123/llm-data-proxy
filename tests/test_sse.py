"""Unit tests for SSE reconstruction."""

import json

from llmdataproxy.sse import (
    reconstruct_chat_response,
    reconstruct_completion_response,
)


def _sse_chunks(*lines):
    """Helper: build SSE chunk bytes from data lines."""
    text = "\n".join(lines)
    return [text.encode("utf-8")]


class TestReconstructChatResponse:
    def test_basic_chat_stream(self):
        chunks = _sse_chunks(
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"Hello"}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":" world"}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        )
        result = reconstruct_chat_response(chunks)
        assert result is not None
        assert result["id"] == "c1"
        assert result["object"] == "chat.completion"
        assert result["model"] == "m"
        assert result["choices"][0]["message"]["content"] == "Hello world"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_tool_calls_stream(self):
        chunks = _sse_chunks(
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather"}}]}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":\\"B"}}]}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"eijing\\"}"}}]}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        )
        result = reconstruct_chat_response(chunks)
        assert result is not None
        msg = result["choices"][0]["message"]
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "call_1"
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"city":"Beijing"}'

    def test_usage_collected(self):
        chunks = _sse_chunks(
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"Hi"}}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        )
        result = reconstruct_chat_response(chunks)
        assert result is not None
        assert result["usage"]["prompt_tokens"] == 10

    def test_reasoning_content_accumulated(self):
        chunks = _sse_chunks(
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"reasoning_content":"Let me think..."}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"Answer"}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        )
        result = reconstruct_chat_response(chunks)
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "Let me think..."
        assert msg["content"] == "Answer"

    def test_empty_stream(self):
        chunks = _sse_chunks("data: [DONE]")
        result = reconstruct_chat_response(chunks)
        assert result is None

    def test_rl_fields_accumulated(self):
        chunks = _sse_chunks(
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"Hi"},"token_ids":[1,2]}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":" there"},"token_ids":[3,4]}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        )
        result = reconstruct_chat_response(chunks)
        c = result["choices"][0]
        assert c["token_ids"] == [1, 2, 3, 4]

    def test_chunks_across_boundaries(self):
        """SSE data lines may be split across multiple bytes chunks."""
        chunk1 = b'data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"rol'
        chunk2 = b'e":"assistant"}}]}\ndata: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"Hi"}}]}\ndata: [DONE]\n'
        result = reconstruct_chat_response([chunk1, chunk2])
        assert result is not None
        assert result["choices"][0]["message"]["content"] == "Hi"


class TestReconstructCompletionResponse:
    def test_basic_completion_stream(self):
        chunks = _sse_chunks(
            'data: {"id":"cmpl1","object":"text_completion.chunk","model":"m","choices":[{"index":0,"text":"Hello"}]}',
            'data: {"id":"cmpl1","object":"text_completion.chunk","model":"m","choices":[{"index":0,"text":" world"}]}',
            'data: {"id":"cmpl1","object":"text_completion.chunk","model":"m","choices":[{"index":0,"text":"","finish_reason":"stop"}]}',
            "data: [DONE]",
        )
        result = reconstruct_completion_response(chunks)
        assert result is not None
        assert result["id"] == "cmpl1"
        assert result["object"] == "text_completion"
        assert result["choices"][0]["text"] == "Hello world"

    def test_empty_stream(self):
        chunks = _sse_chunks("data: [DONE]")
        result = reconstruct_completion_response(chunks)
        assert result is None

    def test_usage_collected(self):
        chunks = _sse_chunks(
            'data: {"id":"cmpl1","object":"text_completion.chunk","model":"m","choices":[{"index":0,"text":"Hi"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}',
            'data: {"id":"cmpl1","object":"text_completion.chunk","model":"m","choices":[{"index":0,"text":"","finish_reason":"stop"}]}',
            "data: [DONE]",
        )
        result = reconstruct_completion_response(chunks)
        assert result["usage"]["prompt_tokens"] == 5
