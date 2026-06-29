"""Shared fixtures for llmdataproxy tests."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(host, port, timeout=10):
    """Poll until the server accepts connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
# Mock upstream server
# ---------------------------------------------------------------------------
def make_mock_upstream():
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "mock-model", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        last_msg = messages[-1].get("content", "") if messages else ""
        is_stream = body.get("stream", False)
        has_tools = bool(body.get("tools"))

        if is_stream:
            return StreamingResponse(
                _stream_chat_response(last_msg, has_tools, body.get("model", "mock")),
                media_type="text/event-stream",
            )
        else:
            return JSONResponse(
                _nonstream_chat_response(last_msg, has_tools, body.get("model", "mock"))
            )

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        is_stream = body.get("stream", False)

        if is_stream:
            return StreamingResponse(
                _stream_completion_response(prompt, body.get("model", "mock")),
                media_type="text/event-stream",
            )
        else:
            return JSONResponse(
                _nonstream_completion_response(prompt, body.get("model", "mock"))
            )

    @app.get("/v1/error_test")
    async def error_test():
        return JSONResponse(
            {"error": {"message": "test error", "type": "test"}}, status_code=500
        )

    return app


def _nonstream_chat_response(user_text, has_tools, model):
    if has_tools:
        return {
            "id": "chatcmpl-mock-001",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_mock_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Beijing"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }
    return {
        "id": "chatcmpl-mock-001",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"[mock reply to: {user_text}]",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


async def _stream_chat_response(user_text, has_tools, model):
    rid = "chatcmpl-mock-stream"
    yield f'data: {json.dumps({"id": rid, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})}\n\n'
    if has_tools:
        yield f'data: {{"id":"{rid}","object":"chat.completion.chunk","model":"{model}","choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":0,"id":"call_s1","type":"function","function":{{"name":"get_weather"}}}}]}}}}]}}\n\n'
        yield f'data: {{"id":"{rid}","object":"chat.completion.chunk","model":"{model}","choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":0,"function":{{"arguments":"{{\\"city\\":\\"Beijing\\"}}"}}}}]}}}}]}}\n\n'
    else:
        for w in ["Hello", " there", " from", " mock!"]:
            yield f'data: {json.dumps({"id": rid, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"content": w}}]})}\n\n'
    yield f'data: {json.dumps({"id": rid, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}\n\n'
    yield "data: [DONE]\n\n"


def _nonstream_completion_response(prompt, model):
    return {
        "id": "cmpl-mock-001",
        "object": "text_completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": f" completion for: {prompt[-30:]}",
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


async def _stream_completion_response(prompt, model):
    rid = "cmpl-mock-stream"
    for word in [" comp", "letion", " mock"]:
        yield f'data: {json.dumps({"id": rid, "object": "text_completion.chunk", "model": model, "choices": [{"index": 0, "text": word}]})}\n\n'
    yield f'data: {json.dumps({"id": rid, "object": "text_completion.chunk", "model": model, "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]})}\n\n'
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mock_upstream_url():
    """Start a mock upstream server and return its URL."""
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    # Kill anything on the port from a previous run
    subprocess.run(["fuser", "-k", f"{port}/tcp"], stderr=subprocess.DEVNULL)

    app = make_mock_upstream()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": project_root}
    proc = subprocess.Popen(
        [sys.executable, "-c",
         f"import uvicorn; "
         f"from tests.conftest import make_mock_upstream; "
         f"uvicorn.run(make_mock_upstream(), host='127.0.0.1', port={port}, log_level='warning')"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )

    if not wait_for_port("127.0.0.1", port, timeout=5):
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        proc.kill()
        proc.wait()
        pytest.fail(f"Mock upstream failed to start:\n{stderr}")

    yield url

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture
def chatml_dir():
    """Temporary directory for ChatML output."""
    d = tempfile.mkdtemp(prefix="llm_proxy_test_")
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def proxy_url(mock_upstream_url, chatml_dir):
    """Start llm_proxy against mock upstream, return proxy URL."""
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    subprocess.run(["fuser", "-k", f"{port}/tcp"], stderr=subprocess.DEVNULL)

    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(package_dir, "src")

    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "llmdataproxy",
         "--host", "127.0.0.1",
         "--port", str(port),
         "--base-url", mock_upstream_url,
         "--log-folder", chatml_dir,
         "--log-chatml", "multi",
         "--session-name", "test_sess"],
        env={**os.environ, "PYTHONPATH": src_dir},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    if not wait_for_port("127.0.0.1", port, timeout=5):
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        proc.kill()
        proc.wait()
        pytest.fail(f"Proxy failed to start:\n{stderr}")

    yield url

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture
def test_client(proxy_url):
    """httpx client pointed at the proxy."""
    with httpx.Client(timeout=30) as client:
        client._proxy_url = proxy_url
        yield client
