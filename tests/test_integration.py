"""Integration tests — proxy against mock upstream."""

import json
import os
import glob
import sys

import httpx
import pytest


class TestProxyEndpoints:
    """Tests that require mock_upstream + proxy running."""

    def test_proxyhealth(self, test_client):
        r = test_client.get(f"{test_client._proxy_url}/proxyhealth")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_newsession(self, test_client):
        r = test_client.post(
            f"{test_client._proxy_url}/newsession",
            json={"session_name": "test_sess"},
        )
        assert r.status_code == 200
        assert r.json()["session_name"] == "test_sess"

        r2 = test_client.post(f"{test_client._proxy_url}/newsession")
        assert r2.status_code == 200
        assert r2.json()["session_name"].startswith("sess_")

    def test_nonstream_chat(self, test_client):
        body = {
            "model": "mock-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "stream": False,
        }
        r = test_client.post(
            f"{test_client._proxy_url}/v1/chat/completions", json=body
        )
        assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]}"
        data = r.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_stream_chat(self, test_client):
        body = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        parts = []
        with test_client.stream(
            "POST", f"{test_client._proxy_url}/v1/chat/completions", json=body
        ) as resp:
            for chunk in resp.iter_bytes():
                parts.append(chunk.decode("utf-8", errors="replace"))
        text = "".join(parts)
        assert "data: " in text
        assert "[DONE]" in text

    def test_multi_round(self, test_client):
        body1 = {
            "model": "mock",
            "messages": [{"role": "user", "content": "Q1"}],
            "stream": False,
        }
        r1 = test_client.post(
            f"{test_client._proxy_url}/v1/chat/completions", json=body1
        )
        assert r1.status_code == 200
        a1 = r1.json()["choices"][0]["message"]["content"]

        body2 = {
            "model": "mock",
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": "Q2"},
            ],
            "stream": False,
        }
        r2 = test_client.post(
            f"{test_client._proxy_url}/v1/chat/completions", json=body2
        )
        assert r2.status_code == 200
        a2 = r2.json()["choices"][0]["message"]["content"]

        body3 = {
            "model": "mock",
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": a2},
                {"role": "user", "content": "Q3"},
            ],
            "stream": False,
        }
        r3 = test_client.post(
            f"{test_client._proxy_url}/v1/chat/completions", json=body3
        )
        assert r3.status_code == 200

    def test_tool_calls(self, test_client):
        body = {
            "model": "mock",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
            "stream": False,
        }
        r = test_client.post(
            f"{test_client._proxy_url}/v1/chat/completions", json=body
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert msg["tool_calls"] is not None
        assert len(msg["tool_calls"]) > 0

    def test_tool_calls_stream(self, test_client):
        body = {
            "model": "mock",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
            "stream": True,
        }
        parts = []
        with test_client.stream(
            "POST", f"{test_client._proxy_url}/v1/chat/completions", json=body
        ) as resp:
            for chunk in resp.iter_bytes():
                parts.append(chunk.decode("utf-8", errors="replace"))
        text = "".join(parts)
        assert "get_weather" in text

    def test_catchall(self, test_client):
        r = test_client.get(f"{test_client._proxy_url}/v1/models")
        assert r.status_code == 200
        assert "data" in r.json()

    def test_upstream_error(self, test_client):
        r = test_client.get(f"{test_client._proxy_url}/v1/error_test")
        assert r.status_code == 500

    def test_newsession_dump(self, test_client, chatml_dir):
        # This test is self-contained: first make a chat request, then
        # call /newsession, which triggers dump_all() of the old session.
        # NOTE: We must use a fresh session name via /newsession first,
        # then do a chat request under the *new* name, then call
        # /newsession again to dump it.

        # Switch to a known session name
        r = test_client.post(
            f"{test_client._proxy_url}/newsession",
            json={"session_name": "dump_test_src", "session_path": chatml_dir},
        )
        assert r.status_code == 200

        body = {
            "model": "mock",
            "messages": [{"role": "user", "content": "Dump test"}],
            "stream": False,
        }
        test_client.post(
            f"{test_client._proxy_url}/v1/chat/completions", json=body
        )
        # dump_all() runs before switching to new name
        test_client.post(
            f"{test_client._proxy_url}/newsession",
            json={"session_name": "next_sess", "session_path": chatml_dir},
        )

        files = glob.glob(os.path.join(chatml_dir, "dump_test_src*.json"))
        assert len(files) >= 1, f"No ChatML files found in {chatml_dir} (files={glob.glob(os.path.join(chatml_dir, '*'))})"
        filepath = files[0]
        with open(filepath) as f:
            data = json.load(f)
        assert "messages" in data
        assert "remarks" in data
        assert len(data["messages"]) >= 2

    def test_override_model(self, test_client):
        # Set override
        r = test_client.post(
            f"{test_client._proxy_url}/change_override_model",
            json={"model": "forced-model"},
        )
        assert r.status_code == 200
        assert r.json()["model_override"] == "forced-model"

        # Clear override
        r = test_client.post(
            f"{test_client._proxy_url}/change_override_model",
            json={"model": "none"},
        )
        assert r.status_code == 200
        assert r.json()["model_override"] is None

    def test_session_chats(self, test_client):
        r = test_client.get(f"{test_client._proxy_url}/session_chats")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
