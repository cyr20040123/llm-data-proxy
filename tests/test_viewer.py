"""Tests for the read-only trajectory viewer."""

import base64
import json
import os
import tempfile

from fastapi.testclient import TestClient

from llmdataproxy.viewer import create_viewer_app


class FakeSessionManager:
    def __init__(self, path):
        self.session_path = path
        self.log_folder = path
        self.enabled = True

    def get_session_chats(self):
        return [{
            "messages": [
                {"role": "system", "content": "sys", "timestamp": ""},
                {"role": "user", "content": "live question", "timestamp": "t1"},
                {"role": "assistant", "content": "live answer", "timestamp": "t2"},
            ]
        }]


def _write_session(directory, name="task_demo.chatml.json",
                   instruction="Please do the thing."):
    image = b"\x89PNG\r\n\x1a\nDATA"
    os.makedirs(os.path.join(directory, "images"), exist_ok=True)
    with open(os.path.join(directory, "images", "pic.png"), "wb") as f:
        f.write(image)
    data = {
        "messages": [
            {"role": "system", "content": "You are an agent.", "timestamp": ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(image).decode()}},
                    {"type": "image_url", "image_url": {"url": "images/pic.png"}},
                ],
                "timestamp": "2026-01-01T00:00:01Z",
            },
            {
                "role": "assistant",
                "content": "Done.",
                "reasoning_content": "because",
                "timestamp": "2026-01-01T00:00:02Z",
            },
        ],
        "remarks": {"incomplete": False},
    }
    with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
        json.dump(data, f)
    return name


def _make_client():
    directory = tempfile.mkdtemp()
    name = _write_session(directory)
    app = create_viewer_app(FakeSessionManager(directory))
    return TestClient(app), directory, name


def test_list_shows_sessions_and_live():
    client, directory, name = _make_client()
    r = client.get("/")
    assert r.status_code == 200
    assert name in r.text
    assert "Please do the thing." in r.text
    assert "live" in r.text


def test_view_renders_roles_images_and_reasoning():
    client, _, name = _make_client()
    r = client.get("/view", params={"src": "file", "name": name})
    assert r.status_code == 200
    assert 'role-user' in r.text
    assert 'role-assistant' in r.text
    assert '--role-bg' in r.text
    assert r.text.count("<img") >= 2  # inline data URL + external file embedded
    assert base64.b64encode(b"\x89PNG\r\n\x1a\nDATA").decode() in r.text
    assert "reasoning" in r.text
    assert "Msg-1 Turn-1" in r.text
    assert "Msg-3 Turn-1" in r.text


def test_view_live_session():
    client, _, _ = _make_client()
    r = client.get("/view", params={"src": "live", "idx": 0})
    assert r.status_code == 200
    assert "live question" in r.text
    assert "live answer" in r.text
    assert client.get("/view", params={"src": "live", "idx": 9}).status_code == 404


def test_message_and_turn_labels():
    directory = tempfile.mkdtemp()
    data = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "tool result"},
        {"role": "assistant", "content": "a2"},
    ]}
    name = "turns.chatml.json"
    with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
        json.dump(data, f)
    client = TestClient(create_viewer_app(FakeSessionManager(directory)))
    r = client.get("/view", params={"src": "file", "name": name})
    assert r.status_code == 200
    assert "Msg-1 Turn-1" in r.text
    assert "Msg-2 Turn-1" in r.text
    assert "Msg-3 Turn-1" in r.text
    assert "Msg-4 Turn-2" in r.text
    assert "Msg-5 Turn-2" in r.text


def test_raw_endpoints():
    client, _, name = _make_client()
    r = client.get("/raw", params={"src": "file", "name": name})
    assert r.status_code == 200
    assert json.loads(r.text)["remarks"]["incomplete"] is False
    r = client.get("/raw", params={"src": "live", "idx": 0})
    assert r.status_code == 200


def test_invalid_names_and_traversal():
    client, _, _ = _make_client()
    assert client.get("/view", params={"src": "file", "name": "../secret.chatml.json"}).status_code == 400
    assert client.get("/view", params={"src": "file", "name": "/etc/passwd"}).status_code == 400
    assert client.get("/view", params={"src": "file", "name": "missing.chatml.json"}).status_code == 404
    assert client.get("/health").json() == {"status": "ok"}

def test_recursive_directories_dropdown():
    root = tempfile.mkdtemp()
    a = os.path.join(root, "run_a", "traj")
    b = os.path.join(root, "run_b", "traj")
    os.makedirs(a)
    os.makedirs(b)
    _write_session(a, "a.chatml.json", "task A")
    _write_session(b, "b.chatml.json", "task B")
    mgr = FakeSessionManager(a)
    mgr.seen_session_paths = [a]
    client = TestClient(create_viewer_app(mgr, root=root))
    r = client.get("/")
    assert a in r.text and b in r.text
    assert "a.chatml.json" in r.text
    assert "b.chatml.json" not in r.text  # only direct sessions
    r = client.get("/", params={"dir": b})
    assert "b.chatml.json" in r.text
    assert "a.chatml.json" not in r.text
    assert client.get("/view", params={"dir": b, "src": "file",
                                       "name": "b.chatml.json"}).status_code == 200


def test_seen_external_directory_outside_root():
    root = tempfile.mkdtemp()
    ext = tempfile.mkdtemp()
    _write_session(ext, "external.chatml.json", "external task")
    mgr = FakeSessionManager(root)
    mgr.seen_session_paths = [ext]
    client = TestClient(create_viewer_app(mgr, root=root))
    r = client.get("/")
    assert ext in r.text
    r = client.get("/", params={"dir": ext})
    assert "external.chatml.json" in r.text


def test_directory_traversal_rejected():
    client, _, name = _make_client()
    assert client.get("/", params={"dir": "/etc"}).status_code == 400
    assert client.get("/view", params={"dir": "/etc", "src": "file",
                                       "name": name}).status_code == 400

