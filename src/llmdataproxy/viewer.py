"""Read-only HTTP viewer for ChatML trajectory files and live sessions."""

from __future__ import annotations

import base64
import html
import json
import os
import re
import socket
import threading
import time
from datetime import datetime
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

ROLE_COLORS = {
    "system": ("#f3f4f6", "#6b7280"),
    "user": ("#eff6ff", "#2563eb"),
    "assistant": ("#f0fdf4", "#16a34a"),
    "tool": ("#fffbeb", "#d97706"),
}
DEFAULT_COLORS = ("#fafafa", "#9ca3af")

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f8fafc; color: #0f172a;
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 28px 20px 64px; }
h1 { font-size: 23px; margin: 0 0 4px; }
.sub { color: #64748b; font-size: 13px; }
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
nav { display: flex; justify-content: space-between; align-items: center;
  gap: 12px; margin-bottom: 16px; }
nav .links { display: flex; gap: 14px; font-size: 13px; }
.cards { display: grid; gap: 12px; margin-top: 18px; }
.card { display: block; background: #fff; border: 1px solid #e2e8f0;
  border-radius: 12px; padding: 13px 16px; box-shadow: 0 1px 2px rgba(15,23,42,.04);
  transition: .15s; }
.card:hover { border-color: #93c5fd; box-shadow: 0 4px 14px rgba(37,99,235,.10); }
.card .name { font-weight: 600; font-size: 15px; word-break: break-all; }
.card .row { display: flex; gap: 12px; color: #64748b; font-size: 12.5px;
  margin-top: 4px; flex-wrap: wrap; }
.card .instr { color: #334155; font-size: 13.5px; margin-top: 6px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 999px;
  background: #e0e7ff; color: #3730a3; font-size: 11px; font-weight: 600; }
.badge.live { background: #dcfce7; color: #166534; }
.msg { background: var(--role-bg); border: 1px solid #e2e8f0;
  border-left: 4px solid var(--role-fg); border-radius: 10px;
  padding: 12px 14px; margin: 14px 0; }
.msg .meta { display: flex; gap: 10px; align-items: center; margin-bottom: 6px; }
.msg .role { color: var(--role-fg); font-weight: 700; text-transform: uppercase;
  font-size: 11px; letter-spacing: .06em; }
.msg .ts { color: #94a3b8; font-size: 11.5px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.msg .seq { margin-left: auto; color: #64748b; font-size: 11.5px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.text { white-space: pre-wrap; word-break: break-word; font-size: 14px; }
pre { white-space: pre-wrap; word-break: break-word; background: #fff;
  border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px;
  font-size: 12.5px; overflow: auto; }
img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0;
  margin-top: 8px; display: block; background: #fff; }
.muted { color: #94a3b8; font-size: 12.5px; }
details.reasoning { margin: 6px 0; }
details.reasoning summary { cursor: pointer; color: #64748b; font-size: 12.5px; }
.empty { background: #fff; border: 1px dashed #cbd5e1; border-radius: 12px;
  padding: 26px; text-align: center; color: #64748b; margin-top: 20px; }
select { max-width: 100%; padding: 6px 8px; border: 1px solid #cbd5e1;
  border-radius: 8px; background: #fff; color: #0f172a; font-size: 13px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
"""


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return ""


def _count_images(messages) -> int:
    n = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            n += sum(1 for p in content
                     if isinstance(p, dict) and p.get("type") == "image_url")
    return n


def _summarize(data) -> dict:
    messages = data.get("messages", []) if isinstance(data, dict) else []
    instruction = ""
    for msg in messages:
        if msg.get("role") == "user":
            instruction = _text_of(msg.get("content")).strip().replace("\n", " ")
            if instruction:
                break
    return {
        "messages": len(messages),
        "images": _count_images(messages),
        "instruction": instruction,
        "remarks": (data.get("remarks") if isinstance(data, dict) else None),
        "error": None,
    }


def _session_files(base: str):
    try:
        names = os.listdir(base)
    except OSError:
        return []
    files = []
    for name in names:
        if not name.endswith(".chatml.json"):
            continue
        path = os.path.join(base, name)
        if os.path.isfile(path):
            st = os.stat(path)
            files.append((name, path, st.st_mtime_ns, st.st_size))
    files.sort(key=lambda item: item[2], reverse=True)
    return files


def _resolve_image(url, base_dir: str):
    """Return a browser-displayable src, or None."""
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        return url
    if url.startswith(("http://", "https://")):
        return url
    rel = url.replace("\\", "/")
    if rel.startswith("/"):
        return None
    base = os.path.realpath(base_dir)
    path = os.path.realpath(os.path.join(base, rel))
    if path != base and not path.startswith(base + os.sep):
        return None
    if not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
    }.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        payload = f.read()
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def _render_content(content, base_dir: str) -> str:
    if isinstance(content, str):
        return f'<div class="text">{_esc(content)}</div>'
    if not isinstance(content, list):
        return f"<pre>{_esc(json.dumps(content, ensure_ascii=False, indent=2))}</pre>"
    out = []
    for part in content:
        if isinstance(part, str):
            out.append(f'<div class="text">{_esc(part)}</div>')
            continue
        if not isinstance(part, dict):
            out.append(f"<pre>{_esc(json.dumps(part, ensure_ascii=False))}</pre>")
            continue
        kind = part.get("type")
        if kind == "text":
            out.append(f'<div class="text">{_esc(part.get("text", ""))}</div>')
        elif kind == "image_url":
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            src = _resolve_image(url, base_dir)
            if src:
                out.append(f'<a href="{_esc(src)}" target="_blank" rel="noopener">'
                           f'<img loading="lazy" src="{_esc(src)}"></a>')
            else:
                out.append(f'<div class="muted">[image unavailable: {_esc(url)[:160]}]</div>')
        else:
            out.append(f"<pre>{_esc(json.dumps(part, ensure_ascii=False, indent=2))}</pre>")
    return "".join(out)


def _assign_turns(messages):
    """Turn numbering: a non-assistant message immediately followed by an
    assistant starts a new turn; both messages belong to that turn."""
    turns = [0] * len(messages)
    turn = 1
    i = 0
    while i < len(messages):
        role = messages[i].get("role")
        if (role != "assistant" and i + 1 < len(messages)
                and messages[i + 1].get("role") == "assistant"):
            turns[i] = turn
            turns[i + 1] = turn
            turn += 1
            i += 2
        else:
            turns[i] = max(1, turn - 1) if role == "assistant" else turn
            i += 1
    return turns


def _render_message(msg, base_dir: str, message_idx: int = 1,
                    turn_idx: int = 1) -> str:
    role = str(msg.get("role", "unknown"))
    safe_role = re.sub(r"[^a-z0-9_-]+", "_", role.lower())
    bg, fg = ROLE_COLORS.get(role, DEFAULT_COLORS)
    parts = [f'<section class="msg role-{safe_role}" '
             f'style="--role-bg:{bg};--role-fg:{fg}">']
    parts.append('<div class="meta">')
    parts.append(f'<span class="role">{_esc(role)}</span>')
    if msg.get("timestamp"):
        parts.append(f'<span class="ts">{_esc(msg["timestamp"])}</span>')
    parts.append(f'<span class="seq">Msg-{message_idx} Turn-{turn_idx}</span>')
    parts.append("</div>")
    if msg.get("reasoning_content"):
        parts.append('<details class="reasoning"><summary>reasoning</summary>'
                     f'<div class="text">{_esc(msg["reasoning_content"])}</div></details>')
    parts.append(_render_content(msg.get("content"), base_dir))
    if msg.get("tool_calls"):
        parts.append('<details class="reasoning"><summary>tool_calls</summary>'
                     f'<pre>{_esc(json.dumps(msg["tool_calls"], ensure_ascii=False, indent=2))}</pre>'
                     "</details>")
    parts.append("</section>")
    return "".join(parts)


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_esc(title)}</title><style>{CSS}</style></head>"
            f"<body><div class=\"wrap\">{body}</div></body></html>")


def _render_list(selected: str, directories, file_summaries, live_entries) -> str:
    if selected not in directories:
        directories = [selected] + list(directories)
    options = []
    for directory in directories:
        mark = " selected" if directory == selected else ""
        options.append(f'<option value="{_esc(directory)}"{mark}>{_esc(directory)}</option>')
    body = [
        '<nav><div><h1>Trajectories</h1>'
        '<div class="sub">trajectory directory (shows only its direct sessions)</div>'
        '<select onchange="location.href=\'/?dir=\'+encodeURIComponent(this.value)">'
        f'{"".join(options)}</select></div>'
        '<div class="links"><a href="/">refresh</a></div></nav>'
    ]
    if live_entries:
        body.append('<div class="sub" style="margin-top:16px">Live sessions</div>')
        body.append('<div class="cards">')
        for idx, summary in live_entries:
            body.append(
                f'<a class="card" href="/view?src=live&idx={idx}&dir={quote(selected, safe="")}">'
                f'<div class="name"><span class="badge live">live</span> session #{idx}</div>'
                f'<div class="row"><span>{summary["messages"]} messages</span>'
                f'<span>{summary["images"]} images</span></div>'
                f'<div class="instr">{_esc(summary["instruction"][:160])}</div></a>')
        body.append("</div>")

    if file_summaries:
        body.append(f'<div class="sub" style="margin-top:16px">'
                    f'Dumped sessions in {_esc(selected)}</div>')
        body.append('<div class="cards">')
        for name, summary, mtime_ns, size in file_summaries:
            mtime = datetime.fromtimestamp(mtime_ns / 1e9).strftime("%Y-%m-%d %H:%M:%S")
            href = (f'/view?src=file&dir={quote(selected, safe="")}'
                    f'&name={quote(name, safe="")}')
            body.append(
                f'<a class="card" href="{_esc(href)}">'
                f'<div class="name">{_esc(name)}</div>'
                f'<div class="row"><span>{summary["messages"]} messages</span>'
                f'<span>{summary["images"]} images</span>'
                f'<span>{_human_size(size)}</span><span>{_esc(mtime)}</span></div>'
                f'<div class="instr">{_esc(summary["instruction"][:160])}</div></a>')
        body.append("</div>")
    else:
        body.append('<div class="empty">No dumped sessions in this directory.</div>')
    return _page("Trajectories", "".join(body))


def _render_session(title: str, data, base_dir: str, raw_url: str, subtitle: str,
                    back_url: str = "/") -> str:
    messages = data.get("messages", []) if isinstance(data, dict) else []
    remarks = data.get("remarks") if isinstance(data, dict) else None
    body = [
        f'<nav><div><h1>{_esc(title)}</h1><div class="sub">{_esc(subtitle)}</div></div>'
        f'<div class="links"><a href="{_esc(back_url)}">back</a>'
        f'<a href="{_esc(raw_url)}">raw JSON</a></div></nav>'
    ]
    if remarks:
        body.append(f'<div class="sub">remarks: {_esc(json.dumps(remarks, ensure_ascii=False))}</div>')
    if not messages:
        body.append('<div class="empty">This session has no messages.</div>')
    turns = _assign_turns(messages)
    for idx, msg in enumerate(messages, start=1):
        body.append(_render_message(msg, base_dir, idx, turns[idx - 1]))
    return _page(title, "".join(body))


def _find_free_port(preferred: int) -> int:
    candidates = [preferred]
    candidates.extend(p for p in range(preferred + 1, min(preferred + 100, 65535) + 1))
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def create_viewer_app(session_manager, root: str | None = None) -> FastAPI:
    app = FastAPI(title="llm-data-proxy trajectory viewer", docs_url=None,
                  redoc_url=None, openapi_url=None)
    cache: dict[str, tuple] = {}
    root_dir = os.path.realpath(os.path.expanduser(root)) if root else None

    def current_dir() -> str:
        path = session_manager.session_path or session_manager.log_folder
        return os.path.realpath(os.path.expanduser(path))

    def seen_dirs():
        result = []
        for path in getattr(session_manager, "seen_session_paths", []) or []:
            real = os.path.realpath(os.path.expanduser(path))
            if os.path.isdir(real) and real not in result:
                result.append(real)
        current = current_dir()
        if os.path.isdir(current) and current not in result:
            result.append(current)
        return result

    def discover_dirs():
        found = set(seen_dirs())
        scan_root = root_dir or current_dir()
        if scan_root and os.path.isdir(scan_root):
            for dirpath, dirnames, filenames in os.walk(scan_root):
                dirnames[:] = [d for d in dirnames if d not in
                               (".git", "__pycache__", "images", "node_modules")]
                if any(name.endswith(".chatml.json") for name in filenames):
                    found.add(os.path.realpath(dirpath))
        return sorted(found)

    def resolve_dir(raw: str, dirs) -> str:
        current = current_dir()
        if not raw:
            return current
        real = os.path.realpath(os.path.expanduser(raw))
        if real not in dirs:
            raise HTTPException(status_code=400, detail="directory not allowed")
        return real

    def file_summary(path: str, mtime_ns: int, size: int) -> dict:
        key = (path, mtime_ns, size)
        cached = cache.get(path)
        if cached and cached[0] == key:
            return cached[1]
        try:
            with open(path, encoding="utf-8") as f:
                summary = _summarize(json.load(f))
        except Exception as exc:  # partial write / invalid JSON
            summary = {"messages": 0, "images": 0, "instruction": "",
                       "remarks": None, "error": str(exc)[:120]}
        cache[path] = (key, summary)
        return summary

    def live_entries():
        if not getattr(session_manager, "enabled", True):
            return []
        try:
            chats = session_manager.get_session_chats()
        except Exception:
            return []
        return [(i, _summarize({"messages": entry.get("messages", [])}))
                for i, entry in enumerate(chats)]

    @app.get("/", response_class=HTMLResponse)
    def index(dir: str = ""):
        dirs = discover_dirs()
        selected = resolve_dir(dir, dirs)
        files = []
        if os.path.isdir(selected):
            for name, path, mtime_ns, size in _session_files(selected):
                files.append((name, file_summary(path, mtime_ns, size), mtime_ns, size))
        live = live_entries() if selected == current_dir() else []
        return _render_list(selected, dirs, files, live)

    @app.get("/view", response_class=HTMLResponse)
    def view(src: str = "file", name: str = "", idx: int = -1, dir: str = ""):
        if src == "live":
            base = current_dir()
            chats = (session_manager.get_session_chats()
                     if getattr(session_manager, "enabled", True) else [])
            if idx < 0 or idx >= len(chats):
                raise HTTPException(status_code=404, detail="live session not found")
            entry = chats[idx]
            data = {"messages": entry.get("messages", []),
                    "remarks": entry.get("remarks")}
            back = f"/?dir={quote(base, safe='')}"
            return _render_session(f"live session #{idx}", data, base,
                                   f"/raw?src=live&idx={idx}",
                                   f"live · {len(data['messages'])} messages", back)
        dirs = discover_dirs()
        base = resolve_dir(dir, dirs)
        if not name or os.path.basename(name) != name or not name.endswith(".chatml.json"):
            raise HTTPException(status_code=400, detail="invalid session name")
        known = {item[0] for item in _session_files(base)}
        if name not in known:
            raise HTTPException(status_code=404, detail="session not found")
        path = os.path.join(base, name)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read session: {exc}")
        size = os.path.getsize(path)
        back = f"/?dir={quote(base, safe='')}"
        raw = f"/raw?src=file&dir={quote(base, safe='')}&name={quote(name, safe='')}"
        return _render_session(name, data, base, raw,
                               f"{len(data.get('messages', []))} messages · {_human_size(size)}",
                               back)

    @app.get("/raw")
    def raw(src: str = "file", name: str = "", idx: int = -1, dir: str = ""):
        if src == "live":
            chats = (session_manager.get_session_chats()
                     if getattr(session_manager, "enabled", True) else [])
            if idx < 0 or idx >= len(chats):
                raise HTTPException(status_code=404, detail="live session not found")
            return chats[idx]
        dirs = discover_dirs()
        base = resolve_dir(dir, dirs)
        if not name or os.path.basename(name) != name or not name.endswith(".chatml.json"):
            raise HTTPException(status_code=400, detail="invalid session name")
        known = {item[0] for item in _session_files(base)}
        if name not in known:
            raise HTTPException(status_code=404, detail="session not found")
        with open(os.path.join(base, name), encoding="utf-8") as f:
            return PlainTextResponse(f.read(), media_type="application/json")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def start_viewer(session_manager, preferred_port: int, logger,
                 host: str = "127.0.0.1", root: str | None = None):
    """Start the viewer in a daemon thread and return (server, port)."""
    port = _find_free_port(preferred_port)
    app = create_viewer_app(session_manager, root=root)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning",
                            access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="traj-viewer", daemon=True)
    thread.start()
    for _ in range(100):
        if getattr(server, "started", False):
            break
        time.sleep(0.05)
    if getattr(server, "started", False):
        logger.info("trajectory viewer: http://%s:%d", host, port)
    else:
        logger.warning("trajectory viewer did not start on port %d", port)
    return server, port
