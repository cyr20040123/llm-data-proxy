#!/usr/bin/env python3
"""Send a /newsession request to a running llm_proxy."""

import argparse
import json
import os
from urllib.request import Request, urlopen

import yaml

CONFIG_FILE_NAME = "llm_proxy.yaml"


def _load_yaml_config():
    """Load proxy host/port from YAML config."""
    cwd_path = os.path.join(os.getcwd(), CONFIG_FILE_NAME)
    xdg_config_home = os.environ.get(
        "XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config")
    )
    xdg_path = os.path.join(xdg_config_home, "llmdataproxy", CONFIG_FILE_NAME)

    config_path = None
    if os.path.isfile(cwd_path):
        config_path = cwd_path
    elif os.path.isfile(xdg_path):
        config_path = xdg_path

    if not config_path:
        return {}, None

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    defaults = data.get("DEFAULT", {}) or {}
    recents = data.get("RECENT", {}) or {}
    return {**defaults, **recents}, config_path


def main():
    parser = argparse.ArgumentParser(
        description="Trigger /newsession on a running llm_proxy"
    )
    parser.add_argument("session_name", help="New session name")
    parser.add_argument("session_path", nargs="?", default=None,
                        help="New session path (optional)")
    parser.add_argument("--host", default=None,
                        help="Proxy host (default: from config or 127.0.0.1)")
    parser.add_argument("--port", default=None, type=int,
                        help="Proxy port (default: from config or 8030)")
    args = parser.parse_args()

    config, config_path = _load_yaml_config()
    host = args.host or config.get("host", "127.0.0.1")
    port = args.port or config.get("port", 8030)
    if config_path:
        print(f"Using config: {config_path}")
        print(f"  host={host}, port={port}")

    url = f"http://{host}:{port}/newsession"
    if args.session_path and args.session_path != "":
        body = json.dumps({
            "session_name": args.session_name,
            "session_path": args.session_path,
        }).encode()
    else:
        body = json.dumps({"session_name": args.session_name}).encode()

    print(f"\nRequest sending to {url} with session_name={args.session_name}, "
          f"session_path={args.session_path}\n")

    req = Request(url, data=body, headers={"Content-Type": "application/json"},
                  method="POST")
    with urlopen(req) as resp:
        result = json.loads(resp.read())

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
