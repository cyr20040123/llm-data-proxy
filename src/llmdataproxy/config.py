"""Configuration loading — CLI args + YAML config file resolution."""

import argparse
import os
from datetime import datetime

import yaml

CONFIG_FILE_NAME = "llm_proxy.yaml"

# Sentinel for CLI args not explicitly set
_UNSET = object()

# Hardcoded defaults (lowest priority)
_HARD_DEFAULTS = {
    "host": "0.0.0.0",
    "port": 8030,
    "base_url": "",
    "api_key": "",
    "log_folder": "./logs/",
    "log_chatml": "none",
    "session_name": None,
    "session_path": "",
    "temperature": -1.0,
    "rl": False,
    "default_model": None,
    "override_model": False,
}

# Fields persisted to YAML RECENT (in order)
_CONFIG_FIELDS = [
    "host", "port", "base_url", "api_key", "log_folder", "log_chatml",
    "session_name", "session_path", "temperature", "rl", "default_model",
    "override_model",
]

# Fields that are always written to RECENT (even if matching DEFAULT)
_ALWAYS_RECENT = set()


def _find_config_file():
    """Locate the YAML config file.

    Lookup order:
      1. CWD / llm_proxy.yaml
      2. $XDG_CONFIG_HOME/llmdataproxy/llm_proxy.yaml
    Returns the path if found, or None if neither exists.
    """
    cwd_path = os.path.join(os.getcwd(), CONFIG_FILE_NAME)
    if os.path.isfile(cwd_path):
        return cwd_path

    xdg_config_home = os.environ.get(
        "XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config")
    )
    xdg_path = os.path.join(xdg_config_home, "llmdataproxy", CONFIG_FILE_NAME)
    if os.path.isfile(xdg_path):
        return xdg_path

    return None


def _get_config_write_path():
    """Return the path for writing YAML config.  Prefer CWD; fall back to XDG dir."""
    cwd_path = os.path.join(os.getcwd(), CONFIG_FILE_NAME)
    if os.path.isfile(cwd_path) or not os.path.isfile(
        os.path.join(
            os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config")),
            "llmdataproxy", CONFIG_FILE_NAME,
        )
    ):
        return cwd_path

    xdg_config_home = os.environ.get(
        "XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config")
    )
    xdg_dir = os.path.join(xdg_config_home, "llmdataproxy")
    os.makedirs(xdg_dir, exist_ok=True)
    return os.path.join(xdg_dir, CONFIG_FILE_NAME)


def _load_yaml(path):
    """Load YAML config file. Returns empty dict if missing or unreadable."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _cli_to_yaml_key(cli_key):
    """Convert argparse dest (underscores) to YAML key (kebab-case)."""
    return cli_key.replace("_", "-")


def _yaml_to_cli_key(yaml_key):
    """Convert YAML key (kebab-case) to argparse dest (underscores)."""
    return yaml_key.replace("-", "_")


def _yaml_value(key, value):
    """Coerce a YAML value to the correct Python type for the given config key."""
    if key in ("port",):
        return int(value)
    if key in ("temperature",):
        return float(value)
    if key in ("rl",):
        return bool(value)
    if value is None:
        return None
    return value


def parse_args():
    # -- CLI parser -------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible LLM proxy with ChatML logging",
    )
    parser.add_argument("--host", default=_UNSET, help="Proxy listen host")
    parser.add_argument("--port", default=_UNSET, type=int, help="Proxy listen port")
    parser.add_argument("--base-url", default=_UNSET, help="Upstream LLM service base URL")
    parser.add_argument("--api-key", default=_UNSET, help="Upstream API key")
    parser.add_argument("--log-folder", default=_UNSET, help="Log and output directory")
    parser.add_argument("--log-chatml", default=_UNSET,
                        choices=["none", "multi", "single"],
                        help="ChatML recording mode: none, multi, single")
    parser.add_argument("--session-name", default=_UNSET, help="Initial session name")
    parser.add_argument("--session-path", default=_UNSET,
                        help="ChatML output path (defaults to --log-folder)")
    parser.add_argument("--temperature", default=_UNSET, type=float,
                        help="Default temperature for upstream requests")
    parser.add_argument("--rl", default=None, action="store_true",
                        help="Enable RL-specific ChatML logging")
    parser.add_argument("--override-model", default=None, action="store_true",
                        help="Force all requests to use --default-model")
    parser.add_argument("--default-model", default=_UNSET,
                        help="Default model for requests with empty or 'none' model")
    parser.add_argument("--preset", default=_UNSET,
                        help="Name of a YAML config group to load (e.g. DEEPSEEK)")
    parser.add_argument("--config-file", default=None,
                        help="Path to YAML config file (default: llm_proxy.yaml in CWD or XDG)")

    cli_args = parser.parse_args()

    # -- Locate config file -----------------------------------------------
    if cli_args.config_file:
        config_path = cli_args.config_file
    else:
        config_path = _find_config_file()

    # -- Load YAML config -------------------------------------------------
    if config_path:
        yaml_data = _load_yaml(config_path)
    else:
        yaml_data = {}

    defaults = yaml_data.get("DEFAULT", {}) or {}
    preset_name = None
    if cli_args.preset is not _UNSET:
        preset_name = cli_args.preset
        if preset_name not in yaml_data:
            available = [k for k in yaml_data if k not in ("DEFAULT", "RECENT")]
            parser.error(f"preset '{preset_name}' not found in config file. "
                         f"Available presets: {', '.join(available) if available else '(none)'}")
    preset = yaml_data.get(preset_name, {}) or {} if preset_name else {}

    # -- Resolve each field: CLI > preset > DEFAULT > hardcoded --
    # (RECENT is no longer used for resolution — only written to on save)
    def _cli_val(key):
        """Get CLI value for field, returning _UNSET if not explicitly provided."""
        # argparse action="store_true" always returns True/False, never _UNSET,
        # so detect via the actual dest attr
        if key in ("rl", "override_model"):
            return _UNSET if getattr(cli_args, key) is None else getattr(cli_args, key)
        return getattr(cli_args, key)

    resolved = {}
    _base = {}  # DEFAULT-level baseline for diff logging
    session_explicit = False
    for key in _HARD_DEFAULTS:
        yk = _cli_to_yaml_key(key)
        # Compute DEFAULT-level baseline first
        if yk in defaults:
            _base[key] = _yaml_value(key, defaults[yk])
        else:
            _base[key] = _HARD_DEFAULTS[key]
        # Resolve with full priority
        cli_v = _cli_val(key)
        if cli_v is not _UNSET:
            resolved[key] = cli_v
            if key == "session_name":
                session_explicit = True
            continue
        if preset_name and yk in preset:
            resolved[key] = _yaml_value(key, preset[yk])
            continue
        if yk in defaults:
            resolved[key] = _yaml_value(key, defaults[yk])
            continue
        resolved[key] = _HARD_DEFAULTS[key]

    resolved["_defaults"] = _base
    resolved["_config_path"] = config_path

    preset_name_str = preset_name if preset_name else None
    resolved["preset"] = preset_name_str

    # Auto session name if not set
    if not resolved["session_name"]:
        resolved["session_name"] = "sess_" + datetime.now().strftime("%m%d_%H%M%S")
    if not resolved["session_path"]:
        resolved["session_path"] = resolved["log_folder"]

    # Validate required base-url
    if not resolved["base_url"]:
        parser.error("--base-url is required (set via CLI, YAML DEFAULT, or --preset)")

    # -- Write back YAML --------------------------------------------------
    _save_config(resolved, session_explicit)

    # Return a namespace for backward compatibility
    return argparse.Namespace(**resolved)


def _save_config(args, session_explicit):
    """Write back YAML config:
    - RECENT: only fields that differ from DEFAULT (plus _ALWAYS_RECENT)
    - Clean up other non-DEFAULT groups: remove fields matching DEFAULT
    """
    config_path = args.get("_config_path") if isinstance(args, dict) else getattr(args, "_config_path", None)
    write_path = config_path or _get_config_write_path()

    yaml_data = _load_yaml(write_path)
    defaults = yaml_data.get("DEFAULT", {}) or {}

    # Determine which fields differ from DEFAULT
    recent = {}
    for key in _CONFIG_FIELDS:
        yk = _cli_to_yaml_key(key)
        val = args.get(key) if isinstance(args, dict) else getattr(args, key)
        if key == "session_name" and not session_explicit:
            continue  # auto-generated names are not persisted
        if yk in _ALWAYS_RECENT:
            recent[yk] = val
        elif val != defaults.get(yk):
            recent[yk] = val

    # Update RECENT
    if recent:
        yaml_data["RECENT"] = recent
    elif "RECENT" in yaml_data:
        del yaml_data["RECENT"]

    # Clean up other groups: remove fields matching DEFAULT
    for group_name in list(yaml_data.keys()):
        if group_name in ("DEFAULT", "RECENT"):
            continue
        group = yaml_data[group_name]
        if not isinstance(group, dict):
            continue
        cleaned = {}
        for yk, val in group.items():
            if val != defaults.get(yk):
                cleaned[yk] = val
        if cleaned:
            yaml_data[group_name] = cleaned
        else:
            del yaml_data[group_name]

    # Write
    os.makedirs(os.path.dirname(write_path) or ".", exist_ok=True)
    with open(write_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
