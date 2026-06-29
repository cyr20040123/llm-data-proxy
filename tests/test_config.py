"""Unit tests for config resolution logic."""

import os
import tempfile

import pytest
import yaml

from llmdataproxy.config import (
    _HARD_DEFAULTS,
    _CONFIG_FIELDS,
    _cli_to_yaml_key,
    _yaml_to_cli_key,
    _yaml_value,
    _load_yaml,
    _find_config_file,
    _get_config_write_path,
)


class TestKeyConversion:
    def test_cli_to_yaml(self):
        assert _cli_to_yaml_key("base_url") == "base-url"
        assert _cli_to_yaml_key("session_name") == "session-name"
        assert _cli_to_yaml_key("log_chatml") == "log-chatml"

    def test_yaml_to_cli(self):
        assert _yaml_to_cli_key("base-url") == "base_url"
        assert _yaml_to_cli_key("session-name") == "session_name"


class TestYamlValue:
    def test_port_int(self):
        assert _yaml_value("port", "8080") == 8080
        assert _yaml_value("port", 8080) == 8080

    def test_temperature_float(self):
        assert _yaml_value("temperature", "0.7") == 0.7

    def test_rl_bool(self):
        assert _yaml_value("rl", True) is True
        assert _yaml_value("rl", False) is False

    def test_other_passthrough(self):
        assert _yaml_value("host", "0.0.0.0") == "0.0.0.0"
        assert _yaml_value("session_name", None) is None


class TestLoadYaml:
    def test_load_valid_yaml(self):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "test.yaml")
            with open(path, "w") as f:
                yaml.dump({"DEFAULT": {"host": "127.0.0.1"}}, f)
            data = _load_yaml(path)
            assert data["DEFAULT"]["host"] == "127.0.0.1"
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_load_missing_file(self):
        assert _load_yaml("/nonexistent/path.yaml") == {}

    def test_load_non_dict(self):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "test.yaml")
            with open(path, "w") as f:
                yaml.dump([1, 2, 3], f)
            data = _load_yaml(path)
            assert data == {}
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


class TestFindConfigFile:
    def test_cwd_found(self, monkeypatch):
        d = tempfile.mkdtemp()
        try:
            config_path = os.path.join(d, "llm_proxy.yaml")
            with open(config_path, "w") as f:
                yaml.dump({"DEFAULT": {}}, f)
            monkeypatch.chdir(d)
            result = _find_config_file()
            assert result == config_path
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_xdg_fallback(self, monkeypatch):
        d = tempfile.mkdtemp()
        try:
            xdg_dir = os.path.join(d, "llmdataproxy")
            os.makedirs(xdg_dir)
            config_path = os.path.join(xdg_dir, "llm_proxy.yaml")
            with open(config_path, "w") as f:
                yaml.dump({"DEFAULT": {}}, f)
            monkeypatch.setenv("XDG_CONFIG_HOME", d)
            # Change to empty temp dir so CWD has no config
            empty = tempfile.mkdtemp()
            try:
                monkeypatch.chdir(empty)
                result = _find_config_file()
                assert result == config_path
            finally:
                import shutil
                shutil.rmtree(empty, ignore_errors=True)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_not_found(self, monkeypatch):
        empty = tempfile.mkdtemp()
        try:
            monkeypatch.chdir(empty)
            monkeypatch.setenv("XDG_CONFIG_HOME", empty)
            assert _find_config_file() is None
        finally:
            import shutil
            shutil.rmtree(empty, ignore_errors=True)


class TestHardDefaults:
    def test_required_fields_exist(self):
        for f in _CONFIG_FIELDS:
            assert f in _HARD_DEFAULTS, f"Missing {f} in _HARD_DEFAULTS"

    def test_base_url_empty_by_default(self):
        assert _HARD_DEFAULTS["base_url"] == ""

    def test_default_port(self):
        assert _HARD_DEFAULTS["port"] == 8030
