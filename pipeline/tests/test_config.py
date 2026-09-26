from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import config


class TestEnvInt:
    def test_returns_default_when_missing(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert config._env_int("MISSING_VAR", 42) == 42

    def test_parses_valid_integer(self, monkeypatch):
        monkeypatch.setenv("SOME_VAR", "123")
        assert config._env_int("SOME_VAR", 0) == 123

    def test_ignores_whitespace_and_uses_default_on_bad_value(self, monkeypatch, caplog):
        monkeypatch.setenv("SOME_VAR", "abc")
        with caplog.at_level(logging.WARNING):
            assert config._env_int("SOME_VAR", 7) == 7
        assert "invalid int for SOME_VAR" in caplog.text


class TestDefaultDbPath:
    def test_repo_root_defaults_to_news_db(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config, "PY_DIR", tmp_path / "pipeline")
        assert config._default_db_path() == "news.db"

    def test_py_dir_defaults_to_parent(self, monkeypatch, tmp_path):
        py_dir = tmp_path / "pipeline"
        py_dir.mkdir()
        monkeypatch.chdir(py_dir)
        monkeypatch.setattr(config, "PY_DIR", py_dir)
        assert config._default_db_path() == str(tmp_path / "news.db")


class TestLoadConfig:
    def test_uses_defaults_and_empty_feeds(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr(config, "PY_DIR", tmp_path)
        monkeypatch.delenv("DB_PATH", raising=False)
        monkeypatch.delenv("MODEL_NAME", raising=False)
        # avoid writing to real fs
        monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
        with caplog.at_level(logging.WARNING):
            cfg = config.load_config()
        assert cfg.db_path == str(tmp_path / "news.db") or cfg.db_path == "news.db"
        assert cfg.model_name == config.DEFAULT_MODEL_NAME
        assert cfg.scoring_method == "centroid"
        assert cfg.batch_size == config.DEFAULT_BATCH_SIZE
        assert cfg.log_level == "INFO"
        assert cfg.feeds == []

    def test_loads_feeds_toml(self, monkeypatch, tmp_path):
        feeds_toml = tmp_path / "feeds.toml"
        feeds_toml.write_text(
            '[[feeds]]\nurl = "https://example.com/feed.xml"\ntitle = "Example"\n'
        )
        monkeypatch.setattr(config, "PY_DIR", tmp_path)
        monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
        monkeypatch.delenv("DB_PATH", raising=False)
        cfg = config.load_config()
        assert len(cfg.feeds) == 1
        assert cfg.feeds[0].url == "https://example.com/feed.xml"
        assert cfg.feeds[0].title == "Example"

    def test_env_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "PY_DIR", tmp_path)
        monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
        monkeypatch.setenv("DB_PATH", "/tmp/over.db")
        monkeypatch.setenv("MODEL_NAME", "other-model")
        monkeypatch.setenv("BATCH_SIZE", "4")
        monkeypatch.setenv("LOG_LEVEL", "debug")
        monkeypatch.setenv("SCORING_METHOD", "logistic")
        cfg = config.load_config()
        assert cfg.db_path == "/tmp/over.db"
        assert cfg.model_name == "other-model"
        assert cfg.batch_size == 4
        assert cfg.log_level == "DEBUG"
        assert cfg.scoring_method == "logistic"

    def test_loads_feeds_toml_skips_missing_url(self, monkeypatch, tmp_path, caplog):
        feeds_toml = tmp_path / "feeds.toml"
        feeds_toml.write_text(
            '[[feeds]]\nurl = "https://example.com/feed.xml"\n'
            '[[feeds]]\ntitle = "No URL"\n'
        )
        monkeypatch.setattr(config, "PY_DIR", tmp_path)
        monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
        monkeypatch.delenv("DB_PATH", raising=False)
        with caplog.at_level(logging.WARNING):
            cfg = config.load_config()
        assert len(cfg.feeds) == 1
        assert cfg.feeds[0].url == "https://example.com/feed.xml"
        assert "feed entry missing url" in caplog.text


class TestSetupLogging:
    def test_sets_level(self):
        logging.getLogger().setLevel(logging.NOTSET)
        config.setup_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG
