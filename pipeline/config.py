"""Configuration loading for the SmartFeed Python pipeline.

Reads environment variables and ``pipeline/feeds.toml``. The pipeline is
intended to be run from the repository root (``python pipeline/main.py``), so
paths default to that layout. ``DB_PATH`` can always be overridden.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# pipeline/ directory regardless of the current working directory.
PY_DIR = Path(__file__).resolve().parent
REPO_ROOT = PY_DIR.parent

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_AGE_DAYS = 30
DEFAULT_RETENTION_DAYS = 60
DEFAULT_RETRAIN_INTERVAL = 0
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SCORING_METHOD = "centroid"

USER_AGENT = "smartfeed-pipeline/0.1 (+https://github.com/local/smartfeed-rs)"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("invalid int for %s=%r, using default %d", name, raw, default)
        return default


def _default_db_path() -> str:
    """Pick a sensible default SQLite path.

    If the user runs from the repo root, ``news.db`` is correct. If they run
    from inside ``pipeline/``, ``../news.db`` is correct. ``DB_PATH`` always wins.
    """
    cwd = Path.cwd()
    if cwd.resolve() == PY_DIR.resolve():
        return str(PY_DIR.parent / "news.db")
    return "news.db"


@dataclass
class FeedConfig:
    url: str
    title: str | None = None


@dataclass
class Config:
    db_path: str
    model_name: str
    batch_size: int
    max_age_days: int
    retention_days: int
    retrain_interval: int
    log_level: str
    scoring_method: str
    feeds: list[FeedConfig] = field(default_factory=list)
    cache_dir: Path = field(default_factory=lambda: PY_DIR / ".cache")
    feeds_toml_path: Path = field(default_factory=lambda: PY_DIR / "feeds.toml")


def _load_feeds(path: Path) -> list[FeedConfig]:
    if not path.exists():
        log.warning("feeds file not found at %s; no feeds configured", path)
        return []
    with path.open("rb") as fh:
        data: dict[str, Any] = tomllib.load(fh)
    feeds: list[FeedConfig] = []
    for entry in data.get("feeds", []):
        url = entry.get("url")
        if not url:
            log.warning("feed entry missing url: %r", entry)
            continue
        feeds.append(FeedConfig(url=url, title=entry.get("title")))
    return feeds


def load_config() -> Config:
    """Build a :class:`Config` from environment variables and feeds.toml."""
    from dotenv import load_dotenv

    load_dotenv(PY_DIR / ".env")

    db_path = os.environ.get("DB_PATH") or _default_db_path()
    scoring_method = (
        (os.environ.get("SCORING_METHOD") or DEFAULT_SCORING_METHOD).strip().lower()
    )

    cfg = Config(
        db_path=db_path,
        model_name=os.environ.get("MODEL_NAME") or DEFAULT_MODEL_NAME,
        batch_size=_env_int("BATCH_SIZE", DEFAULT_BATCH_SIZE),
        max_age_days=_env_int("MAX_AGE_DAYS", DEFAULT_MAX_AGE_DAYS),
        retention_days=_env_int("RETENTION_DAYS", DEFAULT_RETENTION_DAYS),
        retrain_interval=_env_int("RETRAIN_INTERVAL", DEFAULT_RETRAIN_INTERVAL),
        log_level=(os.environ.get("LOG_LEVEL") or DEFAULT_LOG_LEVEL).upper(),
        scoring_method=scoring_method,
    )

    cfg.feeds = _load_feeds(cfg.feeds_toml_path)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def setup_logging(level: str) -> None:
    log_level = getattr(logging, level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Ensure the root logger is set even when basicConfig was already configured.
    logging.getLogger().setLevel(log_level)
