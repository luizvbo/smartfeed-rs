"""SQLite access layer for the SmartFeed Python pipeline.

Shares the ``news_items``, ``feeds`` and ``votes`` tables with the Rust axum
web app. Creates them with ``CREATE TABLE IF NOT EXISTS`` so the Rust app's
schema is never overwritten. Also owns the Python-only ``news_item_embeddings``
and ``pipeline_state`` tables.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager

from config import Config

SCHEMA_SHARED: list[str] = [
    # news_items is owned jointly with the Rust app. Column names must match
    # src/models.rs exactly.
    """
    CREATE TABLE IF NOT EXISTS news_items (
        id           INTEGER PRIMARY KEY,
        title        TEXT NOT NULL,
        url          TEXT NOT NULL UNIQUE,
        source_feed  TEXT NOT NULL,
        summary      TEXT,
        image_url    TEXT,
        published_at INTEGER,
        fetched_at   INTEGER NOT NULL,
        model_score  REAL,
        opened_at    INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feeds (
        id            INTEGER PRIMARY KEY,
        url           TEXT NOT NULL UNIQUE,
        title         TEXT,
        last_fetch    INTEGER,
        last_status   TEXT,
        error_message TEXT,
        is_active     INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS votes (
        id           INTEGER PRIMARY KEY,
        news_item_id INTEGER NOT NULL UNIQUE,
        vote         TEXT NOT NULL,
        created_at   INTEGER NOT NULL
    )
    """,
]

SCHEMA_PYTHON: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS news_item_embeddings (
        news_item_id INTEGER PRIMARY KEY REFERENCES news_items(id) ON DELETE CASCADE,
        embedding    BLOB NOT NULL,
        updated_at   INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_state (
        key   TEXT PRIMARY KEY,
        value TEXT
    )
    """,
]

SCHEMA_INDICES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_news_items_published_at ON news_items(published_at)",
    "CREATE INDEX IF NOT EXISTS idx_news_items_fetched_at ON news_items(fetched_at)",
    "CREATE INDEX IF NOT EXISTS idx_feeds_is_active ON feeds(is_active)",
]


def now() -> int:
    return int(time.time())


def connect(cfg: Config) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and sane pragmas."""
    conn = sqlite3.connect(cfg.db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables idempotently and seed pipeline_state."""
    cur = conn.cursor()
    for stmt in SCHEMA_SHARED + SCHEMA_PYTHON + SCHEMA_INDICES:
        cur.execute(stmt)
    cur.execute(
        "INSERT OR IGNORE INTO pipeline_state(key, value) VALUES ('last_train_at', '0')"
    )
    conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Commit on success, rollback on error."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# --- feed helpers ---------------------------------------------------------


def upsert_feed(conn: sqlite3.Connection, url: str, title: str | None) -> None:
    conn.execute(
        "INSERT INTO feeds(url, title, is_active) VALUES (?, ?, 1) "
        "ON CONFLICT(url) DO UPDATE SET title=COALESCE(excluded.title, feeds.title)",
        (url, title),
    )


def update_feed_status(
    conn: sqlite3.Connection,
    url: str,
    status: str,
    error_message: str | None,
) -> None:
    conn.execute(
        "UPDATE feeds SET last_fetch=?, last_status=?, error_message=? WHERE url=?",
        (now(), status, error_message, url),
    )


# --- news_item helpers ----------------------------------------------------


def url_exists(conn: sqlite3.Connection, url: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM news_items WHERE url=? LIMIT 1", (url,)
    ).fetchone()
    return row is not None


def insert_news_item(
    conn: sqlite3.Connection,
    *,
    title: str,
    url: str,
    source_feed: str,
    summary: str | None,
    image_url: str | None,
    published_at: int | None,
) -> int | None:
    """Insert a news item. Returns the new id, or None if the URL already exists."""
    cur = conn.execute(
        """
        INSERT INTO news_items
            (title, url, source_feed, summary, image_url, published_at, fetched_at, opened_at, model_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        ON CONFLICT(url) DO NOTHING
        """,
        (title, url, source_feed, summary, image_url, published_at, now()),
    )
    if cur.rowcount == 0:
        return None
    return cur.lastrowid


def list_new_item_ids_without_embedding(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        """
        SELECT ni.id FROM news_items ni
        LEFT JOIN news_item_embeddings e ON e.news_item_id = ni.id
        WHERE e.news_item_id IS NULL
        ORDER BY ni.id
        """
    ).fetchall()
    return [r[0] for r in rows]


def list_all_item_ids(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute("SELECT id FROM news_items ORDER BY id").fetchall()
    return [r[0] for r in rows]


def get_item_text(conn: sqlite3.Connection, news_item_id: int) -> str:
    row = conn.execute(
        "SELECT title, COALESCE(summary, '') FROM news_items WHERE id=?",
        (news_item_id,),
    ).fetchone()
    if row is None:
        return ""
    return f"{row[0]} {row[1]}".strip()


# --- embedding cache ------------------------------------------------------


def get_embedding(conn: sqlite3.Connection, news_item_id: int) -> bytes | None:
    row = conn.execute(
        "SELECT embedding FROM news_item_embeddings WHERE news_item_id=?",
        (news_item_id,),
    ).fetchone()
    return row[0] if row else None


def upsert_embedding(conn: sqlite3.Connection, news_item_id: int, blob: bytes) -> None:
    conn.execute(
        """
        INSERT INTO news_item_embeddings(news_item_id, embedding, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(news_item_id) DO UPDATE SET embedding=excluded.embedding, updated_at=excluded.updated_at
        """,
        (news_item_id, blob, now()),
    )


# --- votes / state --------------------------------------------------------


def count_votes_since(conn: sqlite3.Connection, since_ts: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM votes WHERE created_at > ?", (since_ts,)
    ).fetchone()
    return int(row[0]) if row else 0


def list_votes_with_embeddings(
    conn: sqlite3.Connection,
) -> list[tuple[int, str, bytes]]:
    rows = conn.execute(
        """
        SELECT v.news_item_id, v.vote, e.embedding
        FROM votes v
        JOIN news_item_embeddings e ON e.news_item_id = v.news_item_id
        """
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM pipeline_state WHERE key=?", (key,)
    ).fetchone()
    return row[0] if row else None


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO pipeline_state(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def update_model_score(
    conn: sqlite3.Connection, news_item_id: int, score: float
) -> None:
    conn.execute(
        "UPDATE news_items SET model_score=? WHERE id=?", (score, news_item_id)
    )
