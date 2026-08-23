from __future__ import annotations

import pytest

import db as dbmod


class TestSchema:
    def test_tables_created(self, conn):
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"news_items", "feeds", "votes", "news_item_embeddings", "pipeline_state"} <= tables

    def test_wal_mode(self, conn):
        # in-memory db falls back to "memory" journal mode
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row is not None


class TestFeeds:
    def test_upsert_feed(self, conn):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        row = conn.execute("SELECT * FROM feeds WHERE url=?", ("https://example.com/feed.xml",)).fetchone()
        assert row["title"] == "Example"
        assert row["is_active"] == 1

    def test_update_feed_status(self, conn):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.update_feed_status(conn, "https://example.com/feed.xml", "ok", None)
        row = conn.execute("SELECT * FROM feeds WHERE url=?", ("https://example.com/feed.xml",)).fetchone()
        assert row["last_status"] == "ok"
        assert row["error_message"] is None


class TestNewsItems:
    def test_insert_and_url_exists(self, conn):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        inserted = dbmod.insert_news_item(
            conn,
            title="Hello",
            url="https://example.com/a",
            source_feed="https://example.com/feed.xml",
            summary="summary",
            image_url=None,
            published_at=1234567890,
        )
        assert inserted is not None
        assert dbmod.url_exists(conn, "https://example.com/a") is True

    def test_insert_duplicate_url_returns_none(self, conn):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.insert_news_item(
            conn,
            title="Hello",
            url="https://example.com/a",
            source_feed="https://example.com/feed.xml",
            summary="summary",
            image_url=None,
            published_at=1234567890,
        )
        inserted = dbmod.insert_news_item(
            conn,
            title="Hello again",
            url="https://example.com/a",
            source_feed="https://example.com/feed.xml",
            summary="summary",
            image_url=None,
            published_at=1234567890,
        )
        assert inserted is None

    def test_get_item_text(self, conn):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.insert_news_item(
            conn,
            title="Title",
            url="https://example.com/a",
            source_feed="https://example.com/feed.xml",
            summary="Summary",
            image_url=None,
            published_at=123,
        )
        item_id = conn.execute("SELECT id FROM news_items WHERE url=?", ("https://example.com/a",)).fetchone()[0]
        assert dbmod.get_item_text(conn, item_id) == "Title Summary"

    def test_get_item_text_missing(self, conn):
        assert dbmod.get_item_text(conn, 999) == ""

    def test_update_model_score(self, conn):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.insert_news_item(
            conn,
            title="Hello",
            url="https://example.com/a",
            source_feed="https://example.com/feed.xml",
            summary="summary",
            image_url=None,
            published_at=1,
        )
        item_id = conn.execute("SELECT id FROM news_items WHERE url=?", ("https://example.com/a",)).fetchone()[0]
        dbmod.update_model_score(conn, item_id, 0.75)
        row = conn.execute("SELECT model_score FROM news_items WHERE id=?", (item_id,)).fetchone()
        assert abs(row[0] - 0.75) < 1e-9


class TestEmbeddings:
    def test_upsert_and_get_embedding(self, conn):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.insert_news_item(
            conn,
            title="Hello",
            url="https://example.com/a",
            source_feed="https://example.com/feed.xml",
            summary="summary",
            image_url=None,
            published_at=1,
        )
        item_id = conn.execute("SELECT id FROM news_items WHERE url=?", ("https://example.com/a",)).fetchone()[0]
        dbmod.upsert_embedding(conn, item_id, b"fake-embedding-bytes")
        assert dbmod.get_embedding(conn, item_id) == b"fake-embedding-bytes"

    def test_list_new_item_ids_without_embedding(self, conn):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.insert_news_item(
            conn,
            title="Hello",
            url="https://example.com/a",
            source_feed="https://example.com/feed.xml",
            summary="summary",
            image_url=None,
            published_at=1,
        )
        ids = dbmod.list_new_item_ids_without_embedding(conn)
        assert len(ids) == 1


class TestVotesAndState:
    def test_count_votes_since_and_list_votes(self, conn):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.insert_news_item(
            conn,
            title="Hello",
            url="https://example.com/a",
            source_feed="https://example.com/feed.xml",
            summary="summary",
            image_url=None,
            published_at=1,
        )
        item_id = conn.execute("SELECT id FROM news_items WHERE url=?", ("https://example.com/a",)).fetchone()[0]
        dbmod.upsert_embedding(conn, item_id, b"\x00" * 12)
        conn.execute("INSERT INTO votes(news_item_id, vote, created_at) VALUES (?, 'up', 100)", (item_id,))
        conn.commit()

        assert dbmod.count_votes_since(conn, 0) == 1
        votes = dbmod.list_votes_with_embeddings(conn)
        assert len(votes) == 1
        assert votes[0][1] == "up"

    def test_state_get_set(self, conn):
        dbmod.set_state(conn, "last_train_at", "12345")
        assert dbmod.get_state(conn, "last_train_at") == "12345"


class TestTransaction:
    def test_rolls_back_on_error(self, conn):
        with pytest.raises(ValueError):
            with dbmod.transaction(conn):
                dbmod.set_state(conn, "tx_test", "1")
                raise ValueError("boom")
        assert dbmod.get_state(conn, "tx_test") is None

    def test_commits_on_success(self, conn):
        with dbmod.transaction(conn):
            dbmod.set_state(conn, "tx_test", "ok")
        assert dbmod.get_state(conn, "tx_test") == "ok"
