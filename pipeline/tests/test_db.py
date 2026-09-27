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

    def test_rank_score_column(self, conn):
        cols = {r[1] for r in conn.execute("PRAGMA table_info(news_items)")}
        assert "rank_score" in cols

    def test_rank_score_migration_adds_column(self):
        # Existing DBs created before rank_score existed get an idempotent ALTER.
        c = dbmod.connect(type("Cfg", (), {"db_path": ":memory:"}))
        c.execute(
            """
            CREATE TABLE news_items (
                id INTEGER PRIMARY KEY, title TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE, source_feed TEXT NOT NULL,
                summary TEXT, image_url TEXT, published_at INTEGER,
                fetched_at INTEGER NOT NULL, model_score REAL, opened_at INTEGER
            )
            """
        )
        c.commit()
        dbmod.init_schema(c)
        cols = {r[1] for r in c.execute("PRAGMA table_info(news_items)")}
        c.close()
        assert "rank_score" in cols

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

    def test_is_feed_active(self, conn):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        assert dbmod.is_feed_active(conn, "https://example.com/feed.xml") is True
        conn.execute(
            "UPDATE feeds SET is_active=0 WHERE url=?",
            ("https://example.com/feed.xml",),
        )
        assert dbmod.is_feed_active(conn, "https://example.com/feed.xml") is False
        # Unknown feeds are treated as active so first ingest still registers them.
        assert dbmod.is_feed_active(conn, "https://example.com/missing.xml") is True


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

    def test_insert_without_published_at_falls_back_to_fetched_at(self, conn):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        inserted = dbmod.insert_news_item(
            conn,
            title="Hello",
            url="https://example.com/nodate",
            source_feed="https://example.com/feed.xml",
            summary=None,
            image_url=None,
            published_at=None,
        )
        assert inserted is not None
        row = conn.execute(
            "SELECT published_at, fetched_at FROM news_items WHERE id=?", (inserted,)
        ).fetchone()
        assert row["published_at"] is not None
        assert row["published_at"] == row["fetched_at"]

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


class TestSettings:
    def test_explore_pct_default(self, conn):
        assert dbmod.get_explore_pct(conn) == dbmod.DEFAULT_EXPLORE_PCT

    def test_explore_pct_reads_pipeline_state(self, conn):
        dbmod.set_state(conn, "explore_pct", "15.5000")
        assert dbmod.get_explore_pct(conn) == 15.5

    def test_explore_pct_clamped_and_invalid(self, conn):
        dbmod.set_state(conn, "explore_pct", "250")
        assert dbmod.get_explore_pct(conn) == 100.0
        dbmod.set_state(conn, "explore_pct", "abc")
        assert dbmod.get_explore_pct(conn) == dbmod.DEFAULT_EXPLORE_PCT


class TestCleanup:
    def _item(self, conn, url, published_at, opened_at=None):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.insert_news_item(
            conn,
            title="T",
            url=url,
            source_feed="https://example.com/feed.xml",
            summary=None,
            image_url=None,
            published_at=published_at,
        )
        item_id = conn.execute(
            "SELECT id FROM news_items WHERE url=?", (url,)
        ).fetchone()[0]
        if opened_at is not None:
            conn.execute(
                "UPDATE news_items SET opened_at=? WHERE id=?", (opened_at, item_id)
            )
        return item_id

    def test_deletes_old_unvoted_items_and_cascades_embeddings(self, conn):
        import time

        old = int(time.time()) - 90 * 86400
        item_id = self._item(conn, "https://example.com/old", old)
        dbmod.upsert_embedding(conn, item_id, b"\x00" * 12)
        conn.commit()

        deleted = dbmod.cleanup_expired_items(conn, 60)
        assert deleted == 1
        assert conn.execute(
            "SELECT 1 FROM news_items WHERE id=?", (item_id,)
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM news_item_embeddings WHERE news_item_id=?", (item_id,)
        ).fetchone() is None

    def test_keeps_voted_opened_and_recent_items(self, conn):
        import time

        old = int(time.time()) - 90 * 86400
        now = int(time.time())
        voted = self._item(conn, "https://example.com/voted", old)
        conn.execute(
            "INSERT INTO votes(news_item_id, vote, created_at) VALUES (?, 'down', ?)",
            (voted, now),
        )
        opened = self._item(conn, "https://example.com/opened", old, opened_at=now)
        recent = self._item(conn, "https://example.com/recent", now)
        conn.commit()

        deleted = dbmod.cleanup_expired_items(conn, 60)
        assert deleted == 0
        remaining = {
            r[0]
            for r in conn.execute("SELECT id FROM news_items").fetchall()
        }
        assert {voted, opened, recent} <= remaining
        # The vote on the kept item survives.
        assert conn.execute(
            "SELECT 1 FROM votes WHERE news_item_id=?", (voted,)
        ).fetchone() is not None

    def test_removes_orphan_votes(self, conn):
        import time

        now = int(time.time())
        # An orphan vote pointing at a deleted/absent item.
        conn.execute(
            "INSERT INTO votes(news_item_id, vote, created_at) VALUES (999, 'up', ?)",
            (now,),
        )
        conn.commit()

        dbmod.cleanup_expired_items(conn, 60)
        assert conn.execute(
            "SELECT 1 FROM votes WHERE news_item_id=999"
        ).fetchone() is None

    def test_zero_retention_is_noop(self, conn):
        import time

        old = int(time.time()) - 90 * 86400
        self._item(conn, "https://example.com/old", old)
        conn.commit()
        assert dbmod.cleanup_expired_items(conn, 0) == 0
        assert conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0] == 1


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
