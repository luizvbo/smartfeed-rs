from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import db as dbmod
import embed
import model
from model import _cosine, _score_centroid, _train_centroid, should_retrain, score_all, train


def _make_cfg(cache_dir=None, retrain_interval=0, scoring_method="centroid"):
    if cache_dir is None:
        import tempfile

        cache_dir = Path(tempfile.mkdtemp())
    return SimpleNamespace(
        cache_dir=cache_dir,
        retrain_interval=retrain_interval,
        scoring_method=scoring_method,
    )


def _insert_voted_item(conn, url, vote):
    dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
    dbmod.insert_news_item(
        conn,
        title="Hello",
        url=url,
        source_feed="https://example.com/feed.xml",
        summary="s",
        image_url=None,
        published_at=1,
    )
    item_id = conn.execute("SELECT id FROM news_items WHERE url=?", (url,)).fetchone()[0]
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32) if vote == "up" else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    dbmod.upsert_embedding(conn, item_id, emb.tobytes())
    conn.execute("INSERT INTO votes(news_item_id, vote, created_at) VALUES (?, ?, ?)", (item_id, vote, int(time.time())))
    conn.commit()
    return item_id, emb


class TestCosine:
    def test_zero_with_zero_vector(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        assert _cosine(a, b) == 0.0

    def test_one_with_same_vector(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert _cosine(a, a) == pytest.approx(1.0)


class TestScoreCentroid:
    def test_both(self):
        liked = np.array([1.0, 0.0], dtype=np.float32)
        disliked = np.array([0.0, 1.0], dtype=np.float32)
        emb = np.array([1.0, 0.0], dtype=np.float32)
        score = _score_centroid(emb, liked, disliked)
        assert score > 0

    def test_only_liked(self):
        liked = np.array([1.0, 0.0], dtype=np.float32)
        emb = np.array([1.0, 0.0], dtype=np.float32)
        score = _score_centroid(emb, liked, None)
        assert score == pytest.approx(1.0)

    def test_only_disliked(self):
        disliked = np.array([1.0, 0.0], dtype=np.float32)
        emb = np.array([1.0, 0.0], dtype=np.float32)
        score = _score_centroid(emb, None, disliked)
        assert score == pytest.approx(-1.0)

    def test_none(self):
        assert _score_centroid(np.array([1.0, 0.0]), None, None) == 0.0


class TestTrainCentroid:
    def test_saves_files(self, conn, tmp_path):
        _insert_voted_item(conn, "https://example.com/up1", "up")
        _insert_voted_item(conn, "https://example.com/down1", "down")
        cfg = _make_cfg(cache_dir=tmp_path)
        _train_centroid(conn, cfg)
        assert (tmp_path / "liked_centroid.npy").exists()
        assert (tmp_path / "disliked_centroid.npy").exists()


class TestShouldRetrain:
    def test_force(self, conn):
        cfg = _make_cfg()
        assert should_retrain(conn, cfg, force=True) is True

    def test_no_votes(self, conn):
        cfg = _make_cfg()
        assert should_retrain(conn, cfg, force=False) is False

    def test_new_votes(self, conn):
        _insert_voted_item(conn, "https://example.com/up1", "up")
        cfg = _make_cfg()
        dbmod.set_state(conn, "last_train_at", str(int(time.time()) - 10))
        conn.commit()
        assert should_retrain(conn, cfg, force=False) is True

    def test_interval(self, conn):
        cfg = _make_cfg(retrain_interval=1)
        dbmod.set_state(conn, "last_train_at", str(int(time.time()) - 2))
        conn.commit()
        assert should_retrain(conn, cfg, force=False) is True

    def test_unchanged_votes_after_train(self, conn, tmp_path):
        _insert_voted_item(conn, "https://example.com/up1", "up")
        cfg = _make_cfg(cache_dir=tmp_path)
        train(conn, cfg)
        assert should_retrain(conn, cfg, force=False) is False

    def test_vote_deletion_triggers_retrain(self, conn, tmp_path):
        _insert_voted_item(conn, "https://example.com/up1", "up")
        cfg = _make_cfg(cache_dir=tmp_path)
        train(conn, cfg)
        conn.execute("DELETE FROM votes")
        conn.commit()
        assert should_retrain(conn, cfg, force=False) is True


class TestTrain:
    def test_train_sets_state(self, conn, tmp_path):
        _insert_voted_item(conn, "https://example.com/up1", "up")
        cfg = _make_cfg(cache_dir=tmp_path, scoring_method="centroid")
        train(conn, cfg)
        assert dbmod.get_state(conn, "last_train_at") is not None

    def test_clears_files_with_no_votes(self, conn, tmp_path):
        (tmp_path / "liked_centroid.npy").write_bytes(b"")
        (tmp_path / "disliked_centroid.npy").write_bytes(b"")
        cfg = _make_cfg(cache_dir=tmp_path)
        _train_centroid(conn, cfg)
        assert not (tmp_path / "liked_centroid.npy").exists()
        assert not (tmp_path / "disliked_centroid.npy").exists()

    def test_only_up_clears_disliked(self, conn, tmp_path):
        (tmp_path / "liked_centroid.npy").write_bytes(b"")
        (tmp_path / "disliked_centroid.npy").write_bytes(b"")
        _insert_voted_item(conn, "https://example.com/up1", "up")
        cfg = _make_cfg(cache_dir=tmp_path)
        _train_centroid(conn, cfg)
        assert (tmp_path / "liked_centroid.npy").exists()
        assert not (tmp_path / "disliked_centroid.npy").exists()

    def test_only_down_clears_liked(self, conn, tmp_path):
        (tmp_path / "liked_centroid.npy").write_bytes(b"")
        (tmp_path / "disliked_centroid.npy").write_bytes(b"")
        _insert_voted_item(conn, "https://example.com/down1", "down")
        cfg = _make_cfg(cache_dir=tmp_path)
        _train_centroid(conn, cfg)
        assert not (tmp_path / "liked_centroid.npy").exists()
        assert (tmp_path / "disliked_centroid.npy").exists()


class TestScoreAndLogistic:
    def test_score_all_updates_model_score(self, conn, tmp_path, monkeypatch):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.insert_news_item(
            conn,
            title="Hello",
            url="https://example.com/a",
            source_feed="https://example.com/feed.xml",
            summary="s",
            image_url=None,
            published_at=1,
        )
        item_id = conn.execute("SELECT id FROM news_items WHERE url=?", ("https://example.com/a",)).fetchone()[0]
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dbmod.upsert_embedding(conn, item_id, emb.tobytes())

        _insert_voted_item(conn, "https://example.com/up1", "up")
        cfg = _make_cfg(cache_dir=tmp_path, scoring_method="centroid")
        train(conn, cfg)

        monkeypatch.setattr(embed, "compute_missing_embeddings", lambda *a, **k: 0)
        embedder = SimpleNamespace()
        scored = score_all(conn, cfg, embedder)
        assert scored == 2
        row = conn.execute("SELECT model_score FROM news_items WHERE id=?", (item_id,)).fetchone()
        assert row[0] is not None

    def test_score_all_with_no_votes(self, conn, tmp_path, monkeypatch):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.insert_news_item(
            conn,
            title="Hello",
            url="https://example.com/a",
            source_feed="https://example.com/feed.xml",
            summary="s",
            image_url=None,
            published_at=1,
        )
        item_id = conn.execute("SELECT id FROM news_items WHERE url=?", ("https://example.com/a",)).fetchone()[0]
        dbmod.upsert_embedding(conn, item_id, np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes())
        cfg = _make_cfg(cache_dir=tmp_path, scoring_method="centroid")
        monkeypatch.setattr(embed, "compute_missing_embeddings", lambda *a, **k: 0)
        scored = score_all(conn, cfg, SimpleNamespace())
        row = conn.execute("SELECT model_score FROM news_items WHERE id=?", (item_id,)).fetchone()
        assert scored == 1
        assert row[0] == 0.0

    def test_train_logistic_no_sklearn_fallback(self, conn, tmp_path):
        _insert_voted_item(conn, "https://example.com/up1", "up")
        cfg = _make_cfg(cache_dir=tmp_path, scoring_method="logistic")
        train(conn, cfg)
        assert dbmod.get_state(conn, "last_train_at") is not None

    def test_score_all_logistic(self, conn, monkeypatch):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.insert_news_item(
            conn,
            title="Hello",
            url="https://example.com/a",
            source_feed="https://example.com/feed.xml",
            summary="s",
            image_url=None,
            published_at=1,
        )
        item_id = conn.execute("SELECT id FROM news_items WHERE url=?", ("https://example.com/a",)).fetchone()[0]
        dbmod.upsert_embedding(conn, item_id, np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes())

        class FakeClf:
            classes_ = np.array([0, 1])
            def predict_proba(self, x):
                return np.array([[0.2, 0.8]])

        monkeypatch.setattr(model, "_load_logistic", lambda cfg: FakeClf())
        monkeypatch.setattr(embed, "compute_missing_embeddings", lambda *a, **k: 0)
        cfg = _make_cfg(scoring_method="logistic")
        scored = score_all(conn, cfg, SimpleNamespace())
        row = conn.execute("SELECT model_score FROM news_items WHERE id=?", (item_id,)).fetchone()
        assert scored == 1
        assert abs(row[0] - 0.8) < 1e-9


class TestApplyExploration:
    def _seed_items(self, conn, scores, votes=None):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        ids = []
        for i, score in enumerate(scores):
            dbmod.insert_news_item(
                conn,
                title="T",
                url=f"https://example.com/i{i}",
                source_feed="https://example.com/feed.xml",
                summary=None,
                image_url=None,
                published_at=1,
            )
            item_id = conn.execute(
                "SELECT id FROM news_items WHERE url=?",
                (f"https://example.com/i{i}",),
            ).fetchone()[0]
            if score is not None:
                dbmod.update_model_score(conn, item_id, score)
            ids.append(item_id)
        for idx, vote in (votes or {}).items():
            conn.execute(
                "INSERT INTO votes(news_item_id, vote, created_at) VALUES (?, ?, 100)",
                (ids[idx], vote),
            )
        conn.commit()
        return ids

    def _rank(self, conn, item_id):
        return conn.execute(
            "SELECT rank_score FROM news_items WHERE id=?", (item_id,)
        ).fetchone()[0]

    def test_boosts_fraction_of_eligible_items(self, conn):
        import random

        # 10 unvoted items; median of scores = 0.6 → eligible = {0.1..0.5}.
        ids = self._seed_items(conn, [i / 10 for i in range(1, 11)])
        dbmod.set_state(conn, "explore_pct", "40")
        conn.commit()

        boosted = model.apply_exploration(conn, rng=random.Random(7))
        assert boosted == 2  # round(5 * 40%)

        boosted_ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM news_items WHERE rank_score > model_score + 0.001"
            ).fetchall()
        ]
        assert len(boosted_ids) == 2
        # Boosted items come only from the eligible (below-median) pool and
        # land in the top-decile range (0.9..1.0).
        for item_id in boosted_ids:
            model_score = conn.execute(
                "SELECT model_score FROM news_items WHERE id=?", (item_id,)
            ).fetchone()[0]
            assert model_score < 0.6
            assert 0.9 <= self._rank(conn, item_id) <= 1.0
        # Non-boosted items mirror their model score.
        for i, item_id in enumerate(ids, start=1):
            if item_id not in boosted_ids:
                assert abs(self._rank(conn, item_id) - i / 10) < 1e-9

    def test_only_eligible_items_touched(self, conn):
        import random

        # index 0 is voted (down) with a low score — must never be boosted.
        # index 1 has NULL model_score — it is eligible.
        ids = self._seed_items(
            conn,
            [0.05, None, 0.9, 0.95, 1.0],
            votes={0: "down"},
        )
        dbmod.set_state(conn, "explore_pct", "100")
        conn.commit()

        boosted = model.apply_exploration(conn, rng=random.Random(3))
        # Median of unvoted scores = 0.95 → eligible = {NULL-score item,
        # 0.9 item}; pct=100 boosts both. The voted item is never touched.
        assert boosted == 2
        assert self._rank(conn, ids[0]) == 0.05  # voted item untouched
        for item_id in (ids[1], ids[2]):
            rank = self._rank(conn, item_id)
            model_score = conn.execute(
                "SELECT model_score FROM news_items WHERE id=?", (item_id,)
            ).fetchone()[0]
            assert rank is not None and rank > (model_score or 0.0)
        for item_id in ids[3:]:
            assert self._rank(conn, item_id) == conn.execute(
                "SELECT model_score FROM news_items WHERE id=?", (item_id,)
            ).fetchone()[0]

    def test_zero_pct_mirrors_model_score(self, conn):
        import random

        ids = self._seed_items(conn, [0.2, 0.8, None])
        dbmod.set_state(conn, "explore_pct", "0")
        conn.commit()

        boosted = model.apply_exploration(conn, rng=random.Random(1))
        assert boosted == 0
        assert self._rank(conn, ids[0]) == 0.2
        assert self._rank(conn, ids[1]) == 0.8
        assert self._rank(conn, ids[2]) is None

    def test_no_scores_no_boost(self, conn):
        import random

        self._seed_items(conn, [None, None])
        dbmod.set_state(conn, "explore_pct", "50")
        conn.commit()
        boosted = model.apply_exploration(conn, rng=random.Random(1))
        assert boosted == 0
