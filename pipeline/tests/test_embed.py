from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import db as dbmod
import embed


class FakeModel:
    def __init__(self, dim=3):
        self.dim = dim

    def encode(self, texts, **kwargs):
        return np.ones((len(texts), self.dim), dtype=np.float64)


def _make_embedder(dim=3):
    cfg = SimpleNamespace(model_name="fake-model", batch_size=2)
    e = embed.Embedder(cfg)
    e._model = FakeModel(dim)
    e._dim = dim
    return e


class TestEmbedder:
    def test_load_using_dim_method(self, monkeypatch):
        import sentence_transformers

        class FakeST:
            def __init__(self, name, device=None):
                self.name = name

            def get_sentence_embedding_dimension(self):
                return 7

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
        cfg = SimpleNamespace(model_name="fake-model", batch_size=2)
        e = embed.Embedder(cfg)
        assert e.dim == 7

    def test_load_using_probe(self, monkeypatch):
        import sentence_transformers

        class FakeST:
            def __init__(self, name, device=None):
                pass

            def encode(self, texts, **kwargs):
                return np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
        cfg = SimpleNamespace(model_name="fake-model", batch_size=2)
        e = embed.Embedder(cfg)
        assert e.dim == 5

    def test_dim(self):
        e = _make_embedder(4)
        assert e.dim == 4

    def test_encode(self):
        e = _make_embedder(3)
        out = e.encode(["hello", "world"])
        assert out.shape == (2, 3)
        assert out.dtype == np.float32

    def test_encode_one(self):
        e = _make_embedder(3)
        out = e.encode_one("hello")
        assert out.shape == (3,)

    def test_encode_empty(self):
        e = _make_embedder(3)
        out = e.encode([])
        assert out.shape == (0, 3)


class TestComputeMissing:
    def test_no_missing_returns_zero(self, conn):
        e = _make_embedder(3)
        assert embed.compute_missing_embeddings(conn, e) == 0

    def test_computes_and_caches(self, conn):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.insert_news_item(
            conn,
            title="Hello",
            url="https://example.com/a",
            source_feed="https://example.com/feed.xml",
            summary="Summary",
            image_url=None,
            published_at=1,
        )
        e = _make_embedder(3)
        count = embed.compute_missing_embeddings(conn, e)
        assert count == 1

        item_id = conn.execute("SELECT id FROM news_items WHERE url=?", ("https://example.com/a",)).fetchone()[0]
        blob = dbmod.get_embedding(conn, item_id)
        assert blob is not None
        arr = np.frombuffer(blob, dtype=np.float32)
        assert arr.shape == (3,)


class TestLoadEmbedding:
    def test_round_trip(self, conn):
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
        dbmod.upsert_embedding(conn, item_id, np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes())
        arr = embed.load_embedding(conn, item_id)
        assert arr is not None
        np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0])

    def test_missing(self, conn):
        assert embed.load_embedding(conn, 999) is None
