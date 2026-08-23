"""Sentence-transformer embedding computation and caching.

The transformer is loaded once at startup and stays frozen. Embeddings are
computed from ``title + " " + summary`` and stored in
``news_item_embeddings`` as ``numpy.float32`` bytes.
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np

from config import Config
from db import (
    get_item_text,
    list_new_item_ids_without_embedding,
    upsert_embedding,
)

log = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformer model and the SQLite cache."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._model = None
        self._dim: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        # Imported lazily so importing embed.py does not require the heavy dep.
        from sentence_transformers import SentenceTransformer

        log.info(
            "loading sentence-transformer model %s (device=cpu)", self.cfg.model_name
        )
        self._model = SentenceTransformer(self.cfg.model_name, device="cpu")
        # Determine the embedding dimension from the model itself.
        dim = getattr(self._model, "get_sentence_embedding_dimension", None)
        if callable(dim):
            self._dim = int(dim())
        else:
            # Fallback: encode a tiny probe.
            probe = self._model.encode(["probe"], convert_to_numpy=True)
            self._dim = int(probe.shape[1])
        log.info("model loaded; embedding dim=%d", self._dim)

    @property
    def dim(self) -> int:
        self._load()
        assert self._dim is not None
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        assert self._model is not None
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return embeddings.astype(np.float32, copy=False)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def compute_missing_embeddings(conn: sqlite3.Connection, embedder: Embedder) -> int:
    """Compute and cache embeddings for any item lacking one. Returns count."""
    ids = list_new_item_ids_without_embedding(conn)
    if not ids:
        return 0
    log.info("computing embeddings for %d item(s)", len(ids))
    batch_size = max(1, embedder.cfg.batch_size)
    total = 0
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        texts = [get_item_text(conn, item_id) for item_id in chunk]
        # Skip empty texts (should not happen since title is NOT NULL).
        embeddings = embedder.encode(texts)
        for item_id, emb in zip(chunk, embeddings):
            blob = emb.astype(np.float32).tobytes()
            upsert_embedding(conn, item_id, blob)
            total += 1
    conn.commit()
    log.info("stored %d embedding(s)", total)
    return total


def load_embedding(conn: sqlite3.Connection, news_item_id: int) -> np.ndarray | None:
    from db import get_embedding

    blob = get_embedding(conn, news_item_id)
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()
