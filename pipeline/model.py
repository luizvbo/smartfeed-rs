"""Preference model training and scoring.

The transformer stays frozen; this module trains a lightweight preference
model on top of cached embeddings.

Two strategies are supported:

* ``centroid`` (default, no extra deps): maintains liked/disliked centroids
  and scores via cosine similarity differences.
* ``logistic`` (optional, requires scikit-learn): fits a LogisticRegression
  on the voted embeddings and uses ``predict_proba`` for the "up" class.

Centroid artifacts are stored as ``.npy`` files in ``pipeline/.cache/``. The
logistic classifier is pickled into the same directory.
"""

from __future__ import annotations

import logging
import pickle
import random
import sqlite3
import time

import numpy as np

from config import Config
from db import (
    get_explore_pct,
    get_state,
    list_all_item_ids,
    list_votes_with_embeddings,
    set_state,
    update_model_score,
    update_rank_score,
)
from embed import Embedder, load_embedding

log = logging.getLogger(__name__)

LIKED_CENTROID_FILE = "liked_centroid.npy"
DISLIKED_CENTROID_FILE = "disliked_centroid.npy"
LOGISTIC_FILE = "logistic.pkl"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _gather_votes(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray, np.ndarray]:
    rows = list_votes_with_embeddings(conn)
    if not rows:
        return [], np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    ids = [r[0] for r in rows]
    X = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
    # Map vote strings to labels: up=1, down=0; anything else is ignored upstream.
    y = np.array([1 if r[1].lower() == "up" else 0 for r in rows], dtype=np.int32)
    return ids, X, y


# --- centroid strategy ----------------------------------------------------


def _train_centroid(conn: sqlite3.Connection, cfg: Config) -> bool:
    ids, X, y = _gather_votes(conn)
    if len(ids) == 0:
        log.info("no votes yet; clearing centroid cache")
        for name in (LIKED_CENTROID_FILE, DISLIKED_CENTROID_FILE):
            p = cfg.cache_dir / name
            if p.exists():
                p.unlink()
        return True

    liked = X[y == 1]
    disliked = X[y == 0]

    if len(liked) > 0:
        liked_centroid = liked.mean(axis=0).astype(np.float32)
        np.save(cfg.cache_dir / LIKED_CENTROID_FILE, liked_centroid)
        log.info("liked centroid saved from %d vote(s)", len(liked))
    else:
        p = cfg.cache_dir / LIKED_CENTROID_FILE
        if p.exists():
            p.unlink()
        log.info("no 'up' votes; liked centroid cleared")

    if len(disliked) > 0:
        disliked_centroid = disliked.mean(axis=0).astype(np.float32)
        np.save(cfg.cache_dir / DISLIKED_CENTROID_FILE, disliked_centroid)
        log.info("disliked centroid saved from %d vote(s)", len(disliked))
    else:
        p = cfg.cache_dir / DISLIKED_CENTROID_FILE
        if p.exists():
            p.unlink()
        log.info("no 'down' votes; disliked centroid cleared")
    return True


def _load_centroids(cfg: Config) -> tuple[np.ndarray | None, np.ndarray | None]:
    liked_path = cfg.cache_dir / LIKED_CENTROID_FILE
    disliked_path = cfg.cache_dir / DISLIKED_CENTROID_FILE
    liked = np.load(liked_path) if liked_path.exists() else None
    disliked = np.load(disliked_path) if disliked_path.exists() else None
    return liked, disliked


def _score_centroid(
    emb: np.ndarray, liked: np.ndarray | None, disliked: np.ndarray | None
) -> float:
    if liked is not None and disliked is not None:
        score = _cosine(emb, liked) - _cosine(emb, disliked)
    elif liked is not None:
        score = _cosine(emb, liked)
    elif disliked is not None:
        score = -_cosine(emb, disliked)
    else:
        return 0.0
    return float(max(-1.0, min(1.0, score)))


# --- logistic strategy ----------------------------------------------------


def _train_logistic(conn: sqlite3.Connection, cfg: Config) -> bool:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        log.warning("scikit-learn not installed; falling back to centroid training")
        return _train_centroid(conn, cfg)

    ids, X, y = _gather_votes(conn)
    if len(ids) < 2 or len(set(y.tolist())) < 2:
        log.info("not enough diverse votes for logistic; clearing classifier")
        p = cfg.cache_dir / LOGISTIC_FILE
        if p.exists():
            p.unlink()
        # Still keep centroids up to date as a fallback scorer.
        _train_centroid(conn, cfg)
        return True

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)
    with open(cfg.cache_dir / LOGISTIC_FILE, "wb") as fh:
        pickle.dump(clf, fh)
    log.info("logistic classifier trained on %d vote(s)", len(ids))
    # Also refresh centroids so centroid scoring remains a valid fallback.
    _train_centroid(conn, cfg)
    return True


def _load_logistic(cfg: Config):
    p = cfg.cache_dir / LOGISTIC_FILE
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to load logistic classifier: %s", exc)
        return None


def _score_logistic(emb: np.ndarray, clf) -> float:
    proba = clf.predict_proba(emb.reshape(1, -1))
    classes = list(clf.classes_)
    if 1 in classes:
        idx = classes.index(1)
        return float(proba[0, idx])
    return float(proba[0, 0])


# --- public API -----------------------------------------------------------


def _votes_fingerprint(conn: sqlite3.Connection) -> str:
    """Fingerprint of the votes table: ``count:newest_created_at``.

    Inserts and vote changes bump ``MAX(created_at)``; deletions shrink the
    count, so any vote mutation — including un-voting — is detected.
    """
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(created_at), 0) FROM votes"
    ).fetchone()
    return f"{row[0]}:{row[1]}"


def should_retrain(conn: sqlite3.Connection, cfg: Config, force: bool) -> bool:
    if force:
        return True
    last = get_state(conn, "last_train_at")
    last_ts = int(last) if last and last.isdigit() else 0
    if cfg.retrain_interval > 0:
        if time.time() - last_ts >= cfg.retrain_interval:
            return True
    # Retrain whenever the votes table differs from the fingerprint stored by
    # train() — covers new votes, changed votes, and deletions.
    stored = get_state(conn, "votes_fingerprint") or "0:0"
    return stored != _votes_fingerprint(conn)


def train(conn: sqlite3.Connection, cfg: Config) -> bool:
    """Train the configured preference model. Returns True if training ran."""
    if cfg.scoring_method == "logistic":
        ok = _train_logistic(conn, cfg)
    else:
        ok = _train_centroid(conn, cfg)
    set_state(conn, "last_train_at", str(int(time.time())))
    set_state(conn, "votes_fingerprint", _votes_fingerprint(conn))
    conn.commit()
    log.info("training complete; last_train_at updated")
    return ok


def score_all(conn: sqlite3.Connection, cfg: Config, embedder: Embedder) -> int:
    """Score every news item and write ``model_score``. Returns count scored."""
    ids = list_all_item_ids(conn)
    if not ids:
        return 0

    # Make sure every item has an embedding before scoring.
    from embed import compute_missing_embeddings

    compute_missing_embeddings(conn, embedder)

    if cfg.scoring_method == "logistic":
        clf = _load_logistic(cfg)
        if clf is None:
            log.info(
                "logistic classifier not available; using centroid scoring for this run"
            )
            method = "centroid"
        else:
            method = "logistic"
    else:
        method = "centroid"

    liked, disliked = (None, None)
    if method == "centroid":
        liked, disliked = _load_centroids(cfg)

    scored = 0
    for item_id in ids:
        emb = load_embedding(conn, item_id)
        if emb is None:
            continue
        if method == "logistic":
            score = _score_logistic(emb, clf)
        else:
            score = _score_centroid(emb, liked, disliked)
        update_model_score(conn, item_id, score)
        scored += 1
    conn.commit()
    log.info("scored %d item(s) using %s strategy", scored, method)
    return scored


def apply_exploration(
    conn: sqlite3.Connection, rng: random.Random | None = None
) -> int:
    """Recompute ``rank_score`` for every item (epsilon-greedy exploration).

    ``rank_score = model_score`` normally; but for a randomly selected
    ``explore_pct``% of *unvoted, low-scored* items (``model_score`` NULL or
    below the median of unvoted items) a synthetic ``rank_score`` is drawn
    uniformly from the top decile of current scores — so a slice of
    low-ranked items still interleaves with good items in the score-sorted
    feed and the model gets fresh signals instead of sitting in a local
    maximum. Returns the number of boosted items.
    """
    rng = rng or random.Random()
    pct = get_explore_pct(conn)

    rows = conn.execute(
        """
        SELECT ni.id, ni.model_score,
               EXISTS(SELECT 1 FROM votes v WHERE v.news_item_id = ni.id) AS voted
        FROM news_items ni
        """
    ).fetchall()
    if not rows:
        return 0

    scores = sorted(
        float(r["model_score"]) for r in rows if r["model_score"] is not None
    )
    unvoted = [r for r in rows if not r["voted"]]
    unvoted_scores = sorted(
        float(r["model_score"]) for r in unvoted if r["model_score"] is not None
    )
    median = unvoted_scores[len(unvoted_scores) // 2] if unvoted_scores else None

    eligible = [
        r
        for r in unvoted
        if r["model_score"] is None
        or (median is not None and float(r["model_score"]) < median)
    ]

    # Synthetic boosts are drawn from the top decile of all current scores.
    lo = hi = None
    if scores:
        top = scores[-max(1, len(scores) // 10) :]
        lo, hi = top[0], top[-1]

    boosted: set[int] = set()
    if pct > 0 and eligible and lo is not None:
        count = round(len(eligible) * pct / 100)
        for r in rng.sample(eligible, count):
            update_rank_score(
                conn, r["id"], rng.uniform(lo, hi) if hi > lo else lo
            )
            boosted.add(r["id"])

    for r in rows:
        if r["id"] not in boosted:
            update_rank_score(conn, r["id"], r["model_score"])

    conn.commit()
    log.info(
        "exploration: %d of %d eligible low-ranked item(s) boosted (explore_pct=%.1f)",
        len(boosted),
        len(eligible),
        pct,
    )
    return len(boosted)
