# SmartFeed Python pipeline

This directory contains the data pipeline for SmartFeed. It ingests RSS/Atom
feeds, computes multilingual sentence embeddings, trains a lightweight
preference model on the user's votes, and updates `news_items.model_score` in
the shared SQLite database (`news.db`). The Rust axum web app reads those rows
and writes `votes` back; the Python pipeline never touches the `votes` table
except to read it for training.

## Layout

```
pipeline/
├── main.py             # entry point / CLI
├── config.py           # env + feeds.toml loading
├── db.py               # SQLite schema and helpers
├── rss.py              # feed fetching and parsing
├── embed.py            # sentence-transformer embeddings + cache
├── model.py            # centroid / logistic preference model
├── pyproject.toml      # project metadata + dependencies (uv)
├── uv.lock             # locked dependency versions
├── feeds.toml.example  # copy to feeds.toml and edit
└── README.md
```

## Install

Dependencies are managed with [uv](https://docs.astral.sh/uv/). From the repo
root:

```bash
uv sync --directory pipeline            # runtime deps
uv sync --directory pipeline --group dev  # + pytest for tests
```

For the optional logistic scoring strategy, also install scikit-learn:

```bash
uv sync --directory pipeline --extra logistic
```

## Configure feeds

```bash
cp pipeline/feeds.toml.example pipeline/feeds.toml
$EDITOR pipeline/feeds.toml
```

Each `[[feeds]]` block takes a `url` (required) and an optional `title`:

```toml
[[feeds]]
url = "https://hnrss.org/frontpage"
title = "Hacker News"
```

## Run

Run from the **repository root** (not from inside `pipeline/`):

```bash
uv run --directory pipeline python pipeline/main.py                 # run once
uv run --directory pipeline python pipeline/main.py --loop 300      # loop every 300 seconds
uv run --directory pipeline python pipeline/main.py --retrain       # force retrain
uv run --directory pipeline python pipeline/main.py --dry-run       # no DB writes
```

Or activate `pipeline/.venv` and run `python pipeline/main.py` directly.

Running from the repo root makes the default database path `news.db` resolve
correctly next to the Rust app. If you run from inside `pipeline/`, the default
becomes `../news.db`. You can always override with the `DB_PATH` environment
variable:

```bash
DB_PATH=/path/to/news.db uv run --directory pipeline python pipeline/main.py
```

## Environment variables

| Variable           | Default                                       | Description                                             |
| ------------------ | --------------------------------------------- | ------------------------------------------------------- |
| `DB_PATH`          | `news.db` (or `../news.db` from `pipeline/`)  | SQLite database file                                    |
| `MODEL_NAME`       | `paraphrase-multilingual-MiniLM-L12-v2` | sentence-transformers model name                        |
| `BATCH_SIZE`       | `32`                                    | embedding batch size                                    |
| `MAX_AGE_DAYS`     | `30`                                    | skip feed entries older than this                       |
| `RETENTION_DAYS`   | `60`                                    | delete `news_items` older than this (never voted/opened) |
| `RETRAIN_INTERVAL` | `0`                                     | retrain at most every N seconds (0 = only on new votes) |
| `LOG_LEVEL`        | `INFO`                                  | logging level                                           |
| `SCORING_METHOD`   | `centroid`                              | `centroid` (default) or `logistic` (needs scikit-learn) |

`python-dotenv` is supported: put variables in `pipeline/.env` and they will
be loaded automatically when running the pipeline.

## Database

The pipeline connects to the same SQLite file the Rust web app uses. It:

- Enables `PRAGMA journal_mode=WAL` so the web app and the pipeline can run
  concurrently.
- Creates `news_items`, `feeds`, and `votes` with `CREATE TABLE IF NOT EXISTS`
  so the Rust app's schema is never overwritten. Column names match
  `src/models.rs` exactly.
- Owns two Python-only tables: `news_item_embeddings` (cached embeddings) and
  `pipeline_state` (key/value state such as `last_train_at`).
- Never drops or truncates `news_items`, `feeds`, or `votes`.

## Model

The sentence-transformer (`paraphrase-multilingual-MiniLM-L12-v2` by default)
is **frozen**. The pipeline only trains a lightweight preference model on top
of the cached embeddings.

### Centroid scoring (default, no extra deps)

- `liked_centroid` = mean embedding of items voted `up`.
- `disliked_centroid` = mean embedding of items voted `down`.
- Score = `cosine(item, liked) - cosine(item, disliked)`, clipped to `[-1, 1]`.
- Falls back gracefully when only one side has votes, or when there are no
  votes yet (score = 0).

Centroids are cached as `.npy` files in `pipeline/.cache/`.

### Logistic scoring (optional, needs scikit-learn)

Set `SCORING_METHOD=logistic`. Fits a `LogisticRegression` on the voted
embeddings and uses `predict_proba` for the `up` class as the score. The
classifier is pickled into `pipeline/.cache/`. Centroids are still refreshed as a
fallback scorer.

Retraining is skipped unless the votes table changed since the last run
(inserted, changed, or deleted votes — tracked via a `votes_fingerprint` key
in `pipeline_state`), `RETRAIN_INTERVAL` has elapsed, or `--retrain` is passed.

## Exploration (epsilon-greedy ranking)

After every `score_all`, the pipeline recomputes `news_items.rank_score`:

- Normally `rank_score = model_score`.
- For a randomly selected `explore_pct`% of *unvoted, low-scored* items
  (`model_score` NULL or below the median of unvoted items), `rank_score` is
  drawn uniformly from the top decile of current scores — so a slice of
  low-ranked items still surfaces in the score-sorted feed and the model gets
  fresh signals instead of sitting in a local maximum.

`explore_pct` lives in `pipeline_state` and is written by the web app
(`POST /settings`, percent 0–100, default 10) — the pipeline re-reads it on
each run, so changes take effect without restarting anything. Voted items are
never boosted. The web app orders "Top score" by
`COALESCE(rank_score, model_score)` and badges boosted cards `explore` while
showing the real `model_score`.

## Retention cleanup

Each run ends with `cleanup_expired_items`: `news_items` older than
`RETENTION_DAYS` (by `COALESCE(published_at, fetched_at)`) are deleted —
**unless** they have a vote or `opened_at` set. Embeddings cascade via the FK;
votes have no FK so orphan votes are deleted explicitly. A
`PRAGMA wal_checkpoint(TRUNCATE)` follows the cleanup.

## Shared schema with the Rust app

The Rust web app (repo root) reads `news_items` and `feeds`, and writes
`votes`. The Python pipeline writes `news_items`, `feeds`, and
`news_item_embeddings`, and reads `votes` for training. Both layers share the
same SQLite file and rely on WAL mode for concurrent access.
