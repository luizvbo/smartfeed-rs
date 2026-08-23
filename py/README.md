# SmartFeed Python pipeline

This directory contains the data pipeline for SmartFeed. It ingests RSS/Atom
feeds, computes multilingual sentence embeddings, trains a lightweight
preference model on the user's votes, and updates `news_items.model_score` in
the shared SQLite database (`news.db`). The Rust axum web app reads those rows
and writes `votes` back; the Python pipeline never touches the `votes` table
except to read it for training.

## Layout

```
py/
├── main.py             # entry point / CLI
├── config.py           # env + feeds.toml loading
├── db.py               # SQLite schema and helpers
├── rss.py              # feed fetching and parsing
├── embed.py            # sentence-transformer embeddings + cache
├── model.py            # centroid / logistic preference model
├── pyproject.toml      # pip install -e py/
├── requirements.txt    # alternative flat dependency list
├── feeds.toml.example  # copy to feeds.toml and edit
└── README.md
```

## Install

Either install the dependencies directly:

```bash
pip install -r py/requirements.txt
```

or install the project in editable mode (also pulls in the same deps):

```bash
pip install -e py/
```

For the optional logistic scoring strategy, also install scikit-learn:

```bash
pip install "scikit-learn>=1.4"
# or
pip install -e "py/[logistic]"
```

## Configure feeds

```bash
cp py/feeds.toml.example py/feeds.toml
$EDITOR py/feeds.toml
```

Each `[[feeds]]` block takes a `url` (required) and an optional `title`:

```toml
[[feeds]]
url = "https://hnrss.org/frontpage"
title = "Hacker News"
```

## Run

Run from the **repository root** (not from inside `py/`):

```bash
python py/main.py                 # run once
python py/main.py --loop 300      # loop every 300 seconds
python py/main.py --retrain       # force retrain of the preference model
python py/main.py --dry-run       # do not write to the DB
```

Running from the repo root makes the default database path `news.db` resolve
correctly next to the Rust app. If you run from inside `py/`, the default
becomes `../news.db`. You can always override with the `DB_PATH` environment
variable:

```bash
DB_PATH=/path/to/news.db python py/main.py
```

## Environment variables

| Variable           | Default                                 | Description                                             |
| ------------------ | --------------------------------------- | ------------------------------------------------------- |
| `DB_PATH`          | `news.db` (or `../news.db` from `py/`)  | SQLite database file                                    |
| `MODEL_NAME`       | `paraphrase-multilingual-MiniLM-L12-v2` | sentence-transformers model name                        |
| `BATCH_SIZE`       | `32`                                    | embedding batch size                                    |
| `MAX_AGE_DAYS`     | `30`                                    | skip feed entries older than this                       |
| `RETRAIN_INTERVAL` | `0`                                     | retrain at most every N seconds (0 = only on new votes) |
| `LOG_LEVEL`        | `INFO`                                  | logging level                                           |
| `SCORING_METHOD`   | `centroid`                              | `centroid` (default) or `logistic` (needs scikit-learn) |

`python-dotenv` is supported: put variables in `py/.env` and they will be
loaded automatically when running the pipeline.

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

Centroids are cached as `.npy` files in `py/.cache/`.

### Logistic scoring (optional, needs scikit-learn)

Set `SCORING_METHOD=logistic`. Fits a `LogisticRegression` on the voted
embeddings and uses `predict_proba` for the `up` class as the score. The
classifier is pickled into `py/.cache/`. Centroids are still refreshed as a
fallback scorer.

Retraining is skipped unless new votes have arrived since `last_train_at`,
`RETRAIN_INTERVAL` has elapsed, or `--retrain` is passed.

## Shared schema with the Rust app

The Rust web app (repo root) reads `news_items` and `feeds`, and writes
`votes`. The Python pipeline writes `news_items`, `feeds`, and
`news_item_embeddings`, and reads `votes` for training. Both layers share the
same SQLite file and rely on WAL mode for concurrent access.
