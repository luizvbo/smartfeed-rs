# SmartFeed

A mobile-first, server-rendered personal news reader built in Rust. It reads from a SQLite database that is shared with a companion Python data pipeline in [`pipeline/`](pipeline/), so the Rust layer is intentionally thin and read-mostly.

## Tech stack

- Rust + axum 0.8
- Askama templates
- HTMX for interactivity
- toasty ORM with the SQLite driver
- tower-http for static files and tracing
- anyhow + tracing

## Running

```bash
cargo build
cargo run
```

The server listens on `127.0.0.1:3000` by default. Open http://127.0.0.1:3000 in a browser.

## Configuration (environment variables)

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:./news.db` | SQLite connection URL |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `3000` | Port |
| `RUST_LOG` | `info,tower_http=debug` | tracing-subscriber filter, e.g. `info`, `smartfeed=debug`, or `tower_http=trace` |

## Seeding sample data

If you have not run the Python pipeline yet (see `pipeline/README.md` for ingestion and scoring), a seed binary is provided to insert sample rows so the UI is visible immediately:

```bash
cargo run --bin seed
```

It upserts sample `Feed` and `NewsItem` rows and a couple of `Vote` rows, so it is safe to run more than once. It does **not** run automatically on startup.

## Database

The SQLite file path comes from `DATABASE_URL` (default `./news.db`). On startup the app:

1. Connects with toasty.
2. Runs `PRAGMA journal_mode=WAL;` so the Python pipeline can write concurrently with the web app.
3. Initializes the schema without resetting data:
   - On an empty database, it calls `db.push_schema()`.
   - If the Python pipeline has already created some of the tables, it uses idempotent `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` statements to create any missing tables.

The expected schema is defined by the toasty models in `src/models.rs`. The Python pipeline is expected to insert rows into `news_items` and `feeds` using the same column layout; the Rust app owns the `votes` table.

## Endpoints

- `GET /` - home page with filter tabs and paginated cards
- `GET /items?page&sort&filter` - HTMX fragment for the next page of cards
- `GET /items/{id}` - detail page for a single item
- `POST /items/{id}/vote` - vote `up` or `down` (returns re-rendered card for HTMX, redirect otherwise)
- `GET /read/{id}` - marks the item as opened (sets `opened_at`) and returns an HTTP 302 redirect to the original URL
- `GET /feeds` - list of configured feeds (with add-feed form and per-feed enable/disable toggle)
- `POST /feeds` - register a feed URL (the pipeline ingests it on its next run)
- `POST /feeds/{id}/toggle` - flip `feeds.is_active`; the pipeline skips disabled feeds
- `POST /settings` - saves `explore_pct` (0-100) to `pipeline_state` for the pipeline's exploration step; the "Rescore now" button additionally sets `rescore_requested=1`, which the pipeline clears and honors by forcing retrain+score on its next loop iteration
- `GET /static/*` - static assets served from `static/`

Query parameters:

- `filter=all|opened|voted` (default `all`)
- `sort=new|score` (default `new`)
- `page` (1-indexed, default `1`, page size `15`)

## Exploration (epsilon-greedy ranking)

To avoid filter-bubble local maxima, the "Top score" sort doesn't rank by `model_score` alone. The pipeline maintains a `rank_score` column: equal to `model_score` normally, but a configurable percentage (`explore_pct`, default 10%) of unvoted, low-scored items receives a synthetic boost into the top-decile range so they still surface and can collect votes.

- `explore_pct` is set from the app — the small form under the filter bar writes it to `pipeline_state` (`POST /settings`). The pipeline reads it on every run, so no restart is needed on either side.
- `sort=score` orders by `COALESCE(rank_score, model_score) DESC` — backfill-safe for rows scored before the column existed.
- Boosted cards show a subtle `explore` badge; the `Score:` label always keeps showing the true `model_score`.

## Filters

- `all` (default) hides items that already have a vote — rated items don't resurface.
- `opened` lists items the user clicked through (`opened_at` set) and has not yet voted on, most recent first — for rating after returning from an external tab.
- `voted` shows only items that have a vote.

## Mobile-first design notes

- The viewport meta tag in `base.html` makes the app phone-friendly.
- Cards are a single vertical column on narrow screens and widen to a two-column grid on desktop.
- Touch targets for vote buttons and "Read more" are at least 48×48 px.
- The primary pagination pattern is a large "Load more" button at the bottom. It uses HTMX to append the next page in place; the `href` provides a no-JS fallback to `/?page=N&filter=...&sort=...`.
- The "Opened" filter lists items the user has already clicked through (`opened_at` set) and has not yet voted on, sorted by most recently opened. This makes it easy to rate an article after returning from an external browser tab.
- No sticky or floating bars obscure content on small screens.

## HTMX behavior

- Vote forms use `hx-post`, `hx-target="closest .news-card"`, and `hx-swap="outerHTML"`.
- The "Load more" button uses `hx-get`, `hx-target="this"`, and `hx-swap="outerHTML"`.
- Images are lazy-loaded and have a one-line `onerror` handler that hides broken images and shows the CSS placeholder.

## Production notes

- `db.push_schema()` is used for the MVP. For a real deployment, switch to toasty's migration workflow. Columns added later (e.g. `rank_score`) are covered by an idempotent `PRAGMA table_info` + `ALTER TABLE` check in `init_db`.
- `toasty::Db` is held in an `Arc` and cloned per handler because it is cheap to clone.

## Running on a home server (systemd)

`deploy/` contains two unit files:

- `deploy/smartfeed-web.service` — runs `target/release/smartfeed` with `DATABASE_URL`/`HOST`/`PORT` env, `Restart=always`.
- `deploy/smartfeed-pipeline.service` — runs `uv run --directory pipeline python pipeline/main.py --loop 300` with `WorkingDirectory` at the repo root, `Restart=always`.

```bash
cargo build --release
sudo cp deploy/smartfeed-{web,pipeline}.service /etc/systemd/system/
sudo $EDITOR /etc/systemd/system/smartfeed-*.service   # adjust paths/User
sudo systemctl daemon-reload
sudo systemctl enable --now smartfeed-web smartfeed-pipeline
```

For LAN access, set `HOST=0.0.0.0` in the web unit. If the app is exposed beyond the LAN, put it behind Caddy (TLS) or Tailscale and add authentication — the app has no built-in auth and should not be wide open on the internet.
