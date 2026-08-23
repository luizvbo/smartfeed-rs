# SmartFeed

A mobile-first, server-rendered personal news reader built in Rust. It reads from a SQLite database that is shared with a separate Python data pipeline, so the Rust layer is intentionally thin and read-mostly.

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

Because the Python pipeline is not part of this repo, a seed binary is provided to insert sample rows so the UI is visible immediately:

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
- `GET /feeds` - list of configured feeds
- `GET /static/*` - static assets served from `static/`

Query parameters:

- `filter=all|opened` (default `all`)
- `sort=new|score` (default `new`)
- `page` (1-indexed, default `1`, page size `15`)

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

- `db.push_schema()` is used for the MVP. For a real deployment, switch to toasty's migration workflow.
- `toasty::Db` is held in an `Arc` and cloned per handler because it is cheap to clone.
