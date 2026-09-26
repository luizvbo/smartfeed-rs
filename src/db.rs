use std::collections::HashSet;

use anyhow::Context;

pub async fn init_db(url: &str) -> anyhow::Result<toasty::Db> {
    tracing::info!(%url, "connecting to database");

    let mut db = toasty::Db::builder()
        .models(toasty::models!(crate::*))
        .connect(url)
        .await
        .with_context(|| format!("failed to connect to database at {url}"))?;

    // Enable WAL mode so the Python pipeline can write concurrently.
    // PRAGMA returns a row, so use query() rather than statement().
    toasty::sql::query("PRAGMA journal_mode=WAL;")
        .exec(&mut db)
        .await
        .context("failed to enable WAL mode")?;
    tracing::info!("enabled SQLite WAL mode");

    // Wait instead of erroring with SQLITE_BUSY while the pipeline holds a
    // write transaction (e.g. during ingestion or scoring).
    toasty::sql::query("PRAGMA busy_timeout=30000;")
        .exec(&mut db)
        .await
        .context("failed to set busy_timeout")?;
    tracing::info!("set SQLite busy_timeout");

    ensure_schema(&mut db).await?;

    Ok(db)
}

async fn ensure_schema(db: &mut toasty::Db) -> anyhow::Result<()> {
    let rows = toasty::sql::query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('news_items','feeds','votes')",
    )
    .exec(db)
    .await
    .context("failed to query sqlite_master")?;

    let existing: HashSet<String> = rows
        .iter()
        .filter_map(|row| {
            row.as_record()
                .and_then(|record| record.first())
                .and_then(|value| value.as_str())
                .map(String::from)
        })
        .collect();

    if existing.is_empty() {
        // Clean database: let toasty push its full schema.
        db.push_schema().await.context("failed to push schema")?;
        tracing::info!("created schema with toasty push_schema");
    } else if existing.len() < 3 {
        // The Python pipeline may have created some tables before the Rust app
        // starts. Use idempotent CREATE statements to fill in the rest without
        // touching existing tables.
        tracing::info!(missing = 3 - existing.len(), "filling in missing tables");
        ensure_table(db, "news_items", NEWS_ITEMS_DDL).await?;
        ensure_table(db, "feeds", FEEDS_DDL).await?;
        ensure_table(db, "votes", VOTES_DDL).await?;
    } else {
        tracing::info!("schema already present");
    }

    Ok(())
}

async fn ensure_table(db: &mut toasty::Db, name: &str, ddl: &[&str]) -> anyhow::Result<()> {
    for stmt in ddl {
        toasty::sql::statement(*stmt)
            .exec(db)
            .await
            .with_context(|| format!("failed to ensure table {name}"))?;
    }
    Ok(())
}

const NEWS_ITEMS_DDL: &[&str] = &[
    r#"CREATE TABLE IF NOT EXISTS "news_items" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "title" TEXT NOT NULL,
    "url" TEXT NOT NULL,
    "source_feed" TEXT NOT NULL,
    "summary" TEXT,
    "image_url" TEXT,
    "published_at" BIGINT,
    "fetched_at" BIGINT NOT NULL,
    "model_score" REAL,
    "opened_at" BIGINT
);"#,
    r#"CREATE UNIQUE INDEX IF NOT EXISTS "index_news_items_by_url" ON "news_items" ("url");"#,
];

const FEEDS_DDL: &[&str] = &[
    r#"CREATE TABLE IF NOT EXISTS "feeds" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "url" TEXT NOT NULL,
    "title" TEXT,
    "last_fetch" BIGINT,
    "last_status" TEXT,
    "error_message" TEXT,
    "is_active" BOOLEAN NOT NULL
);"#,
    r#"CREATE UNIQUE INDEX IF NOT EXISTS "index_feeds_by_url" ON "feeds" ("url");"#,
];

const VOTES_DDL: &[&str] = &[
    r#"CREATE TABLE IF NOT EXISTS "votes" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "news_item_id" INTEGER NOT NULL,
    "vote" TEXT NOT NULL,
    "created_at" BIGINT NOT NULL
);"#,
    r#"CREATE UNIQUE INDEX IF NOT EXISTS "index_votes_by_news_item_id" ON "votes" ("news_item_id");"#,
];
