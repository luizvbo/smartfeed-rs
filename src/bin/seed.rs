use anyhow::Context;
use smartfeed::db::init_db;
use smartfeed::models::{Feed, NewsItem, Vote};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let database_url =
        std::env::var("DATABASE_URL").unwrap_or_else(|_| "sqlite:./news.db".to_string());

    let mut db = init_db(&database_url)
        .await
        .context("failed to initialize database")?;

    let now = time::OffsetDateTime::now_utc().unix_timestamp();

    let feeds = [
        ("https://example.com/feed.xml", "Example News"),
        ("https://example.org/feed.rss", "Example Blog"),
    ];

    for (url, title) in feeds {
        Feed::upsert_by_url(url)
            .title(Some(title.to_string()))
            .is_active(true)
            .exec(&mut db)
            .await
            .context("failed to seed feed")?;
    }

    let items = [
        (
            "Rust 1.96 released with new features",
            "https://example.com/rust-1-96",
            "Example News",
            Some("The latest Rust release brings performance improvements and new language features."),
            Some("https://example.com/rust.jpg"),
            now - 3600,
            now,
            Some(0.92),
        ),
        (
            "Async Rust in 2026",
            "https://example.com/async-rust-2026",
            "Example News",
            Some("A look at how async runtimes have evolved and what is next for the ecosystem."),
            None,
            now - 7200,
            now - 120,
            Some(0.85),
        ),
        (
            "SQLite concurrency patterns",
            "https://example.com/sqlite-concurrency",
            "Example Blog",
            Some("How WAL mode enables readers and writers to share the same database file."),
            Some("https://example.com/sqlite.jpg"),
            now - 86400,
            now - 300,
            Some(0.78),
        ),
        (
            "Mobile-first web design",
            "https://example.org/mobile-first",
            "Example Blog",
            Some("Designing for small screens first creates faster, more focused experiences."),
            None,
            now - 90000,
            now - 600,
            Some(0.65),
        ),
        (
            "HTMX vs full SPA frameworks",
            "https://example.org/htmx-spa",
            "Example Blog",
            Some("A comparison of hypermedia-driven applications and modern JavaScript frameworks."),
            Some("https://example.org/htmx.png"),
            now - 172800,
            now - 1800,
            None,
        ),
        (
            "Training smaller language models",
            "https://example.com/small-llms",
            "Example News",
            Some("Smaller models can be surprisingly capable when trained on high-quality data."),
            Some("https://example.com/llm.jpg"),
            now - 200000,
            now - 2400,
            Some(0.91),
        ),
        (
            "Open source sustainability",
            "https://example.org/oss-sustainability",
            "Example Blog",
            Some("Funding models for open source projects continue to evolve."),
            None,
            now - 260000,
            now - 3600,
            Some(0.55),
        ),
        (
            "A new database ORM for Rust",
            "https://example.com/toasty-orm",
            "Example News",
            Some("Toasty is an async ORM that supports SQLite, PostgreSQL, MySQL, and DynamoDB."),
            Some("https://example.com/toasty.jpg"),
            now - 300000,
            now - 4000,
            Some(0.88),
        ),
    ];

    for (title, url, source, summary, image, published, fetched, score) in items {
        NewsItem::upsert_by_url(url)
            .title(title.to_string())
            .source_feed(source.to_string())
            .summary(summary.map(String::from))
            .image_url(image.map(String::from))
            .published_at(Some(published))
            .fetched_at(fetched)
            .model_score(score)
            .opened_at(None)
            .exec(&mut db)
            .await
            .context("failed to seed news item")?;
    }

    // Add a couple of votes for the first two items so the UI shows the state.
    for (news_item_id, vote) in [(1, "up"), (2, "down")] {
        Vote::upsert_by_news_item_id(news_item_id)
            .vote(vote.to_string())
            .created_at(now)
            .exec(&mut db)
            .await
            .context("failed to seed vote")?;
    }

    println!(
        "Seeded {} news items and {} feeds.",
        items.len(),
        feeds.len()
    );

    Ok(())
}
