use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use axum::routing::{get, post};
use axum::Router;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use smartfeed::db::init_db;
use smartfeed::handlers::{
    add_feed, feeds, index, item_detail, items_fragment, read_item, toggle_feed, update_settings,
    vote_item, AppState,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        "info,tower_http=debug,toasty=warn"
            .parse()
            .expect("default filter is valid")
    });

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .init();

    let database_url =
        std::env::var("DATABASE_URL").unwrap_or_else(|_| "sqlite:./news.db".to_string());
    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(3000);

    let db = init_db(&database_url)
        .await
        .context("failed to initialize database")?;

    let state = AppState { db: Arc::new(db) };

    let app = Router::new()
        .route("/", get(index))
        .route("/items", get(items_fragment))
        .route("/items/{id}", get(item_detail))
        .route("/items/{id}/vote", post(vote_item))
        .route("/read/{id}", get(read_item))
        .route("/feeds", get(feeds))
        .route("/feeds", post(add_feed))
        .route("/feeds/{id}/toggle", post(toggle_feed))
        .route("/settings", post(update_settings))
        .nest_service("/static", ServeDir::new("static"))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .context("invalid host or port")?;

    tracing::info!(%addr, %database_url, "starting server");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await.context("server error")?;

    Ok(())
}
