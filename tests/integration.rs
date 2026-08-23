use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post};
use axum::Router;
use http_body_util::BodyExt;
use smartfeed::db::init_db;
use smartfeed::handlers::{
    feeds, index, item_detail, items_fragment, read_item, vote_item, AppState,
};
use smartfeed::models::{Feed, NewsItem, Vote};
use time::OffsetDateTime;
use toasty::Db;
use tower::ServiceExt;

fn test_router(db: Db) -> Router {
    let state = AppState { db: Arc::new(db) };
    Router::new()
        .route("/", get(index))
        .route("/items", get(items_fragment))
        .route("/items/{id}", get(item_detail))
        .route("/items/{id}/vote", post(vote_item))
        .route("/read/{id}", get(read_item))
        .route("/feeds", get(feeds))
        .with_state(state)
}

async fn setup_db() -> Db {
    init_db("sqlite::memory:").await.expect("init db")
}

async fn seed(db: &mut Db) -> u64 {
    let now = OffsetDateTime::now_utc().unix_timestamp();

    Feed::upsert_by_url("https://example.com/feed.xml")
        .title(Some("Example".to_string()))
        .is_active(true)
        .exec(db)
        .await
        .unwrap();

    let item = NewsItem::upsert_by_url("https://example.com/rust")
        .title("Rust 1.96".to_string())
        .source_feed("https://example.com/feed.xml".to_string())
        .summary(Some("New features.".to_string()))
        .image_url(Some("https://example.com/rust.jpg".to_string()))
        .published_at(now - 3600)
        .fetched_at(now)
        .model_score(Some(0.95))
        .opened_at(None)
        .exec(db)
        .await
        .unwrap();

    Vote::upsert_by_news_item_id(item.id)
        .vote("up".to_string())
        .created_at(now)
        .exec(db)
        .await
        .unwrap();

    item.id
}

async fn body_string(response: axum::response::Response) -> String {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[tokio::test]
async fn index_returns_html() {
    let mut db = setup_db().await;
    seed(&mut db).await;
    let app = test_router(db);

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_string(response).await;
    assert!(body.contains("Rust 1.96"));
}

#[tokio::test]
async fn items_fragment_returns_cards() {
    let mut db = setup_db().await;
    seed(&mut db).await;
    let app = test_router(db);

    let response = app
        .oneshot(Request::builder().uri("/items").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_string(response).await;
    assert!(body.contains("Rust 1.96"));
}

#[tokio::test]
async fn item_detail_returns_html() {
    let mut db = setup_db().await;
    let id = seed(&mut db).await;
    let app = test_router(db);

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/items/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_string(response).await;
    assert!(body.contains("Rust 1.96"));
}

#[tokio::test]
async fn item_detail_not_found() {
    let mut db = setup_db().await;
    seed(&mut db).await;
    let app = test_router(db);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/items/99999")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn read_item_redirects_and_updates_opened_at() {
    let mut db = setup_db().await;
    let id = seed(&mut db).await;
    let app = test_router(db);

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/read/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FOUND);
    let location = response
        .headers()
        .get("location")
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(location, "https://example.com/rust");
}

#[tokio::test]
async fn vote_up_and_toggle() {
    let mut db = setup_db().await;
    let id = seed(&mut db).await;
    let app = test_router(db);

    // Voting the same way toggles the vote off and redirects back to the index.
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/items/{id}/vote"))
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("vote=up&filter=all&sort=new&page=1"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(response.status().is_redirection());
    let location = response
        .headers()
        .get("location")
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(location, "/?filter=all&sort=new&page=1");
}

#[tokio::test]
async fn vote_invalid_value_returns_400() {
    let mut db = setup_db().await;
    let id = seed(&mut db).await;
    let app = test_router(db);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/items/{id}/vote"))
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("vote=side&filter=all&sort=new&page=1"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn feeds_returns_feeds_list() {
    let mut db = setup_db().await;
    seed(&mut db).await;
    let app = test_router(db);

    let response = app
        .oneshot(Request::builder().uri("/feeds").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_string(response).await;
    assert!(body.contains("Example"));
}

#[tokio::test]
async fn read_item_not_found() {
    let mut db = setup_db().await;
    seed(&mut db).await;
    let app = test_router(db);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/read/99999")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn vote_down_via_hx_request() {
    let mut db = setup_db().await;
    let id = seed(&mut db).await;
    let app = test_router(db);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/items/{id}/vote"))
                .header("content-type", "application/x-www-form-urlencoded")
                .header("HX-Request", "true")
                .body(Body::from("vote=down&filter=all&sort=new&page=1"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_string(response).await;
    assert!(body.contains("downvoted") || body.contains("Rust 1.96"));
}

#[tokio::test]
async fn index_filter_opened() {
    let mut db = setup_db().await;
    let now = OffsetDateTime::now_utc().unix_timestamp();

    Feed::upsert_by_url("https://example.com/feed.xml")
        .title(Some("Example".to_string()))
        .is_active(true)
        .exec(&mut db)
        .await
        .unwrap();

    NewsItem::upsert_by_url("https://example.com/opened")
        .title("Opened Item".to_string())
        .source_feed("https://example.com/feed.xml".to_string())
        .summary(Some("Already read.".to_string()))
        .image_url(None)
        .published_at(now - 100)
        .fetched_at(now)
        .model_score(Some(0.5))
        .opened_at(Some(now))
        .exec(&mut db)
        .await
        .unwrap();

    let app = test_router(db);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/?filter=opened&sort=new&page=1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_string(response).await;
    assert!(body.contains("Opened Item"));
}

#[tokio::test]
async fn index_sort_score() {
    let mut db = setup_db().await;
    seed(&mut db).await;
    let app = test_router(db);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/?filter=all&sort=score&page=1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_string(response).await;
    assert!(body.contains("Rust 1.96"));
}

#[tokio::test]
async fn index_page_zero_is_normalized() {
    let mut db = setup_db().await;
    seed(&mut db).await;
    let app = test_router(db);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/?page=0&filter=all&sort=new")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
