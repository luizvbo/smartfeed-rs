use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{anyhow, Context};
use askama::Template;
use axum::extract::{Form, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Redirect, Response};
use serde::Deserialize;

use crate::models::{Feed, NewsItem, Vote};
use crate::templates::{
    CardContext, CardsTemplate, DetailTemplate, ErrorTemplate, FeedContext, FeedsTemplate,
    IndexTemplate, ItemCardTemplate, QueryParams,
};

const PAGE_SIZE: usize = 15;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<toasty::Db>,
}

#[derive(Debug)]
pub struct AppError {
    status: StatusCode,
    err: anyhow::Error,
}

impl AppError {
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            err: anyhow!(msg.into()),
        }
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            err: anyhow!(msg.into()),
        }
    }

    pub fn internal(err: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            err,
        }
    }

    fn from_toasty(err: toasty::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            err: anyhow!(err.to_string()),
        }
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        Self::internal(err)
    }
}

impl From<toasty::Error> for AppError {
    fn from(err: toasty::Error) -> Self {
        Self::from_toasty(err)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        tracing::error!(status = %self.status, error = %self.err, "request failed");
        let body = ErrorTemplate {
            message: self.err.to_string(),
        }
        .render()
        .unwrap_or_else(|_| "Internal server error".to_string());
        (self.status, Html(body)).into_response()
    }
}

pub type AppResult<T> = Result<T, AppError>;

fn render(template: impl askama::Template) -> AppResult<Html<String>> {
    let html = template
        .render()
        .map_err(|e| AppError::internal(anyhow!("template render error: {e}")))?;
    Ok(Html(html))
}

pub async fn index(
    State(state): State<AppState>,
    Query(query): Query<QueryParams>,
) -> AppResult<Html<String>> {
    let mut db = state.db.as_ref().clone();
    let query = query.normalize();
    let (items, has_more) = list_cards(&mut db, &query).await?;
    let explore_pct = read_explore_pct(&mut db).await;
    let next_page = query.page() + 1;
    let next_path = query.path_for(next_page);
    let next_query = query.query_for(next_page);
    render(IndexTemplate {
        items,
        query,
        has_more,
        next_page,
        next_path,
        next_query,
        explore_pct,
    })
}

pub async fn items_fragment(
    State(state): State<AppState>,
    Query(query): Query<QueryParams>,
) -> AppResult<Html<String>> {
    let mut db = state.db.as_ref().clone();
    let query = query.normalize();
    let (items, has_more) = list_cards(&mut db, &query).await?;
    let next_page = query.page() + 1;
    let next_path = query.path_for(next_page);
    let next_query = query.query_for(next_page);
    render(CardsTemplate {
        items,
        query,
        has_more,
        next_page,
        next_path,
        next_query,
    })
}

pub async fn item_detail(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> AppResult<Response> {
    let mut db = state.db.as_ref().clone();
    let item = load_card(&mut db, id).await?;
    let html = render(DetailTemplate {
        item,
        query: QueryParams::default(),
    })?;
    Ok(html.into_response())
}

pub async fn read_item(State(state): State<AppState>, Path(id): Path<u64>) -> AppResult<Response> {
    let mut db = state.db.as_ref().clone();
    let mut item = NewsItem::filter_by_id(id)
        .first()
        .exec(&mut db)
        .await?
        .ok_or_else(|| AppError::not_found("news item not found"))?;

    let now = time::OffsetDateTime::now_utc().unix_timestamp();
    toasty::update!(item {
        opened_at: Some(now)
    })
    .exec(&mut db)
    .await?;

    tracing::info!(news_item_id = id, url = %item.url, "marked news item as read");

    let location = axum::http::HeaderValue::from_str(&item.url)
        .map_err(|_| AppError::bad_request("item URL is not a valid header value"))?;

    Ok(axum::response::Response::builder()
        .status(StatusCode::FOUND)
        .header(axum::http::header::LOCATION, location)
        .body(axum::body::Body::empty())
        .unwrap()
        .into_response())
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct VoteForm {
    pub vote: String,
    pub filter: String,
    pub sort: String,
    pub page: usize,
}

impl Default for VoteForm {
    fn default() -> Self {
        Self {
            vote: String::new(),
            filter: "all".to_string(),
            sort: "new".to_string(),
            page: 1,
        }
    }
}

pub async fn vote_item(
    State(state): State<AppState>,
    Path(id): Path<u64>,
    headers: HeaderMap,
    Form(form): Form<VoteForm>,
) -> AppResult<Response> {
    if form.vote != "up" && form.vote != "down" {
        return Err(AppError::bad_request("vote must be 'up' or 'down'"));
    }

    let mut db = state.db.as_ref().clone();

    // Ensure the news item exists before recording a vote.
    NewsItem::filter_by_id(id)
        .first()
        .exec(&mut db)
        .await?
        .ok_or_else(|| AppError::not_found("news item not found"))?;

    let now = time::OffsetDateTime::now_utc().unix_timestamp();
    let existing = Vote::filter_by_news_item_id(id)
        .first()
        .exec(&mut db)
        .await
        .context("failed to load existing vote")?;

    match existing.as_ref().map(|v| v.vote.as_str()) {
        Some(v) if v == form.vote => {
            // Clicking the same vote again toggles it off.
            Vote::filter_by_news_item_id(id)
                .delete()
                .exec(&mut db)
                .await
                .context("failed to delete vote")?;
            tracing::info!(news_item_id = id, "removed vote");
        }
        _ => {
            // New vote or changing from up->down / down->up.
            Vote::upsert_by_news_item_id(id)
                .vote(form.vote.clone())
                .created_at(now)
                .exec(&mut db)
                .await
                .context("failed to upsert vote")?;
            tracing::info!(news_item_id = id, vote = %form.vote, "recorded vote");
        }
    }

    if headers.get("HX-Request").and_then(|v| v.to_str().ok()) == Some("true") {
        let item = load_card(&mut db, id).await?;
        let query = QueryParams {
            filter: form.filter,
            sort: form.sort,
            page: if form.page == 0 { 1 } else { form.page },
        };
        let html = render(ItemCardTemplate { item, query })?;
        Ok(html.into_response())
    } else {
        let page = if form.page == 0 { 1 } else { form.page };
        let query = QueryParams {
            filter: form.filter,
            sort: form.sort,
            page,
        };
        Ok(Redirect::to(&query.path_for(page)).into_response())
    }
}

pub async fn feeds(State(state): State<AppState>) -> AppResult<Html<String>> {
    let mut db = state.db.as_ref().clone();
    let feeds = Feed::all()
        .order_by((Feed::fields().title().asc(), Feed::fields().url().asc()))
        .exec(&mut db)
        .await?;
    render(FeedsTemplate {
        feeds: feeds.into_iter().map(FeedContext::new).collect(),
    })
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct FeedForm {
    pub url: String,
}

/// `POST /feeds` — register a feed URL. The pipeline picks it up on its next
/// run (`feeds.toml` feeds and app-added feeds share the `feeds` table).
pub async fn add_feed(
    State(state): State<AppState>,
    Form(form): Form<FeedForm>,
) -> AppResult<Response> {
    let url = form.url.trim();
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(AppError::bad_request(
            "feed url must start with http:// or https://",
        ));
    }

    let mut db = state.db.as_ref().clone();
    Feed::upsert_by_url(url)
        .is_active(true)
        .exec(&mut db)
        .await
        .context("failed to upsert feed")?;
    tracing::info!(%url, "feed added via app");

    Ok(Redirect::to("/feeds").into_response())
}

/// `POST /feeds/{id}/toggle` — flip `feeds.is_active`; the pipeline skips
/// disabled feeds when ingesting.
pub async fn toggle_feed(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> AppResult<Response> {
    let mut db = state.db.as_ref().clone();
    let mut feed = Feed::filter_by_id(id)
        .first()
        .exec(&mut db)
        .await?
        .ok_or_else(|| AppError::not_found("feed not found"))?;

    let new_state = !feed.is_active;
    toasty::update!(feed {
        is_active: new_state
    })
    .exec(&mut db)
    .await?;
    tracing::info!(feed_id = id, is_active = new_state, "feed toggled");

    Ok(Redirect::to("/feeds").into_response())
}

async fn list_cards(
    db: &mut toasty::Db,
    query: &QueryParams,
) -> AppResult<(Vec<CardContext>, bool)> {
    let page = query.page();
    let offset = (page - 1) * PAGE_SIZE;

    let (items, total) = match query.filter() {
        "opened" => {
            let mut opened = NewsItem::filter(NewsItem::fields().opened_at().is_some())
                .order_by(NewsItem::fields().opened_at().desc())
                .exec(db)
                .await?;

            let votes = load_votes(db, &opened).await?;
            let voted: HashSet<u64> = votes.keys().copied().collect();
            opened.retain(|item| !voted.contains(&item.id));

            let total = opened.len();
            let page_items: Vec<_> = opened.into_iter().skip(offset).take(PAGE_SIZE).collect();
            (page_items, total)
        }
        _ => {
            // filter=all hides already-voted items; filter=voted shows them.
            list_items_page(db, query, offset, query.filter() == "voted").await?
        }
    };

    let cards = build_cards(db, items).await?;
    let has_more = offset + cards.len() < total;
    Ok((cards, has_more))
}

/// Paginated listing for `filter=all` / `filter=voted`.
///
/// `filter=all` excludes items that already have a vote (either direction —
/// rated items don't need to resurface in the default feed); `filter=voted`
/// shows only them. Raw SQL provides the ordered id page because toasty
/// can't express `NOT EXISTS` or `COALESCE` ordering — the rank order is
/// pipeline-owned (`rank_score`, falling back to `model_score`).
async fn list_items_page(
    db: &mut toasty::Db,
    query: &QueryParams,
    offset: usize,
    voted: bool,
) -> AppResult<(Vec<NewsItem>, usize)> {
    let exists = if voted { "EXISTS" } else { "NOT EXISTS" };
    let where_clause = format!("{exists} (SELECT 1 FROM votes v WHERE v.news_item_id = ni.id)");
    let order = if query.sort() == "score" {
        "ORDER BY COALESCE(ni.rank_score, ni.model_score) DESC, ni.fetched_at DESC"
    } else {
        "ORDER BY ni.fetched_at DESC"
    };

    let rows = toasty::sql::query(format!(
        "SELECT ni.id FROM news_items ni WHERE {where_clause} {order} LIMIT ?1 OFFSET ?2"
    ))
    .bind(PAGE_SIZE as i64)
    .bind(offset as i64)
    .exec(db)
    .await
    .context("failed to list news items")?;

    let ids: Vec<u64> = rows
        .iter()
        .filter_map(|row| {
            row.as_record()
                .and_then(|record| record.first())
                .and_then(value_to_u64)
        })
        .collect();

    let total_rows = toasty::sql::query(format!(
        "SELECT COUNT(*) FROM news_items ni WHERE {where_clause}"
    ))
    .exec(db)
    .await
    .context("failed to count news items")?;
    let total = total_rows
        .first()
        .and_then(|row| row.as_record())
        .and_then(|record| record.first())
        .and_then(value_to_u64)
        .unwrap_or(0) as usize;

    let items = items_in_id_order(db, ids).await?;
    Ok((items, total))
}

/// Hydrate NewsItems for an ordered id list, preserving the SQL order.
async fn items_in_id_order(db: &mut toasty::Db, ids: Vec<u64>) -> AppResult<Vec<NewsItem>> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let mut items: Vec<NewsItem> = NewsItem::filter(NewsItem::fields().id().in_list(ids.clone()))
        .exec(db)
        .await?;
    let order: HashMap<u64, usize> = ids.iter().enumerate().map(|(pos, id)| (*id, pos)).collect();
    items.sort_by_key(|item| order.get(&item.id).copied().unwrap_or(usize::MAX));
    Ok(items)
}

fn value_to_u64(value: &toasty::stmt::Value) -> Option<u64> {
    use toasty::stmt::Value as V;
    match value {
        V::I64(v) => u64::try_from(*v).ok(),
        V::U64(v) => Some(*v),
        V::I32(v) => u64::try_from(*v).ok(),
        V::U32(v) => Some(u64::from(*v)),
        _ => None,
    }
}

const DEFAULT_EXPLORE_PCT: f64 = 10.0;

/// Current `explore_pct` setting shared with the pipeline via
/// `pipeline_state`. Falls back to the default when the pipeline has not
/// created the table/key yet.
async fn read_explore_pct(db: &mut toasty::Db) -> f64 {
    let table = toasty::sql::query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_state'",
    )
    .exec(db)
    .await;
    if !matches!(table, Ok(rows) if !rows.is_empty()) {
        return DEFAULT_EXPLORE_PCT;
    }

    let rows = toasty::sql::query("SELECT value FROM pipeline_state WHERE key='explore_pct'")
        .exec(db)
        .await;

    rows.ok()
        .and_then(|r| r.first().cloned())
        .and_then(|row| row.as_record().and_then(|rec| rec.first().cloned()))
        .and_then(|value| value.as_str().map(String::from))
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| (0.0..=100.0).contains(v))
        .unwrap_or(DEFAULT_EXPLORE_PCT)
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct SettingsForm {
    pub explore_pct: f64,
    pub filter: String,
    pub sort: String,
    /// Set by the "Rescore now" button — the pipeline clears the flag and
    /// forces retrain+score on its next loop iteration.
    pub rescore: u8,
}

impl Default for SettingsForm {
    fn default() -> Self {
        Self {
            explore_pct: DEFAULT_EXPLORE_PCT,
            filter: "all".to_string(),
            sort: "new".to_string(),
            rescore: 0,
        }
    }
}

/// `POST /settings` — persists app-tunable settings for the pipeline into
/// `pipeline_state` (`explore_pct`: percent of low-ranked items boosted into
/// the score-sorted feed by the epsilon-greedy exploration step).
pub async fn update_settings(
    State(state): State<AppState>,
    Form(form): Form<SettingsForm>,
) -> AppResult<Response> {
    if !form.explore_pct.is_finite() || !(0.0..=100.0).contains(&form.explore_pct) {
        return Err(AppError::bad_request(
            "explore_pct must be between 0 and 100",
        ));
    }

    let mut db = state.db.as_ref().clone();
    // pipeline_state is Python-owned; create it if the pipeline has not run yet.
    toasty::sql::statement(
        "CREATE TABLE IF NOT EXISTS pipeline_state (key TEXT PRIMARY KEY, value TEXT)",
    )
    .exec(&mut db)
    .await
    .context("failed to ensure pipeline_state")?;
    toasty::sql::statement(
        "INSERT INTO pipeline_state(key, value) VALUES ('explore_pct', ?1) \
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
    )
    .bind(format!("{:.4}", form.explore_pct))
    .exec(&mut db)
    .await
    .context("failed to save explore_pct")?;

    tracing::info!(explore_pct = %form.explore_pct, "updated exploration setting");

    if form.rescore == 1 {
        toasty::sql::statement(
            "INSERT INTO pipeline_state(key, value) VALUES ('rescore_requested', '1') \
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        )
        .exec(&mut db)
        .await
        .context("failed to request rescore")?;
        tracing::info!("rescore requested for the next pipeline run");
    }

    let query = QueryParams {
        filter: form.filter,
        sort: form.sort,
        page: 1,
    };
    Ok(Redirect::to(&query.path_for(1)).into_response())
}

async fn load_card(db: &mut toasty::Db, id: u64) -> AppResult<CardContext> {
    let item = NewsItem::filter_by_id(id)
        .first()
        .exec(db)
        .await?
        .ok_or_else(|| AppError::not_found("news item not found"))?;

    let vote = Vote::filter_by_news_item_id(id)
        .first()
        .exec(db)
        .await?
        .map(|v| v.vote);

    Ok(CardContext::new(item, vote))
}

async fn build_cards(db: &mut toasty::Db, items: Vec<NewsItem>) -> AppResult<Vec<CardContext>> {
    if items.is_empty() {
        return Ok(vec![]);
    }
    let votes = load_votes(db, &items).await?;
    Ok(items
        .into_iter()
        .map(|item| {
            let vote = votes.get(&item.id).cloned();
            CardContext::new(item, vote)
        })
        .collect())
}

async fn load_votes(db: &mut toasty::Db, items: &[NewsItem]) -> AppResult<HashMap<u64, String>> {
    if items.is_empty() {
        return Ok(HashMap::new());
    }
    let ids: Vec<u64> = items.iter().map(|i| i.id).collect();
    let votes: Vec<Vote> = Vote::filter(Vote::fields().news_item_id().in_list(ids))
        .exec(db)
        .await?;
    Ok(votes
        .into_iter()
        .map(|v| (v.news_item_id, v.vote))
        .collect())
}
