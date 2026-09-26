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
            let mut q = NewsItem::all();
            match query.sort() {
                "score" => {
                    q = q.order_by((
                        NewsItem::fields().model_score().desc(),
                        NewsItem::fields().fetched_at().desc(),
                    ));
                }
                _ => {
                    q = q.order_by(NewsItem::fields().fetched_at().desc());
                }
            }
            q = q.limit(PAGE_SIZE).offset(offset);
            let items: Vec<NewsItem> = q.exec(db).await?;
            let total: u64 = NewsItem::all().count().exec(db).await?;
            (items, total as usize)
        }
    };

    let cards = build_cards(db, items).await?;
    let has_more = offset + cards.len() < total;
    Ok((cards, has_more))
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
