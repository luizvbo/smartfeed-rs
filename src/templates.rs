use crate::models::{Feed, NewsItem};
use askama::Template;
use serde::Deserialize;

static DATETIME_FMT: &[time::format_description::FormatItem<'_>] =
    time::macros::format_description!("[day] [month repr:short] [year] [hour]:[minute]");

fn format_unix(ts: i64) -> String {
    time::OffsetDateTime::from_unix_timestamp(ts)
        .ok()
        .and_then(|dt| dt.format(DATETIME_FMT).ok())
        .unwrap_or_else(|| ts.to_string())
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct QueryParams {
    pub filter: String,
    pub sort: String,
    pub page: usize,
}

impl QueryParams {
    pub fn filter(&self) -> &str {
        if self.filter.is_empty() {
            "all"
        } else {
            &self.filter
        }
    }

    pub fn sort(&self) -> &str {
        if self.sort.is_empty() {
            "new"
        } else {
            &self.sort
        }
    }

    pub fn page(&self) -> usize {
        if self.page == 0 {
            1
        } else {
            self.page
        }
    }

    /// Build a query string for a given page, always including filter and sort.
    pub fn query_for(&self, page: usize) -> String {
        format!(
            "filter={}&sort={}&page={}",
            self.filter(),
            self.sort(),
            page
        )
    }

    /// Build a path with query string for a given page.
    pub fn path_for(&self, page: usize) -> String {
        format!("/?{}", self.query_for(page))
    }

    pub fn normalize(mut self) -> Self {
        if self.filter.is_empty() {
            self.filter = "all".to_string();
        }
        if self.sort.is_empty() {
            self.sort = "new".to_string();
        }
        if self.page == 0 {
            self.page = 1;
        }
        self
    }
}

#[derive(Debug, Clone)]
pub struct CardContext {
    pub item: NewsItem,
    pub vote: Option<String>,
}

impl CardContext {
    pub fn new(item: NewsItem, vote: Option<String>) -> Self {
        Self { item, vote }
    }

    pub fn upvoted(&self) -> bool {
        self.vote.as_deref() == Some("up")
    }

    pub fn downvoted(&self) -> bool {
        self.vote.as_deref() == Some("down")
    }

    pub fn score_text(&self) -> Option<String> {
        self.item.model_score.map(|s| format!("{:.2}", s))
    }

    /// True when the pipeline's exploration step boosted this item's
    /// `rank_score` above its real `model_score` — the card shows a subtle
    /// badge while "Score:" keeps displaying the true model score.
    pub fn exploring(&self) -> bool {
        match self.item.rank_score {
            Some(rank) => rank > self.item.model_score.unwrap_or(f64::MIN) + 0.001,
            None => false,
        }
    }

    pub fn published_at_human(&self) -> String {
        format_unix(self.item.published_at.unwrap_or(self.item.fetched_at))
    }

    pub fn summary_short(&self) -> String {
        match &self.item.summary {
            None => String::new(),
            Some(s) => {
                if s.len() <= 200 {
                    s.clone()
                } else {
                    let mut end = 200;
                    while end > 0 && !s.is_char_boundary(end) {
                        end -= 1;
                    }
                    if let Some(pos) = s[..end].rfind(' ') {
                        end = pos;
                    }
                    format!("{}…", &s[..end])
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct FeedContext {
    pub feed: Feed,
}

impl FeedContext {
    pub fn new(feed: Feed) -> Self {
        Self { feed }
    }

    pub fn last_fetch_human(&self) -> String {
        self.feed
            .last_fetch
            .map(format_unix)
            .unwrap_or_else(|| "never".to_string())
    }
}

#[derive(Template)]
#[template(path = "index.html")]
pub struct IndexTemplate {
    pub items: Vec<CardContext>,
    pub query: QueryParams,
    pub has_more: bool,
    pub next_page: usize,
    pub next_path: String,
    pub next_query: String,
    pub explore_pct: f64,
}

#[derive(Template)]
#[template(path = "cards.html")]
pub struct CardsTemplate {
    pub items: Vec<CardContext>,
    pub query: QueryParams,
    pub has_more: bool,
    pub next_page: usize,
    pub next_path: String,
    pub next_query: String,
}

#[derive(Template)]
#[template(path = "item_card.html")]
pub struct ItemCardTemplate {
    pub item: CardContext,
    pub query: QueryParams,
}

#[derive(Template)]
#[template(path = "detail.html")]
pub struct DetailTemplate {
    pub item: CardContext,
    pub query: QueryParams,
}

#[derive(Template)]
#[template(path = "feeds.html")]
pub struct FeedsTemplate {
    pub feeds: Vec<FeedContext>,
}

#[derive(Template)]
#[template(path = "error.html")]
pub struct ErrorTemplate {
    pub message: String,
}
