#[derive(Debug, Clone, toasty::Model)]
#[table = "news_items"]
pub struct NewsItem {
    #[key]
    #[auto]
    pub id: u64,

    pub title: String,

    #[unique]
    pub url: String,

    pub source_feed: String,

    pub summary: Option<String>,

    pub image_url: Option<String>,

    pub published_at: Option<i64>,

    pub fetched_at: i64,

    pub model_score: Option<f64>,

    pub opened_at: Option<i64>,
}

#[derive(Debug, Clone, toasty::Model)]
#[table = "feeds"]
pub struct Feed {
    #[key]
    #[auto]
    pub id: u64,

    #[unique]
    pub url: String,

    pub title: Option<String>,

    pub last_fetch: Option<i64>,

    pub last_status: Option<String>,

    pub error_message: Option<String>,

    #[default(true)]
    pub is_active: bool,
}

#[derive(Debug, Clone, toasty::Model)]
#[table = "votes"]
pub struct Vote {
    #[key]
    #[auto]
    pub id: u64,

    #[unique]
    pub news_item_id: u64,

    pub vote: String,

    pub created_at: i64,
}
