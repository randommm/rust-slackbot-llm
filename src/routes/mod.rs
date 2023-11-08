mod error_handling;
mod llm;
mod pages;
use axum::{
    extract::FromRef,
    routing::{get, post},
    Router,
};
use error_handling::AppError;
use sqlx::SqlitePool;

use llm::{Model, ModelBuilder};

#[derive(Clone, FromRef)]
pub struct SlackOAuthToken(pub String);

#[derive(Clone, FromRef)]
pub struct SlackSigningSecret(pub String);

#[derive(Clone, FromRef)]
pub struct LlmApiToken(pub String);

#[derive(Clone, FromRef)]
pub struct AppState {
    pub db_pool: SqlitePool,
    pub slack_oauth_token: SlackOAuthToken,
    pub slack_signing_secret: SlackSigningSecret,
    pub llm_api_token: LlmApiToken,
    pub llm_model: Model,
}

pub async fn create_routes(
    db_pool: SqlitePool,
    slack_oauth_token: String,
    slack_signing_secret: String,
    llm_api_token: String,
) -> Result<Router, Box<dyn std::error::Error>> {
    let app_state = AppState {
        db_pool,
        slack_oauth_token: SlackOAuthToken(slack_oauth_token),
        slack_signing_secret: SlackSigningSecret(slack_signing_secret),
        llm_api_token: LlmApiToken(llm_api_token),
        llm_model: ModelBuilder::default().build()?,
    };

    let api = Router::new()
        .route("/slack_events", post(pages::get_slack_events))
        .route("/", get(pages::index))
        .with_state(app_state.clone());

    Ok(Router::new()
        .nest("/v1", api)
        .fallback(get(pages::not_found_json)))
}
