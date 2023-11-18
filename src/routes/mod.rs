mod error_handling;
mod llm;
mod pages;
use axum::{
    extract::FromRef,
    routing::{get, post},
    Router,
};
use error_handling::AppError;
use llm::ModelBuilder;
use sqlx::SqlitePool;
use std::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot::Sender as OneShotSender;

type LLMSender = Sender<(String, Vec<u32>, OneShotSender<(String, Vec<u32>)>)>;
type LLMReceiver = Receiver<(String, Vec<u32>, OneShotSender<(String, Vec<u32>)>)>;

#[derive(Clone, FromRef)]
pub struct SlackOAuthToken(pub String);

#[derive(Clone, FromRef)]
pub struct SlackSigningSecret(pub String);

#[derive(Clone, FromRef)]
pub struct AppState {
    pub db_pool: SqlitePool,
    pub slack_oauth_token: SlackOAuthToken,
    pub slack_signing_secret: SlackSigningSecret,
    pub llm_model_sender: LLMSender,
}

pub async fn create_routes(
    db_pool: SqlitePool,
    slack_oauth_token: String,
    slack_signing_secret: String,
) -> Result<Router, Box<dyn std::error::Error>> {
    let model = ModelBuilder::default().build()?;

    let tx = model.run().ok_or("Failed to start LLM model")?;

    let app_state = AppState {
        db_pool,
        slack_oauth_token: SlackOAuthToken(slack_oauth_token),
        slack_signing_secret: SlackSigningSecret(slack_signing_secret),
        llm_model_sender: tx,
    };

    let api = Router::new()
        .route("/slack_events", post(pages::get_slack_events))
        .route("/", get(pages::index))
        .with_state(app_state.clone());

    Ok(Router::new()
        .nest("/v1", api)
        .fallback(get(pages::not_found_json)))
}
