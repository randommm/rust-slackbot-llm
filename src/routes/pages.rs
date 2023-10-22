use super::{AppError, LlmApiToken, SlackOAuthToken, SlackSigningSecret};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};

use futures::future::TryFutureExt;
use hmac::{Hmac, Mac};
use reqwest::{header::AUTHORIZATION, multipart};
use serde_json::Value;
use sha2::Sha256;
use sqlx::SqlitePool;
use std::time::SystemTime;

pub async fn get_slack_events(
    State(db_pool): State<SqlitePool>,
    State(slack_signing_secret): State<SlackSigningSecret>,
    State(slack_oauth_token): State<SlackOAuthToken>,
    State(llm_api_token): State<LlmApiToken>,
    headers: HeaderMap,
    body: String,
) -> Result<impl IntoResponse, AppError> {
    // println!("Slack event body: {:?}", body);
    // println!("Slack event headers: {:?}", headers);

    let provided_timestamp = headers
        .get("X-Slack-Request-Timestamp")
        .ok_or(AppError::new_wum(
            "X-Slack-Request-Timestamp header is required",
        ))?
        .to_str()?;
    let provided_signature = headers
        .get("X-Slack-Signature")
        .ok_or(AppError::new_wum("X-Slack-Signature header is required"))?
        .to_str()?
        .get(3..)
        .ok_or("Proposed slack signature is smaller than 4 characters")?;

    // Convert provided_signature (&str of hex chars) to bytes
    let provided_signature = provided_signature
        .chars()
        .collect::<Vec<_>>()
        .windows(2)
        .step_by(2)
        .map(|i| u8::from_str_radix(i.iter().collect::<String>().as_str(), 16))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Invalid hex on proposed slack signature: {e}"))?;
    // println!("provided_signature: {:?}", provided_signature);

    // check timestamp
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|e| format!("Error: {:?}", e))?
        .as_secs() as f64;
    if !(-60_f64..=60_f64).contains(&(provided_timestamp.parse::<f64>()? - now)) {
        return Err(AppError::new_wum("Timestamp is too old or in the future"));
    }

    let sig_basestring = "v0:".to_owned() + provided_timestamp + ":" + body.as_str();
    let mut mac = Hmac::<Sha256>::new_from_slice(slack_signing_secret.0.as_bytes())
        .map_err(|e| format!("Hmac could not ingest slack_signing_secret: {e}"))?;
    mac.update(sig_basestring.as_bytes());
    mac.verify_slice(&provided_signature[..])
        .map_err(|e| format!("Hmac slack_signing_secret verification failed: {e}"))?;

    let query: Value = serde_json::from_str(body.as_str())?;

    let mut api_response: Value = Default::default();
    if let Some(challenge) = query.get("challenge").and_then(|x| x.as_str()) {
        api_response["challenge"] = Value::String(challenge.to_owned());
    }
    tokio::spawn(process_slack_events(
        slack_oauth_token,
        llm_api_token,
        db_pool,
        query,
    ));

    Ok(Json(api_response))
}

pub async fn process_slack_events(
    slack_oauth_token: SlackOAuthToken,
    llm_api_token: LlmApiToken,
    db_pool: SqlitePool,
    query: Value,
) -> Result<(), AppError> {
    // println!("Slack event received by processor: {:?}", query);

    if let Some(event) = query.get("event") {
        if event.get("bot_id").is_none() {
            if let Some(type_) = event.get("type") {
                let type_ = type_.as_str().ok_or("type is not a string")?;
                if type_ == "message" {
                    if let Some(text) = event.get("text") {
                        let text = text.as_str().ok_or("text is not a string")?;
                        if let Some(channel) = event.get("channel") {
                            let channel = channel.as_str().ok_or("channel is not a string")?;

                            let reply = if text == "delete" {
                                let _ = sqlx::query(
                                    "DELETE FROM sessions
                                    WHERE channel = $1",
                                )
                                .bind(channel)
                                .execute(&db_pool)
                                .await;

                                "Section deleted".to_owned()
                            } else {
                                let mut payload: Value = Default::default();
                                payload["inputs"] = Value::default();
                                payload["options"] = Value::default();

                                // select check model state exists
                                let query: Result<(Vec<u8>,), _> = sqlx::query_as(
                                    r#"SELECT id,
                                        FROM sessions WHERE channel = $1;"#,
                                )
                                .bind(channel)
                                .fetch_one(&db_pool)
                                .await;

                                let mut past_user_inputs = Vec::<Value>::default();
                                let mut generated_responses = Vec::<Value>::default();
                                if let Ok(query) = query {
                                    let (model_state,) = query;
                                    let deserialized: Result<(_, _), _> =
                                        bincode::deserialize(&model_state[..]);
                                    if let Ok(deserialized) = deserialized {
                                        past_user_inputs = deserialized.0;
                                        generated_responses = deserialized.1;
                                        payload["inputs"]["past_user_inputs"] =
                                            serde_json::to_value(&past_user_inputs)?;
                                        payload["inputs"]["generated_responses"] =
                                            serde_json::to_value(&generated_responses)?;
                                    } else {
                                        println!("Failed to deserialize model state");
                                    }
                                } else {
                                    let timestamp = SystemTime::now()
                                        .duration_since(SystemTime::UNIX_EPOCH)
                                        .map_err(|e| format!("Error: {:?}", e))?
                                        .as_secs()
                                        as i64;
                                    sqlx::query(
                                        "INSERT OR IGNORE INTO trades_resampled
                                        (channel, created_at, updated_at)
                                        VALUES ($1, $2, $3);",
                                    )
                                    .bind(channel)
                                    .bind(timestamp)
                                    .bind(timestamp)
                                    .execute(&db_pool)
                                    .await?;
                                }

                                payload["options"]["wait_for_model"] = Value::Bool(true);
                                payload["inputs"]["text"] = Value::String(text.to_owned());

                                let reqw_client = reqwest::Client::new();
                                let reply = reqw_client
                                    .post("https://api-inference.huggingface.co/models/bert-base-uncased")
                                    .header(AUTHORIZATION, format!("Bearer {}", llm_api_token.0))
                                    .json(&payload)
                                    .send()
                                    .and_then(|reqw_response| async move {reqw_response.text().await}).await.map_err(|e| {
                                        println!("Failed to send request to LLM: {e}");
                                    });

                                if let Ok(reply) = reply {
                                    past_user_inputs.push(Value::String(text.to_owned()));
                                    generated_responses.push(Value::String(reply.to_owned()));
                                    let model_state = (past_user_inputs, generated_responses);
                                    let encoded: Vec<u8> =
                                        bincode::serialize(&model_state).unwrap();
                                    let timestamp = SystemTime::now()
                                        .duration_since(SystemTime::UNIX_EPOCH)
                                        .map_err(|e| format!("Error: {:?}", e))?
                                        .as_secs()
                                        as i64;
                                    sqlx::query(
                                        "INSERT INTO trades_resampled
                                        (channel, created_at, updated_at, model_state)
                                        VALUES ($1, $2, $3, $4)
                                        ON CONFLICT (channel)
                                        DO UPDATE SET
                                        model_state = EXCLUDED.model_state,
                                        updated_at = EXCLUDED.updated_at;",
                                    )
                                    .bind(channel)
                                    .bind(timestamp)
                                    .bind(timestamp)
                                    .bind(encoded)
                                    .execute(&db_pool)
                                    .await?;
                                    "Reply from LLM:\n".to_owned()
                                        + reply.as_str()
                                        + "\n to delete section, send \"delete\" (without clauses)"
                                } else {
                                    "Failed to get reply from LLM".to_owned()
                                }
                            };

                            let reqw_client = reqwest::Client::new();
                            let form = multipart::Form::new()
                                .text("text", reply)
                                .text("channel", channel.to_owned());
                            let reqw_response = reqw_client
                                .post("https://slack.com/api/chat.postMessage")
                                .header(AUTHORIZATION, format!("Bearer {}", slack_oauth_token.0))
                                .multipart(form)
                                .send()
                                .await?;
                            reqw_response.text().await.map_err(|e| {
                                format!("Failed to read reqwest response body: {e}")
                            })?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

pub async fn index() -> Result<impl IntoResponse, AppError> {
    Ok(Json("Welcome to the Rust Slackbot LLM API!"))
}

pub async fn not_found_json() -> AppError {
    AppError::new("Endpoint not found")
        .with_user_message("Endpoint not found")
        .with_code(StatusCode::NOT_FOUND)
}
