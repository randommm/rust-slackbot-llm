use super::{AppError, SlackOAuthToken, SlackSigningSecret};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};
use hmac::{Hmac, Mac};
use log::{error, info, trace};
use regex::Regex;
use reqwest::{header::AUTHORIZATION, multipart};
use serde_json::Value;
use sha2::Sha256;
use sqlx::SqlitePool;
use std::time::SystemTime;

pub async fn receive_slack_events(
    State(db_pool): State<SqlitePool>,
    State(slack_signing_secret): State<SlackSigningSecret>,
    State(slack_oauth_token): State<SlackOAuthToken>,
    headers: HeaderMap,
    body: String,
) -> Result<impl IntoResponse, AppError> {
    trace!("Slack event body: {:?}", body);
    trace!("Slack event headers: {:?}", headers);

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
    trace!("provided_signature: {:?}", provided_signature);

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
    tokio::spawn(try_process_slack_events(slack_oauth_token, db_pool, query));

    Ok(Json(api_response))
}

pub async fn try_process_slack_events(
    slack_oauth_token: SlackOAuthToken,
    db_pool: SqlitePool,
    query: Value,
) -> Result<(), AppError> {
    let value = process_slack_events(slack_oauth_token, db_pool, &query).await;

    if let Err(ref value) = value {
        error!(
            "failed to process Slack event.\nGot error:\n{:?}\nGot payload:{:?} ",
            value, query
        );
    }

    value
}

async fn process_slack_events(
    slack_oauth_token: SlackOAuthToken,
    db_pool: SqlitePool,
    query: &Value,
) -> Result<(), AppError> {
    let event = query.get("event").ok_or("event is found on query")?;

    // filters out messages from the bot itself
    // avoid infinite loops of the bot talking to itself
    if event.get("bot_id").is_some() {
        return Ok(());
    }

    let type_ = event.get("type").ok_or("type not found on query")?;
    let type_ = type_.as_str().ok_or("type is not a string")?;
    if type_ != "message" && type_ != "app_mention" {
        return Ok(());
    }

    let text = event.get("text").ok_or("text not found on query")?;
    let text = text.as_str().ok_or("text is not a string")?;

    let channel = event.get("channel").ok_or("channel not found on query")?;
    let channel = channel.as_str().ok_or("channel is not a string")?;

    let user = event.get("user").and_then(|x| x.as_str());
    let user = match user {
        Some(x) => get_email_given_slack_user_id(x.to_owned(), slack_oauth_token.clone())
            .await
            .unwrap_or(x.to_owned()),
        None => "unknown".to_owned(),
    };
    info!("from user {user} at channel {channel} and type {type_}, received message: {text}. ");

    let thread_ts = event.get("thread_ts");
    let thread_ts = match thread_ts {
        Some(x) => x,
        None => event
            .get("event_ts")
            .ok_or("neither thread_ts nor event_ts found")?,
    };
    let thread_ts = thread_ts.as_str().ok_or("thread_ts is not a string")?;

    let text = match Regex::new(r" ?<@.*> ?") {
        Ok(pattern) if type_ == "app_mention" => {
            let text = pattern.replace_all(text, " ");
            text.as_ref().trim().to_owned()
        }
        _ => text.trim().to_owned(),
    };

    info!("Processed message: {text}.");

    let reply_to_user = if text == "delete" || text == "\"delete\"" {
        let _ = sqlx::query("DELETE FROM sessions WHERE channel = $1 AND thread_ts = $2")
            .bind(channel)
            .bind(thread_ts)
            .execute(&db_pool)
            .await;
        let _ = sqlx::query("DELETE FROM queue WHERE channel = $1 AND thread_ts = $2")
            .bind(channel)
            .bind(thread_ts)
            .execute(&db_pool)
            .await;

        "Ok, the LLM section was deleted. A new message will start a fresh LLM section.".to_owned()
    } else if text == "plot" || text == "\"plot\"" {
        return plot_random_stuff(
            channel.to_owned(),
            thread_ts.to_owned(),
            slack_oauth_token.clone(),
        )
        .await;
    } else {
        let created_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| format!("Error: {:?}", e))?
            .as_secs() as i64;
        sqlx::query(
            "INSERT INTO queue (text, channel, thread_ts, created_at, leased_at)
            VALUES ($1, $2, $3, $4, 0);",
        )
        .bind(text)
        .bind(channel)
        .bind(thread_ts)
        .bind(created_at)
        .execute(&db_pool)
        .await?;

        let mut initial_message = "Placed on message on the LLM queue.".to_owned();
        let _ = sqlx::query_as("SELECT COUNT(*) FROM queue")
            .fetch_one(&db_pool)
            .await
            .map(|(row,): (i64,)| {
                initial_message.push_str(format!(" Current queue size: {}.", row).as_str())
            });
        initial_message.push_str("\nNote: to delete the LLM chat section, send \"delete\".");
        initial_message
    };

    send_user_message(
        &slack_oauth_token,
        channel.to_owned(),
        thread_ts.to_owned(),
        reply_to_user,
    )
    .await?;

    Ok(())
}

pub async fn send_user_message(
    slack_oauth_token: &SlackOAuthToken,
    channel: String,
    thread_ts: String,
    text: String,
) -> Result<(), AppError> {
    let form = multipart::Form::new()
        .text("text", text)
        .text("channel", channel)
        .text("thread_ts", thread_ts);

    let reqw_response = reqwest::Client::new()
        .post("https://slack.com/api/chat.postMessage")
        .header(AUTHORIZATION, format!("Bearer {}", slack_oauth_token.0))
        .multipart(form)
        .send()
        .await?;
    reqw_response
        .text()
        .await
        .map_err(|e| format!("Failed to read reqwest response body: {e}"))?;
    Ok(())
}

pub async fn plot_random_stuff(
    channel: String,
    thread_ts: String,
    slack_oauth_token: SlackOAuthToken,
) -> Result<(), AppError> {
    let mut buffer_ = vec![0; 640 * 480 * 3];
    {
        // just an example adapted from plotters homepage
        // https://github.com/plotters-rs/plotters
        use plotters::prelude::*;
        let root = BitMapBackend::with_buffer(&mut buffer_, (640, 480)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("Some plots", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-1f32..1f32, -0.1..2f32)?;

        chart.configure_mesh().draw()?;

        chart
            .draw_series(LineSeries::new(
                (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x.powi(2))),
                &RED,
            ))?
            .label("y = x^2")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        chart
            .draw_series(LineSeries::new(
                (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x.powi(4))),
                &GREEN,
            ))?
            .label("y = x^4")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

        chart
            .draw_series(LineSeries::new(
                (-50..=50)
                    .map(|x| x as f32 / 50.0)
                    .map(|x| (x, x.powi(4) + x.powi(2))),
                &BLUE,
            ))?
            .label("y = x^2 + x^4")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        root.present()?;
    }

    let image = image::RgbImage::from_raw(640, 480, buffer_)
        .ok_or("Failed to generate image from buffer")?;
    let mut bytes: Vec<u8> = Vec::new();
    image
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .map_err(|e| format!("image write_to error: {e}"))?;

    let reqw_client = reqwest::Client::new();
    let part = multipart::Part::stream(bytes)
        .file_name("plot.png")
        .mime_str("image/png")?;
    let form = multipart::Form::new()
        .text("channels", channel)
        .text("title", "A plot for ya")
        .part("file", part)
        .text("thread_ts", thread_ts);
    let reqw_response = reqw_client
        .post("https://slack.com/api/files.upload")
        .header(AUTHORIZATION, format!("Bearer {}", slack_oauth_token.0))
        .multipart(form)
        .send()
        .await?;
    let reqw_response = reqw_response
        .text()
        .await
        .map_err(|e| format!("Failed to read response body: {e}"))?;
    let reqw_response: Value = serde_json::from_str(&reqw_response)
        .map_err(|e| format!("Could not parse response body: {e}"))?;
    trace!("Received send plot response {:?}", reqw_response);

    Ok(())
}

pub async fn get_email_given_slack_user_id(
    slack_user_id: String,
    slack_oauth_token: SlackOAuthToken,
) -> Result<String, AppError> {
    let reqw_client = reqwest::Client::new();
    let reqw_response = reqw_client
        .get(format!(
            "https://slack.com/api/users.info?user={slack_user_id}"
        ))
        .header(AUTHORIZATION, format!("Bearer {}", slack_oauth_token.0))
        .send()
        .await?;
    let body = reqw_response
        .text()
        .await
        .map_err(|e| format!("Failed to read response body: {e}"))?;
    let slack_user: Value =
        serde_json::from_str(&body).map_err(|e| format!("Could not parse response body: {e}"))?;
    Ok(slack_user["user"]["profile"]["email"]
        .as_str()
        .map(|x| x.to_owned())
        .ok_or("Could not find user id in response")?)
}

pub async fn index() -> Result<impl IntoResponse, AppError> {
    Ok(Json("Welcome to the Rust Slackbot LLM API!"))
}

pub async fn not_found_json() -> AppError {
    AppError::new("Endpoint not found")
        .with_user_message("Endpoint not found")
        .with_code(StatusCode::NOT_FOUND)
}
