use dotenvy::var;
use rust_slackbot_llm::run;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let slack_oauth_token = var("SLACK_OAUTH_TOKEN")
        .map_err(|_| "Expected SLACK_OAUTH_TOKEN in the environment or .env file")?;
    let slack_signing_secret = var("SLACK_SIGNING_SECRET")
        .map_err(|_| "Expected SLACK_SIGNING_SECRET in the environment or .env file")?;
    let llm_api_token = var("LLM_API_TOKEN")
        .map_err(|_| "Expected LLM_API_TOKEN in the environment or .env file")?;
    let database_url =
        var("DATABASE_URL").map_err(|e| format!("Failed to get DATABASE_URL: {}", e))?;
    run(
        database_url,
        slack_oauth_token,
        slack_signing_secret,
        llm_api_token,
    )
    .await
}
