use axum_test::{TestServer, TestServerConfig};
use hmac::{Hmac, Mac};
use http::{HeaderName, HeaderValue};
use rust_slackbot_llm::routes::create_routes;
use serde_json::{json, Value};
use sha2::Sha256;
use std::{fmt::Write, time::SystemTime};

async fn new_test_app() -> TestServer {
    let app = create_routes().await.unwrap();
    let config = TestServerConfig::builder().mock_transport().build();

    TestServer::new_with_config(app, config).unwrap()
}

#[tokio::test]
async fn test_index() {
    let server = new_test_app().await;

    let response = server.get("/v1").await;

    assert_eq!(response.status_code(), 200);
    let expected_response: Value = "Welcome to the Rust Slackbot LLM API!".into();
    response.assert_json(&expected_response);
}

#[tokio::test]
async fn test_challenge() {
    let slack_signing_secret = "some_signing_secret";
    std::env::set_var("SLACK_SIGNING_SECRET", slack_signing_secret);

    let server = new_test_app().await;
    let now = std::time::SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as f64;
    let now = now.to_string();
    let payload = json!({
        "token": "some_token",
        "challenge": "some_challenge",
        "type": "url_verification",
    });

    let body = payload.to_string();
    let body = "v0:".to_owned() + now.as_str() + ":" + body.as_str();

    let mut mac = Hmac::<Sha256>::new_from_slice(slack_signing_secret.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(body.as_bytes());
    let mac_result = mac.finalize();
    let slack_signature = mac_result.into_bytes();
    let slack_signature = slack_signature.iter().fold(String::new(), |mut output, b| {
        let _ = write!(output, "{b:02x}");
        output
    });
    let slack_signature = "v0=".to_owned() + slack_signature.as_str();

    let response = server
        .post("/v1/slack_events")
        .json(&payload)
        .add_header(
            HeaderName::from_static("x-slack-request-timestamp"),
            HeaderValue::from_str(now.as_str()).unwrap(),
        )
        .add_header(
            HeaderName::from_static("x-slack-signature"),
            HeaderValue::from_str(slack_signature.as_str()).unwrap(),
        )
        .await;

    response.assert_json(&json!({
        "challenge": "some_challenge",
    }));
    assert_eq!(response.status_code(), 200);
}
