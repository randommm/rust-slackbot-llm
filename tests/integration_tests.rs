use axum_test::{TestServer, TestServerConfig};
use rust_slackbot_llm::routes::create_routes;
use serde_json::Value;

async fn new_test_app() -> TestServer {
    let app = create_routes().await.unwrap();
    let config = TestServerConfig::builder()
        .expect_success_by_default()
        .mock_transport()
        .build();

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
