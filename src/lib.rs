mod routes;
use sqlx::sqlite::SqlitePoolOptions;

pub async fn run(
    database_url: String,
    slack_oauth_token: String,
    slack_signing_secret: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect(database_url.as_str())
        .await
        .map_err(|e| format!("DB connection failed: {}", e))?;

    let app = routes::create_routes(db_pool, slack_oauth_token, slack_signing_secret).await?;
    let bind_addr = &"0.0.0.0:51005"
        .parse()
        .map_err(|e| format!("Failed to parse address: {}", e))?;
    println!("Listening on: http://localhost:51005");
    axum::Server::bind(bind_addr)
        .serve(app.into_make_service())
        .await
        .map_err(|e| format!("Server error: {}", e))?;
    Ok(())
}
