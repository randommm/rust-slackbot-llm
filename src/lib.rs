mod llm;
mod routes;

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let app = routes::create_routes().await?;
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
