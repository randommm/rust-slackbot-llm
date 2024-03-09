use rust_slackbot_llm::run;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "rust_slackbot_llm=info");
    }
    pretty_env_logger::init_timed();
    run().await
}
