This is work in progress

An Slack bot written in Rust that allows the user to interact with a LLM.

How to use:

Create an `.env` file at the root of the repository (same folder as the `Cargo.toml` file) with:

        SLACK_OAUTH_TOKEN=""
        SLACK_SIGNING_SECRET=""
        LLM_API_TOKEN=""
        DATABASE_URL="sqlite://db/db.sqlite3"

and run without Docker compose:

```bash
docker compose build && docker compose up
```

or with Docker, but without Docker compose:

```bash
docker build . -t rust-slackbot-llm && docker run --rm -it -v $(pwd)/.env:/app/.env rust-slackbot-llm
```

or with plain `cargo`:

```bash
cargo run
```
