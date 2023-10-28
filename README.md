An Slack bot written in Rust that allows the user to interact with an LLM.

How to use:

## Creating an App on Slack, first steps

* Go to https://api.slack.com/apps and create a new app from scratch.

* Navigate to "OAuth & Permissions" (https://api.slack.com/apps/YOURAPPID/oauth). Go to section "Bot Token Scopes", click on "Add an Oauth Scope", select scope "chat:write". Then click on "install to Workspace". You will obtain the "Bot User OAuth Token" on this page then.

* Navigate to "Basic Information" (https://api.slack.com/apps/YOURAPPID/general). Here you will obtain the "Signing secret".

* Now you need to deploy your app following the steps of the next section before continuing with the Slack app configuration

## Using this repo


Create an `.env` file at the root of the repository (same folder as the `Cargo.toml` file) with:

        SLACK_OAUTH_TOKEN="bot_user_oauth_token_from_previous_step"
        SLACK_SIGNING_SECRET="signing_secret_from_previous_step"
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

# Concluding the deployment of your Slack app

* Navigate to "Event Subscriptions" (https://api.slack.com/apps/YOURAPPID/event-subscriptions). On the requested URL section, fill in https://your_domain_here/v1/slack_events and if everything was properly configured in the previous step, you should receive a "verified" status.

* On "Subscribe to bot events", add scopes "app_mention" and "message.im" and click on "Save changes".

* A yellow upper box will show up requesting you the reinstall the App, proceed with that.

* Go to App Home (https://api.slack.com/apps/YOURAPPID/app-home) and check the box "Allow users to send Slash commands and messages from the messages tab".
