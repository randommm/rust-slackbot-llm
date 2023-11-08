A Slack chat bot written in Rust that allows the user to interact with a Mistral large language model. For previous version that used the Hugging Face API, see commit [246011b01](https://github.com/randommm/rust-slackbot-llm/tree/246011b01cd9089a3bf4dfa08f431909df8c7b60).

<img src="https://github.com/randommm/rust-slackbot-llm/assets/4267674/d1da83f2-4dd4-43f7-943d-36e12cee232a" alt="screenshot" width="300"/>

## Creating an App on Slack, first steps

* Go to https://api.slack.com/apps and create a new app from scratch.

* Navigate to "OAuth & Permissions" (https://api.slack.com/apps/YOURAPPID/oauth). Go to section "Bot Token Scopes", click on "Add an Oauth Scope", select scopes "app_mentions:read", "chat:write", "files:write", "im:history". Then click on "install to Workspace". You will obtain the "Bot User OAuth Token" on this page then.

* Navigate to "Basic Information" (https://api.slack.com/apps/YOURAPPID/general). Here you will obtain the "Signing secret".

* Now you need to deploy your app following the steps of the next section before continuing with the Slack app configuration.

## Configuring, compiling and running

Create an `.env` file at the root of the repository (same folder as the `Cargo.toml` file) with:

        SLACK_OAUTH_TOKEN="bot_user_oauth_token_from_previous_step"
        SLACK_SIGNING_SECRET="signing_secret_from_previous_step"
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

## Public IP address

When you run the App, it's exposed via port 51005 and on HTTP only, therefore, it's recommend that install Nginx and add the following configuration (probably at `/etc/nginx/sites-available/default`) to:

        server {
            server_name your_domain_here.com;
            location / {
                proxy_pass http://127.0.0.1:51005;
                proxy_set_header Host $host;
            }
            listen 80;
        }

And then (after pointing your DNS to your domain) run `sudo certbot --nginx`.

If you don't own a domain, you might wanna try your luck with https://nip.io/.

You also need to have a reacheable IP address for Slack to deliver the payloads to your bot. Sadly this is not possible with many residential internet ISPs which now use carrier grade NAT, so as a work around you can get a simple machine on a cloud provider (e.g.: AFAIK, Google Cloud has a always free tier, use at your own risk) and run the app there or run at your local machine and redirect the port to cloud machine using SSH `ssh your_domain_here.com -R 127.0.0.1:51005:127.0.0.1:51005`.

## Concluding the deployment of your Slack app

* Navigate to "Event Subscriptions" (https://api.slack.com/apps/YOURAPPID/event-subscriptions). On the requested URL section, fill in https://your_domain_here.com/v1/slack_events and if everything was properly configured in the previous step, you should receive a "verified" status.

* On "Subscribe to bot events", add scopes "app_mention" and "message.im" and click on "Save changes".

* A yellow upper box will show up requesting you the reinstall the App, proceed with that.

* Go to App Home (https://api.slack.com/apps/YOURAPPID/app-home) and check the box "Allow users to send Slash commands and messages from the messages tab".

## An extra: plotting

If configured everything correctly, you should have you also have support for plotting out of the box. Just send plot as message to chat bot to get an example:

<img src="https://github.com/randommm/rust-slackbot-llm/assets/4267674/ab651be4-2ebb-4607-9977-1515be80e2e6" alt="screenshot" width="300"/>

