mod utils;
use super::routes::pages::send_user_message;
use super::routes::SlackOAuthToken;
use log::error;
use reqwest::{header::AUTHORIZATION, multipart};
use sqlx::SqlitePool;
use std::thread;
use std::time::SystemTime;
use tokio::runtime::Handle;
use utils::Model;

pub async fn start_llm_worker(db_pool: SqlitePool, slack_oauth_token: SlackOAuthToken) {
    let async_handle = Handle::current();
    thread::spawn(move || {
        let sample_len = 1000_usize;
        let temperature = Some(0.8);
        let top_p = None;
        let seed = None;
        let repeat_penalty = 1.1;
        let repeat_last_n = 64;
        loop {
            let res = thread::scope(|s| {
                s.spawn(|| {
                    thread_priority::set_current_thread_priority(
                        thread_priority::ThreadPriority::Min,
                    )
                    .unwrap_or_default();
                    let mut llm_model = Model::start_model(
                        temperature,
                        top_p,
                        seed,
                        sample_len,
                        repeat_penalty,
                        repeat_last_n,
                    )?;

                    loop {
                        // async task to select a task from the queue
                        let (task_id, prompt_str, channel, thread_ts) =
                            async_handle.block_on(async {
                                get_next_task(&db_pool)
                                    .await
                                    .map_err(|e| format!("Failed to get next task from queue: {e}"))
                            })?;

                        // async task to get the state if it exists
                        let pre_prompt_tokens = async_handle.block_on(async {
                            get_session_state(&db_pool, &channel, &thread_ts, &slack_oauth_token)
                                .await
                                .map_err(|e| format!("Failed to get session state: {e}"))
                        })?;

                        let (next_pre_prompt_tokens, generated_text) =
                            llm_model.run_model_iteraction(prompt_str, pre_prompt_tokens)?;

                        let encoded: Vec<u8> = bincode::serialize(&next_pre_prompt_tokens)
                            .map_err(|e| format!("Failed to encode model {e}"))?;

                        async_handle
                            .block_on(async {
                                let now = SystemTime::now()
                                    .duration_since(SystemTime::UNIX_EPOCH)
                                    .map_err(|e| format!("Error: {:?}", e))?
                                    .as_secs() as i64;
                                sqlx::query("DELETE FROM queue WHERE id = $1;")
                                    .bind(task_id)
                                    .execute(&db_pool)
                                    .await?;
                                sqlx::query(
                                    "INSERT INTO sessions
                                    (channel, thread_ts, created_at, updated_at, model_state)
                                    VALUES ($1, $2, $3, $4, $5)
                                    ON CONFLICT (channel, thread_ts)
                                    DO UPDATE SET
                                    model_state = EXCLUDED.model_state,
                                    updated_at = EXCLUDED.updated_at;",
                                )
                                .bind(&channel)
                                .bind(&thread_ts)
                                .bind(now)
                                .bind(now)
                                .bind(encoded)
                                .execute(&db_pool)
                                .await?;
                                Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
                            })
                            .unwrap_or_else(|e| {
                                error!("Failed to save model state:\n{e}");
                            });

                        let reply_to_user = "Reply from the LLM:\n".to_owned()
                            + &generated_text[1..generated_text.len() - 4];
                        async_handle
                            .block_on(send_user_message(
                                &slack_oauth_token,
                                channel,
                                thread_ts,
                                reply_to_user,
                            ))
                            .unwrap_or_else(|e| {
                                error!("{:?}", e);
                            });
                    }

                    #[allow(unreachable_code)]
                    Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
                })
                .join()
            });
            error!("LLM worker thread exited with message: {res:?}, restarting in 5 seconds");
            thread::sleep(std::time::Duration::from_secs(5));
        }
    });
}

async fn get_next_task(
    db_pool: &SqlitePool,
) -> Result<(i64, String, String, String), Box<dyn std::error::Error + Send + Sync>> {
    let (task_id, prompt_str, channel, thread_ts) = loop {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| format!("Error: {:?}", e))?
            .as_secs() as i64;
        let mut tx = db_pool.begin().await?;
        match sqlx::query_as(
            "
        SELECT id,text,channel,thread_ts FROM queue
        WHERE leased_at <= $1
        ORDER BY created_at ASC
        LIMIT 0,1
        ",
        )
        .bind(now - 600)
        .fetch_one(&mut *tx)
        .await
        {
            Ok(res) => {
                let (task_id, prompt_str, channel, thread_ts) = res;

                if sqlx::query(
                    "
                UPDATE queue SET leased_at = $1
                WHERE id = $2
                ",
                )
                .bind(now)
                .bind(task_id)
                .execute(&mut *tx)
                .await
                .is_ok()
                    && tx.commit().await.is_ok()
                {
                    break (task_id, prompt_str, channel, thread_ts);
                }
            }
            Err(_) => tokio::time::sleep(tokio::time::Duration::from_secs(1)).await,
        }
    };
    Ok((task_id, prompt_str, channel, thread_ts))
}

async fn get_session_state(
    db_pool: &SqlitePool,
    channel: &str,
    thread_ts: &str,
    slack_oauth_token: &SlackOAuthToken,
) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
    let mut initial_message = "Running LLM ".to_owned();
    let query: Result<(Vec<u8>,), _> = sqlx::query_as(
        r#"SELECT model_state FROM sessions WHERE channel = $1 AND thread_ts = $2;"#,
    )
    .bind(channel)
    .bind(thread_ts)
    .fetch_one(db_pool)
    .await;
    let pre_prompt_tokens = if let Ok(query) = query {
        initial_message.push_str("reusing section. ");
        let (model_state,) = query;
        let deserialized = bincode::deserialize(&model_state[..]);
        deserialized.unwrap_or_default()
    } else {
        initial_message.push_str("with new section. ");
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| format!("Error: {:?}", e))?
            .as_secs() as i64;
        sqlx::query(
            r#"INSERT OR IGNORE INTO
            sessions (channel, thread_ts, created_at, updated_at)
            VALUES ($1, $2, $3, $4);"#,
        )
        .bind(channel)
        .bind(thread_ts)
        .bind(timestamp)
        .bind(timestamp)
        .execute(db_pool)
        .await?;
        Default::default()
    };

    let reqw_client = reqwest::Client::new();
    let form = multipart::Form::new()
        .text("text", initial_message)
        .text("channel", channel.to_owned())
        .text("thread_ts", thread_ts.to_owned());
    tokio::spawn(
        reqw_client
            .post("https://slack.com/api/chat.postMessage")
            .header(AUTHORIZATION, format!("Bearer {}", slack_oauth_token.0))
            .multipart(form)
            .send(),
    );

    Ok(pre_prompt_tokens)
}
