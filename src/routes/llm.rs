// Adapted from https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized/main.rs
// which have licenses
// https://github.com/huggingface/candle/blob/main/LICENSE-APACHE
// https://github.com/huggingface/candle/blob/main/LICENSE-MIT

use super::SlackOAuthToken;
use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::thread;
use tokenizers::Tokenizer;
use tokio::runtime::Handle;

use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;
use sqlx::SqlitePool;

use reqwest::{header::AUTHORIZATION, multipart};

use std::time::SystemTime;

pub fn build_model_weights() -> Result<ModelWeights, Box<dyn std::error::Error>> {
    let repo = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF";
    // let filename = "mistral-7b-instruct-v0.1.Q5_K_M.gguf";
    let filename = "mistral-7b-instruct-v0.1.Q2_K.gguf";

    let api = hf_hub::api::sync::Api::new()?;
    let api = api.model(repo.to_string());
    let model_path = api.get(filename)?;

    let mut file = std::fs::File::open(model_path)?;
    let start = std::time::Instant::now();

    let model = {
        let model = gguf_file::Content::read(&mut file)?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.blck_size();
        }
        println!(
            "loaded {:?} tensors ({}) in {:.2}s",
            model.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );
        ModelWeights::from_gguf(model, &mut file)?
    };
    println!("model built");
    Ok(model)
}

pub async fn start_llm_worker(db_pool: SqlitePool, slack_oauth_token: SlackOAuthToken) {
    let async_handle = Handle::current();
    thread::spawn(move || {
        let sample_len = 1000_usize;
        let temperature = Some(0.8);
        let top_p = None;
        let seed = None;
        let repeat_penalty = 1.1;
        let repeat_last_n = 64;
        println!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle::utils::with_avx(),
            candle::utils::with_neon(),
            candle::utils::with_simd128(),
            candle::utils::with_f16c()
        );
        println!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            {
                #[allow(clippy::unnecessary_literal_unwrap)]
                temperature.unwrap_or_default()
            },
            repeat_penalty,
            repeat_last_n
        );
        loop {
            let res = thread::scope(|s| {
                s.spawn(|| {
                    thread_priority::set_current_thread_priority(
                        thread_priority::ThreadPriority::Min,
                    )
                    .unwrap_or_default();
                    let mut model_weights = build_model_weights().unwrap();
                    let api = hf_hub::api::sync::Api::new()?;
                    let repo = "mistralai/Mistral-7B-v0.1";
                    let api = api.model(repo.to_string());
                    let tokenizer_path = api.get("tokenizer.json")?;
                    let tokenizer = Tokenizer::from_file(tokenizer_path)
                        .map_err(|e| format!("Error loading tokenizer: {e}"))?;

                    let mut logits_processor = LogitsProcessor::new(
                        #[allow(clippy::unnecessary_literal_unwrap)]
                        seed.unwrap_or_else(|| {
                            let seed = std::time::SystemTime::now()
                                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();
                            println!("Using {} as LogitsProcessor RNG seed", seed);
                            seed
                        }),
                        temperature,
                        top_p,
                    );
                    println!("Starting LLM model");

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

                        let prompt_str = format!("[INST] {prompt_str} [/INST]");
                        // print!("{}", &prompt_str);
                        let tokens = tokenizer
                            .encode(prompt_str, true)
                            .map_err(|e| format!("Error encoding tokenizer: {e}"))?;

                        let prompt_tokens =
                            [pre_prompt_tokens, tokens.get_ids().to_owned()].concat();
                        let mut to_sample = sample_len.saturating_sub(1);
                        let prompt_tokens = if prompt_tokens.len() + to_sample
                            > model::MAX_SEQ_LEN - 10
                        {
                            let to_remove =
                                prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
                            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
                        } else {
                            prompt_tokens
                        };
                        let mut all_tokens = vec![];

                        let device = Device::Cpu;
                        let start_prompt_processing = std::time::Instant::now();
                        let mut next_token = {
                            let input =
                                Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
                            let logits = model_weights.forward(&input, 0)?;
                            let logits = logits.squeeze(0)?;
                            logits_processor.sample(&logits)?
                        };
                        let prompt_dt = start_prompt_processing.elapsed();
                        all_tokens.push(next_token);
                        let mut generated_text = String::new();
                        extract_token(next_token, &tokenizer, &mut generated_text);

                        let eos_token = *tokenizer.get_vocab(true).get("</s>").unwrap();

                        let start_post_prompt = std::time::Instant::now();
                        for index in 0..to_sample {
                            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                            let logits =
                                model_weights.forward(&input, prompt_tokens.len() + index)?;
                            let logits = logits.squeeze(0)?;
                            let logits = if repeat_penalty == 1. {
                                logits
                            } else {
                                let start_at = all_tokens.len().saturating_sub(repeat_last_n);
                                candle_transformers::utils::apply_repeat_penalty(
                                    &logits,
                                    repeat_penalty,
                                    &all_tokens[start_at..],
                                )?
                            };
                            next_token = logits_processor.sample(&logits)?;
                            all_tokens.push(next_token);
                            extract_token(next_token, &tokenizer, &mut generated_text);
                            if next_token == eos_token {
                                to_sample = index + 1;
                                break;
                            };
                        }
                        let dt = start_post_prompt.elapsed();
                        println!(
                            "\n\n{:4} prompt tokens processed: {:.2} token/s",
                            prompt_tokens.len(),
                            prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
                        );
                        println!(
                            "{:4} tokens generated: {:.2} token/s",
                            to_sample,
                            to_sample as f64 / dt.as_secs_f64(),
                        );

                        let next_pre_prompt_tokens =
                            [prompt_tokens.as_slice(), all_tokens.as_slice()].concat();

                        // println!("Saving model state");

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
                                println!("Failed to save model state:\n{e}");
                            });

                        async_handle
                            .block_on(async {
                                let reply_to_user = "Reply from the LLM:\n".to_owned()
                                    + &generated_text[1..generated_text.len() - 4];

                                let form = multipart::Form::new()
                                    .text("text", reply_to_user)
                                    .text("channel", channel.to_owned())
                                    .text("thread_ts", thread_ts.clone());

                                let reqw_response = reqwest::Client::new()
                                    .post("https://slack.com/api/chat.postMessage")
                                    .header(
                                        AUTHORIZATION,
                                        format!("Bearer {}", slack_oauth_token.0),
                                    )
                                    .multipart(form)
                                    .send()
                                    .await?;
                                reqw_response.text().await.map_err(|e| {
                                    format!("Failed to read reqwest response body: {e}")
                                })?;
                                Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
                            })
                            .unwrap_or_else(|e| {
                                println!("Failed to send user message:\n{e}");
                            });
                    }

                    #[allow(unreachable_code)]
                    Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
                })
                .join()
            });
            println!("LLM worker thread exited with message: {res:?}, restarting in 5 seconds");
            thread::sleep(std::time::Duration::from_secs(5));
        }
    });
}

fn extract_token(next_token: u32, tokenizer: &Tokenizer, output: &mut String) {
    // Extracting the last token as a string is complicated, here we just apply some simple
    // heuristics as it seems to work well enough for this example. See the following for more
    // details:
    // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141

    if let Some(text) = tokenizer.id_to_token(next_token) {
        let text = text.replace('▁', " ");
        let ascii = text
            .strip_prefix("<0x")
            .and_then(|t| t.strip_suffix('>'))
            .and_then(|t| u8::from_str_radix(t, 16).ok());

        match ascii {
            None => output.push_str(text.as_str()),
            Some(ascii) => {
                if let Some(chr) = char::from_u32(ascii as u32) {
                    if chr.is_ascii() {
                        output.push(chr);
                    }
                }
            }
        }
    }
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
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
            Err(_) => {
                // println!("No work to do, sleeping {e} {now}");
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await
            }
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

// #[cfg(test)]
// mod tests {
//     use super::ModelBuilder;
//     use tokio::sync::oneshot;

//     #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
//     async fn sequential_dialog() {
//         let model = ModelBuilder {
//             sample_len: 30,
//             ..Default::default()
//         }
//         .build()
//         .unwrap();
//         let tx = model.run().await;

//         let prompt = "Create a Rust program in 20 words".to_string();
//         let pre_prompt_tokens = vec![];

//         let (oneshot_tx, oneshot_rx) = oneshot::channel();
//         tx.send((prompt, pre_prompt_tokens, oneshot_tx)).unwrap();
//         let (output, pre_prompt_tokens) = oneshot_rx.await.unwrap();
//         println!("{output}");

//         let prompt = "Give me the Cargo.toml in 20 words".to_string();
//         let (oneshot_tx, oneshot_rx) = oneshot::channel();
//         tx.send((prompt, pre_prompt_tokens, oneshot_tx)).unwrap();
//         let (output, _) = oneshot_rx.await.unwrap();
//         println!("{output}");
//     }
// }
