// Adapted from https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized/main.rs
// which have licenses
// https://github.com/huggingface/candle/blob/main/LICENSE-APACHE
// https://github.com/huggingface/candle/blob/main/LICENSE-MIT

use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

use super::{LLMReceiver, LLMSender};
use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::sync::mpsc::channel;
use std::thread;

use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;

pub struct ModelBuilder {
    pub sample_len: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub llm_receiver: Option<LLMReceiver>,
}
pub struct Model {
    model_weights: ModelWeights,
    logits_processor: LogitsProcessor,
    tokenizer: Tokenizer,
    sample_len: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
    mpsc_receiver: LLMReceiver,
    mpsc_sender: Option<LLMSender>,
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self {
            sample_len: 1000,
            temperature: 0.8,
            top_p: None,
            seed: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            llm_receiver: None,
        }
    }
}

impl ModelBuilder {
    pub fn build(self) -> Result<Model, Box<dyn std::error::Error>> {
        let temperature = if self.temperature == 0. {
            None
        } else {
            Some(self.temperature)
        };

        println!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle::utils::with_avx(),
            candle::utils::with_neon(),
            candle::utils::with_simd128(),
            candle::utils::with_f16c()
        );

        println!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            temperature.unwrap_or_default(),
            self.repeat_penalty,
            self.repeat_last_n
        );

        let model_weights = build_model_weights()?;

        let api = hf_hub::api::sync::Api::new()?;
        let repo = "mistralai/Mistral-7B-v0.1";
        let api = api.model(repo.to_string());
        let tokenizer_path = api.get("tokenizer.json")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Error loading tokenizer: {e}"))?;

        let logits_processor = LogitsProcessor::new(
            self.seed.unwrap_or_else(|| {
                let seed = std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                println!("Using {} as LogitsProcessor RNG seed", seed);
                seed
            }),
            temperature,
            self.top_p,
        );

        let (tx, rx) = if let Some(llm_receiver) = self.llm_receiver {
            (None, llm_receiver)
        } else {
            let (tx, rx) = channel();
            (Some(tx), rx)
        };

        println!("Starting LLM model");

        let model = Model {
            model_weights,
            logits_processor,
            tokenizer,
            sample_len: self.sample_len,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            mpsc_receiver: rx,
            mpsc_sender: tx,
        };

        Ok(model)
    }
}

pub fn build_model_weights() -> Result<ModelWeights, Box<dyn std::error::Error>> {
    //let repo = "TheBloke/Mistral-7B-v0.1-GGUF";
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

impl Model {
    #[allow(clippy::await_holding_lock)]
    pub fn run(mut self) -> Option<LLMSender> {
        let mpsc_sender = self.mpsc_sender.take();
        thread::spawn(move || {
            let mpsc_receiver = Arc::new(Mutex::new(self.mpsc_receiver));
            let mut break_sign = false;
            loop {
                if break_sign {
                    break;
                }
                thread::scope(|s| {
                    s.spawn(|| {
                        loop {
                            let (prompt_str, pre_prompt_tokens, oneshot_tx) = mpsc_receiver
                                .lock()
                                .ok()
                                .and_then(|x| x.recv().ok())
                                .unwrap_or_else(|| {
                                    // if mutex is poisoned or channel is closed
                                    // we set break_sign to true
                                    // to avoid new threads being spawned
                                    // and then panic
                                    break_sign = true;
                                    panic!("Mutex is poisoned or channel is closed")
                                });

                            thread_priority::set_current_thread_priority(
                                thread_priority::ThreadPriority::Min,
                            )
                            .unwrap_or_default();
                            let prompt_str = format!("[INST] {prompt_str} [/INST]");
                            // print!("{}", &prompt_str);
                            let tokens = self
                                .tokenizer
                                .encode(prompt_str, true)
                                .map_err(|e| format!("Error encoding tokenizer: {e}"))?;

                            let prompt_tokens =
                                [pre_prompt_tokens, tokens.get_ids().to_owned()].concat();
                            let mut to_sample = self.sample_len.saturating_sub(1);
                            let prompt_tokens =
                                if prompt_tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
                                    let to_remove =
                                        prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
                                    prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..]
                                        .to_vec()
                                } else {
                                    prompt_tokens
                                };
                            let mut all_tokens = vec![];

                            let device = Device::Cpu;
                            let start_prompt_processing = std::time::Instant::now();
                            let mut next_token = {
                                let input =
                                    Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
                                let logits = self.model_weights.forward(&input, 0)?;
                                let logits = logits.squeeze(0)?;
                                self.logits_processor.sample(&logits)?
                            };
                            let prompt_dt = start_prompt_processing.elapsed();
                            all_tokens.push(next_token);
                            let mut output = String::new();
                            extract_token(next_token, &self.tokenizer, &mut output);

                            let eos_token = *self.tokenizer.get_vocab(true).get("</s>").unwrap();

                            let start_post_prompt = std::time::Instant::now();
                            for index in 0..to_sample {
                                let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                                let logits = self
                                    .model_weights
                                    .forward(&input, prompt_tokens.len() + index)?;
                                let logits = logits.squeeze(0)?;
                                let logits = if self.repeat_penalty == 1. {
                                    logits
                                } else {
                                    let start_at =
                                        all_tokens.len().saturating_sub(self.repeat_last_n);
                                    candle_transformers::utils::apply_repeat_penalty(
                                        &logits,
                                        self.repeat_penalty,
                                        &all_tokens[start_at..],
                                    )?
                                };
                                next_token = self.logits_processor.sample(&logits)?;
                                all_tokens.push(next_token);
                                extract_token(next_token, &self.tokenizer, &mut output);
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
                            oneshot_tx
                                .send((output, next_pre_prompt_tokens))
                                .unwrap_or_default();
                        }
                        #[allow(unreachable_code)]
                        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
                    });
                });
                thread::sleep(std::time::Duration::from_millis(500));
            }
        });
        mpsc_sender
    }
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

#[cfg(test)]
mod tests {
    use super::ModelBuilder;
    use tokio::sync::oneshot;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn sequential_dialog() {
        let model = ModelBuilder {
            sample_len: 30,
            ..Default::default()
        }
        .build()
        .unwrap();
        let tx = model.run().unwrap();

        let prompt = "Create a Rust program in 20 words".to_string();
        let pre_prompt_tokens = vec![];

        let (oneshot_tx, oneshot_rx) = oneshot::channel();
        tx.send((prompt, pre_prompt_tokens, oneshot_tx)).unwrap();
        let (output, pre_prompt_tokens) = oneshot_rx.await.unwrap();
        println!("{output}");

        let prompt = "Give me the Cargo.toml in 20 words".to_string();
        let (oneshot_tx, oneshot_rx) = oneshot::channel();
        tx.send((prompt, pre_prompt_tokens, oneshot_tx)).unwrap();
        let (output, _) = oneshot_rx.await.unwrap();
        println!("{output}");
    }
}
