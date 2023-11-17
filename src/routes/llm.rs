// Adapted from https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized/main.rs
// which have licenses
// https://github.com/huggingface/candle/blob/main/LICENSE-APACHE
// https://github.com/huggingface/candle/blob/main/LICENSE-MIT

use tokenizers::Tokenizer;

use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;
use std::sync::{Arc, Mutex, TryLockError};
pub struct ModelBuilder {
    sample_len: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

#[derive(Clone)]
pub struct Model {
    model_weights: Arc<Mutex<ModelWeights>>,
    tokenizer: Tokenizer,
    sample_len: usize,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self {
            sample_len: 1000,
            temperature: 0.8,
            top_p: None,
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
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

        let model = Model {
            model_weights: Arc::new(Mutex::new(model_weights)),
            tokenizer,
            sample_len: self.sample_len,
            temperature,
            top_p: self.top_p,
            seed: self.seed,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
        };

        Ok(model)
    }
}

pub fn build_model_weights() -> Result<ModelWeights, Box<dyn std::error::Error>> {
    //let repo = "TheBloke/Mistral-7B-v0.1-GGUF";
    let repo = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF";

    // let filename = h"mistral-7b-instruct-v0.1.Q4_K_S.gguf";
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
    pub async fn interact(
        &self,
        prompt_str: String,
        pre_prompt_tokens: &Vec<u32>,
    ) -> Result<(String, Vec<u32>), Box<dyn std::error::Error>> {
        let mut model_weights = loop {
            match self.model_weights.try_lock() {
                Ok(model_weights) => break model_weights,
                Err(TryLockError::Poisoned(e)) => {
                    let guard = e.into_inner();
                    // waiting for https://github.com/rust-lang/rust/issues/96469
                    // *guard = build_model_weights()?;
                    // self.model_weights.clear_poison();
                    // println!("Note: model_weights mutex was poisoned, will try to rebuild");
                    break guard;
                }
                Err(TryLockError::WouldBlock) => {}
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        };

        tokio::task::block_in_place(move || {
            thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Min)
                .unwrap_or_default();
            let prompt_str = format!("[INST] {prompt_str} [/INST]");
            // print!("{}", &prompt_str);
            let tokens = self
                .tokenizer
                .encode(prompt_str, true)
                .map_err(|e| format!("Error encoding tokenizer: {e}"))?;

            let prompt_tokens = [pre_prompt_tokens, tokens.get_ids()].concat();
            let to_sample = self.sample_len.saturating_sub(1);
            let prompt_tokens = if prompt_tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
                let to_remove = prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
                prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
            } else {
                prompt_tokens
            };
            let mut all_tokens = vec![];
            let mut logits_processor =
                LogitsProcessor::new(self.seed, self.temperature, self.top_p);

            let device = Device::Cpu;
            let start_prompt_processing = std::time::Instant::now();
            let mut next_token = {
                let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
                let logits = model_weights.forward(&input, 0)?;
                let logits = logits.squeeze(0)?;
                logits_processor.sample(&logits)?
            };
            let prompt_dt = start_prompt_processing.elapsed();
            all_tokens.push(next_token);
            let mut output = String::new();
            extract_token(next_token, &self.tokenizer, &mut output);

            let eos_token = *self.tokenizer.get_vocab(true).get("</s>").unwrap();

            let start_post_prompt = std::time::Instant::now();
            for index in 0..to_sample {
                let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                let logits = model_weights.forward(&input, prompt_tokens.len() + index)?;
                let logits = logits.squeeze(0)?;
                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = all_tokens.len().saturating_sub(self.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        self.repeat_penalty,
                        &all_tokens[start_at..],
                    )?
                };
                next_token = logits_processor.sample(&logits)?;
                all_tokens.push(next_token);
                extract_token(next_token, &self.tokenizer, &mut output);
                if next_token == eos_token {
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

            let next_pre_prompt_tokens = [prompt_tokens.as_slice(), all_tokens.as_slice()].concat();
            Ok((output, next_pre_prompt_tokens))
        })
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn sequential_dialog() {
        let model = ModelBuilder {
            sample_len: 30,
            ..Default::default()
        }
        .build()
        .unwrap();
        let prompt = "Create a Rust program in 20 words".to_string();
        let pre_prompt_tokens = vec![];
        let (output, pre_prompt_tokens) = model.interact(prompt, &pre_prompt_tokens).await.unwrap();
        println!("{output}");

        let prompt = "Give me the Cargo.toml in 20 words".to_string();
        let (output, _) = model.interact(prompt, &pre_prompt_tokens).await.unwrap();
        println!("{output}");
    }

    // waiting for https://github.com/rust-lang/rust/issues/96469
    // #[tokio::test]
    // async fn poisoning_rebuild() {
    //     let model = ModelBuilder::default().build().unwrap();
    //     let c_model = model.clone();

    //     #[allow(unused_variables, unreachable_code)]
    //     std::thread::spawn(move || {
    //         let lock = c_model.model_weights.lock().unwrap();
    //         panic!();
    //         drop(lock);
    //     })
    //     .join()
    //     .unwrap_or_default();

    //     assert!(model.model_weights.is_poisoned());

    //     let prompt = "Create a basic Rust program".to_string();
    //     let pre_prompt_tokens = vec![];
    //     let (output, _) = model.interact(prompt, &pre_prompt_tokens).await.unwrap();
    //     println!("{output}");
    // }
}
