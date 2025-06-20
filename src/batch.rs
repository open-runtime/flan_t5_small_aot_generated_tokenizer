use crate::{FlanT5Tokenizer, Result, TokenizerError};
use crossbeam::channel::{bounded, Sender, Receiver};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub batch_timeout: Duration,
    pub num_workers: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            batch_timeout: Duration::from_millis(10),
            num_workers: num_cpus::get(),
        }
    }
}

pub struct BatchTokenizer {
    tokenizer: Arc<FlanT5Tokenizer>,
    config: BatchConfig,
    sender: Sender<BatchRequest>,
    result_receiver: Arc<Mutex<Receiver<BatchResult>>>,
}

struct BatchRequest {
    id: usize,
    text: String,
}

struct BatchResult {
    id: usize,
    tokens: Result<Vec<u32>>,
}

impl BatchTokenizer {
    pub fn new(tokenizer: FlanT5Tokenizer, config: BatchConfig) -> Self {
        let tokenizer = Arc::new(tokenizer);
        let (request_sender, request_receiver) = bounded(1000);
        let (result_sender, result_receiver) = bounded(1000);
        
        // Spawn batch processing thread
        let tokenizer_clone = tokenizer.clone();
        let config_clone = config.clone();
        std::thread::spawn(move || {
            Self::batch_worker(
                request_receiver,
                result_sender,
                tokenizer_clone,
                config_clone,
            );
        });
        
        Self {
            tokenizer,
            config,
            sender: request_sender,
            result_receiver: Arc::new(Mutex::new(result_receiver)),
        }
    }
    
    /// Process a batch of texts
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        if texts.len() > self.config.max_batch_size * 10 {
            return Err(TokenizerError::BatchTooLarge {
                size: texts.len(),
                max_size: self.config.max_batch_size * 10,
            });
        }
        
        // Use parallel processing for large batches
        #[cfg(feature = "parallel")]
        if texts.len() > self.config.max_batch_size {
            return texts.par_iter()
                .map(|text| self.tokenizer.encode(text))
                .collect();
        }
        
        #[cfg(not(feature = "parallel"))]
        if texts.len() > self.config.max_batch_size {
            return texts.iter()
                .map(|text| self.tokenizer.encode(text))
                .collect();
        }
        
        // Use batch queue for smaller batches
        let mut results = vec![None; texts.len()];
        
        // Send all requests
        for (id, text) in texts.iter().enumerate() {
            self.sender.send(BatchRequest {
                id,
                text: text.to_string(),
            }).map_err(|_| TokenizerError::TokenNotFound("Batch queue closed".into()))?;
        }
        
        // Collect results
        let receiver = self.result_receiver.lock();
        for _ in 0..texts.len() {
            let result = receiver.recv()
                .map_err(|_| TokenizerError::TokenNotFound("Result queue closed".into()))?;
            results[result.id] = Some(result.tokens?);
        }
        
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }
    
    fn batch_worker(
        receiver: Receiver<BatchRequest>,
        sender: Sender<BatchResult>,
        tokenizer: Arc<FlanT5Tokenizer>,
        config: BatchConfig,
    ) {
        let mut batch = Vec::with_capacity(config.max_batch_size);
        
        loop {
            // Collect batch
            let deadline = std::time::Instant::now() + config.batch_timeout;
            
            while batch.len() < config.max_batch_size {
                let timeout = deadline.saturating_duration_since(std::time::Instant::now());
                match receiver.recv_timeout(timeout) {
                    Ok(request) => batch.push(request),
                    Err(_) => break,
                }
            }
            
            if batch.is_empty() {
                // Check if we should exit
                if receiver.is_empty() {
                    // Try to receive with a small timeout to check if channel is still alive
                    match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
                        Ok(request) => batch.push(request),
                        Err(_) => break, // Channel closed or timeout
                    }
                    continue;
                }
            }
            
            // Process batch
            #[cfg(feature = "parallel")]
            let results: Vec<_> = batch
                .par_drain(..)
                .map(|request| {
                    let tokens = tokenizer.encode(&request.text);
                    BatchResult {
                        id: request.id,
                        tokens,
                    }
                })
                .collect();
                
            #[cfg(not(feature = "parallel"))]
            let results: Vec<_> = batch
                .drain(..)
                .map(|request| {
                    let tokens = tokenizer.encode(&request.text);
                    BatchResult {
                        id: request.id,
                        tokens,
                    }
                })
                .collect();
            
            // Send results
            for result in results {
                if sender.send(result).is_err() {
                    break;
                }
            }
        }
    }
} 