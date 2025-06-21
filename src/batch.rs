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
}

impl BatchTokenizer {
    pub fn new(tokenizer: FlanT5Tokenizer, config: BatchConfig) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            config,
        }
    }
    
    /// Process a batch of texts with zero-copy
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
        
        // For smaller batches or non-parallel builds
        texts.iter()
            .map(|text| self.tokenizer.encode(text))
            .collect()
    }
    
    /// Process a batch with pre-allocated output
    pub fn encode_batch_preallocated(&self, texts: &[&str], output: &mut Vec<Vec<u32>>) -> Result<()> {
        output.clear();
        output.reserve(texts.len());
        
        for text in texts {
            output.push(self.tokenizer.encode(text)?);
        }
        
        Ok(())
    }
    
    /// Process streaming batches with a callback
    pub fn encode_stream<F>(&self, texts: &[&str], mut callback: F) -> Result<()>
    where
        F: FnMut(usize, Vec<u32>) -> Result<()>,
    {
        #[cfg(feature = "parallel")]
        if texts.len() > self.config.max_batch_size {
            let results: Vec<_> = texts.par_iter()
                .enumerate()
                .map(|(idx, text)| (idx, self.tokenizer.encode(text)))
                .collect();
                
            for (idx, result) in results {
                callback(idx, result?)?;
            }
            return Ok(());
        }
        
        for (idx, text) in texts.iter().enumerate() {
            callback(idx, self.tokenizer.encode(text)?)?;
        }
        
        Ok(())
    }
}

/// Zero-copy batch request for advanced usage
pub struct BatchRequest<'a> {
    pub id: usize,
    pub text: &'a str,
}

/// Batch result
pub struct BatchResult {
    pub id: usize,
    pub tokens: Result<Vec<u32>>,
}

/// Advanced batch processor with worker threads
pub struct AsyncBatchTokenizer {
    tokenizer: Arc<FlanT5Tokenizer>,
    config: BatchConfig,
    sender: Sender<BatchRequest<'static>>,
    receiver: Arc<Mutex<Receiver<BatchResult>>>,
}

impl AsyncBatchTokenizer {
    /// Create a new async batch tokenizer
    pub fn new(tokenizer: FlanT5Tokenizer, config: BatchConfig) -> Self {
        let tokenizer = Arc::new(tokenizer);
        let (tx, rx) = bounded::<BatchRequest<'static>>(1000);
        let (result_tx, result_rx) = bounded::<BatchResult>(1000);
        
        // Spawn worker threads
        let num_workers = config.num_workers;
        for _ in 0..num_workers {
            let tokenizer_clone = Arc::clone(&tokenizer);
            let rx_clone = rx.clone();
            let tx_clone = result_tx.clone();
            
            std::thread::spawn(move || {
                while let Ok(request) = rx_clone.recv() {
                    let tokens = tokenizer_clone.encode(request.text);
                    let _ = tx_clone.send(BatchResult {
                        id: request.id,
                        tokens,
                    });
                }
            });
        }
        
        Self {
            tokenizer,
            config,
            sender: tx,
            receiver: Arc::new(Mutex::new(result_rx)),
        }
    }
    
    /// Process a batch asynchronously
    pub fn encode_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        let mut results = vec![None; texts.len()];
        
        // Send all requests
        for (id, text) in texts.iter().enumerate() {
            // We need to ensure the text lives long enough
            // In practice, the caller should ensure this
            let request = BatchRequest {
                id,
                text: unsafe { std::mem::transmute(*text) },
            };
            
            self.sender.send(request)
                .map_err(|_| TokenizerError::TokenNotFound("Batch queue closed".into()))?;
        }
        
        // Collect results
        let receiver = self.receiver.lock();
        for _ in 0..texts.len() {
            let result = receiver.recv()
                .map_err(|_| TokenizerError::TokenNotFound("Result queue closed".into()))?;
            results[result.id] = Some(result.tokens?);
        }
        
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }
} 