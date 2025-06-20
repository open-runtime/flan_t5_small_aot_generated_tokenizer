#![doc = include_str!("../README.md")]

mod tokenizer;
mod batch;
mod error;
mod sentencepiece;

#[cfg(feature = "candle")]
mod candle_integration;
#[cfg(feature = "candle")]
mod pool;

pub use tokenizer::{FlanT5Tokenizer, TokenizerConfig};
pub use batch::{BatchTokenizer, BatchConfig};
pub use error::{TokenizerError, Result};

#[cfg(feature = "candle")]
pub use candle_integration::{TokenizerCandle, TokenizedTensor};
#[cfg(feature = "candle")]
pub use pool::TensorPool;

// Re-export generated constants
include!(concat!(env!("OUT_DIR"), "/tokenizer_data.rs")); 