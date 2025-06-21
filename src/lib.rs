#![doc = include_str!("../README.md")]

mod error;
mod tokenizer;
mod batch;

#[cfg(feature = "candle")]
mod candle_integration;

#[cfg(feature = "candle")]
mod pool;

pub use error::{Result, TokenizerError};
pub use tokenizer::FlanT5Tokenizer;
pub use batch::{BatchTokenizer, BatchConfig, AsyncBatchTokenizer, BatchRequest, BatchResult};

// Re-export Candle integration if feature is enabled
#[cfg(feature = "candle")]
pub use candle_integration::{TokenizerCandle, TokenizedTensor};

#[cfg(feature = "candle")]
pub use pool::TensorPool;

// Import generated constants
include!(concat!(env!("OUT_DIR"), "/tokenizer_data.rs"));

/// Configuration for the tokenizer
#[derive(Clone, Debug)]
pub struct TokenizerConfig {
    pub add_prefix_space: bool,
    pub max_length: usize,
    pub pad_to_max_length: bool,
    pub add_eos_token: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            add_prefix_space: true,
            max_length: 512,
            pad_to_max_length: false,
            add_eos_token: true,
        }
    }
}

// Additional exports for rust_tokenizers compatibility
pub use crate::PAD_TOKEN_ID as PAD_ID;
pub use crate::UNK_TOKEN_ID as UNK_ID;
pub use crate::EOS_TOKEN_ID as EOS_ID;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generated_constants() {
        // Test that vocabulary was generated correctly
        assert!(TOKEN_TO_ID.len() == VOCAB_SIZE, "Vocabulary size mismatch");
        
        // Test special tokens
        assert_eq!(TOKEN_TO_ID.get("<pad>"), Some(&PAD_TOKEN_ID));
        assert_eq!(TOKEN_TO_ID.get("</s>"), Some(&EOS_TOKEN_ID));
        assert_eq!(TOKEN_TO_ID.get("<unk>"), Some(&UNK_TOKEN_ID));
        
        // Test reverse mapping
        assert_eq!(id_to_token(PAD_TOKEN_ID), Some("<pad>"));
        assert_eq!(id_to_token(EOS_TOKEN_ID), Some("</s>"));
        assert_eq!(id_to_token(UNK_TOKEN_ID), Some("<unk>"));
        
        // Test helper functions
        assert!(is_control('\u{0000}'));
        assert!(!is_control(' '));
        
        // Test whitespace detection
        assert!(is_whitespace(&' '));
        assert!(is_whitespace(&'\u{00A0}')); // Non-breaking space
        assert!(!is_whitespace(&'a'));
    }
}