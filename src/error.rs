use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Token not found in vocabulary: {0}")]
    TokenNotFound(String),
    
    #[error("Invalid token ID: {0}")]
    InvalidTokenId(u32),
    
    #[error("Text too long: {length} > {max_length}")]
    TextTooLong { length: usize, max_length: usize },
    
    #[error("Batch size too large: {size} > {max_size}")]
    BatchTooLarge { size: usize, max_size: usize },
    
    #[cfg(feature = "candle")]
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, TokenizerError>; 