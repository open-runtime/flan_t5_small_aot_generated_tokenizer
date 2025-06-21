use crate::{
    error::{Result, TokenizerError},
};
use crate::TokenizerConfig;
use ahash::AHashMap;
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use ahash::AHasher;

// Import generated vocabulary data
include!(concat!(env!("OUT_DIR"), "/tokenizer_data.rs"));

pub struct FlanT5Tokenizer {
    pub config: TokenizerConfig,
    // Zero-copy cache: hash of text -> Arc<Vec<u32>>
    cache: parking_lot::RwLock<AHashMap<u64, Arc<Vec<u32>>>>,
}

impl Clone for FlanT5Tokenizer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            // Create a new empty cache for the cloned instance
            cache: parking_lot::RwLock::new(AHashMap::with_capacity(10_000)),
        }
    }
}

impl FlanT5Tokenizer {
    pub fn new(config: TokenizerConfig) -> Self {
        Self {
            config,
            cache: parking_lot::RwLock::new(AHashMap::with_capacity(10_000)),
        }
    }
    
    pub fn with_default_config() -> Self {
        Self::new(TokenizerConfig::default())
    }
    
    /// Get a stable hash of text for cache key
    #[inline]
    fn hash_text(text: &str) -> u64 {
        let mut hasher = AHasher::default();
        text.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Clone tokenizer for batch processing
    pub fn for_batch(&self) -> Self {
        Self {
            config: self.config.clone(),
            cache: parking_lot::RwLock::new(AHashMap::new()),
        }
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return if self.config.add_eos_token {
                Ok(vec![EOS_TOKEN_ID])
            } else {
                Ok(vec![])
            };
        }
        
        if text.len() > self.config.max_length * 6 {
            return Err(TokenizerError::TextTooLong {
                length: text.len(),
                max_length: self.config.max_length * 6,
            });
        }
        
        // Check cache first with zero-copy
        let cache_key = Self::hash_text(text);
        {
            let cache = self.cache.read();
            if let Some(tokens) = cache.get(&cache_key) {
                // Return cloned Arc data (cheap clone)
                return Ok((**tokens).clone());
            }
        }
        
        // Preprocess text and tokenize
        let processed = self.preprocess_text(text);
        let mut tokens = self.tokenize_with_viterbi_zero_copy(&processed)?;
        
        // Apply post-processing
        if self.config.add_eos_token {
            tokens.push(EOS_TOKEN_ID);
        }
        
        // Apply truncation if needed
        if tokens.len() > self.config.max_length {
            tokens.truncate(self.config.max_length);
            if self.config.add_eos_token && tokens.last() != Some(&EOS_TOKEN_ID) {
                tokens[self.config.max_length - 1] = EOS_TOKEN_ID;
            }
        }
        
        // Apply padding if configured
        if self.config.pad_to_max_length && tokens.len() < self.config.max_length {
            tokens.resize(self.config.max_length, PAD_TOKEN_ID);
        }
        
        // Store in cache with Arc
        let arc_tokens = Arc::new(tokens.clone());
        {
            let mut cache = self.cache.write();
            cache.insert(cache_key, Arc::clone(&arc_tokens));
        }
        
        Ok(tokens)
    }
    
    /// Zero-allocation text preprocessing
    fn preprocess_text(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len() * 2);
        
        let mut is_start = true;
        let mut prev_was_whitespace = false;
        
        for ch in text.chars() {
            if is_whitespace(&ch) {
                if !prev_was_whitespace && !is_start {
                    result.push('▁');
                }
                prev_was_whitespace = true;
            } else if is_cjk_char(ch) {
                if !is_start && !prev_was_whitespace {
                    result.push('▁');
                }
                result.push(ch);
                result.push('▁');
                prev_was_whitespace = true;
            } else {
                if is_start && self.config.add_prefix_space {
                    result.push('▁');
                }
                result.push(ch);
                prev_was_whitespace = false;
                is_start = false;
            }
        }
        
        result
    }
    
    /// Decode token IDs back to text with pre-calculated size
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        if token_ids.is_empty() {
            return Ok(String::new());
        }
        
        // Pre-calculate approximate size needed
        let estimated_size = token_ids.len() * 4;
        let mut text = String::with_capacity(estimated_size);
        
        let mut is_first = true;
        
        for &id in token_ids {
            if id == PAD_TOKEN_ID || id == EOS_TOKEN_ID {
                continue;
            }
            
            if let Some(token) = id_to_token(id) {
                if let Some(suffix) = token.strip_prefix(METASPACE_REPLACEMENT) {
                    if !is_first {
                        text.push(' ');
                    }
                    text.push_str(suffix);
                } else if let Some(hex_byte) = token.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
                    if let Ok(byte_val) = u8::from_str_radix(hex_byte, 16) {
                        text.push(byte_val as char);
                    } else {
                        text.push_str(token);
                    }
                } else {
                    text.push_str(token);
                }
                is_first = false;
            } else {
                return Err(TokenizerError::InvalidTokenId(id));
            }
        }
        
        text.shrink_to_fit();
        Ok(text)
    }
    
    /// Check if a token exists in vocabulary
    #[inline]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        TOKEN_TO_ID.get(token).copied()
    }
    
    /// Get token string from ID
    #[inline]
    pub fn id_to_token(&self, id: u32) -> Option<&'static str> {
        id_to_token(id)
    }
    
    /// Get vocabulary size
    #[inline]
    pub fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }
    
    /// Check cache status
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read();
        let size = cache.len();
        let capacity = cache.capacity();
        (size, capacity)
    }
    
    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }
}

// Zero-copy Viterbi implementation
impl FlanT5Tokenizer {
    /// Zero-copy Viterbi tokenization working directly with string slices
    fn tokenize_with_viterbi_zero_copy(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        
        // Check if entire text is a single known token
        if let Some(&token_id) = TOKEN_TO_ID.get(text) {
            return Ok(vec![token_id]);
        }
        
        // For single character
        if text.chars().count() == 1 {
            if let Some(&token_id) = TOKEN_TO_ID.get(text) {
                return Ok(vec![token_id]);
            } else {
                return Ok(vec![UNK_TOKEN_ID]);
            }
        }
        
        // Initialize Viterbi arrays using byte indices
        let text_len = text.len();
        let mut best_score = vec![f64::INFINITY; text_len + 1];
        let mut best_token_id = vec![0u32; text_len + 1];
        let mut best_token_start = vec![0usize; text_len + 1];
        
        best_score[0] = 0.0;
        
        // Process using byte indices
        let mut byte_indices: Vec<(usize, char)> = text.char_indices().collect();
        byte_indices.push((text_len, '\0')); // Sentinel for end
        
        for i in 0..byte_indices.len() - 1 {
            let (start_byte, _) = byte_indices[i];
            
            if best_score[start_byte].is_infinite() {
                continue;
            }
            
            // Check all possible token lengths from this position
            for j in (i + 1)..byte_indices.len().min(i + 100) {
                let (end_byte, _) = byte_indices[j];
                let candidate = &text[start_byte..end_byte];
                
                // Check if this substring exists in vocabulary
                if let Some(&token_id) = TOKEN_TO_ID.get(candidate) {
                    // Use negative log probability as score (from the tokenizer JSON)
                    // More negative = more common = better
                    // Convert to positive cost where lower is better
                    let token_score = -TOKEN_SCORES.get(candidate).copied().unwrap_or(-10.0);
                    let score = best_score[start_byte] + token_score;
                    
                    if score < best_score[end_byte] {
                        best_score[end_byte] = score;
                        best_token_id[end_byte] = token_id;
                        best_token_start[end_byte] = start_byte;
                    }
                }
            }
            
            // Handle unknown character - single character fallback
            if i + 1 < byte_indices.len() {
                let (next_byte, _) = byte_indices[i + 1];
                let single_char = &text[start_byte..next_byte];
                
                // Check if single character exists in vocabulary
                if let Some(&char_token_id) = TOKEN_TO_ID.get(single_char) {
                    let char_score = -TOKEN_SCORES.get(single_char).copied().unwrap_or(-10.0);
                    let score = best_score[start_byte] + char_score;
                    
                    if score < best_score[next_byte] {
                        best_score[next_byte] = score;
                        best_token_id[next_byte] = char_token_id;
                        best_token_start[next_byte] = start_byte;
                    }
                } else {
                    // True unknown character - use high penalty
                    let unk_score = best_score[start_byte] + 100.0;
                    
                    if unk_score < best_score[next_byte] {
                        best_score[next_byte] = unk_score;
                        best_token_id[next_byte] = UNK_TOKEN_ID;
                        best_token_start[next_byte] = start_byte;
                    }
                }
            }
        }
        
        // Backtrack to find the best path
        let mut tokens = Vec::new();
        let mut pos = text_len;
        
        while pos > 0 {
            let token_id = best_token_id[pos];
            tokens.push(token_id);
            pos = best_token_start[pos];
        }
        
        tokens.reverse();
        Ok(tokens)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_tokenization() {
        let tokenizer = FlanT5Tokenizer::with_default_config();
        
        let text = "Hello world!";
        let tokens = tokenizer.encode(text).unwrap();
        
        // Verify tokens were generated
        assert!(!tokens.is_empty());
        
        // Test decode
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded.trim(), text);
    }
    
    #[test]
    fn test_batch_tokenization() {
        use crate::{BatchTokenizer, BatchConfig};
        
        let tokenizer = FlanT5Tokenizer::with_default_config();
        let batch_tokenizer = BatchTokenizer::new(tokenizer, BatchConfig::default());
        
        let texts = vec![
            "First sentence.",
            "Second sentence with more words.",
            "Third one.",
        ];
        
        let results = batch_tokenizer.encode_batch(&texts).unwrap();
        assert_eq!(results.len(), texts.len());
    }
} 