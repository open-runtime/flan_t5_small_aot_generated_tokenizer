use crate::{Result, TokenizerError, VOCAB, VOCAB_SCORES, VOCAB_SIZE, id_to_token};
use crate::{PAD_TOKEN_ID, EOS_TOKEN_ID, UNK_TOKEN_ID};
use ahash::AHashMap;
use std::borrow::Cow;

const WHITESPACE_MARKER: char = '▁';
const BYTE_FALLBACK_PREFIX: &str = "<0x";
const MAX_TOKEN_LENGTH: usize = 16;

#[derive(Clone, Debug)]
pub struct TokenizerConfig {
    pub max_length: usize,
    pub add_eos: bool,
    pub add_bos: bool,
    pub pad_to_max_length: bool,
    pub lowercase: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            add_eos: false,  // HuggingFace doesn't add EOS by default
            add_bos: false,
            pad_to_max_length: false,
            lowercase: false,
        }
    }
}

pub struct FlanT5Tokenizer {
    pub(crate) config: TokenizerConfig,
    // Cache for subword tokenization
    cache: parking_lot::RwLock<AHashMap<String, Vec<u32>>>,
}

impl Clone for FlanT5Tokenizer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            cache: parking_lot::RwLock::new(AHashMap::new()),
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
    
    /// Tokenize a single text string
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let processed = self.preprocess(text);
        let mut tokens = self.tokenize_internal(&processed)?;
        
        // Add special tokens
        if self.config.add_bos {
            tokens.insert(0, 0); // BOS token
        }
        if self.config.add_eos {
            tokens.push(EOS_TOKEN_ID);
        }
        
        // Truncate if needed
        if tokens.len() > self.config.max_length {
            tokens.truncate(self.config.max_length);
            if self.config.add_eos {
                tokens[self.config.max_length - 1] = EOS_TOKEN_ID;
            }
        }
        
        // Pad if needed
        if self.config.pad_to_max_length {
            tokens.resize(self.config.max_length, PAD_TOKEN_ID);
        }
        
        Ok(tokens)
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut text = String::with_capacity(token_ids.len() * 4);
        
        for &id in token_ids {
            // Skip special tokens
            if id == PAD_TOKEN_ID || id == EOS_TOKEN_ID {
                continue;
            }
            
            if let Some(token) = id_to_token(id) {
                // Handle whitespace marker
                if token.starts_with(WHITESPACE_MARKER) {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&token[WHITESPACE_MARKER.len_utf8()..]);
                } else if token.starts_with(BYTE_FALLBACK_PREFIX) {
                    // Handle byte fallback tokens
                    if let Some(byte_val) = parse_byte_fallback(token) {
                        text.push(byte_val as char);
                    } else {
                        text.push_str(token);
                    }
                } else {
                    text.push_str(token);
                }
            } else {
                return Err(TokenizerError::InvalidTokenId(id));
            }
        }
        
        Ok(text)
    }
    
    /// Get vocabulary size
    pub const fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }
    
    /// Clear the tokenization cache
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }
    
    fn preprocess<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if self.config.lowercase {
            Cow::Owned(text.to_lowercase())
        } else {
            Cow::Borrowed(text)
        }
    }
    
    fn tokenize_internal(&self, text: &str) -> Result<Vec<u32>> {
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(tokens) = cache.get(text) {
                return Ok(tokens.clone());
            }
        }
        
        // Perform SentencePiece-style tokenization
        let tokens = self.sentencepiece_tokenize(text)?;
        
        // Update cache
        {
            let mut cache = self.cache.write();
            cache.insert(text.to_string(), tokens.clone());
        }
        
        Ok(tokens)
    }
    
    fn sentencepiece_tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // T5 uses SentencePiece with metaspace preprocessing
        
        // Empty string returns empty tokens
        if text.is_empty() {
            return Ok(vec![]);
        }
        
        // Apply Viterbi to the entire text with proper space handling
        self.viterbi_tokenize_with_metaspace(text)
    }
    
    /// Viterbi algorithm with metaspace handling for T5/SentencePiece
    fn viterbi_tokenize_with_metaspace(&self, text: &str) -> Result<Vec<u32>> {
        // Convert text to handle spaces according to SentencePiece rules
        // - Replace spaces with temporary markers to handle them properly
        // - We'll process the text character by character, treating spaces specially
        
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        
        if n == 0 {
            return Ok(vec![]);
        }
        
        // Dynamic programming arrays
        let mut best_score = vec![f64::INFINITY; n + 1];
        let mut best_prev = vec![0usize; n + 1];
        let mut best_token = vec![0u32; n + 1];
        
        best_score[0] = 0.0;
        
        // For each position in the text
        for i in 0..n {
            if best_score[i] == f64::INFINITY {
                continue;
            }
            
            // Skip if this is a space character - spaces are not tokenized directly
            if chars[i].is_whitespace() {
                // Space transitions to next position with no cost
                if best_score[i] < best_score[i + 1] {
                    best_score[i + 1] = best_score[i];
                    best_prev[i + 1] = i;
                    best_token[i + 1] = 0; // Marker for space
                }
                continue;
            }
            
            // Try all possible tokens starting at position i
            let max_end = (i + 50).min(n);
            
            // Build potential tokens incrementally
            let mut token_str = String::new();
            let mut first_char = true;
            
            for j in i..max_end {
                if chars[j].is_whitespace() {
                    // Stop at space - we'll handle space transitions separately
                    break;
                }
                
                token_str.push(chars[j]);
                
                // For the first token after a space (or at start), try with space marker
                let should_try_with_marker = i == 0 || (i > 0 && chars[i - 1].is_whitespace());
                
                if should_try_with_marker && first_char {
                    // Try with space marker prefix
                    let token_with_marker = format!("▁{}", token_str);
                    if let Some(&token_id) = VOCAB.get(token_with_marker.as_str()) {
                        let score = VOCAB_SCORES.get(token_with_marker.as_str())
                            .copied()
                            .unwrap_or(10.0);
                        
                        let new_score = best_score[i] + score;
                        if new_score < best_score[j + 1] {
                            best_score[j + 1] = new_score;
                            best_prev[j + 1] = i;
                            best_token[j + 1] = token_id;
                        }
                    }
                }
                
                // Always try without space marker
                if let Some(&token_id) = VOCAB.get(token_str.as_str()) {
                    let score = VOCAB_SCORES.get(token_str.as_str())
                        .copied()
                        .unwrap_or(10.0);
                    
                    let new_score = best_score[i] + score;
                    if new_score < best_score[j + 1] {
                        best_score[j + 1] = new_score;
                        best_prev[j + 1] = i;
                        best_token[j + 1] = token_id;
                    }
                }
                
                first_char = false;
            }
            
            // Single character fallback
            if i + 1 <= n && !chars[i].is_whitespace() {
                let ch_str = chars[i].to_string();
                
                // Try with space marker if appropriate
                let should_try_with_marker = i == 0 || (i > 0 && chars[i - 1].is_whitespace());
                
                if should_try_with_marker {
                    let ch_with_marker = format!("▁{}", ch_str);
                    if let Some(&token_id) = VOCAB.get(ch_with_marker.as_str()) {
                        let score = VOCAB_SCORES.get(ch_with_marker.as_str())
                            .copied()
                            .unwrap_or(15.0);
                        
                        let new_score = best_score[i] + score;
                        if new_score < best_score[i + 1] {
                            best_score[i + 1] = new_score;
                            best_prev[i + 1] = i;
                            best_token[i + 1] = token_id;
                        }
                    }
                }
                
                // Try without marker
                if let Some(&token_id) = VOCAB.get(ch_str.as_str()) {
                    let score = VOCAB_SCORES.get(ch_str.as_str())
                        .copied()
                        .unwrap_or(15.0);
                    
                    let new_score = best_score[i] + score;
                    if new_score < best_score[i + 1] {
                        best_score[i + 1] = new_score;
                        best_prev[i + 1] = i;
                        best_token[i + 1] = token_id;
                    }
                } else {
                    // Unknown character - use UNK with high penalty
                    let new_score = best_score[i] + 20.0;
                    if new_score < best_score[i + 1] {
                        best_score[i + 1] = new_score;
                        best_prev[i + 1] = i;
                        best_token[i + 1] = UNK_TOKEN_ID;
                    }
                }
            }
        }
        
        // Backtrack to find the best tokenization
        let mut tokens = Vec::new();
        let mut pos = n;
        
        while pos > 0 {
            let token_id = best_token[pos];
            let prev_pos = best_prev[pos];
            
            // Skip space markers (token_id == 0)
            if token_id != 0 {
                tokens.push(token_id);
            }
            
            pos = prev_pos;
        }
        
        tokens.reverse();
        
        // Handle edge case of trailing space
        if text.ends_with(' ') {
            tokens.push(3); // Space marker token
        }
        
        Ok(tokens)
    }
}

fn parse_byte_fallback(token: &str) -> Option<u8> {
    if token.starts_with(BYTE_FALLBACK_PREFIX) && token.ends_with('>') {
        let hex_str = &token[BYTE_FALLBACK_PREFIX.len()..token.len() - 1];
        u8::from_str_radix(hex_str, 16).ok()
    } else {
        None
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