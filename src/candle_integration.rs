use crate::{FlanT5Tokenizer, BatchTokenizer, BatchConfig, TensorPool, Result, PAD_TOKEN_ID};
use candle_core::{Device, DType, Tensor};
use std::sync::Arc;

pub struct TokenizedTensor {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
}

pub trait TokenizerCandle {
    fn tokenize_to_tensor(&self, text: &str, device: &Device) -> Result<TokenizedTensor>;
    
    fn batch_tokenize_to_tensor(
        &self, 
        texts: &[&str], 
        device: &Device
    ) -> Result<TokenizedTensor>;
}

impl TokenizerCandle for FlanT5Tokenizer {
    fn tokenize_to_tensor(&self, text: &str, device: &Device) -> Result<TokenizedTensor> {
        let tokens = self.encode(text)?;
        let len = tokens.len();
        
        // Create input_ids tensor
        let input_ids = Tensor::from_vec(tokens, len, device)?;
        
        // Create attention mask (1 for real tokens, 0 for padding)
        let attention_mask = if self.config.pad_to_max_length {
            let mut mask = vec![1u32; len];
            mask.resize(self.config.max_length, 0);
            Tensor::from_vec(mask, self.config.max_length, device)?
        } else {
            Tensor::ones(len, DType::U32, device)?
        };
        
        Ok(TokenizedTensor {
            input_ids: input_ids.unsqueeze(0)?, // Add batch dimension
            attention_mask: attention_mask.unsqueeze(0)?,
        })
    }
    
    fn batch_tokenize_to_tensor(
        &self, 
        texts: &[&str], 
        device: &Device
    ) -> Result<TokenizedTensor> {
        let batch_size = texts.len();
        let tokenized: Vec<_> = texts.iter()
            .map(|text| self.encode(text))
            .collect::<Result<Vec<_>>>()?;
        
        // Find max length in batch
        let max_len = tokenized.iter()
            .map(|tokens| tokens.len())
            .max()
            .unwrap_or(0);
        
        let padded_len = if self.config.pad_to_max_length {
            self.config.max_length
        } else {
            max_len
        };
        
        // Create padded tensors
        let mut input_ids_vec = Vec::with_capacity(batch_size * padded_len);
        let mut attention_mask_vec = Vec::with_capacity(batch_size * padded_len);
        
        for tokens in tokenized {
            let len = tokens.len();
            
            // Add tokens
            input_ids_vec.extend_from_slice(&tokens);
            
            // Pad if needed
            if len < padded_len {
                input_ids_vec.resize(input_ids_vec.len() + padded_len - len, PAD_TOKEN_ID);
            }
            
            // Create attention mask
            attention_mask_vec.extend(vec![1u32; len]);
            attention_mask_vec.resize(attention_mask_vec.len() + padded_len - len, 0);
        }
        
        let input_ids = Tensor::from_vec(
            input_ids_vec, 
            (batch_size, padded_len), 
            device
        )?;
        
        let attention_mask = Tensor::from_vec(
            attention_mask_vec,
            (batch_size, padded_len),
            device
        )?;
        
        Ok(TokenizedTensor {
            input_ids,
            attention_mask,
        })
    }
}

// Example usage module
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_candle_integration() {
        let device = Device::Cpu;
        let tokenizer = FlanT5Tokenizer::with_default_config();
        
        let text = "Test sentence for Candle.";
        let tensor = tokenizer.tokenize_to_tensor(text, &device).unwrap();
        
        assert_eq!(tensor.input_ids.dims(), &[1, tensor.input_ids.dims()[1]]);
        assert_eq!(tensor.attention_mask.dims(), tensor.input_ids.dims());
    }
} 