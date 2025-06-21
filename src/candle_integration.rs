use crate::{FlanT5Tokenizer, Result, PAD_TOKEN_ID, EOS_TOKEN_ID};
use candle_core::{Device, DType, Tensor};

#[derive(Debug)]
pub struct TokenizedTensor {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
    pub position_ids: Option<Tensor>,
    pub token_type_ids: Option<Tensor>,
}

impl TokenizedTensor {
    /// Get the sequence length (excluding batch dimension)
    pub fn seq_len(&self) -> usize {
        self.input_ids.dims()[1]
    }
    
    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.input_ids.dims()[0]
    }
    
    /// Move tensors to a different device
    pub fn to_device(&self, device: &Device) -> Result<TokenizedTensor> {
        Ok(TokenizedTensor {
            input_ids: self.input_ids.to_device(device)?,
            attention_mask: self.attention_mask.to_device(device)?,
            position_ids: self.position_ids.as_ref()
                .map(|t| t.to_device(device))
                .transpose()?,
            token_type_ids: self.token_type_ids.as_ref()
                .map(|t| t.to_device(device))
                .transpose()?,
        })
    }
}

pub trait TokenizerCandle {
    /// Tokenize a single text to tensor
    fn tokenize_to_tensor(&self, text: &str, device: &Device) -> Result<TokenizedTensor>;
    
    /// Tokenize multiple texts to a batched tensor
    fn batch_tokenize_to_tensor(
        &self, 
        texts: &[&str], 
        device: &Device
    ) -> Result<TokenizedTensor>;
    
    /// Decode token IDs from tensor back to text
    fn decode_from_tensor(&self, input_ids: &Tensor) -> Result<Vec<String>>;
    
    /// Create position IDs for the input
    fn create_position_ids(&self, seq_len: usize, batch_size: usize, device: &Device) -> Result<Tensor>;
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
            position_ids: None,
            token_type_ids: None,
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
            position_ids: None,
            token_type_ids: None,
        })
    }
    
    fn decode_from_tensor(&self, input_ids: &Tensor) -> Result<Vec<String>> {
        let dims = input_ids.dims();
        
        // Handle both 1D and 2D tensors
        let (batch_size, seq_len) = match dims.len() {
            1 => (1, dims[0]),
            2 => (dims[0], dims[1]),
            _ => return Err(crate::TokenizerError::TokenNotFound(
                "Input tensor must be 1D or 2D".to_string()
            )),
        };
        
        // Convert tensor to Vec<u32>
        let ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        
        // Decode each sequence in the batch
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * seq_len;
            let end = start + seq_len;
            let sequence = &ids_vec[start..end];
            
            // Find where padding starts (if any)
            let actual_len = sequence.iter()
                .position(|&id| id == PAD_TOKEN_ID)
                .unwrap_or(sequence.len());
            
            let decoded = self.decode(&sequence[..actual_len])?;
            results.push(decoded);
        }
        
        Ok(results)
    }
    
    fn create_position_ids(&self, seq_len: usize, batch_size: usize, device: &Device) -> Result<Tensor> {
        // Create position IDs [0, 1, 2, ..., seq_len-1]
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        
        // Create tensor and expand for batch
        let pos_tensor = Tensor::from_vec(position_ids, seq_len, device)?;
        
        // Expand to batch size
        if batch_size > 1 {
            pos_tensor.unsqueeze(0)?.expand((batch_size, seq_len))
        } else {
            Ok(pos_tensor.unsqueeze(0)?)
        }
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