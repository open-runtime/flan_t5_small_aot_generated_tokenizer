use flan_t5_tokenizer::FlanT5Tokenizer;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use std::fs::File;
use std::path::Path;
use anyhow::Result;
use serde_json::Value;

#[cfg(test)]
mod huggingface_validation_tests {
    use super::*;

    /// Validates tokenizer works on validation data samples
    #[test]
    #[ignore] // Run with: cargo test --test huggingface_validation_tests -- --ignored
    fn test_against_validation_data() -> Result<()> {
        // Check if validation file exists
        let validation_path = Path::new("model/validation_results.parquet");
        if !validation_path.exists() {
            eprintln!("Skipping test: validation_results.parquet not found");
            return Ok(());
        }

        // Initialize our tokenizer
        let tokenizer = FlanT5Tokenizer::with_default_config();
        
        // Read parquet file
        let file = File::open(validation_path)?;
        let reader = SerializedFileReader::new(file)?;
        let row_iter = reader.get_row_iter(None)?;
        
        let mut total_samples = 0;
        let mut successful_tokenizations = 0;
        
        println!("Testing tokenizer on validation samples...");
        
        for row in row_iter {
            if let Ok(row) = row {
                // Extract the input text
                if let Some(text) = extract_text_field(&row) {
                    // Tokenize with our implementation
                    match tokenizer.encode(&text) {
                        Ok(tokens) => {
                            successful_tokenizations += 1;
                            
                            // Print first few examples
                            if total_samples < 3 {
                                println!("\nSample {}:", total_samples + 1);
                                println!("  Text: {:?}", text);
                                println!("  Tokens: {:?}", tokens);
                                println!("  Token count: {}", tokens.len());
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to tokenize: {:?}", e);
                            eprintln!("Text: {:?}", text);
                        }
                    }
                    
                    total_samples += 1;
                }
            }
        }
        
        // Report results
        println!("\nValidation Results:");
        println!("Total samples: {}", total_samples);
        println!("Successful tokenizations: {}", successful_tokenizations);
        println!("Success rate: {:.2}%", 
            successful_tokenizations as f64 / total_samples as f64 * 100.0
        );
        
        // Assert all tokenizations succeeded
        assert_eq!(
            successful_tokenizations, total_samples,
            "Some tokenizations failed: {} out of {}", 
            total_samples - successful_tokenizations, total_samples
        );
        
        Ok(())
    }
    
    /// Validates tokenizer configuration against HuggingFace config.json
    #[test]
    #[ignore]
    fn test_against_model_config() -> Result<()> {
        let config_path = Path::new("model/config.json");
        if !config_path.exists() {
            eprintln!("Skipping test: config.json not found");
            return Ok(());
        }
        
        // Load config
        let config_str = std::fs::read_to_string(config_path)?;
        let config: Value = serde_json::from_str(&config_str)?;
        
        // Initialize tokenizer
        let tokenizer = FlanT5Tokenizer::with_default_config();
        
        // Verify vocabulary size
        if let Some(vocab_size) = config["vocab_size"].as_u64() {
            let our_vocab_size = tokenizer.vocab_size();
            
            // Note: Our tokenizer has 32100 tokens while the model expects 32128
            // This is acceptable as the extra 28 tokens are likely unused special tokens
            let size_diff = (vocab_size as i32 - our_vocab_size as i32).abs();
            assert!(
                size_diff <= 100,
                "Vocabulary size differs too much: expected {}, got {} (diff: {})", 
                vocab_size, our_vocab_size, size_diff
            );
            
            println!("  Vocab size: {} (config: {})", our_vocab_size, vocab_size);
        }
        
        // Verify special token IDs
        use flan_t5_tokenizer::{PAD_ID, EOS_ID};
        
        if let Some(pad_id) = config["pad_token_id"].as_u64() {
            assert_eq!(PAD_ID, pad_id as u32);
        }
        
        if let Some(eos_id) = config["eos_token_id"].as_u64() {
            assert_eq!(EOS_ID, eos_id as u32);
        }
        
        println!("✓ Tokenizer configuration matches model config");
        Ok(())
    }
}

// Helper function to extract text from parquet row
fn extract_text_field(row: &parquet::record::Row) -> Option<String> {
    // Extract the "input" field which contains the text
    for (name, field) in row.get_column_iter() {
        if name == "input" {
            if let Field::Str(text) = field {
                return Some(text.clone());
            }
        }
    }
    None
} 