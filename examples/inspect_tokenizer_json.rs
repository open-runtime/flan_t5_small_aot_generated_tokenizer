use serde_json::Value;
use std::fs;

fn main() {
    println!("=== Inspecting FLAN-T5 Tokenizer JSON ===\n");
    
    // Read the tokenizer JSON
    let json_str = fs::read_to_string("flan_t5_small_tokenizer.json")
        .expect("Failed to read tokenizer JSON");
    
    let json: Value = serde_json::from_str(&json_str)
        .expect("Failed to parse JSON");
    
    // Check the structure
    if let Some(model) = json.get("model") {
        println!("Model type: {}", model.get("type").unwrap_or(&Value::Null));
        
        if let Some(vocab) = model.get("vocab") {
            if let Some(vocab_array) = vocab.as_array() {
                println!("Vocabulary size: {}", vocab_array.len());
                
                // Show first 20 tokens
                println!("\nFirst 20 vocabulary entries:");
                for (i, entry) in vocab_array.iter().take(20).enumerate() {
                    if let Some(arr) = entry.as_array() {
                        if arr.len() >= 2 {
                            let token = arr[0].as_str().unwrap_or("?");
                            let score = arr[1].as_f64().unwrap_or(0.0);
                            println!("  {}: {:?} (score: {})", i, token, score);
                        }
                    }
                }
                
                // Look for special tokens
                println!("\nLooking for special tokens:");
                for (i, entry) in vocab_array.iter().enumerate() {
                    if let Some(arr) = entry.as_array() {
                        if let Some(token) = arr[0].as_str() {
                            if token.starts_with("<") && token.ends_with(">") {
                                println!("  {}: {:?}", i, token);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Check added tokens
    if let Some(added_tokens) = json.get("added_tokens") {
        if let Some(tokens_array) = added_tokens.as_array() {
            println!("\nAdded tokens ({}):", tokens_array.len());
            for token in tokens_array.iter().take(10) {
                if let Some(obj) = token.as_object() {
                    let id = obj.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                    let content = obj.get("content").and_then(|v| v.as_str()).unwrap_or("?");
                    let special = obj.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
                    println!("  ID {}: {:?} (special: {})", id, content, special);
                }
            }
        }
    }
    
    // Check normalizer
    if let Some(normalizer) = json.get("normalizer") {
        println!("\nNormalizer: {}", normalizer.get("type").unwrap_or(&Value::Null));
    }
    
    // Check pre_tokenizer
    if let Some(pre_tokenizer) = json.get("pre_tokenizer") {
        println!("\nPre-tokenizer: {}", pre_tokenizer.get("type").unwrap_or(&Value::Null));
    }
    
    // Check post_processor
    if let Some(post_processor) = json.get("post_processor") {
        println!("\nPost-processor: {}", serde_json::to_string_pretty(post_processor).unwrap_or_default());
    }
} 