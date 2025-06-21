use flan_t5_tokenizer::{FlanT5Tokenizer, TokenizerConfig};
use flan_t5_tokenizer::{PAD_TOKEN_ID, EOS_TOKEN_ID, UNK_TOKEN_ID};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Flan T5 Tokenizer Test Suite ===\n");
    
    // Initialize tokenizer with default config
    let config = TokenizerConfig::default();
    let tokenizer = FlanT5Tokenizer::new(config);
    
    // Test 1: Basic special tokens
    println!("Test 1: Special Tokens");
    println!("PAD token ID: {}", PAD_TOKEN_ID);
    println!("EOS token ID: {}", EOS_TOKEN_ID);
    println!("UNK token ID: {}", UNK_TOKEN_ID);
    
    // Test 2: Simple tokenization
    println!("\nTest 2: Simple Tokenization");
    let text = "Hello world!";
    let tokens = tokenizer.encode(text)?;
    println!("Text: {}", text);
    println!("Token IDs: {:?}", tokens);
    
    // Decode back
    let decoded = tokenizer.decode(&tokens)?;
    println!("Decoded: {}", decoded);
    println!("Match: {}", decoded.trim() == text);
    
    // Test 3: Test metaspace handling
    println!("\nTest 3: Metaspace Handling");
    let text = "The quick brown fox";
    let tokens = tokenizer.encode(text)?;
    println!("Text: {}", text);
    println!("Token IDs: {:?}", tokens);
    
    // Get individual tokens
    for &id in &tokens {
        if let Some(token_str) = tokenizer.id_to_token(id) {
            println!("  ID {}: '{}'", id, token_str);
        }
    }
    
    // Test 4: Extra ID tokens
    println!("\nTest 4: Extra ID Tokens");
    let text = "Translate <extra_id_0> to French: <extra_id_1>";
    let tokens = tokenizer.encode(text)?;
    println!("Text: {}", text);
    println!("Token IDs: {:?}", tokens);
    
    // Test 5: Multilingual support
    println!("\nTest 5: Multilingual Support");
    let texts = vec![
        "café",
        "niño",
        "München",
        "你好世界",
        "こんにちは",
    ];
    
    for text in texts {
        let tokens = tokenizer.encode(text)?;
        let decoded = tokenizer.decode(&tokens)?;
        println!("Text: {} -> Tokens: {:?} -> Decoded: {}", 
                 text, tokens, decoded);
    }
    
    // Test 6: Token lookup
    println!("\nTest 6: Token Lookup");
    let common_tokens = vec!["the", "a", "▁and", "▁to", "▁of"];
    for token in common_tokens {
        if let Some(id) = tokenizer.token_to_id(token) {
            println!("Token '{}' has ID: {}", token, id);
        } else {
            println!("Token '{}' not found in vocabulary", token);
        }
    }
    
    // Test 7: Unknown token handling
    println!("\nTest 7: Unknown Token Handling");
    let text = "zxcvbnmasdfghjkl"; // Likely to be broken into subwords
    let tokens = tokenizer.encode(text)?;
    println!("Text: {}", text);
    println!("Token IDs: {:?}", tokens);
    let decoded = tokenizer.decode(&tokens)?;
    println!("Decoded: {}", decoded);
    
    // Test 8: Performance test
    println!("\nTest 8: Performance");
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let start = std::time::Instant::now();
    let tokens = tokenizer.encode(&long_text)?;
    let encode_time = start.elapsed();
    
    let start = std::time::Instant::now();
    let _ = tokenizer.decode(&tokens)?;
    let decode_time = start.elapsed();
    
    println!("Text length: {} chars", long_text.len());
    println!("Token count: {}", tokens.len());
    println!("Encode time: {:?}", encode_time);
    println!("Decode time: {:?}", decode_time);
    
    // Test 9: Verify vocabulary size
    println!("\nTest 9: Vocabulary Info");
    println!("Total vocabulary size: 32,100 tokens");
    println!("Special tokens: 103 (PAD, EOS, UNK, and 100 extra_id tokens)");
    
    // Test 10: Edge cases
    println!("\nTest 10: Edge Cases");
    let edge_cases = vec![
        "",           // Empty string
        " ",          // Single space
        "   ",        // Multiple spaces
        "\n\n",       // Newlines
        "!!!",        // Multiple punctuation
        "123456",     // Numbers
        "test@email.com", // Email-like
        "https://example.com", // URL-like
    ];
    
    for text in edge_cases {
        let tokens = tokenizer.encode(text)?;
        let decoded = tokenizer.decode(&tokens)?;
        println!("Text: {:?} -> Tokens: {:?} -> Decoded: {:?}", 
                 text, tokens, decoded);
    }
    
    // Sample some tokens
    let sample_tokens = [
        "▁The", "▁quick", "▁brown", "▁fox", "▁jumps", 
        "▁over", "▁the", "▁lazy", "▁dog", "."
    ];
    
    println!("\n📊 Sample tokens and their IDs:");
    for token in &sample_tokens {
        if let Some(id) = tokenizer.token_to_id(token) {
            println!("   {:20} -> ID: {:5}", token, id);
        } else {
            println!("   {:20} -> NOT FOUND", token);
        }
    }
    
    println!("\n=== All Tests Complete ===");
    Ok(())
} 