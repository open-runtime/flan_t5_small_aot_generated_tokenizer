use flan_t5_tokenizer::{FlanT5Tokenizer, TOKEN_TO_ID, TOKEN_SCORES};

fn main() {
    println!("=== Detailed Tokenization Trace ===\n");
    
    let tokenizer = FlanT5Tokenizer::with_default_config();
    let text = "hello";
    
    println!("Input: {:?}", text);
    println!("This should tokenize to [21820] (▁hello)\n");
    
    // Let's manually trace what should happen
    println!("Manual trace of Viterbi algorithm:");
    println!("Text at start, so we need to check with ▁ prefix\n");
    
    // Check what tokens are available
    println!("Available tokens from position 0:");
    let chars: Vec<char> = text.chars().collect();
    
    // Check each possible substring from position 0
    for len in 1..=chars.len() {
        let substr: String = chars[0..len].iter().collect();
        
        // Check with space marker (since we're at the beginning)
        let with_marker = format!("▁{}", substr);
        if let Some(&id) = TOKEN_TO_ID.get(&with_marker) {
            let score = TOKEN_SCORES.get(&with_marker).copied().unwrap_or(999.0);
            println!("  {:?} -> ID {} (score: {:.2})", with_marker, id, score);
        }
        
        // Also check without marker
        if let Some(&id) = TOKEN_TO_ID.get(&substr) {
            let score = TOKEN_SCORES.get(&substr).copied().unwrap_or(999.0);
            println!("  {:?} -> ID {} (score: {:.2})", substr, id, score);
        }
    }
    
    // Now run our actual tokenizer
    println!("\n\nActual tokenization result:");
    let tokens = tokenizer.encode(text).expect("Tokenization failed");
    println!("Tokens: {:?}", tokens);
    
    // Decode to verify
    let decoded = tokenizer.decode(&tokens).expect("Decode failed");
    println!("Decoded: {:?}", decoded);
    
    // Show what each token is
    println!("\nToken breakdown:");
    for (i, &token_id) in tokens.iter().enumerate() {
        if let Some(token_str) = flan_t5_tokenizer::id_to_token(token_id) {
            let score = TOKEN_SCORES.get(token_str).copied().unwrap_or(999.0);
            println!("  [{}] ID {} = {:?} (score: {:.2})", i, token_id, token_str, score);
        }
    }
    
    // Test with a space prefix to see if that changes things
    println!("\n\nTesting with space prefix:");
    let spaced_text = format!(" {}", text);
    let spaced_tokens = tokenizer.encode(&spaced_text).expect("Tokenization failed");
    println!("Input: {:?}", spaced_text);
    println!("Tokens: {:?}", spaced_tokens);
    
    // Compare with HuggingFace
    println!("\n\nComparison with HuggingFace:");
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HF tokenizer");
    let hf_encoding = hf_tokenizer.encode(text, false).expect("HF encode failed");
    let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
    println!("HF tokens: {:?}", hf_tokens);
} 