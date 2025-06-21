use flan_t5_tokenizer::{FlanT5Tokenizer, TokenizerConfig};

// Reimplement a simple version to understand the issue
fn debug_tokenize(text: &str) {
    println!("=== Debug Tokenization for: '{}' ===", text);
    
    // First, preprocess the text
    let mut preprocessed = String::new();
    preprocessed.push('▁'); // Add prefix space
    
    for ch in text.chars() {
        if ch == ' ' {
            preprocessed.push('▁');
        } else {
            preprocessed.push(ch);
        }
    }
    
    println!("Preprocessed: '{}'", preprocessed);
    
    // Create tokenizer to check vocabulary
    let tokenizer = FlanT5Tokenizer::with_default_config();
    
    // Check what tokens exist for substrings
    println!("\nChecking substrings:");
    let test_substrings = [
        "▁Hello",
        "▁Hell",
        "▁Hel", 
        "▁He",
        "▁H",
        "▁",
        "Hello",
        "world",
        "▁world",
        "▁world!",
        "!",
    ];
    
    for substr in &test_substrings {
        if let Some(id) = tokenizer.token_to_id(substr) {
            println!("  '{}' -> Token ID {}", substr, id);
        }
    }
    
    // Now tokenize with the actual tokenizer
    let tokens = tokenizer.encode(text).unwrap();
    println!("\nActual tokenization:");
    println!("Token count: {}", tokens.len());
    println!("Token IDs: {:?}", tokens);
    
    // Decode to verify
    let decoded = tokenizer.decode(&tokens).unwrap();
    println!("\nDecoded: '{}'", decoded);
    println!("Match: {}", text == decoded);
}

fn main() {
    println!("Debug Viterbi Tokenization\n");
    
    // Test cases
    debug_tokenize("Hello world!");
    
    println!("\n{}\n", "=".repeat(50));
    
    debug_tokenize("Hello");
    
    println!("\n{}\n", "=".repeat(50));
    
    // Test if the issue is with the preprocessing
    let tokenizer = FlanT5Tokenizer::with_default_config();
    
    // Try encoding preprocessed text directly
    println!("Testing direct token lookup:");
    let test_tokens = ["▁Hello", "▁world", "!", "▁The", "▁quick"];
    for token in &test_tokens {
        if let Some(id) = tokenizer.token_to_id(token) {
            println!("  Token '{}' -> ID {}", token, id);
            
            // Try to encode just this token
            let encoded = tokenizer.encode(token).unwrap();
            println!("    Encoding '{}' produces: {:?}", token, encoded);
        }
    }
} 