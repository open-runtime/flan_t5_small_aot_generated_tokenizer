use flan_t5_tokenizer::FlanT5Tokenizer;

fn main() {
    let tokenizer = FlanT5Tokenizer::with_default_config();
    
    // Test preprocessing
    println!("Preprocessing test:");
    let text = "Hello world!";
    
    // We need to access the preprocessed text somehow
    // Let's encode and decode to see what's happening
    let tokens = tokenizer.encode(text).unwrap();
    println!("Text: '{}'", text);
    println!("Token count: {}", tokens.len());
    println!("Token IDs: {:?}", tokens);
    
    // Compare with HuggingFace expected behavior
    println!("\nExpected behavior:");
    println!("HuggingFace would produce ~3 tokens for 'Hello world!'");
    println!("We're producing {} tokens", tokens.len());
    
    // Test some specific tokens
    println!("\nVocabulary test:");
    let test_words = ["▁Hello", "Hello", "▁world", "world", "!", "▁", "H", "e", "l", "o"];
    for word in &test_words {
        if let Some(id) = tokenizer.token_to_id(word) {
            println!("  '{}' -> ID {}", word, id);
        } else {
            println!("  '{}' -> NOT FOUND", word);
        }
    }
    
    // Test longer text
    println!("\nLonger text test:");
    let long_text = "The quick brown fox jumps over the lazy dog.";
    let long_tokens = tokenizer.encode(long_text).unwrap();
    println!("Text: '{}'", long_text);
    println!("Chars: {}, Tokens: {}", long_text.len(), long_tokens.len());
    
    // Decode back
    let decoded = tokenizer.decode(&tokens).unwrap();
    println!("\nDecode test:");
    println!("Original: '{}'", text);
    println!("Decoded:  '{}'", decoded);
    println!("Match: {}", text == decoded);
} 