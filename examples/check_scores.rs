use flan_t5_tokenizer::FlanT5Tokenizer;

fn main() {
    let tokenizer = FlanT5Tokenizer::with_default_config();
    
    println!("Checking token scores for character vs word tokens:\n");
    
    // Check character tokens
    println!("Character tokens:");
    let chars = ['H', 'e', 'l', 'o', 'w', 'r', 'd', '!', '▁'];
    for ch in &chars {
        if let Some(id) = tokenizer.token_to_id(&ch.to_string()) {
            println!("  '{}' -> ID {} (character)", ch, id);
        }
    }
    
    println!("\nWord/subword tokens:");
    let words = ["▁Hello", "▁world", "Hello", "world", "▁He", "▁Hell", "▁", "!"];
    for word in &words {
        if let Some(id) = tokenizer.token_to_id(word) {
            println!("  '{}' -> ID {}", word, id);
        }
    }
    
    // The key issue: check if we can trace through the Viterbi path
    println!("\n\nViterbi path simulation for 'Hello world!':");
    println!("After preprocessing: '▁Hello▁world!'\n");
    
    // Position 0: starting with ▁
    println!("From position 0:");
    let test_spans = [
        ("▁", 0, 1),
        ("▁H", 0, 2), 
        ("▁He", 0, 3),
        ("▁Hel", 0, 4),
        ("▁Hell", 0, 5),
        ("▁Hello", 0, 6),
        ("▁Hello▁", 0, 7),
        ("▁Hello▁w", 0, 8),
        ("▁Hello▁world", 0, 12),
        ("▁Hello▁world!", 0, 13),
    ];
    
    for (span, start, end) in &test_spans {
        if let Some(id) = tokenizer.token_to_id(span) {
            println!("  '{}' [{}:{}] -> Found token ID {}", span, start, end, id);
        }
    }
    
    // The issue might be with how we're iterating through the string
    println!("\n\nChecking byte indices:");
    let text = "▁Hello▁world!";
    let char_indices: Vec<(usize, char)> = text.char_indices().collect();
    println!("Text: '{}'", text);
    println!("Char indices:");
    for (i, (byte_idx, ch)) in char_indices.iter().enumerate() {
        println!("  [{}] byte {} -> '{}'", i, byte_idx, ch);
    }
    
    // Test substrings by byte indices
    println!("\nSubstrings by byte indices:");
    for i in 0..char_indices.len() {
        for j in (i+1)..=char_indices.len().min(i+10) {
            let start_byte = char_indices[i].0;
            let end_byte = if j < char_indices.len() {
                char_indices[j].0
            } else {
                text.len()
            };
            let substr = &text[start_byte..end_byte];
            if tokenizer.token_to_id(substr).is_some() {
                println!("  [{},{}] -> '{}'", start_byte, end_byte, substr);
            }
        }
    }
} 