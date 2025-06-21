use flan_t5_tokenizer::{TOKEN_TO_ID, TOKEN_SCORES};

fn main() {
    println!("=== Viterbi Algorithm Debug ===\n");
    
    let text = "hello";
    println!("Input text: {:?}\n", text);
    
    // Test what tokens we can find
    println!("Token lookups:");
    
    // Check individual characters
    for ch in text.chars() {
        let ch_str = ch.to_string();
        if let Some(&id) = TOKEN_TO_ID.get(&ch_str) {
            let score = TOKEN_SCORES.get(&ch_str).copied().unwrap_or(999.0);
            println!("  {:?} -> ID: {}, Score: {:.2}", ch_str, id, score);
        } else {
            println!("  {:?} -> NOT FOUND", ch_str);
        }
    }
    
    // Check with space marker
    println!("\nWith space marker:");
    for ch in text.chars() {
        let ch_with_marker = format!("▁{}", ch);
        if let Some(&id) = TOKEN_TO_ID.get(&ch_with_marker) {
            let score = TOKEN_SCORES.get(&ch_with_marker).copied().unwrap_or(999.0);
            println!("  {:?} -> ID: {}, Score: {:.2}", ch_with_marker, id, score);
        } else {
            println!("  {:?} -> NOT FOUND", ch_with_marker);
        }
    }
    
    // Check the full word
    println!("\nFull word:");
    if let Some(&id) = TOKEN_TO_ID.get(text) {
        let score = TOKEN_SCORES.get(text).copied().unwrap_or(999.0);
        println!("  {:?} -> ID: {}, Score: {:.2}", text, id, score);
    } else {
        println!("  {:?} -> NOT FOUND", text);
    }
    
    // Check with space marker
    let text_with_marker = format!("▁{}", text);
    if let Some(&id) = TOKEN_TO_ID.get(&text_with_marker) {
        let score = TOKEN_SCORES.get(&text_with_marker).copied().unwrap_or(999.0);
        println!("  {:?} -> ID: {}, Score: {:.2}", text_with_marker, id, score);
    } else {
        println!("  {:?} -> NOT FOUND", text_with_marker);
    }
    
    // Now let's simulate what our algorithm should be doing
    println!("\n\nSimulating tokenization:");
    println!("Text starts at beginning, so we should look for tokens with ▁ prefix");
    
    // For position 0 (start of text)
    println!("\nPosition 0:");
    
    // Check progressively longer substrings
    let chars: Vec<char> = text.chars().collect();
    for len in 1..=chars.len() {
        let substr: String = chars[0..len].iter().collect();
        let substr_with_marker = format!("▁{}", substr);
        
        if let Some(&id) = TOKEN_TO_ID.get(&substr_with_marker) {
            let score = TOKEN_SCORES.get(&substr_with_marker).copied().unwrap_or(999.0);
            println!("  {:?} -> ID: {}, Score: {:.2} ✓", substr_with_marker, id, score);
        } else {
            println!("  {:?} -> NOT FOUND", substr_with_marker);
        }
    }
    
    // Compare scores
    println!("\n\nScore comparison:");
    println!("Character-by-character path:");
    let mut char_score = 0.0;
    for ch in text.chars() {
        if let Some(_) = TOKEN_TO_ID.get(&ch.to_string()) {
            let score = TOKEN_SCORES.get(&ch.to_string()).copied().unwrap_or(999.0);
            println!("  '{}': {:.2}", ch, score);
            char_score += score;
        }
    }
    println!("  Total: {:.2}", char_score);
    
    println!("\nWhole word path:");
    if let Some(&id) = TOKEN_TO_ID.get(&text_with_marker) {
        let score = TOKEN_SCORES.get(&text_with_marker).copied().unwrap_or(999.0);
        println!("  {:?} (ID {}): {:.2}", text_with_marker, id, score);
        println!("  Total: {:.2}", score);
    }
    
    println!("\nLower score wins! The whole word should be chosen.");
} 