use flan_t5_tokenizer::{FlanT5Tokenizer, TOKEN_TO_ID, id_to_token};

fn main() {
    println!("=== Detailed Tokenizer Debugging ===\n");
    
    // Initialize tokenizers
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    // Test case 1: Simple word tokenization
    println!("=== Test 1: Simple Words ===");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "hello");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "world");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "Hello");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "World");
    
    // Test case 2: Phrases with spaces
    println!("\n=== Test 2: Phrases with Spaces ===");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "hello world");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "Hello world");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "Hello World");
    
    // Test case 3: Metaspace handling
    println!("\n=== Test 3: Space Handling ===");
    test_tokenization(&our_tokenizer, &hf_tokenizer, " hello");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "hello ");
    test_tokenization(&our_tokenizer, &hf_tokenizer, " hello ");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "  hello  ");
    
    // Test case 4: Special subwords
    println!("\n=== Test 4: Subword Tokenization ===");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "recurring");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "recur");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "ring");
    
    // Test case 5: Problematic examples from tests
    println!("\n=== Test 5: Problematic Examples ===");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "Create a recurring meeting for team standup");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "Define serendipity");
    test_tokenization(&our_tokenizer, &hf_tokenizer, "undefined");
    
    // Test case 6: Check specific tokens
    println!("\n=== Test 6: Specific Token Checks ===");
    check_token_mapping(&our_tokenizer, &hf_tokenizer, "▁");
    check_token_mapping(&our_tokenizer, &hf_tokenizer, "▁recur");
    check_token_mapping(&our_tokenizer, &hf_tokenizer, "recur");
    check_token_mapping(&our_tokenizer, &hf_tokenizer, "▁recurring");
    check_token_mapping(&our_tokenizer, &hf_tokenizer, "recurring");
    check_token_mapping(&our_tokenizer, &hf_tokenizer, "serend");
    check_token_mapping(&our_tokenizer, &hf_tokenizer, "ipity");
    check_token_mapping(&our_tokenizer, &hf_tokenizer, "serendipity");
    check_token_mapping(&our_tokenizer, &hf_tokenizer, "▁serendipity");
    check_token_mapping(&our_tokenizer, &hf_tokenizer, "un");
    check_token_mapping(&our_tokenizer, &hf_tokenizer, "defined");
    check_token_mapping(&our_tokenizer, &hf_tokenizer, "undefined");
    
    // Test case 7: Direct vocabulary lookup
    println!("\n=== Test 7: Vocabulary Analysis ===");
    analyze_vocabulary();
}

fn test_tokenization(our: &FlanT5Tokenizer, hf: &tokenizers::Tokenizer, text: &str) {
    let our_tokens = our.encode(text).expect("Our tokenizer failed");
    let hf_encoding = hf.encode(text, false).expect("HF tokenizer failed");
    let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
    
    println!("\nText: {:?}", text);
    println!("Our: {:?}", our_tokens);
    println!("HF:  {:?}", hf_tokens);
    
    if our_tokens != hf_tokens {
        println!("❌ MISMATCH");
        
        // Decode each tokenizer's output
        let our_decoded = our.decode(&our_tokens).unwrap_or_else(|_| "DECODE_ERROR".to_string());
        let hf_decoded = hf.decode(&hf_tokens, false).unwrap_or_else(|_| "DECODE_ERROR".to_string());
        
        println!("Our decoded: {:?}", our_decoded);
        println!("HF decoded:  {:?}", hf_decoded);
        
        // Show token-by-token breakdown
        println!("Token breakdown:");
        for (i, &token_id) in hf_tokens.iter().enumerate() {
            let token_str = hf.decode(&[token_id], false).unwrap_or_else(|_| format!("<{}>", token_id));
            println!("  HF[{}]: {} -> {:?}", i, token_id, token_str);
        }
        for (i, &token_id) in our_tokens.iter().enumerate() {
            let token_str = id_to_token(token_id).unwrap_or("<UNK>");
            println!("  Our[{}]: {} -> {:?}", i, token_id, token_str);
        }
    } else {
        println!("✅ MATCH");
    }
}

fn check_token_mapping(_our: &FlanT5Tokenizer, hf: &tokenizers::Tokenizer, token: &str) {
    println!("Checking token: {}", token);
    
    // Check if token exists in vocabulary
    if let Some(&id) = TOKEN_TO_ID.get(token) {
        println!("  Found in vocab with ID: {}", id);
        
        // See what HF encodes this as
        let hf_encoding = hf.encode(token, false).expect("HF encode failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        println!("  HF encodes as: {:?}", hf_tokens);
    } else {
        println!("  NOT in vocabulary");
    }
}

fn analyze_vocabulary() {
    use std::collections::HashMap;
    
    println!("\nVocabulary statistics:");
    
    // Count tokens by prefix
    let mut prefix_counts: HashMap<&str, usize> = HashMap::new();
    let mut token_lengths: HashMap<usize, usize> = HashMap::new();
    let mut sample_tokens: Vec<(&str, u32)> = Vec::new();
    
    for (token, &id) in TOKEN_TO_ID.entries() {
        // Count by prefix
        if token.starts_with("▁") {
            *prefix_counts.entry("▁ (space)").or_insert(0) += 1;
        } else if token.starts_with("<0x") {
            *prefix_counts.entry("<0x (byte)").or_insert(0) += 1;
        } else if token.starts_with("<") {
            *prefix_counts.entry("< (special)").or_insert(0) += 1;
        } else {
            *prefix_counts.entry("(regular)").or_insert(0) += 1;
        }
        
        // Count by length
        *token_lengths.entry(token.len()).or_insert(0) += 1;
        
        // Collect samples
        if sample_tokens.len() < 20 && token.contains("recur") {
            sample_tokens.push((token, id));
        }
    }
    
    println!("\nToken prefix distribution:");
    for (prefix, count) in &prefix_counts {
        println!("  {}: {}", prefix, count);
    }
    
    println!("\nToken length distribution (top 10):");
    let mut lengths: Vec<_> = token_lengths.into_iter().collect();
    lengths.sort_by_key(|&(len, _)| len);
    for (len, count) in lengths.iter().take(10) {
        println!("  Length {}: {} tokens", len, count);
    }
    
    println!("\nSample tokens containing 'recur':");
    for (token, id) in &sample_tokens {
        println!("  {:?} -> {}", token, id);
    }
    
    // Check specific problematic tokens
    println!("\nChecking specific tokens:");
    let check_tokens = vec!["recurring", "▁recurring", "recur", "▁recur", "ring", "▁ring"];
    for token in check_tokens {
        if let Some(&id) = TOKEN_TO_ID.get(token) {
            println!("  {:?} -> {}", token, id);
        } else {
            println!("  {:?} -> NOT FOUND", token);
        }
    }
} 