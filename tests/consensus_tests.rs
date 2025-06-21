/// Comprehensive consensus tests comparing our tokenizer implementation
/// against HuggingFace tokenizers and rust_tokenizers

use flan_t5_tokenizer::FlanT5Tokenizer;
use rust_tokenizers::tokenizer::{T5Tokenizer, Tokenizer as RustTokenizer, TruncationStrategy};

/// Test cases covering various scenarios
const TEST_CASES: &[(&str, &str)] = &[
    // Basic text
    ("Hello world!", "simple_greeting"),
    ("The quick brown fox jumps over the lazy dog.", "pangram"),
    
    // Empty and whitespace
    ("", "empty"),
    (" ", "single_space"),
    ("   ", "multiple_spaces"),
    ("\n", "newline"),
    ("\t", "tab"),
    
    // Punctuation heavy
    ("Hello, world! How are you?", "punctuation"),
    ("test@email.com", "email"),
    ("https://example.com", "url"),
    ("$123.45", "currency"),
    
    // Numbers
    ("123", "simple_number"),
    ("3.14159", "decimal"),
    ("1,234,567", "formatted_number"),
    
    // Special tokens
    ("<pad>", "pad_token"),
    ("</s>", "eos_token"),
    ("<unk>", "unk_token"),
    ("<extra_id_0>", "extra_id_0"),
    ("<extra_id_99>", "extra_id_99"),
    
    // Mixed content
    ("Translate <extra_id_0> to French: <extra_id_1>", "t5_template"),
    
    // Unicode and multilingual
    ("café", "accented"),
    ("naïve", "diaeresis"),
    ("Zürich", "umlaut"),
    ("€100", "euro_symbol"),
    ("你好", "chinese"),
    ("こんにちは", "japanese"),
    ("مرحبا", "arabic"),
    ("🚀", "emoji"),
    
    // Edge cases
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "long_single_char"),
    ("word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word ", "repeated_word"),
    ("CamelCaseWord", "camelcase"),
    ("snake_case_word", "snakecase"),
    ("UPPERCASE", "uppercase"),
    
    // Sentence boundaries
    ("First sentence. Second sentence.", "two_sentences"),
    ("Question? Answer!", "question_answer"),
    
    // Code-like text
    ("function() { return 42; }", "code_snippet"),
    ("print('Hello, World!')", "python_code"),
    
    // Real-world examples
    ("The patient was diagnosed with COVID-19.", "medical"),
    ("Stock price increased by 5.2% today.", "financial"),
    ("Machine learning models require significant computational resources.", "technical"),
];

// Special tokens that should be handled specially
const SPECIAL_TOKENS: &[&str] = &[
    "<pad>",
    "</s>",
    "<unk>",
    "<extra_id_0>",
    "<extra_id_1>",
    "<extra_id_99>",
];

#[test]
fn test_tokenization_consensus() {
    // Initialize all tokenizers
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    let rust_tokenizer = T5Tokenizer::from_file("model/spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    println!("\n=== Tokenization Consensus Test ===");
    println!("Comparing: Our tokenizer vs HuggingFace vs rust_tokenizers\n");
    
    let mut total_tests = 0;
    let mut all_agree = 0;
    let mut our_hf_agree = 0;
    let mut our_rust_agree = 0;
    let mut hf_rust_agree = 0;
    let mut consensus_against_ours = 0;
    
    // Test basic tokenization
    for (text, expected_behavior) in TEST_CASES {
        total_tests += 1;
        
        let our_tokens = our_tokenizer.encode(text).expect("Our tokenizer failed");
        let hf_encoding = hf_tokenizer.encode(*text, false).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        let rust_encoding = rust_tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0);
        let rust_tokens: Vec<u32> = rust_encoding.token_ids.iter().map(|&id| id as u32).collect();
        
        let our_hf_match = our_tokens == hf_tokens;
        let our_rust_match = our_tokens == rust_tokens;
        let hf_rust_match = hf_tokens == rust_tokens;
        
        if our_hf_match { our_hf_agree += 1; }
        if our_rust_match { our_rust_agree += 1; }
        if hf_rust_match { hf_rust_agree += 1; }
        
        if our_hf_match && our_rust_match && hf_rust_match {
            all_agree += 1;
        } else {
            println!("❌ Disagreement on: \"{}\" ({})", text, expected_behavior);
            println!("   Our tokens:  {:?}", our_tokens);
            println!("   HF tokens:   {:?}", hf_tokens);
            println!("   Rust tokens: {:?}", rust_tokens);
            println!();
        }
        
        // Check if HF and rust agree but ours differs
        if hf_rust_match && !our_hf_match {
            consensus_against_ours += 1;
        }
    }
    
    // Test special tokens
    println!("\n--- Special Token Handling ---");
    for token in SPECIAL_TOKENS {
        total_tests += 1;
        
        let our_tokens = our_tokenizer.encode(token).expect("Our tokenizer failed");
        let hf_encoding = hf_tokenizer.encode(*token, false).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        let rust_encoding = rust_tokenizer.encode(token, None, 512, &TruncationStrategy::LongestFirst, 0);
        let rust_tokens: Vec<u32> = rust_encoding.token_ids.iter().map(|&id| id as u32).collect();
        
        let our_hf_match = our_tokens == hf_tokens;
        let our_rust_match = our_tokens == rust_tokens;
        let hf_rust_match = hf_tokens == rust_tokens;
        
        if our_hf_match { our_hf_agree += 1; }
        if our_rust_match { our_rust_agree += 1; }
        if hf_rust_match { hf_rust_agree += 1; }
        
        if our_hf_match && our_rust_match && hf_rust_match {
            all_agree += 1;
        } else {
            println!("Special token: {} -> Our: {:?}, HF: {:?}, Rust: {:?}", 
                token, our_tokens, hf_tokens, rust_tokens);
        }
        
        // Check if HF and rust agree but ours differs
        if hf_rust_match && !our_hf_match {
            consensus_against_ours += 1;
        }
    }
    
    // Print summary
    println!("\n=== Consensus Summary ===");
    println!("Total tests: {}", total_tests);
    println!("All three agree: {} ({:.1}%)", all_agree, all_agree as f64 / total_tests as f64 * 100.0);
    
    println!("\nPairwise agreement:");
    println!("Our ↔ HuggingFace: {} ({:.1}%)", our_hf_agree, our_hf_agree as f64 / total_tests as f64 * 100.0);
    println!("Our ↔ rust_tokenizers: {} ({:.1}%)", our_rust_agree, our_rust_agree as f64 / total_tests as f64 * 100.0);
    println!("HuggingFace ↔ rust_tokenizers: {} ({:.1}%)", hf_rust_agree, hf_rust_agree as f64 / total_tests as f64 * 100.0);
    
    if consensus_against_ours > 0 {
        println!("\nWARNING: {} cases where HF and rust_tokenizers agree but ours differs!", consensus_against_ours);
    }
    
    // Success criteria: Our tokenizer should match HuggingFace closely
    // rust_tokenizers has known differences (always adds EOS, different extra_id mapping)
    if our_hf_agree == total_tests {
        println!("\n✅ SUCCESS: Our tokenizer matches HuggingFace 100%!");
        println!("   (rust_tokenizers differences are expected - it always adds EOS and maps extra_id tokens differently)");
    }
    
    // Assert that we have high agreement with HuggingFace
    assert!(our_hf_agree as f64 / total_tests as f64 >= 0.95, 
        "Our tokenizer should match HuggingFace at least 95% of the time");
    
    // Don't require three-way consensus since rust_tokenizers has expected differences
}

#[test]
fn test_decode_consensus() {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let mut decode_mismatches = Vec::new();
    
    for (text, test_name) in TEST_CASES {
        // Skip empty text as it may have special handling
        if text.is_empty() {
            continue;
        }
        
        // First encode with HF to get tokens
        let hf_encoding = hf_tokenizer.encode(*text, false).unwrap();
        let token_ids: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        // Decode with both tokenizers
        let our_decoded = our_tokenizer.decode(&token_ids)
            .unwrap_or_else(|e| panic!("Our decode failed on {}: {:?}", test_name, e));
        let hf_decoded = hf_tokenizer.decode(&token_ids, false)
            .unwrap_or_else(|e| panic!("HF decode failed on {}: {:?}", test_name, e));
        
        // Compare decoded text (normalize whitespace for comparison)
        let our_normalized = our_decoded.trim();
        let hf_normalized = hf_decoded.trim();
        
        if our_normalized != hf_normalized {
            decode_mismatches.push((test_name, token_ids.clone(), our_decoded.clone(), hf_decoded.clone()));
            println!("\nDecode MISMATCH for {}", test_name);
            println!("  Tokens: {:?}", token_ids);
            println!("  Our decoded: {:?}", our_decoded);
            println!("  HF decoded:  {:?}", hf_decoded);
        }
    }
    
    if !decode_mismatches.is_empty() {
        println!("\n\n=== DECODE CONSENSUS FAILURES ===");
        println!("Found {} decode mismatches", decode_mismatches.len());
        panic!("Decode consensus tests failed!");
    }
}

#[test]
fn test_special_token_handling() {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    // Test all extra_id tokens
    for i in 0..100 {
        let token = format!("<extra_id_{}>", i);
        
        let our_tokens = our_tokenizer.encode(&token).unwrap();
        let hf_tokens: Vec<u32> = hf_tokenizer.encode(&token[..], false).unwrap().get_ids().to_vec();
        
        assert_eq!(our_tokens, hf_tokens, 
            "Extra ID token {} mismatch. Our: {:?}, HF: {:?}", i, our_tokens, hf_tokens);
    }
    
    println!("✓ All 100 extra_id tokens match!");
}

#[test] 
fn test_viterbi_segmentation_quality() {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    // Test cases where segmentation quality matters
    let segmentation_tests = &[
        "unbelievable",
        "preprocessing", 
        "tokenization",
        "internationalization",
        "antidisestablishmentarianism",
        "supercalifragilisticexpialidocious",
        "pneumonoultramicroscopicsilicovolcanoconiosis",
    ];
    
    for &word in segmentation_tests {
        let our_tokens = our_tokenizer.encode(word).unwrap();
        let hf_tokens: Vec<u32> = hf_tokenizer.encode(word, false).unwrap().get_ids().to_vec();
        
        println!("\nSegmentation for '{}': ", word);
        print!("  Our ({}): ", our_tokens.len());
        for &id in &our_tokens {
            if let Some(token) = our_tokenizer.id_to_token(id) {
                print!("{} ", token);
            }
        }
        println!();
        
        print!("  HF ({}):", hf_tokens.len());
        for &id in &hf_tokens {
            if let Some(token) = our_tokenizer.id_to_token(id) {
                print!("{} ", token);
            }
        }
        println!();
        
        assert_eq!(our_tokens, hf_tokens, "Segmentation mismatch for '{}'", word);
    }
}

#[test]
fn test_consistency_across_runs() {
    let tokenizer = FlanT5Tokenizer::with_default_config();
    
    // Tokenize the same text multiple times
    let text = "The quick brown fox jumps over the lazy dog.";
    let mut results = Vec::new();
    
    for _ in 0..10 {
        let tokens = tokenizer.encode(text).unwrap();
        results.push(tokens);
    }
    
    // All results should be identical
    for i in 1..results.len() {
        assert_eq!(results[0], results[i], 
            "Inconsistent tokenization on run {}", i);
    }
    
    println!("✓ Tokenization is consistent across 10 runs");
}

#[test]
fn test_cache_correctness() {
    let tokenizer = FlanT5Tokenizer::with_default_config();
    
    // First tokenization (cache miss)
    let text = "Cache test sentence.";
    let tokens1 = tokenizer.encode(text).unwrap();
    
    // Second tokenization (cache hit)
    let tokens2 = tokenizer.encode(text).unwrap();
    
    assert_eq!(tokens1, tokens2, "Cache returned different results!");
    
    // Clear cache and try again
    tokenizer.clear_cache();
    let tokens3 = tokenizer.encode(text).unwrap();
    
    assert_eq!(tokens1, tokens3, "Different results after cache clear!");
    
    println!("✓ Cache is working correctly");
}

#[test]
fn test_rust_tokenizers_detailed_comparison() {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    
    // Load rust_tokenizers T5 tokenizer
    let rust_tokenizer = T5Tokenizer::from_file("model/spiece.model", false)
        .expect("Failed to load rust_tokenizers T5");
    
    for (text, test_name) in TEST_CASES.iter().take(10) {
        let our_tokens = our_tokenizer.encode(text).unwrap();
        let rust_encoding = rust_tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0);
        let rust_tokens: Vec<u32> = rust_encoding.token_ids.iter().map(|&id| id as u32).collect();
        
        println!("\nComparing with rust_tokenizers for: {}", test_name);
        println!("  Our tokens: {:?}", our_tokens);
        println!("  Rust tokens: {:?}", rust_tokens);
        
        // Note: rust_tokenizers might have different special token handling
        // so we focus on the core tokenization
    }
} 