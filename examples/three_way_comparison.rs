use flan_t5_tokenizer::FlanT5Tokenizer;
use rust_tokenizers::tokenizer::{T5Tokenizer, Tokenizer as RustTokenizer, TruncationStrategy};
use std::path::Path;

fn main() {
    println!("=== Three-Way Tokenizer Comparison ===\n");
    println!("Comparing: HuggingFace vs rust_tokenizers vs Our Implementation\n");
    
    // Initialize tokenizers
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    
    // HuggingFace tokenizer
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    // rust_tokenizers T5 tokenizer - check if model files exist
    let spm_path = Path::new("spiece.model");
    let rust_tokenizer = if spm_path.exists() {
        println!("Loading rust_tokenizers with spiece.model...\n");
        Some(T5Tokenizer::from_file("spiece.model", false)
            .expect("Failed to load rust_tokenizers"))
    } else {
        println!("Warning: spiece.model not found. Skipping rust_tokenizers comparison.\n");
        None
    };
    
    // Test cases
    let test_cases = vec![
        // Simple words
        "hello",
        "world",
        "Hello",
        "World",
        
        // Phrases
        "hello world",
        "Hello World",
        
        // Real-world examples that were failing
        "Schedule a meeting with John tomorrow at 3pm",
        "Set a reminder to call mom this weekend",
        "Create a recurring meeting for team standup",
        "Define serendipity",
        "undefined",
        
        // Edge cases
        "Hi",
        "OK",
        "123",
        "3pm",
        "   ",
        "",
        
        // Special characters
        "hello, world!",
        "test@example.com",
        "#include <iostream>",
        
        // Unicode
        "Café",
        "Hello, 世界!",
        "Русский and English",
    ];
    
    let mut total_tests = 0;
    let mut hf_rust_agree = 0;
    let mut all_agree = 0;
    let mut our_differs = 0;
    
    for text in &test_cases {
        if text.is_empty() {
            // Skip empty string for rust_tokenizers as it might handle it differently
            continue;
        }
        
        total_tests += 1;
        println!("\n{}", "=".repeat(60));
        println!("Text: {:?}", text);
        println!("{}", "-".repeat(60));
        
        // Our tokenizer
        let our_tokens = our_tokenizer.encode(text).expect("Our tokenizer failed");
        println!("Our tokens:          {:?}", our_tokens);
        
        // HuggingFace tokenizer
        // Use true to add special tokens (EOS) to match rust_tokenizers behavior
        let hf_encoding = hf_tokenizer.encode(*text, true).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        println!("HuggingFace tokens:  {:?}", hf_tokens);
        
        // rust_tokenizers (if available)
        let rust_tokens = if let Some(ref tokenizer) = rust_tokenizer {
            let tokenized = tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0);
            let tokens: Vec<u32> = tokenized.token_ids.iter().map(|&id| id as u32).collect();
            println!("rust_tokenizers:     {:?}", tokens);
            Some(tokens)
        } else {
            None
        };
        
        // Compare results
        let hf_rust_match = if let Some(ref rust_tokens) = rust_tokens {
            &hf_tokens == rust_tokens
        } else {
            false
        };
        
        let our_hf_match = our_tokens == hf_tokens;
        let our_rust_match = if let Some(ref rust_tokens) = rust_tokens {
            &our_tokens == rust_tokens
        } else {
            false
        };
        
        // Analysis
        if let Some(_) = rust_tokens {
            if hf_rust_match {
                hf_rust_agree += 1;
                if our_hf_match {
                    all_agree += 1;
                    println!("✅ All three tokenizers agree!");
                } else {
                    our_differs += 1;
                    println!("❌ HF and rust_tokenizers agree, but ours differs!");
                    
                    // Show token-by-token breakdown
                    println!("\nToken breakdown:");
                    for (i, &id) in hf_tokens.iter().enumerate() {
                        let token_str = hf_tokenizer.decode(&[id], false)
                            .unwrap_or_else(|_| format!("<{}>", id));
                        println!("  HF[{}]: {} -> {:?}", i, id, token_str);
                    }
                    for (i, &id) in our_tokens.iter().enumerate() {
                        let token_str = our_tokenizer.decode(&[id])
                            .unwrap_or_else(|_| format!("<{}>", id));
                        println!("  Our[{}]: {} -> {:?}", i, id, token_str);
                    }
                }
            } else {
                println!("⚠️  HF and rust_tokenizers disagree!");
                if our_hf_match {
                    println!("    Our implementation matches HuggingFace");
                } else if our_rust_match {
                    println!("    Our implementation matches rust_tokenizers");
                } else {
                    println!("    All three implementations differ!");
                    
                    // Show all tokenizations for debugging
                    println!("\nToken breakdown:");
                    println!("  HuggingFace: {:?}", hf_tokens);
                    if let Some(ref rust_tokens) = rust_tokens {
                        println!("  rust_tokenizers: {:?}", rust_tokens);
                    }
                    println!("  Our implementation: {:?}", our_tokens);
                }
            }
        } else {
            // Only comparing with HuggingFace
            if our_hf_match {
                println!("✅ Our tokenizer matches HuggingFace");
            } else {
                our_differs += 1;
                println!("❌ Our tokenizer differs from HuggingFace");
            }
        }
    }
    
    // Summary
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!("Total tests: {}", total_tests);
    
    if rust_tokenizer.is_some() {
        println!("HuggingFace and rust_tokenizers agree: {} ({:.1}%)", 
            hf_rust_agree, (hf_rust_agree as f64 / total_tests as f64) * 100.0);
        println!("All three agree: {} ({:.1}%)", 
            all_agree, (all_agree as f64 / total_tests as f64) * 100.0);
        println!("Our implementation differs: {} ({:.1}%)", 
            our_differs, (our_differs as f64 / total_tests as f64) * 100.0);
        
        if hf_rust_agree < total_tests {
            println!("\n⚠️  Note: HuggingFace and rust_tokenizers disagree on {} test(s)!", 
                total_tests - hf_rust_agree);
            println!("This suggests the implementations have fundamental differences.");
        }
    } else {
        println!("Our implementation matches HuggingFace: {} ({:.1}%)", 
            total_tests - our_differs, ((total_tests - our_differs) as f64 / total_tests as f64) * 100.0);
        println!("Our implementation differs: {} ({:.1}%)", 
            our_differs, (our_differs as f64 / total_tests as f64) * 100.0);
    }
} 