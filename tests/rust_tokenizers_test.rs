use flan_t5_tokenizer::FlanT5Tokenizer;
use rust_tokenizers::tokenizer::{T5Tokenizer, Tokenizer as RustTokenizer, TruncationStrategy};
use std::path::PathBuf;

// Helper to download T5 model files for rust_tokenizers
fn download_t5_model_files() -> Result<PathBuf, Box<dyn std::error::Error>> {
    use std::fs;
    use std::io::Write;
    
    let cache_dir = PathBuf::from("tests/t5_model_cache");
    fs::create_dir_all(&cache_dir)?;
    
    let model_path = cache_dir.join("spiece.model");
    
    // Check if already downloaded
    if model_path.exists() {
        return Ok(model_path);
    }
    
    // For testing, we'll use a local sentencepiece model
    // In production, you'd download from HuggingFace
    println!("Note: rust_tokenizers requires a SentencePiece .model file");
    println!("Please download a T5 spiece.model file and place it at: {:?}", model_path);
    
    Err("SentencePiece model file not found. Please download manually.".into())
}

#[test]
#[ignore] // Ignore by default since it requires external model files
fn test_rust_tokenizers_comparison() {
    // Get model path
    let model_path = match download_t5_model_files() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("Skipping rust_tokenizers test: {}", e);
            return;
        }
    };
    
    // Initialize tokenizers
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let rust_tokenizer = T5Tokenizer::from_file(
        model_path.to_str().unwrap(),
        false, // lower_case
    ).expect("Failed to load rust_tokenizers T5");
    
    // Test cases
    let test_cases = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "🌍 🚀 ✨",
    ];
    
    println!("\n=== Rust Tokenizers Comparison ===");
    
    for text in test_cases {
        // Our tokenizer
        let our_tokens = our_tokenizer.encode(text).unwrap();
        
        // rust_tokenizers
        let rust_encoded = rust_tokenizer.encode(
            text,
            None,
            512,
            &TruncationStrategy::LongestFirst,
            0,
        );
        let rust_tokens: Vec<u32> = rust_encoded.token_ids.iter()
            .map(|&id| id as u32)
            .collect();
        
        println!("\nText: \"{}\"", text);
        println!("Our tokens:  {:?}", our_tokens);
        println!("Rust tokens: {:?}", rust_tokens);
        
        // Check if they match
        if our_tokens != rust_tokens {
            println!("❌ MISMATCH");
        } else {
            println!("✅ MATCH");
        }
    }
}

#[test]
fn test_rust_tokenizers_not_available() {
    // This test documents that rust_tokenizers requires external model files
    println!("\n=== rust_tokenizers Information ===");
    println!("rust_tokenizers requires SentencePiece .model files that must be downloaded separately.");
    println!("It does not support loading from HuggingFace tokenizer.json format directly.");
    println!("For T5 models, you need:");
    println!("1. A SentencePiece .model file (e.g., from google/t5-small on HuggingFace)");
    println!("2. The model file path to initialize T5Tokenizer");
    println!("\nThis is different from:");
    println!("- Our tokenizer: Embeds vocabulary at compile-time, no runtime files needed");
    println!("- HuggingFace tokenizers: Can load from tokenizer.json format");
} 