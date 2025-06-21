use flan_t5_tokenizer::FlanT5Tokenizer;

fn main() {
    println!("=== Tokenizer Inspection ===\n");
    
    // Initialize tokenizers
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    // Test simple cases
    let test_cases = vec![
        "Hello",
        "Hello world",
        "a",
        "▁a",
        "▁Hello",
        "Hello▁world",
        "Schedule a meeting",
    ];
    
    println!("Testing simple cases:");
    for text in test_cases {
        println!("\nText: \"{}\"", text);
        
        // Our tokenizer
        match our_tokenizer.encode(text) {
            Ok(tokens) => {
                println!("  Our tokens: {:?}", tokens);
                // Try to decode to see what's happening
                if let Ok(decoded) = our_tokenizer.decode(&tokens) {
                    println!("  Our decoded: \"{}\"", decoded);
                }
            }
            Err(e) => println!("  Our error: {}", e),
        }
        
        // HuggingFace
        match hf_tokenizer.encode(text, false) {
            Ok(encoding) => {
                let tokens: Vec<u32> = encoding.get_ids().to_vec();
                println!("  HF tokens: {:?}", tokens);
                if let Ok(decoded) = hf_tokenizer.decode(&tokens, false) {
                    println!("  HF decoded: \"{}\"", decoded);
                }
            }
            Err(e) => println!("  HF error: {}", e),
        }
    }
    
    // Check special tokens
    println!("\n\n=== Special Token IDs ===");
    println!("Our tokenizer:");
    println!("  PAD: 0");
    println!("  EOS: 1");  
    println!("  UNK: 2");
    
    // Let's see what token ID 3 is in HuggingFace (it appears frequently)
    println!("\nChecking what HF token ID 3 is:");
    if let Ok(decoded) = hf_tokenizer.decode(&[3], false) {
        println!("  Token 3 decoded as: \"{}\"", decoded);
    }
    
    // Check if the tokenizer has a normalizer
    println!("\n=== Tokenizer Components ===");
    println!("HF tokenizer info:");
    let model = hf_tokenizer.get_model();
    println!("  Model type: {:?}", std::any::type_name_of_val(&model));
    
    let normalizer = hf_tokenizer.get_normalizer();
    println!("  Has normalizer: {}", normalizer.is_some());
    
    let pre_tokenizer = hf_tokenizer.get_pre_tokenizer();
    println!("  Has pre-tokenizer: {}", pre_tokenizer.is_some());
    
    // Look at vocabulary for common tokens
    println!("\n=== Checking Common Token Mappings ===");
    let common_words = vec!["the", "a", "is", "to", "of", "and", "in"];
    for word in common_words {
        // Try with and without sentencepiece marker
        for variant in &[word, &format!("▁{}", word)] {
            match our_tokenizer.encode(variant) {
                Ok(tokens) => {
                    println!("{}: {:?}", variant, tokens.iter().filter(|&&t| t != 1).collect::<Vec<_>>());
                }
                Err(_) => {}
            }
        }
    }
} 