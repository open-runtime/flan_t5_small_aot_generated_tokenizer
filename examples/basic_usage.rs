use flan_t5_tokenizer::{FlanT5Tokenizer, TokenizerConfig, BatchTokenizer, BatchConfig};

fn main() -> flan_t5_tokenizer::Result<()> {
    println!("FLAN-T5 Tokenizer Example");
    println!("=========================\n");
    
    // Create tokenizer with default config
    let tokenizer = FlanT5Tokenizer::with_default_config();
    println!("Vocabulary size: {}", tokenizer.vocab_size());
    
    // Test texts
    let test_texts = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Translate English to German: Hello, how are you?",
        "Summarize: The weather today is sunny with a chance of rain later.",
        "🦀 Rust is awesome! 🚀",
    ];
    
    println!("\nSingle tokenization examples:");
    println!("-----------------------------");
    
    for text in &test_texts {
        let tokens = tokenizer.encode(text)?;
        let decoded = tokenizer.decode(&tokens)?;
        
        println!("\nOriginal: {}", text);
        println!("Tokens ({}): {:?}", tokens.len(), &tokens[..tokens.len().min(20)]);
        if tokens.len() > 20 {
            println!("         ... and {} more tokens", tokens.len() - 20);
        }
        println!("Decoded: {}", decoded);
    }
    
    // Test batch tokenization
    println!("\n\nBatch tokenization example:");
    println!("---------------------------");
    
    let batch_tokenizer = BatchTokenizer::new(tokenizer.clone(), BatchConfig::default());
    let batch_texts: Vec<&str> = test_texts.iter().map(|s| s.as_ref()).collect();
    
    let start = std::time::Instant::now();
    let _batch_results = batch_tokenizer.encode_batch(&batch_texts)?;
    let duration = start.elapsed();
    
    println!("Batch tokenized {} texts in {:?}", batch_texts.len(), duration);
    println!("Average time per text: {:?}", duration / batch_texts.len() as u32);
    
    // Test with custom config
    println!("\n\nCustom configuration example:");
    println!("-----------------------------");
    
    let mut custom_config = TokenizerConfig::default();
    custom_config.max_length = 50;
    custom_config.pad_to_max_length = true;
    custom_config.add_eos_token = true;
    
    let custom_tokenizer = FlanT5Tokenizer::new(custom_config);
    let text = "This text will be padded to the max length!";
    let tokens = custom_tokenizer.encode(text)?;
    let decoded = custom_tokenizer.decode(&tokens)?;
    
    println!("Original: {}", text);
    println!("Tokens (padded to {}): {:?}", tokens.len(), &tokens[..20]);
    println!("Decoded: {}", decoded);
    
    Ok(())
} 