#[cfg(feature = "candle")]
use flan_t5_tokenizer::{FlanT5Tokenizer, TokenizerCandle, TokenizerConfig};
#[cfg(feature = "candle")]
use candle_core::{Device, DType};

#[cfg(feature = "candle")]
fn main() -> anyhow::Result<()> {
    println!("=== Candle Integration Example ===\n");
    
    // Create tokenizer with custom config
    let mut config = TokenizerConfig::default();
    config.pad_to_max_length = true;
    config.max_length = 128;
    config.add_eos_token = true;
    
    let tokenizer = FlanT5Tokenizer::new(config);
    let device = Device::Cpu;
    
    // Example 1: Single text tokenization
    println!("Example 1: Single text tokenization");
    let text = "Translate English to French: The weather is beautiful today.";
    
    let tensor = tokenizer.tokenize_to_tensor(text, &device)?;
    println!("Text: \"{}\"", text);
    println!("Input IDs shape: {:?}", tensor.input_ids.dims());
    println!("Attention mask shape: {:?}", tensor.attention_mask.dims());
    println!("Batch size: {}", tensor.batch_size());
    println!("Sequence length: {}", tensor.seq_len());
    
    // Decode back to verify
    let decoded = tokenizer.decode_from_tensor(&tensor.input_ids)?;
    println!("Decoded: {:?}\n", decoded);
    
    // Example 2: Batch tokenization
    println!("Example 2: Batch tokenization");
    let texts = vec![
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "How does natural language processing work?",
        "What are neural networks?",
    ];
    
    let batch_tensor = tokenizer.batch_tokenize_to_tensor(&texts, &device)?;
    println!("Batch size: {}", batch_tensor.batch_size());
    println!("Padded sequence length: {}", batch_tensor.seq_len());
    println!("Input IDs shape: {:?}", batch_tensor.input_ids.dims());
    
    // Decode the batch
    let decoded_batch = tokenizer.decode_from_tensor(&batch_tensor.input_ids)?;
    println!("\nDecoded batch:");
    for (i, (original, decoded)) in texts.iter().zip(decoded_batch.iter()).enumerate() {
        println!("  [{}] Original: \"{}\"", i, original);
        println!("      Decoded:  \"{}\"", decoded);
    }
    
    // Example 3: Working with position IDs
    println!("\nExample 3: Position IDs");
    let position_ids = tokenizer.create_position_ids(
        batch_tensor.seq_len(), 
        batch_tensor.batch_size(), 
        &device
    )?;
    println!("Position IDs shape: {:?}", position_ids.dims());
    
    // Show first sequence's position IDs
    let pos_ids_vec: Vec<u32> = position_ids.get(0)?.to_vec1()?;
    println!("First 10 position IDs: {:?}", &pos_ids_vec[..10.min(pos_ids_vec.len())]);
    
    // Example 4: Device transfer (if CUDA available)
    println!("\nExample 4: Device operations");
    #[cfg(feature = "cuda")]
    {
        if let Ok(cuda_device) = Device::new_cuda(0) {
            let cuda_tensor = batch_tensor.to_device(&cuda_device)?;
            println!("✅ Successfully moved tensors to CUDA device");
            println!("   Input IDs device: {:?}", cuda_tensor.input_ids.device());
        } else {
            println!("ℹ️  CUDA not available, skipping GPU example");
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("ℹ️  CUDA feature not enabled");
    }
    
    // Example 5: Working with special tokens
    println!("\nExample 5: Special tokens");
    let special_texts = vec![
        "<pad>",
        "</s>",
        "<extra_id_0> is a special token",
        "Normal text with <extra_id_1> placeholder",
    ];
    
    for text in &special_texts {
        let tensor = tokenizer.tokenize_to_tensor(text, &device)?;
        let ids: Vec<u32> = tensor.input_ids.squeeze(0)?.to_vec1()?;
        println!("\"{}\" -> {:?}", text, &ids[..ids.iter().position(|&x| x == 0).unwrap_or(ids.len())]);
    }
    
    // Example 6: Memory-efficient batch processing
    println!("\nExample 6: Memory-efficient processing");
    let large_batch: Vec<&str> = (0..20)
        .map(|i| match i % 4 {
            0 => "Short text",
            1 => "This is a medium length sentence with more words.",
            2 => "This is a much longer sentence that contains significantly more tokens and will demonstrate how padding works in the batch tokenizer.",
            _ => "Another example text for variety.",
        })
        .collect();
    
    let large_tensor = tokenizer.batch_tokenize_to_tensor(&large_batch, &device)?;
    
    // Analyze padding efficiency
    let attention_sum: Vec<f32> = large_tensor.attention_mask.sum(1)?.to_vec1()?;
    let actual_lengths: Vec<usize> = attention_sum.iter().map(|&x| x as usize).collect();
    let padding_waste: usize = (large_tensor.batch_size() * large_tensor.seq_len()) 
        - actual_lengths.iter().sum::<usize>();
    
    println!("Batch statistics:");
    println!("  Total tokens: {}", large_tensor.batch_size() * large_tensor.seq_len());
    println!("  Actual tokens: {}", actual_lengths.iter().sum::<usize>());
    println!("  Padding tokens: {}", padding_waste);
    println!("  Padding efficiency: {:.1}%", 
        (actual_lengths.iter().sum::<usize>() as f64 / (large_tensor.batch_size() * large_tensor.seq_len()) as f64) * 100.0);
    
    println!("\n✅ Candle integration example completed successfully!");
    
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    println!("This example requires the 'candle' feature to be enabled.");
    println!("Run with: cargo run --example candle_example --features candle");
} 