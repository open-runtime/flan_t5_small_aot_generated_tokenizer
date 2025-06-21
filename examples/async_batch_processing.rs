use flan_t5_tokenizer::{FlanT5Tokenizer, AsyncBatchTokenizer, BatchConfig};
use std::time::{Duration, Instant};

fn main() {
    println!("=== Async Batch Tokenizer Example ===\n");
    
    // Configure the batch processor
    let config = BatchConfig {
        max_batch_size: 32,
        batch_timeout: Duration::from_millis(10),
        num_workers: 4, // Use 4 worker threads
    };
    
    // Create tokenizer and async batch processor
    let tokenizer = FlanT5Tokenizer::with_default_config();
    let async_tokenizer = AsyncBatchTokenizer::new(tokenizer, config);
    
    // Example 1: Basic async batch processing
    println!("Example 1: Basic batch processing");
    let texts = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require significant computational resources.",
        "Rust programming language provides memory safety without garbage collection.",
        "Natural language processing enables computers to understand human language.",
    ];
    
    let start = Instant::now();
    match async_tokenizer.encode_batch_async(&texts) {
        Ok(results) => {
            let duration = start.elapsed();
            println!("✅ Successfully tokenized {} texts in {:?}", texts.len(), duration);
            println!("   Average: {:?} per text", duration / texts.len() as u32);
            
            // Display first result as example
            println!("\nExample tokenization:");
            println!("   Text: \"{}\"", texts[0]);
            println!("   Tokens: {:?}", &results[0]);
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    
    // Example 2: Large batch processing
    println!("\n\nExample 2: Large batch processing");
    let large_batch: Vec<&str> = (0..100)
        .map(|i| match i % 5 {
            0 => "Translate this sentence to French.",
            1 => "Summarize the following document.",
            2 => "What is the capital of France?",
            3 => "Explain quantum computing in simple terms.",
            _ => "Generate a creative story about AI.",
        })
        .collect();
    
    let start = Instant::now();
    match async_tokenizer.encode_batch_async(&large_batch) {
        Ok(results) => {
            let duration = start.elapsed();
            println!("✅ Successfully tokenized {} texts in {:?}", large_batch.len(), duration);
            println!("   Average: {:?} per text", duration / large_batch.len() as u32);
            println!("   Total tokens generated: {}", 
                results.iter().map(|r| r.len()).sum::<usize>());
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    
    // Example 3: Mixed content types
    println!("\n\nExample 3: Mixed content types");
    let mixed_content = vec![
        // Short texts
        "Hello!",
        "Yes",
        "No",
        
        // Medium texts
        "The weather today is quite pleasant with clear skies.",
        "Please send me the report by end of day tomorrow.",
        
        // Long texts
        "Artificial intelligence has revolutionized many industries, from healthcare \
         to finance, enabling automated decision-making and pattern recognition at \
         unprecedented scales.",
        
        // Technical content
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "SELECT * FROM users WHERE age > 18 AND status = 'active' ORDER BY created_at DESC;",
        
        // Special tokens
        "<extra_id_0>",
        "<pad>",
        "</s>",
    ];
    
    let start = Instant::now();
    match async_tokenizer.encode_batch_async(&mixed_content) {
        Ok(results) => {
            let duration = start.elapsed();
            println!("✅ Successfully tokenized {} mixed texts in {:?}", 
                mixed_content.len(), duration);
            
            // Show length distribution
            let mut length_dist = std::collections::HashMap::new();
            for result in &results {
                let bucket = (result.len() / 10) * 10;
                *length_dist.entry(bucket).or_insert(0) += 1;
            }
            
            println!("\nToken length distribution:");
            let mut buckets: Vec<_> = length_dist.iter().collect();
            buckets.sort_by_key(|&(k, _)| k);
            for (bucket, count) in buckets {
                println!("   {}-{} tokens: {} texts", bucket, bucket + 9, count);
            }
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    
    // Example 4: Performance comparison
    println!("\n\nExample 4: Performance comparison");
    let benchmark_texts: Vec<&str> = vec![
        "This is a test sentence for benchmarking tokenization speed."; 50
    ];
    
    // Async batch processing
    let start = Instant::now();
    let _ = async_tokenizer.encode_batch_async(&benchmark_texts).unwrap();
    let async_duration = start.elapsed();
    
    // Sequential processing for comparison
    let tokenizer = FlanT5Tokenizer::with_default_config();
    let start = Instant::now();
    for text in &benchmark_texts {
        let _ = tokenizer.encode(text).unwrap();
    }
    let sequential_duration = start.elapsed();
    
    println!("Performance comparison for {} texts:", benchmark_texts.len());
    println!("   Async batch: {:?} (avg: {:?}/text)", 
        async_duration, async_duration / benchmark_texts.len() as u32);
    println!("   Sequential: {:?} (avg: {:?}/text)", 
        sequential_duration, sequential_duration / benchmark_texts.len() as u32);
    println!("   Speedup: {:.2}x", 
        sequential_duration.as_secs_f64() / async_duration.as_secs_f64());
    
    println!("\n✨ Async batch tokenizer is ideal for:");
    println!("   - Processing large batches of text");
    println!("   - Web servers handling multiple concurrent requests");
    println!("   - Data preprocessing pipelines");
    println!("   - Real-time text processing applications");
} 