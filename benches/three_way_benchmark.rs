use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use flan_t5_tokenizer::{FlanT5Tokenizer, BatchTokenizer, BatchConfig};
use rust_tokenizers::tokenizer::{T5Tokenizer, Tokenizer as RustTokenizer, TruncationStrategy};
use std::time::{Duration, Instant};

// Test data categorized by character count
const TINY_TEXT: &str = "Hi"; // 2 chars
const SHORT_TEXT: &str = "Hello world!"; // 12 chars
const MEDIUM_TEXT: &str = "The quick brown fox jumps over the lazy dog."; // 44 chars
const LONG_TEXT: &str = "Machine learning models have revolutionized how we process and understand data. These sophisticated algorithms can identify patterns, make predictions, and automate complex tasks that were once thought to require human intelligence."; // 238 chars
const VERY_LONG_TEXT: &str = "Artificial intelligence and machine learning have become integral parts of modern technology. From recommendation systems that suggest what movies to watch or products to buy, to autonomous vehicles navigating complex environments, AI is transforming every aspect of our lives. Natural language processing enables computers to understand and generate human language, while computer vision allows machines to interpret visual information. Deep learning, a subset of machine learning inspired by the human brain's neural networks, has achieved remarkable breakthroughs in areas like image recognition, speech synthesis, and game playing. As we continue to push the boundaries of what's possible with AI, we must also consider the ethical implications and ensure that these powerful technologies are developed and deployed responsibly."; // 814 chars

// Various input types
const UNICODE_TEXT: &str = "Hello 世界 🌍 مرحبا café"; // Mixed scripts
const CODE_TEXT: &str = "function test() { return x => x * 2; }"; // Code
const SPECIAL_TOKENS_TEXT: &str = "Translate <extra_id_0> to French: <extra_id_1>"; // T5 special tokens
const REPEATED_PATTERN: &str = "very very very very very very very very very very"; // Repetitive

fn bench_cold_start(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_start");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(20));
    
    group.bench_function("our_tokenizer", |b| {
        b.iter(|| {
            let _ = FlanT5Tokenizer::with_default_config();
        });
    });
    
    group.bench_function("hf_tokenizer", |b| {
        b.iter(|| {
            let _ = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json");
        });
    });
    
    group.bench_function("rust_tokenizer", |b| {
        b.iter(|| {
            let _ = T5Tokenizer::from_file("spiece.model", false);
        });
    });
    
    group.finish();
}

fn bench_tokenization_by_size(c: &mut Criterion) {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    let rust_tokenizer = T5Tokenizer::from_file("spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    let mut group = c.benchmark_group("tokenization_by_char_count");
    group.measurement_time(Duration::from_secs(15));
    
    let test_cases = [
        (TINY_TEXT, "2_chars"),
        (SHORT_TEXT, "12_chars"),
        (MEDIUM_TEXT, "44_chars"),
        (LONG_TEXT, "238_chars"),
        (VERY_LONG_TEXT, "814_chars"),
    ];
    
    for (text, label) in &test_cases {
        group.throughput(Throughput::Bytes(text.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("our", label),
            text,
            |b, text| {
                b.iter(|| {
                    our_tokenizer.encode(black_box(text))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("huggingface", label),
            text,
            |b, text| {
                b.iter(|| {
                    hf_tokenizer.encode(black_box(*text), false)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("rust_tokenizers", label),
            text,
            |b, text| {
                b.iter(|| {
                    rust_tokenizer.encode(black_box(text), None, 512, &TruncationStrategy::LongestFirst, 0)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_tokenization_by_type(c: &mut Criterion) {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    let rust_tokenizer = T5Tokenizer::from_file("spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    let mut group = c.benchmark_group("tokenization_by_type");
    
    let test_cases = [
        (UNICODE_TEXT, "unicode"),
        (CODE_TEXT, "code"),
        (SPECIAL_TOKENS_TEXT, "special_tokens"),
        (REPEATED_PATTERN, "repeated"),
    ];
    
    for (text, label) in &test_cases {
        group.bench_with_input(
            BenchmarkId::new("our", label),
            text,
            |b, text| {
                b.iter(|| {
                    our_tokenizer.encode(black_box(text))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("huggingface", label),
            text,
            |b, text| {
                b.iter(|| {
                    hf_tokenizer.encode(black_box(*text), false)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("rust_tokenizers", label),
            text,
            |b, text| {
                b.iter(|| {
                    rust_tokenizer.encode(black_box(text), None, 512, &TruncationStrategy::LongestFirst, 0)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let batch_tokenizer = BatchTokenizer::new(our_tokenizer.clone(), BatchConfig {
        max_batch_size: 200,
        ..Default::default()
    });
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    let rust_tokenizer = T5Tokenizer::from_file("spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    let mut group = c.benchmark_group("batch_processing");
    group.measurement_time(Duration::from_secs(20));
    
    for batch_size in [10, 50, 100, 200] {
        let texts: Vec<&str> = vec![MEDIUM_TEXT; batch_size];
        let total_bytes = texts.iter().map(|t| t.len() as u64).sum();
        group.throughput(Throughput::Bytes(total_bytes));
        
        group.bench_with_input(
            BenchmarkId::new("our_batch", batch_size),
            &texts,
            |b, texts| {
                b.iter(|| {
                    batch_tokenizer.encode_batch(black_box(texts))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("hf_sequential", batch_size),
            &texts,
            |b, texts| {
                b.iter(|| {
                    texts.iter()
                        .map(|text| hf_tokenizer.encode(*text, false))
                        .collect::<Result<Vec<_>, _>>()
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("rust_sequential", batch_size),
            &texts,
            |b, texts| {
                b.iter(|| {
                    texts.iter()
                        .map(|text| rust_tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0))
                        .collect::<Vec<_>>()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_token_count_scaling(c: &mut Criterion) {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    let rust_tokenizer = T5Tokenizer::from_file("spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    let mut group = c.benchmark_group("token_count_scaling");
    
    // Generate texts that produce approximately the target token counts
    let base = "The quick brown fox ";
    let texts_with_counts = vec![
        (base.repeat(1), "~5_tokens"),
        (base.repeat(5), "~25_tokens"),
        (base.repeat(10), "~50_tokens"),
        (base.repeat(20), "~100_tokens"),
        (base.repeat(50), "~250_tokens"),
    ];
    
    for (text, label) in &texts_with_counts {
        group.bench_with_input(
            BenchmarkId::new("our", label),
            text,
            |b, text| {
                b.iter(|| {
                    our_tokenizer.encode(black_box(text))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("huggingface", label),
            text,
            |b, text| {
                b.iter(|| {
                    hf_tokenizer.encode(black_box(text.as_str()), false)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("rust_tokenizers", label),
            text,
            |b, text| {
                b.iter(|| {
                    rust_tokenizer.encode(black_box(text), None, 512, &TruncationStrategy::LongestFirst, 0)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_decode_performance(c: &mut Criterion) {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let mut group = c.benchmark_group("decode_performance");
    
    // Pre-tokenize texts
    let texts = vec![SHORT_TEXT, MEDIUM_TEXT, LONG_TEXT];
    let token_sets: Vec<(Vec<u32>, &str)> = texts.iter()
        .zip(["short", "medium", "long"].iter())
        .map(|(text, label)| {
            let tokens = our_tokenizer.encode(text).unwrap();
            (tokens, *label)
        })
        .collect();
    
    for (tokens, label) in &token_sets {
        group.bench_with_input(
            BenchmarkId::new("our", label),
            tokens,
            |b, tokens| {
                b.iter(|| {
                    our_tokenizer.decode(black_box(tokens))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("huggingface", label),
            tokens,
            |b, tokens| {
                b.iter(|| {
                    hf_tokenizer.decode(black_box(tokens), false)
                });
            },
        );
        
        // Note: rust_tokenizers doesn't provide a direct decode method
    }
    
    group.finish();
}

// Measure actual memory usage and detailed performance metrics
fn detailed_performance_report() {
    println!("\n=== DETAILED PERFORMANCE REPORT ===\n");
    
    // Cold start times
    println!("1. COLD START TIMES:");
    
    let start = Instant::now();
    let _ = FlanT5Tokenizer::with_default_config();
    let our_cold = start.elapsed();
    
    let start = Instant::now();
    let _ = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json").unwrap();
    let hf_cold = start.elapsed();
    
    let start = Instant::now();
    let _ = T5Tokenizer::from_file("spiece.model", false).unwrap();
    let rust_cold = start.elapsed();
    
    println!("   Our tokenizer:    {:?}", our_cold);
    println!("   HuggingFace:      {:?}", hf_cold);
    println!("   rust_tokenizers:  {:?}", rust_cold);
    println!("   Speedup vs HF:    {:.2}x", hf_cold.as_secs_f64() / our_cold.as_secs_f64());
    println!("   Speedup vs rust:  {:.2}x", rust_cold.as_secs_f64() / our_cold.as_secs_f64());
    
    // Token count analysis
    println!("\n2. TOKEN COUNT ANALYSIS:");
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json").unwrap();
    let rust_tokenizer = T5Tokenizer::from_file("spiece.model", false).unwrap();
    
    for (text, name) in &[
        (SHORT_TEXT, "Short text"),
        (MEDIUM_TEXT, "Medium text"),
        (LONG_TEXT, "Long text"),
        (UNICODE_TEXT, "Unicode text"),
    ] {
        let our_tokens = our_tokenizer.encode(text).unwrap();
        let hf_tokens = hf_tokenizer.encode(*text, false).unwrap().get_ids().to_vec();
        let rust_tokens = rust_tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0).token_ids;
        
        println!("\n   {}:", name);
        println!("     Chars: {}", text.len());
        println!("     Our tokens: {}", our_tokens.len());
        println!("     HF tokens: {}", hf_tokens.len());
        println!("     Rust tokens: {} (includes EOS)", rust_tokens.len());
    }
    
    // Throughput comparison
    println!("\n3. THROUGHPUT (ops/sec):");
    let iterations = 10000;
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = our_tokenizer.encode(MEDIUM_TEXT);
    }
    let our_throughput = iterations as f64 / start.elapsed().as_secs_f64();
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hf_tokenizer.encode(MEDIUM_TEXT, false);
    }
    let hf_throughput = iterations as f64 / start.elapsed().as_secs_f64();
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = rust_tokenizer.encode(MEDIUM_TEXT, None, 512, &TruncationStrategy::LongestFirst, 0);
    }
    let rust_throughput = iterations as f64 / start.elapsed().as_secs_f64();
    
    println!("   Our tokenizer:    {:.0} ops/sec", our_throughput);
    println!("   HuggingFace:      {:.0} ops/sec", hf_throughput);
    println!("   rust_tokenizers:  {:.0} ops/sec", rust_throughput);
    
    println!("\n=== END OF REPORT ===\n");
}

criterion_group!(
    benches,
    bench_cold_start,
    bench_tokenization_by_size,
    bench_tokenization_by_type,
    bench_batch_processing,
    bench_token_count_scaling,
    bench_decode_performance,
);
criterion_main!(benches); 