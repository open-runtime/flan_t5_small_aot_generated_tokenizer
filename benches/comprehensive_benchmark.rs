use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use flan_t5_tokenizer::{FlanT5Tokenizer, BatchTokenizer, BatchConfig};
use std::time::Duration;

// Test data of various lengths
const SHORT_TEXT: &str = "Hello, world!";
const MEDIUM_TEXT: &str = "The quick brown fox jumps over the lazy dog. Machine learning models require significant computational resources.";
const LONG_TEXT: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.";

// Real-world queries subset for benchmarking
const REAL_WORLD_QUERIES: &[&str] = &[
    "Schedule a meeting with John tomorrow at 3pm",
    "Show me emails from last week",
    "What's on my calendar today?",
    "What's the weather forecast for next week?",
    "Calculate 15% tip on $45.50",
    "How long until Christmas?",
    "begin the next part of the event after Jesus feeds the four thousand",
    "#include <FirebaseESP8266.h> this is an outdated library and you said to use another one",
    "От октаздра, ребро которого а=2",
    "🚀📅⏰💻",
];

fn bench_single_tokenization(c: &mut Criterion) {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let mut group = c.benchmark_group("single_tokenization");
    group.measurement_time(Duration::from_secs(10));
    
    for (text, name) in &[
        (SHORT_TEXT, "short"),
        (MEDIUM_TEXT, "medium"),
        (LONG_TEXT, "long"),
    ] {
        group.throughput(Throughput::Bytes(text.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("our_tokenizer", name),
            text,
            |b, text| {
                b.iter(|| {
                    our_tokenizer.encode(black_box(text))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("hf_tokenizer", name),
            text,
            |b, text| {
                b.iter(|| {
                    hf_tokenizer.encode(black_box(*text), false)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_batch_tokenization(c: &mut Criterion) {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let batch_tokenizer = BatchTokenizer::new(our_tokenizer.clone(), BatchConfig::default());
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let mut group = c.benchmark_group("batch_tokenization");
    group.measurement_time(Duration::from_secs(15));
    
    for batch_size in [10, 50, 100, 500] {
        let texts: Vec<&str> = REAL_WORLD_QUERIES.iter()
            .cycle()
            .take(batch_size)
            .cloned()
            .collect();
        
        let total_bytes: u64 = texts.iter().map(|t| t.len() as u64).sum();
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
        
        // Parallel HuggingFace tokenization using rayon
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("hf_parallel", batch_size),
            &texts,
            |b, texts| {
                use rayon::prelude::*;
                b.iter(|| {
                    texts.par_iter()
                        .map(|text| hf_tokenizer.encode(*text, false))
                        .collect::<Result<Vec<_>, _>>()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_real_world_queries(c: &mut Criterion) {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let mut group = c.benchmark_group("real_world_queries");
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark all real-world queries together
    let total_bytes: u64 = REAL_WORLD_QUERIES.iter().map(|t| t.len() as u64).sum();
    group.throughput(Throughput::Bytes(total_bytes));
    
    group.bench_function("our_tokenizer", |b| {
        b.iter(|| {
            for query in REAL_WORLD_QUERIES {
                let _ = our_tokenizer.encode(black_box(query));
            }
        });
    });
    
    group.bench_function("hf_tokenizer", |b| {
        b.iter(|| {
            for query in REAL_WORLD_QUERIES {
                let _ = hf_tokenizer.encode(black_box(query), false);
            }
        });
    });
    
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let mut group = c.benchmark_group("decode");
    
    // Pre-encode some texts
    let texts = vec![SHORT_TEXT, MEDIUM_TEXT, LONG_TEXT];
    let our_encoded: Vec<Vec<u32>> = texts.iter()
        .map(|t| our_tokenizer.encode(t).unwrap())
        .collect();
    let hf_encoded: Vec<Vec<u32>> = texts.iter()
        .map(|t| hf_tokenizer.encode(t, false).unwrap().get_ids().to_vec())
        .collect();
    
    for (i, name) in ["short", "medium", "long"].iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("our_decode", name),
            &our_encoded[i],
            |b, tokens| {
                b.iter(|| {
                    our_tokenizer.decode(black_box(tokens))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("hf_decode", name),
            &hf_encoded[i],
            |b, tokens| {
                b.iter(|| {
                    hf_tokenizer.decode(black_box(tokens), false)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_cache_effectiveness(c: &mut Criterion) {
    let tokenizer = FlanT5Tokenizer::with_default_config();
    
    let mut group = c.benchmark_group("cache_effectiveness");
    
    // Repeated text (should benefit from cache)
    let repeated_text = "The quick brown fox jumps over the lazy dog.";
    
    // Unique texts (no cache benefit)
    let unique_texts: Vec<String> = (0..100)
        .map(|i| format!("This is unique text number {} for testing cache misses.", i))
        .collect();
    
    group.bench_function("cached_tokenization", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let _ = tokenizer.encode(black_box(repeated_text));
            }
        });
    });
    
    group.bench_function("uncached_tokenization", |b| {
        let mut i = 0;
        b.iter(|| {
            let _ = tokenizer.encode(black_box(&unique_texts[i % unique_texts.len()]));
            i += 1;
        });
    });
    
    group.finish();
}

// Memory usage benchmark (approximate)
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.sample_size(10);
    
    group.bench_function("our_tokenizer_creation", |b| {
        b.iter(|| {
            let _ = FlanT5Tokenizer::with_default_config();
        });
    });
    
    group.bench_function("hf_tokenizer_creation", |b| {
        b.iter(|| {
            let _ = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json");
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_single_tokenization,
    bench_batch_tokenization,
    bench_real_world_queries,
    bench_decode,
    bench_cache_effectiveness,
    bench_memory_usage
);
criterion_main!(benches); 