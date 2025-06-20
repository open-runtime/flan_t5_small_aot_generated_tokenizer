use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use flan_t5_tokenizer::{FlanT5Tokenizer, BatchTokenizer, BatchConfig};

fn bench_single_tokenization(c: &mut Criterion) {
    let tokenizer = FlanT5Tokenizer::with_default_config();
    let texts = vec![
        "Short text",
        "Medium length text with more words to tokenize properly",
        "Very long text that contains many words and will test the performance \
         of the tokenizer on longer sequences that are more representative of \
         real-world usage patterns in production systems",
    ];
    
    let mut group = c.benchmark_group("single_tokenization");
    for text in texts {
        group.bench_with_input(
            BenchmarkId::from_parameter(text.len()),
            text,
            |b, text| {
                b.iter(|| {
                    tokenizer.encode(black_box(text))
                });
            },
        );
    }
    group.finish();
}

fn bench_batch_tokenization(c: &mut Criterion) {
    let tokenizer = FlanT5Tokenizer::with_default_config();
    let batch_tokenizer = BatchTokenizer::new(tokenizer, BatchConfig::default());
    
    let mut group = c.benchmark_group("batch_tokenization");
    
    for size in [10, 50, 100, 500] {
        let texts: Vec<_> = (0..size)
            .map(|i| format!("Sample text number {} for batch processing", i))
            .collect();
        let text_refs: Vec<_> = texts.iter().map(|s| s.as_str()).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &text_refs,
            |b, texts| {
                b.iter(|| {
                    batch_tokenizer.encode_batch(black_box(texts))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_single_tokenization, bench_batch_tokenization);
criterion_main!(benches); 