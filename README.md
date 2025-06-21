# FLAN-T5 Tokenizer

High-performance, pure Rust implementation of the FLAN-T5 tokenizer with compile-time vocabulary embedding and Candle integration.

## Features

- **Zero Runtime Overhead**: Vocabulary embedded at compile time using perfect hash functions
- **Batch Processing**: Lock-free batch tokenization with configurable workers
- **Candle Integration**: Direct tensor creation for ML inference (with `candle` feature)
- **Memory Efficient**: Tensor pooling and cache-friendly data structures
- **Production Ready**: Comprehensive error handling and monitoring

## Performance

- Cold start: ~20ms (vs 400-700ms for JSON-based tokenizers)
- Single tokenization: ~15-50μs
- Batch tokenization: ~5μs per text with batching
- Memory usage: ~50MB per process

## Usage

```rust
use flan_t5_tokenizer::{FlanT5Tokenizer, BatchTokenizer, BatchConfig};

// Single tokenization
let tokenizer = FlanT5Tokenizer::with_default_config();
let tokens = tokenizer.encode("Hello world!").unwrap();
let decoded = tokenizer.decode(&tokens).unwrap();

// Batch tokenization
let batch_tokenizer = BatchTokenizer::new(tokenizer.clone(), Default::default());
let results = batch_tokenizer.encode_batch(&["Text 1", "Text 2"]).unwrap();

// Special tokens handling
let special_tokens = tokenizer.encode("<extra_id_0>").unwrap();
```

### With Candle Integration

Enable the `candle` feature in your `Cargo.toml`:
```toml
flan-t5-tokenizer = { version = "0.1", features = ["candle"] }
```

Then use Candle integration:
```rust,ignore
use flan_t5_tokenizer::{FlanT5Tokenizer, TokenizerCandle};
use candle_core::Device;

let tokenizer = FlanT5Tokenizer::with_default_config();
let device = Device::Cpu;
let tensors = tokenizer.tokenize_to_tensor("ML inference text", &device)?;
```

## Building

The tokenizer vocabulary is embedded at compile time. Ensure `flan_t5_small_tokenizer.json` is in the project root:
```bash
cargo build --release
```

## License

MIT 