# FLAN-T5 Tokenizer

A high-performance, zero-copy, pure Rust implementation of the FLAN-T5 tokenizer optimized for production use. This implementation focuses on memory efficiency and speed while maintaining 100% compatibility with HuggingFace's tokenizer.

## Features

- **Pure Rust Implementation**: No Python dependencies or runtime overhead
- **Zero-Copy Operations**: Uses Arc and memory-efficient data structures to minimize allocations
- **Compile-Time Vocabulary**: 32,128 tokens embedded at compile time for instant startup
- **Cache-Enabled**: Built-in caching with hash-based lookups for repeated tokenizations  
- **Production Ready**: Thoroughly tested against HuggingFace reference implementation
- **Async Support**: Batch processing with async/await support
- **Candle Integration**: Optional integration with Candle ML framework

## Performance

Benchmarked on real-world validation data (4,260 samples):

| Metric | Performance |
|--------|-------------|
| Throughput | 2,500+ samples/second |
| Average Token Length | 12.39 tokens |
| Memory Usage | ~50MB |
| Cache Hit Rate | >90% for repeated text |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
flan-t5-tokenizer = "0.1.0"

# Optional features
flan-t5-tokenizer = { version = "0.1.0", features = ["candle", "parallel"] }
```

## Quick Start

```rust
use flan_t5_tokenizer::FlanT5Tokenizer;

let tokenizer = FlanT5Tokenizer::with_default_config();

// Encode text to token IDs
let tokens = tokenizer.encode("Hello world!").unwrap();
println!("Tokens: {:?}", tokens); // [21820, 296, 55, 1]

// Decode tokens back to text
let decoded = tokenizer.decode(&tokens).unwrap();
println!("Decoded: {}", decoded); // "Hello world!"
```

## Advanced Usage

### Custom Configuration

```rust
use flan_t5_tokenizer::{FlanT5Tokenizer, TokenizerConfig};

let config = TokenizerConfig {
    add_prefix_space: true,
    max_length: 512,
    pad_to_max_length: false,
    add_eos_token: true,
};

let tokenizer = FlanT5Tokenizer::new(config);
```

### Batch Processing

```rust
use flan_t5_tokenizer::{BatchTokenizer, BatchConfig};

let tokenizer = FlanT5Tokenizer::with_default_config();
let batch_tokenizer = BatchTokenizer::new(tokenizer, BatchConfig::default());

let texts = vec![
    "First sentence.",
    "Second sentence.",
    "Third sentence.",
];

let results = batch_tokenizer.encode_batch(&texts).unwrap();
```

### Async Batch Processing

```rust
use flan_t5_tokenizer::AsyncBatchTokenizer;

let async_tokenizer = AsyncBatchTokenizer::new(config);
let handle = async_tokenizer.encode_async(text);
let tokens = handle.await.unwrap();
```

### Candle Integration

Enable the `candle` feature:

```rust
use flan_t5_tokenizer::{FlanT5Tokenizer, TokenizerCandle};
use candle_core::Device;

let tokenizer = FlanT5Tokenizer::with_default_config();
let device = Device::Cpu;

let tensors = tokenizer.tokenize_to_tensor("Hello world!", &device).unwrap();
// tensors.input_ids: Tensor with token IDs
// tensors.attention_mask: Tensor with attention mask
```

## Examples

The `examples/` directory contains various usage demonstrations:

- **`basic_usage.rs`** - Simple tokenization example
- **`validate_with_model_data.rs`** - Validation using real model files
- **`async_batch_processing.rs`** - Asynchronous batch tokenization
- **`candle_example.rs`** - Integration with Candle ML framework
- **`performance_report.rs`** - Comprehensive performance analysis
- **`three_way_comparison.rs`** - Compare with other tokenizer implementations
- **`tokenizer_debug.rs`** - Debugging and inspection tools

Run any example with:
```bash
cargo run --example basic_usage
```

## Project Structure

```
flan-t5-tokenizer/
├── src/
│   ├── lib.rs              # Public API
│   ├── tokenizer.rs        # Core tokenizer implementation
│   ├── batch.rs            # Batch processing
│   ├── candle_integration.rs # Candle ML integration
│   └── error.rs            # Error types
├── model/                  # Model files (tracked with Git LFS)
│   ├── flan_t5_small_tokenizer.json
│   ├── spiece.model
│   ├── config.json
│   └── validation_results.parquet
├── tests/
│   ├── huggingface_validation_tests.rs  # Validation against HF
│   ├── end_to_end_validation.rs         # Full validation suite
│   └── ...                              # Additional test suites
├── benches/                # Performance benchmarks
├── examples/               # Usage examples
└── build.rs               # Compile-time code generation
```

## Building from Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flan-t5-tokenizer
cd flan-t5-tokenizer
```

2. The model files are tracked with Git LFS:
```bash
git lfs pull
```

3. Build and test:
```bash
cargo build --release
cargo test
```

### Build Process Details

The tokenizer uses compile-time code generation to embed the vocabulary:

1. **`build.rs`** (at repository root - required by Cargo):
   - Runs at compile time before the main compilation
   - Reads `model/flan_t5_small_tokenizer.json`
   - Generates `tokenizer_data.rs` in the build output directory

2. **Generated Code** (`target/debug/build/*/out/tokenizer_data.rs`):
   - Contains ~97,000 lines of generated Rust code
   - Perfect Hash Function for O(1) token lookups
   - Static arrays for reverse lookups (ID → token)
   - All token scores and metadata

3. **Compilation**:
   - The generated code is included via `include!()` macro
   - Everything is compiled into the final binary
   - No runtime file I/O needed

This means:
- The tokenizer binary is self-contained
- No tokenizer files needed at runtime
- Instant startup with no deserialization overhead
- Optimal performance with compile-time optimizations

## Testing

The tokenizer is extensively tested for correctness and performance:

```bash
# Run all tests
./test_runner.sh

# Run specific test suites
cargo test --test huggingface_validation_tests -- --ignored
cargo test --test end_to_end_validation

# Run benchmarks
cargo bench
```

### Test Coverage

- **Consensus Tests**: Validates against HuggingFace implementation
- **Validation Tests**: Tests on 4,260 real-world samples
- **Edge Cases**: Unicode, empty strings, very long text
- **Performance Tests**: Ensures consistent performance
- **Cross-Process Tests**: Validates multi-process safety

## Model Files

The project uses model files from HuggingFace's FLAN-T5 repository:

- `flan_t5_small_tokenizer.json`: Tokenizer vocabulary and configuration
- `config.json`: Model configuration for validation
- `validation_results.parquet`: Real-world validation samples
- `spiece.model`: SentencePiece model (optional, for comparison tests)

These files are stored in the `model/` directory and tracked with Git LFS.

## Architecture

The tokenizer uses several optimization techniques:

1. **Compile-Time Vocabulary**: The entire vocabulary is parsed at compile time and embedded using perfect hash functions
2. **Zero-Copy Tokenization**: Uses Arc<Vec<u32>> for token sequences to enable cheap cloning
3. **Viterbi Algorithm**: Optimized dynamic programming for finding the best tokenization
4. **Caching**: LRU cache with hash-based lookups for repeated text

## Repository Organization

Key files at the repository root:

- **`build.rs`** - Must be at root for Cargo's build system to find it
- **`test_runner.sh`** - Main test orchestration script (common practice)
- **`Cargo.toml`** - Rust package manifest (required at root)

All other code is organized into appropriate directories:
- Source code in `src/`
- Examples in `examples/`
- Tests in `tests/`
- Benchmarks in `benches/`
- Model files in `model/`

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `cargo test`
2. Code is formatted: `cargo fmt`
3. No clippy warnings: `cargo clippy`
4. New features include tests

## License

This project is licensed under either of:
- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

## Acknowledgments

This implementation is based on the FLAN-T5 model by Google Research. The tokenizer aims to be fully compatible with the HuggingFace Transformers implementation. 