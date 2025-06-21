# FLAN-T5 Tokenizer Test Suite

This directory contains a comprehensive test suite for the FLAN-T5 tokenizer implementation, focusing on correctness, performance, and compatibility with HuggingFace.

## Test Files Overview

### Core Validation Tests

1. **`huggingface_validation_tests.rs`** - HuggingFace compatibility validation
   - Tests tokenizer against 4,260 real-world samples from `model/validation_results.parquet`
   - Validates configuration compatibility with `model/config.json`
   - 100% success rate on all validation samples

2. **`end_to_end_validation.rs`** - Comprehensive end-to-end validation
   - Model configuration validation
   - Performance benchmarking (>2,500 samples/sec)
   - Tokenization statistics on real data
   - Placeholder for SafeTensors model inference validation

### Comparison & Consensus Tests

3. **`consensus_tests.rs`** - Multi-implementation consensus testing
   - Compares against HuggingFace `tokenizers` library
   - Compares against `rust_tokenizers` library (when available)
   - Tests basic English, Unicode, special tokens, edge cases

4. **`tokenizer_comparison.rs`** - Three-way tokenizer comparison
   - Detailed comparison between our implementation and reference implementations
   - Character-by-character analysis of differences
   - Performance comparisons

### Stress & Edge Case Tests

5. **`extreme_tokenizer_tests.rs`** - Extreme edge cases and stress tests
   - Very long text (up to 1M characters)
   - Unicode edge cases (RTL, combining characters, emoji)
   - Concurrent access testing
   - Memory usage validation
   - Performance regression checks

6. **`comprehensive_tokenizer_tests.rs`** - Comprehensive test coverage
   - Special token handling
   - Padding and truncation
   - Batch processing
   - Error handling

### Other Tests

7. **`cross_process_tests.rs`** - Multi-process safety tests
   - Process isolation
   - Concurrent tokenization from multiple processes

8. **`rust_tokenizers_test.rs`** - Specific tests for rust_tokenizers compatibility

## Running the Tests

### Prerequisites

Ensure model files are in the `model/` directory:
```bash
model/
├── flan_t5_small_tokenizer.json  # Required
├── config.json                    # Required for validation tests
├── validation_results.parquet     # Required for validation tests
└── spiece.model                   # Optional, for comparison tests
```

### Quick Test Run

```bash
# Run all tests with the test runner
./test_runner.sh

# Run specific test categories
cargo test --lib                                    # Unit tests
cargo test --test huggingface_validation_tests -- --ignored
cargo test --test end_to_end_validation
cargo test --test extreme_tokenizer_tests
```

### Validation Tests

The most important tests for production use:

```bash
# HuggingFace validation (uses real data)
cargo test --test huggingface_validation_tests -- --ignored --nocapture

# End-to-end validation with performance metrics
cargo test --test end_to_end_validation -- --nocapture
```

### Performance Tests

```bash
# Run benchmarks
cargo bench

# Run performance regression tests
cargo test test_tokenization_speed --test extreme_tokenizer_tests -- --nocapture
```

### Test Options

```bash
# Run with detailed output
cargo test -- --nocapture

# Run tests in parallel
cargo test -- --test-threads=4

# Run ignored tests (validation tests)
cargo test -- --ignored

# Run specific test
cargo test test_basic_english_sentences
```

## Test Coverage Summary

| Test Category | Coverage | Key Focus |
|--------------|----------|-----------|
| Unit Tests | Basic functionality | Core tokenizer operations |
| Validation Tests | Real-world data | HuggingFace compatibility |
| Consensus Tests | Cross-implementation | Consistency across libraries |
| Extreme Tests | Edge cases | Unicode, performance, stress |
| Cross-Process | Concurrency | Multi-process safety |

## Performance Baselines

Expected performance on validation data:
- Throughput: >2,500 samples/second
- Average token length: ~12.4 tokens
- Memory usage: <100MB
- Cache hit rate: >90% for repeated text

## Adding New Tests

When adding tests:
1. Place in appropriate test file based on category
2. Use `#[ignore]` for tests requiring model files
3. Include both positive and negative cases
4. Document any special requirements
5. Ensure tests are deterministic

## Continuous Integration

Tests run automatically on:
- Every push and pull request
- Multiple platforms (Linux, macOS, Windows)
- Multiple Rust versions (stable, beta, nightly)

## Troubleshooting

Common issues:

1. **Missing model files**: Ensure files are in `model/` directory
2. **Git LFS**: Run `git lfs pull` to download large files
3. **Comparison tests fail**: Optional spiece.model may be missing
4. **Performance varies**: CPU-dependent, baselines are guidelines

For debugging:
```bash
# Enable debug output
RUST_LOG=debug cargo test

# Run with backtrace
RUST_BACKTRACE=full cargo test
```