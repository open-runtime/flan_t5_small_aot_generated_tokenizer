# Changelog

All notable changes to the FLAN-T5 Tokenizer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive validation against HuggingFace's implementation with 4,260 real-world samples
- End-to-end validation test suite with performance benchmarking
- Model files organization in `/model` directory
- Git LFS support for large model files
- Detailed documentation for build process and generated code
- Support for `model/config.json` validation
- Real-world performance metrics in documentation

### Changed
- All model file paths updated to use `/model` directory
- Updated README to reflect actual implementation (removed unimplemented features)
- Reorganized test documentation to focus on core validation tests
- Moved `check_token.rs` to `examples/` directory

### Fixed
- Viterbi algorithm scoring to use proper token scores from vocabulary
- Token count now matches reference implementations
- Vocabulary size handling to accommodate minor differences (32100 vs 32128)

## [0.1.0] - 2024-01-20

### Added
- Initial pure Rust implementation of FLAN-T5 tokenizer
- Zero-copy tokenization with Arc-based token sharing
- Compile-time vocabulary embedding using Perfect Hash Functions
- Viterbi algorithm for optimal tokenization
- Cache support with hash-based lookups
- Batch processing capabilities
- Async batch tokenization
- Optional Candle ML framework integration
- Comprehensive test suite including:
  - Consensus tests against reference implementations
  - Extreme edge case testing
  - Cross-process safety tests
  - Performance benchmarks
- Examples demonstrating various use cases
- CI/CD pipeline with GitHub Actions

### Technical Details
- Vocabulary: 32,100 tokens embedded at compile time
- Performance: 2,500+ samples/second on validation data
- Memory usage: ~50MB per process
- Average tokenization: 12.39 tokens per input text

### Dependencies
- Pure Rust implementation with minimal dependencies
- Optional features: `candle` for ML integration, `parallel` for rayon support
- Build-time code generation using `phf_codegen`

## Build Process Documentation

The tokenizer uses a sophisticated build-time code generation process:

1. **`build.rs`** reads `model/flan_t5_small_tokenizer.json` at compile time
2. Generates `target/*/build/*/out/tokenizer_data.rs` containing:
   - Perfect Hash Function mapping tokens to IDs
   - Reverse mapping arrays for ID to token lookup
   - Token scores for Viterbi algorithm
   - Special token constants
3. This generated code is included via `include!()` in the library
4. Results in zero runtime file I/O - everything is compiled into the binary

This approach provides:
- Instant startup (no file loading)
- Optimal performance (compile-time optimized hash functions)
- Self-contained binaries (no external files needed) 