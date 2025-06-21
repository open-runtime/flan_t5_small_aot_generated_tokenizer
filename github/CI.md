# .github/workflows/test.yml
name: Comprehensive Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run nightly to catch regressions
    - cron: '0 0 * * *'

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  test:
    name: Test on ${{ matrix.os }} / ${{ matrix.rust }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        rust: [stable, beta, nightly]
        exclude:
          # Skip some combinations to save CI time
          - os: windows-latest
            rust: beta
          - os: macos-latest
            rust: beta

    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: actions-rust-lang/setup-rust-toolchain@v1
      with:
        toolchain: ${{ matrix.rust }}
        components: rustfmt, clippy
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/bin/
          ~/.cargo/registry/index/
          ~/.cargo/registry/cache/
          ~/.cargo/git/db/
          target/
        key: ${{ runner.os }}-cargo-${{ matrix.rust }}-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Download tokenizer file
      run: |
        # Download FLAN-T5 tokenizer
        curl -L https://huggingface.co/google/flan-t5-small/resolve/main/tokenizer.json \
          -o flan_t5_small_tokenizer.json
        curl -L https://huggingface.co/google/flan-t5-small/resolve/main/spiece.model \
          -o spiece.model
    
    - name: Check formatting
      if: matrix.rust == 'stable'
      run: cargo fmt -- --check
    
    - name: Run clippy
      if: matrix.rust == 'stable'
      run: cargo clippy -- -D warnings
    
    - name: Build
      run: cargo build --verbose --all-features
    
    - name: Run unit tests
      run: cargo test --lib --verbose
    
    - name: Run consensus tests
      run: cargo test --test consensus_tests --verbose -- --nocapture
    
    - name: Run stress tests
      run: cargo test --test stress_tests --verbose
    
    - name: Run integration tests
      run: cargo test --test integration_tests --verbose
    
    - name: Run doc tests
      run: cargo test --doc --verbose
    
    - name: Run benchmarks (check only)
      run: cargo bench --no-run

  coverage:
    name: Code Coverage
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: actions-rust-lang/setup-rust-toolchain@v1
      with:
        toolchain: stable
    
    - name: Install tarpaulin
      run: cargo install cargo-tarpaulin
    
    - name: Download tokenizer file
      run: |
        curl -L https://huggingface.co/google/flan-t5-small/resolve/main/tokenizer.json \
          -o flan_t5_small_tokenizer.json
        curl -L https://huggingface.co/google/flan-t5-small/resolve/main/spiece.model \
          -o spiece.model
    
    - name: Generate coverage
      run: cargo tarpaulin --out Xml --all-features --verbose
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        files: ./cobertura.xml
        fail_ci_if_error: true

  fuzzing:
    name: Fuzzing
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust nightly
      uses: actions-rust-lang/setup-rust-toolchain@v1
      with:
        toolchain: nightly
    
    - name: Install cargo-fuzz
      run: cargo install cargo-fuzz
    
    - name: Run fuzzer
      run: |
        cd fuzz
        cargo +nightly fuzz run fuzz_tokenizer -- -max_total_time=300 -print_final_stats=1

  miri:
    name: Miri (Undefined Behavior Detection)
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Miri
      run: |
        rustup toolchain install nightly --component miri
        rustup override set nightly
        cargo miri setup
    
    - name: Run Miri
      run: cargo miri test --lib

  security-audit:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions-rust-lang/audit@v1

# tests/README.md
# FLAN-T5 Tokenizer Test Suite

This directory contains a comprehensive test suite for the FLAN-T5 tokenizer implementation, including consensus testing against reference implementations.

## Test Categories

### 1. **Consensus Tests** (`consensus_tests.rs`)
- Compares outputs against HuggingFace's `tokenizers` and `rust_tokenizers` libraries
- Tests basic English sentences, special tokens, punctuation, numbers
- Extensive Unicode testing (multilingual, emoji, RTL text)
- Edge cases (empty strings, very long text, repeated patterns)
- Property-based testing with `proptest`

### 2. **Stress Tests** (`stress_tests.rs`)
- Concurrent access from multiple threads
- Memory usage under heavy load
- Error recovery from invalid inputs
- Performance under sustained load

### 3. **Edge Cases** (`edge_cases.rs`)
- Maximum token IDs
- Zero-width characters
- Unicode normalization forms (NFC, NFD, NFKC, NFKD)
- Control characters
- Surrogate pairs and complex emoji

### 4. **Integration Tests** (`integration_tests.rs`)
- Cross-process tokenizer functionality
- Testing all preset configurations
- Real-world usage patterns

### 5. **Performance Tests**
- Benchmarks comparing different implementations
- Cache effectiveness measurements
- Regression tests to catch performance degradations

## Running the Tests

### Prerequisites

1. Download the tokenizer files:
```bash
# FLAN-T5 tokenizer.json
curl -L https://huggingface.co/google/flan-t5-small/resolve/main/tokenizer.json \
  -o flan_t5_small_tokenizer.json

# SentencePiece model (for rust_tokenizers)
curl -L https://huggingface.co/google/flan-t5-small/resolve/main/spiece.model \
  -o spiece.model
```

2. Install test dependencies:
```toml
[dev-dependencies]
tokenizers = "0.15"
rust_tokenizers = "8.1"
proptest = "1.4"
criterion = "0.5"
tempfile = "3.8"
similar = "2.3"
uuid = { version = "1.6", features = ["v4"] }
unicode-normalization = "0.1"
```

### Running All Tests

```bash
# Run the test script
chmod +x test_runner.sh
./test_runner.sh

# With benchmarks (takes longer)
./test_runner.sh --bench

# With coverage report
./test_runner.sh --coverage

# With fuzzing (requires nightly)
./test_runner.sh --fuzz
```

### Running Individual Test Categories

```bash
# Unit tests only
cargo test --lib

# Consensus tests with output
cargo test --test consensus_tests -- --nocapture

# Stress tests
cargo test --test stress_tests

# Edge cases
cargo test --test edge_cases

# Run specific test
cargo test test_multilingual_text -- --exact
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench single_tokenization

# Compare with baseline
cargo bench -- --baseline
```

### Property-Based Testing

The test suite uses `proptest` for property-based testing:

```rust
proptest! {
    #[test]
    fn test_tokenization_doesnt_panic(s in "\\PC*") {
        let _ = tokenizer.encode(&s);
    }
}
```

This generates random Unicode strings and ensures the tokenizer never panics.

### Fuzzing

For continuous fuzzing with `cargo-fuzz`:

```bash
# Install cargo-fuzz
cargo +nightly install cargo-fuzz

# Run fuzzer
cargo +nightly fuzz run fuzz_tokenizer

# Run for specific duration
cargo +nightly fuzz run fuzz_tokenizer -- -max_total_time=3600
```

## Test Configuration

### Consensus Testing Tolerance

Some tests allow small differences between implementations:

- **Basic Latin text**: 1% difference allowed
- **Unicode/multilingual**: 5% difference allowed
- **Special tokens**: Exact match required

### Performance Regression Thresholds

- Single tokenization: < 10μs
- Batch processing: > 50k tokens/second
- Cache hit rate: > 90% for repeated text

## Debugging Test Failures

### Enable Detailed Output

```bash
# Set environment variables
export RUST_LOG=debug
export RUST_BACKTRACE=full

# Run with nocapture
cargo test failing_test -- --nocapture
```

### Common Issues

1. **Missing tokenizer files**: Ensure `flan_t5_small_tokenizer.json` exists
2. **Version mismatches**: Check that all tokenizer libraries are up to date
3. **Platform differences**: Some Unicode tests may behave differently on Windows

### Generating Test Vectors

To generate reference test vectors:

```bash
cd tests/data
python generate_test_vectors.py
```

This creates `test_vectors.json` with expected outputs from the reference implementation.

## CI/CD Integration

The test suite includes GitHub Actions configuration for:

- Multi-platform testing (Linux, macOS, Windows)
- Multiple Rust versions (stable, beta, nightly)
- Code coverage with Codecov
- Security audits
- Nightly fuzzing runs
- Miri for undefined behavior detection

## Contributing New Tests

When adding new tests:

1. Add to the appropriate test file based on category
2. Include both positive and negative test cases
3. Document any special requirements
4. Ensure consensus with reference implementations
5. Add benchmarks for performance-critical paths

## Test Metrics

Current test coverage targets:
- Line coverage: > 90%
- Branch coverage: > 85%
- Consensus accuracy: > 99%
- Performance: Within 2x of C++ implementation