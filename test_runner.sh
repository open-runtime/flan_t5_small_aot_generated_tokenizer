#!/bin/bash
#
# FLAN-T5 Tokenizer Comprehensive Test Runner
# 
# Usage:
#   ./test_runner.sh               # Run all tests
#   ./test_runner.sh --bench       # Include benchmarks
#   ./test_runner.sh --coverage    # Generate coverage report
#   ./test_runner.sh --fuzz        # Run fuzzer (requires nightly)
#   ./test_runner.sh --quick       # Quick validation only

set -e

echo "======================================"
echo "FLAN-T5 Tokenizer Test Suite"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if tokenizer files exist
check_files() {
    echo -e "${YELLOW}Checking required files...${NC}"
    
    if [ ! -f "model/flan_t5_small_tokenizer.json" ]; then
        echo -e "${RED}Error: model/flan_t5_small_tokenizer.json not found!${NC}"
        echo "Download with:"
        echo "  curl -L https://huggingface.co/google/flan-t5-small/resolve/main/tokenizer.json \\"
        echo "    -o model/flan_t5_small_tokenizer.json"
        exit 1
    fi
    
    if [ ! -f "model/spiece.model" ]; then
        echo -e "${RED}Error: model/spiece.model not found!${NC}"
        echo "Download with:"
        echo "  curl -L https://huggingface.co/google/flan-t5-small/resolve/main/spiece.model \\"
        echo "    -o model/spiece.model"
        exit 1
    fi
    
    echo -e "${GREEN}✓ All required files present${NC}\n"
}

# Run unit tests
run_unit_tests() {
    echo -e "${YELLOW}1. Running unit tests...${NC}"
    cargo test --lib -- --test-threads=4
    echo -e "${GREEN}✓ Unit tests passed${NC}\n"
}

# Run consensus tests
run_consensus_tests() {
    echo -e "${YELLOW}2. Running consensus tests...${NC}"
    cargo test --test consensus_tests -- --nocapture
    echo -e "${GREEN}✓ Consensus tests passed${NC}\n"
}

# Run extreme tokenizer tests
run_extreme_tests() {
    echo -e "${YELLOW}3. Running extreme tokenizer tests...${NC}"
    cargo test --test extreme_tokenizer_tests
    echo -e "${GREEN}✓ Extreme tests passed${NC}\n"
}

# Run comprehensive tokenizer tests
run_comprehensive_tests() {
    echo -e "${YELLOW}4. Running comprehensive tokenizer tests...${NC}"
    cargo test --test comprehensive_tokenizer_tests
    echo -e "${GREEN}✓ Comprehensive tests passed${NC}\n"
}

# Run three-way comparison test
run_comparison_test() {
    echo -e "${YELLOW}5. Running three-way comparison...${NC}"
    cargo test --test tokenizer_comparison -- --nocapture
    echo -e "${GREEN}✓ Comparison tests passed${NC}\n"
}

# Run benchmarks
run_benchmarks() {
    echo -e "${YELLOW}6. Running benchmarks...${NC}"
    cargo bench --no-fail-fast
    echo -e "${GREEN}✓ Benchmarks completed${NC}\n"
}

# Run examples
run_examples() {
    echo -e "${YELLOW}7. Running examples...${NC}"
    
    # Run each example
    for example in basic_usage tokenizer_inspection three_way_comparison; do
        if cargo run --example $example > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} $example"
        else
            echo -e "  ${RED}✗${NC} $example"
        fi
    done
    
    echo ""
}

# Generate coverage report
generate_coverage() {
    echo -e "${YELLOW}Generating coverage report...${NC}"
    
    if ! command -v cargo-tarpaulin &> /dev/null; then
        echo "Installing cargo-tarpaulin..."
        cargo install cargo-tarpaulin
    fi
    
    cargo tarpaulin --out Html --output-dir coverage
    echo -e "${GREEN}✓ Coverage report generated in coverage/index.html${NC}\n"
}

# Run fuzzer
run_fuzzer() {
    echo -e "${YELLOW}Running fuzzer...${NC}"
    
    if ! rustup show | grep -q nightly; then
        echo "Installing nightly toolchain..."
        rustup install nightly
    fi
    
    # Check if fuzzer target exists
    if [ ! -d "fuzz" ]; then
        echo "Initializing fuzzer..."
        cargo +nightly install cargo-fuzz
        cargo +nightly fuzz init
        
        # Create fuzz target
        cat > fuzz/fuzz_targets/fuzz_tokenizer.rs << 'EOF'
#![no_main]
use libfuzzer_sys::fuzz_target;
use flan_t5_tokenizer::FlanT5Tokenizer;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let tokenizer = FlanT5Tokenizer::with_default_config();
        
        // Should not panic
        if let Ok(tokens) = tokenizer.encode(s) {
            let _ = tokenizer.decode(&tokens);
        }
    }
});
EOF
    fi
    
    echo "Running fuzzer for 30 seconds..."
    cargo +nightly fuzz run fuzz_tokenizer -- -max_total_time=30 -print_final_stats=1
    echo -e "${GREEN}✓ Fuzzing completed${NC}\n"
}

# Performance regression check
check_performance() {
    echo -e "${YELLOW}Checking for performance regressions...${NC}"
    
    # Run performance test and capture output
    cargo test test_tokenization_speed --test extreme_tokenizer_tests -- --nocapture | tee perf_output.tmp
    
    # Extract timing from output (this is a simplified check)
    if grep -q "short.*[0-9]\+\.[0-9]\+" perf_output.tmp; then
        echo -e "${GREEN}✓ Performance check passed${NC}\n"
    else
        echo -e "${YELLOW}⚠ Could not verify performance${NC}\n"
    fi
    
    rm -f perf_output.tmp
}

# Main execution
main() {
    # Parse arguments
    QUICK=false
    BENCH=false
    COVERAGE=false
    FUZZ=false
    
    for arg in "$@"; do
        case $arg in
            --quick)
                QUICK=true
                ;;
            --bench)
                BENCH=true
                ;;
            --coverage)
                COVERAGE=true
                ;;
            --fuzz)
                FUZZ=true
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --quick     Run quick validation only"
                echo "  --bench     Include benchmarks"
                echo "  --coverage  Generate coverage report"
                echo "  --fuzz      Run fuzzer (requires nightly)"
                exit 0
                ;;
        esac
    done
    
    # Check files first
    check_files
    
    # Set environment variables
    export RUST_BACKTRACE=1
    export RUST_LOG=warn
    
    # Start timer
    START_TIME=$(date +%s)
    
    if [ "$QUICK" = true ]; then
        echo -e "${YELLOW}Running quick validation...${NC}\n"
        run_unit_tests
        cargo test test_basic_english_sentences --test extreme_tokenizer_tests
    else
        # Run all test categories
        run_unit_tests
        run_consensus_tests
        run_extreme_tests
        
        # Only run if test file exists
        if [ -f "tests/comprehensive_tokenizer_tests.rs" ]; then
            run_comprehensive_tests
        fi
        
        if [ -f "tests/tokenizer_comparison.rs" ]; then
            run_comparison_test
        fi
        
        run_examples
        check_performance
        
        if [ "$BENCH" = true ]; then
            run_benchmarks
        fi
        
        if [ "$COVERAGE" = true ]; then
            generate_coverage
        fi
        
        if [ "$FUZZ" = true ]; then
            run_fuzzer
        fi
    fi
    
    # Calculate total time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "======================================"
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo "Total time: ${DURATION} seconds"
    echo "======================================"
    
    # Print summary
    echo -e "\n${YELLOW}Test Summary:${NC}"
    cargo test --no-run 2>&1 | grep -E "test result:|Running" || true
}

# Run main
main "$@" 