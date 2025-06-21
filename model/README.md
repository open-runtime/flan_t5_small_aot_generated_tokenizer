# Model Files

This directory contains model files from the HuggingFace FLAN-T5 repository. These files are tracked using Git LFS (Large File Storage) due to their size.

## Files

### Required Files

- **`flan_t5_small_tokenizer.json`** (2.3MB) - The main tokenizer vocabulary and configuration
  - Contains all 32,128 tokens and their mappings
  - Includes token scores and special token definitions
  - Used by `build.rs` to generate compile-time vocabulary

- **`config.json`** (1.9KB) - Model configuration
  - Defines model architecture parameters
  - Used for validation tests to ensure compatibility

- **`validation_results.parquet`** (128KB) - Real-world validation data
  - Contains 4,260 text samples with classification results
  - Used for end-to-end validation and performance testing

### Optional Files

- **`spiece.model`** (773KB) - SentencePiece model file
  - Only needed for comparison tests with `rust_tokenizers`
  - Can be omitted if not running comparison tests

## Git LFS Setup

These files are tracked with Git LFS. To work with them:

1. **Initial Clone**:
   ```bash
   git lfs install  # One-time setup
   git clone <repository>
   git lfs pull     # Download the actual files
   ```

2. **Adding New Model Files**:
   ```bash
   git lfs track "model/*.json"
   git lfs track "model/*.model" 
   git lfs track "model/*.parquet"
   git add .gitattributes
   git add model/your_file
   git commit
   ```

3. **Checking LFS Status**:
   ```bash
   git lfs ls-files  # List LFS-tracked files
   git lfs status    # Check status
   ```

## Downloading Files Manually

If you need to download files manually from HuggingFace:

```bash
# Tokenizer JSON (required)
curl -L https://huggingface.co/google/flan-t5-small/resolve/main/tokenizer.json \
  -o model/flan_t5_small_tokenizer.json

# SentencePiece model (optional)
curl -L https://huggingface.co/google/flan-t5-small/resolve/main/spiece.model \
  -o model/spiece.model

# Config (if needed)
curl -L https://huggingface.co/google/flan-t5-small/resolve/main/config.json \
  -o model/config.json
```

## File Descriptions

### flan_t5_small_tokenizer.json

Contains the complete tokenizer configuration including:
- Vocabulary mappings (token → ID)
- Token scores for the Viterbi algorithm
- Special tokens configuration
- Pre/post-processing rules

### config.json

Model configuration including:
- Architecture: T5ForSequenceClassification
- Vocabulary size: 32,128
- Model dimensions and layer counts
- Label mappings for classification

### validation_results.parquet

Parquet file with validation data:
- Column `input`: Text samples
- Column `true_label`: Ground truth labels
- Column `prediction`: Model predictions
- Column `correct`: Whether prediction matches truth

Used for validating tokenizer correctness and performance on real data.

## Note on File Sizes

While these files are relatively small by modern standards, they're tracked with Git LFS to:
1. Keep the repository size manageable
2. Allow efficient cloning without downloading model files
3. Enable versioning of model files separately from code 