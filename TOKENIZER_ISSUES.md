# FLAN-T5 Tokenizer Issues Summary

## Current Issues

Our tokenizer has 0% accuracy compared to HuggingFace tokenizers due to several fundamental issues:

### 1. Token Scores Are Discarded
- The tokenizer JSON contains scores for each token (e.g., `["▁", -2.0122928619384766]`)
- Our build.rs throws away these scores: `for (idx, (token, _score)) in ...`
- These scores are ESSENTIAL for Unigram model tokenization

### 2. Wrong Tokenization Algorithm
- **Current**: Simple greedy longest-match (in `tokenizer.rs`)
- **Required**: Unigram model with Viterbi algorithm (already implemented in `sentencepiece.rs` but unused!)
- Example: "recurring" should be `["", "recurring"]` not `["▁rec", "ur", "ring"]`

### 3. Incorrect Space Handling
- **Current**: Each space becomes a separate ▁ token
- **Required**: Metaspace pre-tokenizer that normalizes consecutive spaces
- Example: "  hello  " should tokenize as `[21820, 3]` not `[3, 21820, 3, 3]`

### 4. Missing Pre-tokenizer Logic
- The tokenizer uses "Metaspace" pre-tokenizer which:
  - Replaces spaces with ▁ 
  - Adds ▁ at the beginning if text doesn't start with space
  - Normalizes multiple spaces to single space

## Solution

### Step 1: Update build.rs to preserve scores
```rust
// Generate vocabulary with scores
pub static VOCAB_SCORES: phf::Map<&'static str, (u32, f32)> = phf::phf_map! {
    "token" => (id, score),
    // ...
};
```

### Step 2: Use the SentencePiece implementation
- The correct implementation already exists in `sentencepiece.rs`!
- It has TrieNode, Viterbi algorithm, proper scoring
- Just need to integrate it into the main tokenizer

### Step 3: Fix Metaspace preprocessing
- Normalize consecutive spaces
- Handle ▁ markers correctly
- Match HuggingFace's Metaspace behavior

## Test Results

Current tokenizer failures:
- "recurring" → Our: `[5026, 450, 1007]` vs HF: `[3, 21557]`
- "  hello  " → Our: `[3, 21820, 3, 3]` vs HF: `[21820, 3]`
- "Define serendipity" → Our: 7 tokens vs HF: 6 tokens
- "undefined" → Our: `[3550, 13536, 26]` vs HF: `[73, 17094]`

## References

The sentencepiece.txt file shows the correct implementation pattern with:
- Viterbi algorithm for optimal tokenization
- Trie-based token lookup
- Score-based path selection
- Proper handling of unknown characters

The issue is we're not using any of this - we have a simplified greedy tokenizer instead! 