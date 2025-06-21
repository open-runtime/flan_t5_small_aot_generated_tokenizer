# Flan T5 Small Tokenizer JSON - Comprehensive Technical Report

## Executive Summary

The Flan T5 Small tokenizer uses a Unigram SentencePiece model with 32,128 tokens. The JSON format follows Hugging Face's tokenizers library v1.0.0 specification, with vocabulary stored as a dictionary mapping numeric IDs to `[token, score]` tuples, NOT as an array of tuples as initially assumed.

## 1. File Structure Overview

### Top-Level Keys
```json
{
  "version": "1.0.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [...],
  "normalizer": {...},
  "pre_tokenizer": {...},
  "post_processor": {...},
  "decoder": null,
  "model": {...}
}
```

## 2. Model Configuration

### 2.1 Basic Properties
- **Type**: `"Unigram"` (SentencePiece Unigram model)
- **UNK ID**: `2` (maps to `<unk>` token)
- **Byte Fallback**: `true` (but no `<0x` tokens found)
- **Vocabulary Size**: 32,128 tokens total

### 2.2 Vocabulary Format ⚠️ CRITICAL
```json
"vocab": [
  ["<pad>", 0.0],
  ["</s>", 0.0],
  ["<unk>", 0.0],
  ["▁", -2.0122928619384766],
  ...
  ["<extra_id_0>", 0.0]
]
```
**Format**: `Vec<(String, f64)>` where:
- Array index = Token ID (0-based)
- Value: Tuple of `[token_string, log_probability_score]`
- Total: 32,100 entries (indices 0-32099)

## 3. Special Tokens Analysis

### 3.1 Core Special Tokens
- **PAD**: `<pad>` (ID: 0, score: 0.0)
- **EOS**: `</s>` (ID: 1, score: 0.0)
- **UNK**: `<unk>` (ID: 2, score: 0.0)

### 3.2 Extra ID Tokens
- 100 extra ID tokens: `<extra_id_0>` through `<extra_id_99>`
- ID range: 32099 (for `<extra_id_0>`) down to 32000 (for `<extra_id_99>`)
- All have score: 0.0
- Pattern: `<extra_id_N>` where N decreases as ID increases

### 3.3 Added Tokens Configuration
- **Total**: 103 added tokens
- **Properties**:
  ```json
  {
    "id": <number>,
    "content": "<token>",
    "single_word": false,
    "lstrip": false,
    "rstrip": false,
    "normalized": false,
    "special": true
  }
  ```

## 4. Token Characteristics

### 4.1 Token Length Distribution
- **Minimum**: 1 character (punctuation, single letters)
- **Maximum**: 16 characters
- **Average**: ~4-6 characters
- No tokens exceed 20 characters

### 4.2 Token Types
1. **Special Tokens**: 103 tokens (score: 0.0)
2. **Single Characters**: Letters, digits, punctuation
3. **Subwords**: With "▁" prefix for word boundaries
4. **Full Words**: Common words with "▁" prefix
5. **Multilingual**: Accented characters (é, ñ, ü, ö, etc.)

### 4.3 Score Distribution
- **0.0**: 103 tokens (all special tokens)
- **-2.01 to -5.0**: High-frequency tokens
- **-5.0 to -10.0**: Medium-frequency tokens
- **-10.0 to -13.59**: Low-frequency tokens
- **Lowest score**: -13.590115547180176 ("▁Internațional")

## 5. Pre/Post Processing Pipeline

### 5.1 Normalizer
```json
{
  "type": "Sequence",
  "normalizers": [
    {
      "type": "Replace",
      "pattern": {"String": " "},
      "content": "▁"
    }
  ]
}
```

### 5.2 Pre-tokenizer
```json
{
  "type": "Metaspace",
  "replacement": "▁",
  "add_prefix_space": true
}
```

### 5.3 Post-processor
```json
{
  "type": "TemplateProcessing",
  "single": [
    {"Sequence": {"id": "A", "type_id": 0}},
    {"SpecialToken": {"id": "</s>", "type_id": 0}}
  ],
  "pair": [
    {"Sequence": {"id": "A", "type_id": 0}},
    {"SpecialToken": {"id": "</s>", "type_id": 0}},
    {"Sequence": {"id": "B", "type_id": 0}},
    {"SpecialToken": {"id": "</s>", "type_id": 0}}
  ]
}
```

## 6. Key Findings & Implementation Requirements

### 6.1 Critical Implementation Details
1. **Vocabulary is an array** - index represents token ID
2. **IDs are implicit** - array position = token ID
3. **Metaspace handling** - "▁" represents spaces
4. **No byte fallback tokens** in `<0x` format
5. **Score interpretation** - negative log probabilities

### 6.2 Build Script Requirements
1. Parse vocab as `Vec<(String, f64)>` where index = ID
2. Preserve implicit token IDs (0-32099)
3. Handle special tokens separately (score = 0.0)
4. Implement Metaspace preprocessing
5. Support template post-processing for EOS tokens

### 6.3 Tokenization Flow
1. **Input text** → Normalize (replace spaces with ▁)
2. **Apply Metaspace** → Add ▁ prefix if needed
3. **Unigram tokenization** → Use scores for Viterbi
4. **Post-process** → Add </s> token(s)

## 7. Compatibility Notes

### 7.1 Differences from Standard SentencePiece
- Uses Hugging Face tokenizers format
- Includes explicit added_tokens list
- Has structured pre/post processors
- No byte fallback tokens in expected format

### 7.2 rust_tokenizers Compatibility
- Similar to T5Tokenizer but with different vocab format
- Needs custom vocab loading logic
- Can reuse text preprocessing utilities
- Requires adaptation of SentencePiece model interface

## 8. Performance Considerations

### 8.1 Memory Layout
- 32,128 tokens × ~20 bytes average = ~640KB base vocabulary
- PHF optimization can reduce to ~400KB
- Chunked reverse mapping prevents compile-time issues

### 8.2 Optimization Opportunities
1. Group tokens by score for faster Viterbi
2. Separate high-frequency token fast path
3. SIMD-friendly character classification
4. Pre-compute common token sequences

## 9. Validation Checklist

- [x] Vocabulary loads correctly with proper IDs
- [x] Special tokens identified (PAD, EOS, UNK)
- [x] Extra ID tokens in correct order
- [x] Scores preserved as f64
- [x] Metaspace preprocessing works
- [x] Post-processing adds EOS tokens
- [ ] Tokenization matches Python implementation
- [ ] Decoding produces correct text
- [ ] Performance meets 20ms target

## 10. Conclusion

The Flan T5 tokenizer JSON follows a specific Hugging Face format that differs from standard SentencePiece models. The build script must handle dictionary-based vocabulary, preserve token IDs, and implement the full preprocessing pipeline to achieve correct tokenization while maintaining the performance benefits of compile-time embedding. 