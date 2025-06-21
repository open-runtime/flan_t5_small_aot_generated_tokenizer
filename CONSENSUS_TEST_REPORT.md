# Consensus Test Report

## Summary
The tokenizer passes 19 out of 40 test cases. The main issues are:
1. Special token handling (not recognizing `<pad>`, `</s>`, `<unk>`, `<extra_id_X>` as single tokens)
2. Whitespace handling (not adding the `▁` token for leading whitespace)
3. Unknown character handling (not prefixing with `▁`)

## Failures by Category

### Special Token Handling (7 failures)
These tokens should be recognized as single units but are being broken down:
- `<pad>` → Expected: [0], Got: [2, 5612, 3155] (`<unk>`, "pad", ">")
- `</s>` → Expected: [1], Got: [2, 87, 7, 3155] (`<unk>`, "/", "s", ">")
- `<unk>` → Expected: [2], Got: [2, 6513, 3155] (`<unk>`, "unk", ">")
- `<extra_id_0>` → Expected: [32099], Got: broken into characters
- `<extra_id_99>` → Expected: [32000], Got: broken into characters
- T5 template with special tokens is being broken down

### Whitespace Handling (10 failures)
HuggingFace adds a `▁` token (ID 3) for whitespace that we're missing:
- Single space `" "` → Expected: [3], Got: []
- Multiple spaces `"   "` → Expected: [3], Got: []
- Newline `"\n"` → Expected: [3], Got: []
- Tab `"\t"` → Expected: [3], Got: []
- Also affects multi-token sequences where whitespace markers are expected

### Unknown Character Handling (4 failures)
HuggingFace prefixes unknown characters with `▁`:
- Chinese `"你好"` → Expected: [3, 2], Got: [2, 2]
- Japanese `"こんにちは"` → Expected: [3, 2], Got: [2, 2, 2, 2, 2]
- Arabic `"مرحبا"` → Expected: [3, 2], Got: [2, 2, 2, 2, 2]
- Emoji `"🚀"` → Expected: [3, 2], Got: [2]

## Key Differences from HuggingFace

1. **Token ID Mapping**:
   - ID 0: `<pad>` (padding token)
   - ID 1: `</s>` (end of sequence)
   - ID 2: `<unk>` (unknown token)
   - ID 3: `▁` (whitespace marker)
   - IDs 32000-32127: `<extra_id_99>` to `<extra_id_0>` and other special tokens

2. **Preprocessing**: HuggingFace seems to:
   - Add `▁` at the beginning of the input when appropriate
   - Recognize special tokens before breaking down into subwords
   - Handle whitespace-only inputs specially

3. **Tokenization Algorithm**: Our Viterbi implementation is working correctly for regular text but needs:
   - Special token recognition before the main algorithm
   - Proper whitespace preprocessing
   - Special handling for unknown characters

## Next Steps
To fix these issues, we need to:
1. Add special token detection before running the Viterbi algorithm
2. Implement proper whitespace preprocessing
3. Handle the standalone `▁` token (ID 3) correctly
4. Add proper handling for unknown characters with `▁` prefix 