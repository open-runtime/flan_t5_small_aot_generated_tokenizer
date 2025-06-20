use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::collections::HashMap;
use anyhow::Result;
use serde::Deserialize;

#[derive(Deserialize)]
struct TokenizerConfig {
    model: ModelConfig,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

#[derive(Deserialize)]
struct ModelConfig {
    #[serde(rename = "type")]
    model_type: String,
    vocab: Vec<(String, f64)>,
    #[serde(default)]
    unk_id: Option<u32>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    special: bool,
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=flan_t5_small_tokenizer.json");
    
    // Parse tokenizer configuration
    let tokenizer_path = env::var("FLAN_T5_TOKENIZER_PATH")
        .unwrap_or_else(|_| "flan_t5_small_tokenizer.json".to_string());
    
    let tokenizer_json = std::fs::read_to_string(&tokenizer_path)?;
    let config: TokenizerConfig = serde_json::from_str(&tokenizer_json)?;
    
    // Convert vocab to HashMap, preserving scores
    let mut vocab = HashMap::new();
    let mut vocab_scores = HashMap::new();
    for (idx, (token, score)) in config.model.vocab.iter().enumerate() {
        vocab.insert(token.clone(), idx as u32);
        vocab_scores.insert(token.clone(), *score);
    }
    
    let out_dir = env::var("OUT_DIR")?;
    let dest_path = Path::new(&out_dir).join("tokenizer_data.rs");
    let mut file = BufWriter::new(File::create(&dest_path)?);
    
    // Generate vocabulary constants
    generate_vocabulary(&mut file, &vocab)?;
    
    // Generate vocabulary scores
    generate_vocabulary_scores(&mut file, &vocab_scores)?;
    
    // Generate special tokens
    generate_special_tokens(&mut file, &config)?;
    
    // Generate reverse mapping for decoding
    generate_reverse_mapping(&mut file, &vocab)?;
    
    // Generate metadata
    generate_metadata(&mut file, &vocab)?;
    
    Ok(())
}

fn generate_vocabulary(file: &mut BufWriter<File>, vocab: &HashMap<String, u32>) -> Result<()> {
    writeln!(file, "// Auto-generated vocabulary data")?;
    writeln!(file, "pub const VOCAB_SIZE: usize = {};", vocab.len())?;
    writeln!(file)?;
    
    // Use PHF for perfect hash function
    writeln!(file, "pub static VOCAB: phf::Map<&'static str, u32> = phf::phf_map! {{")?;
    
    // Sort for deterministic output
    let mut sorted: Vec<_> = vocab.iter().collect();
    sorted.sort_by_key(|(k, _)| k.as_str());
    
    for (token, id) in sorted {
        writeln!(file, "    {:?} => {},", token, id)?;
    }
    
    writeln!(file, "}};")?;
    writeln!(file)?;
    
    Ok(())
}

fn generate_vocabulary_scores(file: &mut BufWriter<File>, vocab_scores: &HashMap<String, f64>) -> Result<()> {
    writeln!(file, "// Vocabulary scores (negative log probabilities)")?;
    writeln!(file, "pub static VOCAB_SCORES: phf::Map<&'static str, f64> = phf::phf_map! {{")?;
    
    // Sort for deterministic output
    let mut sorted: Vec<_> = vocab_scores.iter().collect();
    sorted.sort_by_key(|(k, _)| k.as_str());
    
    for (token, score) in sorted {
        writeln!(file, "    {:?} => {:.17},", token, score)?;
    }
    
    writeln!(file, "}};")?;
    writeln!(file)?;
    
    Ok(())
}

fn generate_special_tokens(file: &mut BufWriter<File>, config: &TokenizerConfig) -> Result<()> {
    writeln!(file, "// Special tokens")?;
    
    // Find special tokens from added_tokens
    let mut pad_token = None;
    let mut eos_token = None;
    let mut unk_token = None;
    
    for token in &config.added_tokens {
        if token.special {
            match token.content.as_str() {
                "<pad>" => pad_token = Some(token),
                "</s>" => eos_token = Some(token),
                "<unk>" => unk_token = Some(token),
                _ => {}
            }
        }
    }
    
    if let Some(unk) = unk_token {
        writeln!(file, "pub const UNK_TOKEN: &str = {:?};", unk.content)?;
        writeln!(file, "pub const UNK_TOKEN_ID: u32 = {};", unk.id)?;
    } else if let Some(unk_id) = config.model.unk_id {
        writeln!(file, "pub const UNK_TOKEN: &str = \"<unk>\";")?;
        writeln!(file, "pub const UNK_TOKEN_ID: u32 = {};", unk_id)?;
    } else {
        writeln!(file, "pub const UNK_TOKEN: &str = \"<unk>\";")?;
        writeln!(file, "pub const UNK_TOKEN_ID: u32 = 2;")?;
    }
    
    if let Some(pad) = pad_token {
        writeln!(file, "pub const PAD_TOKEN: &str = {:?};", pad.content)?;
        writeln!(file, "pub const PAD_TOKEN_ID: u32 = {};", pad.id)?;
    } else {
        writeln!(file, "pub const PAD_TOKEN: &str = \"<pad>\";")?;
        writeln!(file, "pub const PAD_TOKEN_ID: u32 = 0;")?;
    }
    
    if let Some(eos) = eos_token {
        writeln!(file, "pub const EOS_TOKEN: &str = {:?};", eos.content)?;
        writeln!(file, "pub const EOS_TOKEN_ID: u32 = {};", eos.id)?;
    } else {
        writeln!(file, "pub const EOS_TOKEN: &str = \"</s>\";")?;
        writeln!(file, "pub const EOS_TOKEN_ID: u32 = 1;")?;
    }
    
    writeln!(file)?;
    Ok(())
}

fn generate_reverse_mapping(file: &mut BufWriter<File>, vocab: &HashMap<String, u32>) -> Result<()> {
    writeln!(file, "// Reverse mapping for decoding")?;
    
    // Create sorted array for binary search
    let mut id_to_token: Vec<_> = vocab
        .iter()
        .map(|(token, id)| (*id, token.as_str()))
        .collect();
    id_to_token.sort_by_key(|(id, _)| *id);
    
    // Split into chunks for better compile times
    const CHUNK_SIZE: usize = 8192;
    let num_chunks = (id_to_token.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    for (i, chunk) in id_to_token.chunks(CHUNK_SIZE).enumerate() {
        writeln!(file, "pub static ID_TO_TOKEN_CHUNK_{}: &[(u32, &'static str)] = &[", i)?;
        for (id, token) in chunk {
            writeln!(file, "    ({}, {:?}),", id, token)?;
        }
        writeln!(file, "];")?;
    }
    
    writeln!(file)?;
    writeln!(file, "pub const NUM_VOCAB_CHUNKS: usize = {};", num_chunks)?;
    writeln!(file)?;
    
    // Generate lookup function
    writeln!(file, r#"
pub fn id_to_token(id: u32) -> Option<&'static str> {{
    match id / {chunk_size} {{
"#, chunk_size = CHUNK_SIZE)?;
    
    for i in 0..num_chunks {
        writeln!(file, r#"        {} => ID_TO_TOKEN_CHUNK_{}.binary_search_by_key(&id, |(i, _)| *i)
            .ok()
            .map(|idx| ID_TO_TOKEN_CHUNK_{}[idx].1),"#, i, i, i)?;
    }
    
    writeln!(file, r#"        _ => None,
    }}
}}"#)?;
    
    Ok(())
}

fn generate_metadata(file: &mut BufWriter<File>, vocab: &HashMap<String, u32>) -> Result<()> {
    writeln!(file, "// Metadata")?;
    writeln!(file, "pub const VOCAB_SIZE_U32: u32 = {};", vocab.len())?;
    writeln!(file, "pub const MAX_TOKEN_LENGTH: usize = {};", 
        vocab.keys().map(|k| k.len()).max().unwrap_or(0))?;
    
    // Count special tokens
    let sentinel_count = vocab.keys()
        .filter(|k| k.starts_with("<extra_id_"))
        .count();
    writeln!(file, "pub const SENTINEL_TOKEN_COUNT: usize = {};", sentinel_count)?;
    
    Ok(())
} 