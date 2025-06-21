use flan_t5_tokenizer::FlanT5Tokenizer;

fn main() {
    let tokenizer = FlanT5Tokenizer::with_default_config();
    
    println!("Token 3 is: {:?}", tokenizer.id_to_token(3));
    
    // Test encoding <extra_id_0>
    let tokens = tokenizer.encode("<extra_id_0>").unwrap();
    println!("Encoding '<extra_id_0>': {:?}", tokens);
    
    // Test with config matching HuggingFace
    let mut config = flan_t5_tokenizer::TokenizerConfig::default();
    config.add_eos_token = false;
    config.add_prefix_space = false;
    let tokenizer2 = flan_t5_tokenizer::FlanT5Tokenizer::new(config);
    
    let tokens2 = tokenizer2.encode("<extra_id_0>").unwrap();
    println!("Without EOS and prefix space: {:?}", tokens2);
}
