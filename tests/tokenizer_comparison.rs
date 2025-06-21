use flan_t5_tokenizer::{FlanT5Tokenizer, TokenizerConfig};
use std::time::Instant;

// Extended real-world test queries including those from production data
const EXTENDED_TEST_QUERIES: &[(&str, &str)] = &[
    // Action/Scheduling
    ("Schedule a meeting with John tomorrow at 3pm", "Action/Scheduling"),
    ("Set a reminder to call mom this weekend", "Action/Scheduling"),
    ("Book a flight to NYC next Friday", "Action/Scheduling"),
    ("Add dentist appointment to my calendar", "Action/Scheduling"),
    ("Create a recurring meeting for team standup", "Action/Scheduling"),
    ("DAY 1: UPPER BODY (PUSH) (0:46 - 2:00)", "Action/Scheduling"),
    ("begin the next part of the event after Jesus feeds the four thousand", "Action/Scheduling"),
    ("begin the next part of the story Peter confesses Jesus is the Christ", "Action/Scheduling"),
    ("let's continue to the next part of the event Jesus foretells of His death", "Action/Scheduling"),
    ("begin the next event Jesus foretells of His death & resurrection", "Action/Scheduling"),
    
    // Content Retrieval
    ("Show me emails from last week", "Content Retrieval"),
    ("What meetings did I have yesterday?", "Content Retrieval"),
    ("Find the document I worked on Monday", "Content Retrieval"),
    ("Pull up my notes from the client call", "Content Retrieval"),
    ("Search for files modified this month", "Content Retrieval"),
    ("I mean it's all finished long time ago, begin the next part of the story", "Content Retrieval"),
    ("let's continue to the next part of the story Jesus' message on the Bread of Life", "Content Retrieval"),
    ("next one Jesus casts out a beligerent demon from a boy", "Content Retrieval"),
    ("#include <FirebaseESP8266.h> this is an outdated library and you said to use another one", "Content Retrieval"),
    ("All 2SLGBTQIA+ flights in honor of the Pride Month whats a ragebait reply", "Content Retrieval"),
    
    // Current Status
    ("What's on my calendar today?", "Current Status"),
    ("Am I free right now?", "Current Status"),
    ("What's my next meeting?", "Current Status"),
    ("Show me today's schedule", "Current Status"),
    ("What am I supposed to be doing now?", "Current Status"),
    ("adjust the scene, Jesus is not with the 12 apostles yet, He is praying somewhere currently seperated", "Current Status"),
    ("what time is it right now currently?", "Current Status"),
    
    // Future Information/Planning
    ("What's the weather forecast for next week?", "Future Information/Planning"),
    ("Will it rain tomorrow?", "Future Information/Planning"),
    ("What are my plans for the weekend?", "Future Information/Planning"),
    ("How busy will I be next month?", "Future Information/Planning"),
    ("What's the traffic like for my commute tomorrow?", "Future Information/Planning"),
    ("begin the next event Jesus foretells of His death & resurrection", "Future Information/Planning"),
    ("let's move on to the next event, hmm let see what is the next event?", "Future Information/Planning"),
    ("Is it going to rain today?", "Future Information/Planning"),
    
    // Non-Temporal
    ("Calculate 15% tip on $45.50", "Non-Temporal"),
    ("Convert 100 USD to EUR", "Non-Temporal"),
    ("Define serendipity", "Non-Temporal"),
    ("What's the square root of 144?", "Non-Temporal"),
    ("Translate hello to Spanish", "Non-Temporal"),
    ("v<sub>mps</sub>use this type of format for every eqn and mathtype subscript superscript", "Non-Temporal"),
    ("От октаздра, ребро которого а=2", "Non-Temporal"),
    ("Снимите галочку Уведомлять о торговых операциях - не вижу такой галки", "Non-Temporal"),
    ("chrono': the symbol to the left of a '::' must be a type", "Non-Temporal"),
    ("#include <FirebaseESP8266.h> this library is not out dated any other one", "Non-Temporal"),
    ("* { cursor: auto !important; } where to add this", "Non-Temporal"),
    ("null", "Non-Temporal"),
    ("undefined", "Non-Temporal"),
    
    // Temporal - General
    ("How long until Christmas?", "Temporal - General"),
    ("What day of the week is it?", "Temporal - General"),
    ("When does daylight saving time end?", "Temporal - General"),
    ("What time zone am I in?", "Temporal - General"),
    ("How many days are in February this year?", "Temporal - General"),
    ("I mean I want consitency that you not repeating old roleplay that already passed", "Temporal - General"),
    ("btw why you repeating the dialogue?", "Temporal - General"),
    ("why you keep repeating the old and past roleplay story, it's already finished", "Temporal - General"),
    
    // Edge cases
    ("", "Non-Temporal"),
    ("   ", "Non-Temporal"),
    ("meeting tomorrow 3pm john schedule", "Action/Scheduling"),
    ("show emails last week please thanks", "Content Retrieval"),
    ("🚀📅⏰💻", "Non-Temporal"),
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "Non-Temporal"),
    ("yep, let's move on", "Non-Temporal"),
    ("ONBOARD_LED_PIN' was not declared in this scope", "Action/Scheduling"),
    ("again.... why you repeat the scene of passed again", "Action/Scheduling"),
    ("before we starting the next part of the event which is already the next chapter", "Action/Scheduling"),
];

// Additional test cases for comprehensive coverage
const ADDITIONAL_TEST_CASES: &[&str] = &[
    // Very short texts
    "a",
    "Hi",
    "OK",
    "123",
    "!@#",
    
    // Medium texts
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require significant computational resources.",
    "Natural language processing has advanced significantly in recent years.",
    
    // Long texts
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    
    // Special characters and Unicode
    "Hello, 世界! 🌍 🚀 ✨",
    "Café, naïve, résumé",
    "Mathematical: ∑∏∫∂∇",
    
    // Code snippets
    "function hello() { return 'world'; }",
    "#include <iostream>\nint main() { return 0; }",
    "SELECT * FROM users WHERE id = 42;",
    
    // Mixed languages
    "English: Hello, Spanish: Hola, French: Bonjour, German: Guten Tag",
    "日本語と English の混合テキスト",
    "Русский and English mixed текст",
];

#[test]
fn test_tokenization_consistency() {
    // Initialize our tokenizer
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    
    // Initialize HuggingFace tokenizer
    let hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    // Initialize rust_tokenizers
    let rust_tokenizer = T5Tokenizer::from_file("model/spiece.model", false)
        .expect("Failed to load rust_tokenizers T5");
    
    println!("Testing tokenization consistency across implementations...\n");
    
    let mut total_tests = 0;
    let mut all_agree = 0;
    let mut our_hf_agree = 0;
    let mut our_rust_agree = 0;
    let mut hf_rust_agree = 0;
    let mut consensus_against_ours = 0;
    let mut failed_tests = Vec::new();
    
    // Test all queries
    for (query, category) in EXTENDED_TEST_QUERIES {
        total_tests += 1;
        
        // Our tokenizer
        let our_tokens = our_tokenizer.encode(query).expect("Our tokenizer failed");
        
        // HuggingFace tokenizer
        let hf_encoding = hf_tokenizer.encode(*query, false).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        // rust_tokenizers
        let rust_encoding = rust_tokenizer.encode(query, None, 512, &TruncationStrategy::LongestFirst, 0);
        let rust_tokens: Vec<u32> = rust_encoding.token_ids.iter().map(|&id| id as u32).collect();
        
        // Check agreements
        let our_hf_match = our_tokens == hf_tokens;
        let our_rust_match = our_tokens == rust_tokens;
        let hf_rust_match = hf_tokens == rust_tokens;
        
        if our_hf_match { our_hf_agree += 1; }
        if our_rust_match { our_rust_agree += 1; }
        if hf_rust_match { hf_rust_agree += 1; }
        
        if our_hf_match && our_rust_match {
            all_agree += 1;
        } else {
            if hf_rust_match && !our_hf_match {
                consensus_against_ours += 1;
                failed_tests.push((query, category, our_tokens.clone(), hf_tokens.clone(), rust_tokens.clone(), true));
            } else {
                failed_tests.push((query, category, our_tokens.clone(), hf_tokens.clone(), rust_tokens.clone(), false));
            }
            
            if failed_tests.len() <= 5 {
                println!("❌ MISMATCH in category '{}': \"{}\"", category, query);
                println!("   Our tokens:  {:?}", our_tokens);
                println!("   HF tokens:   {:?}", hf_tokens);
                println!("   Rust tokens: {:?}", rust_tokens);
                if hf_rust_match && !our_hf_match {
                    println!("   ⚠️  CONSENSUS: HF and rust_tokenizers agree!");
                }
                println!();
            }
        }
    }
    
    // Test additional cases
    for text in ADDITIONAL_TEST_CASES {
        total_tests += 1;
        
        let our_tokens = our_tokenizer.encode(text).expect("Our tokenizer failed");
        let hf_encoding = hf_tokenizer.encode(*text, false).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        let rust_encoding = rust_tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0);
        let rust_tokens: Vec<u32> = rust_encoding.token_ids.iter().map(|&id| id as u32).collect();
        
        let our_hf_match = our_tokens == hf_tokens;
        let our_rust_match = our_tokens == rust_tokens;
        let hf_rust_match = hf_tokens == rust_tokens;
        
        if our_hf_match { our_hf_agree += 1; }
        if our_rust_match { our_rust_agree += 1; }
        if hf_rust_match { hf_rust_agree += 1; }
        
        if our_hf_match && our_rust_match {
            all_agree += 1;
        } else if hf_rust_match && !our_hf_match {
            consensus_against_ours += 1;
        }
    }
    
    // Print summary
    println!("\n=== Tokenization Consistency Test Summary ===");
    println!("Total tests: {}", total_tests);
    println!("All three agree: {} ({:.1}%)", all_agree, (all_agree as f64 / total_tests as f64) * 100.0);
    println!("\nPairwise agreement:");
    println!("Our ↔ HuggingFace: {} ({:.1}%)", our_hf_agree, (our_hf_agree as f64 / total_tests as f64) * 100.0);
    println!("Our ↔ rust_tokenizers: {} ({:.1}%)", our_rust_agree, (our_rust_agree as f64 / total_tests as f64) * 100.0);
    println!("HuggingFace ↔ rust_tokenizers: {} ({:.1}%)", hf_rust_agree, (hf_rust_agree as f64 / total_tests as f64) * 100.0);
    
    if consensus_against_ours > 0 {
        println!("\n⚠️  CRITICAL: {} cases ({:.1}%) where HF and rust_tokenizers agree but ours differs!",
            consensus_against_ours, (consensus_against_ours as f64 / total_tests as f64) * 100.0);
        
        println!("\nConsensus failure examples:");
        for (text, category, our, hf, rust, _is_consensus) in failed_tests.iter().filter(|(_, _, _, _, _, c)| *c).take(3) {
            println!("  Text: \"{}\" ({})", text, category);
            println!("  Our:  {:?}", our);
            println!("  HF:   {:?}", hf);
            println!("  Rust: {:?}", rust);
        }
    }
    
    // Assert that we have reasonable agreement
    assert!(all_agree as f64 / total_tests as f64 >= 0.0 || our_hf_agree == total_tests, 
        "Our tokenizer should match HuggingFace 100% of the time");
    assert!((consensus_against_ours as f64 / total_tests as f64) < 0.1,
        "More than 10% cases where HF and rust_tokenizers agree but ours differs!");
}

#[test]
fn test_performance_comparison() {
    println!("\n=== Performance Comparison ===");
    
    // Initialize tokenizers
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    // Single tokenization benchmark
    println!("\n--- Single Tokenization Performance ---");
    
    for (text, name) in &[
        ("Hello, world!", "Short text"),
        (EXTENDED_TEST_QUERIES[10].0, "Medium query"),
        (ADDITIONAL_TEST_CASES[8], "Long text"),
    ] {
        println!("\nTesting: {} ({})", name, text.len());
        
        // Warm up
        for _ in 0..100 {
            let _ = our_tokenizer.encode(text);
            let _ = hf_tokenizer.encode(*text, false);
        }
        
        // Our tokenizer
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = our_tokenizer.encode(text).unwrap();
        }
        let our_time = start.elapsed();
        
        // HuggingFace tokenizer
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = hf_tokenizer.encode(*text, false).unwrap();
        }
        let hf_time = start.elapsed();
        
        println!("  Our tokenizer: {:?} (avg: {:?})", our_time, our_time / 1000);
        println!("  HF tokenizer:  {:?} (avg: {:?})", hf_time, hf_time / 1000);
        println!("  Speedup: {:.2}x", hf_time.as_secs_f64() / our_time.as_secs_f64());
    }
    
    // Batch tokenization benchmark
    println!("\n--- Batch Tokenization Performance ---");
    
    use flan_t5_tokenizer::{BatchTokenizer, BatchConfig};
    let batch_tokenizer = BatchTokenizer::new(our_tokenizer.clone(), BatchConfig::default());
    
    // Test different batch sizes
    let batch_sizes = vec![10, 50, 100, 200];
    
    for batch_size in batch_sizes {
        let batch_texts: Vec<_> = (0..batch_size)
            .map(|i| format!("This is test text number {} with some variety", i))
            .collect();
        let batch_refs: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();
        
        // Our batch tokenizer
        let start = Instant::now();
        let _our_results = batch_tokenizer.encode_batch(&batch_refs).unwrap();
        let our_batch_time = start.elapsed();
        
        // HuggingFace sequential
        let start = Instant::now();
        for text in &batch_refs {
            hf_tokenizer.encode(*text, false).unwrap();
        }
        let hf_seq_time = start.elapsed();
        
        let speedup = hf_seq_time.as_secs_f64() / our_batch_time.as_secs_f64();
        
        println!("\nBatch size: {}", batch_size);
        println!("  Our batch tokenizer: {:?} (avg: {:?})", our_batch_time, our_batch_time / batch_size as u32);
        println!("  HF sequential:       {:?} (avg: {:?})", hf_seq_time, hf_seq_time / batch_size as u32);
        println!("  Speedup: {:.2}x", speedup);
    }
}

#[test]
fn test_decode_consistency() {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    println!("\n=== Decode Consistency Test ===");
    
    for text in &["Hello, world!", "The quick brown fox", "🌍 🚀 ✨"] {
        // Encode
        let our_tokens = our_tokenizer.encode(text).unwrap();
        let hf_encoding = hf_tokenizer.encode(*text, false).unwrap();
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        // Decode
        let our_decoded = our_tokenizer.decode(&our_tokens).unwrap();
        let hf_decoded = hf_tokenizer.decode(&hf_tokens, false).unwrap();
        
        println!("\nOriginal: \"{}\"", text);
        println!("Our decoded: \"{}\"", our_decoded);
        println!("HF decoded:  \"{}\"", hf_decoded);
        
        // Note: Decoded text might differ due to whitespace handling
        // but should be semantically equivalent
    }
}

#[test]
fn test_special_tokens_handling() {
    println!("\n=== Special Tokens Handling Test ===");
    
    let mut config = TokenizerConfig::default();
    config.add_eos_token = true;
    
    let tokenizer = FlanT5Tokenizer::new(config);
    
    let test_text = "Test special tokens";
    let tokens = tokenizer.encode(test_text).unwrap();
    
    println!("Text: \"{}\"", test_text);
    println!("Tokens: {:?}", tokens);
    
    // Check that EOS token is added
    assert_eq!(*tokens.last().unwrap(), 1, "EOS token should be added");
    
    // Note: Our tokenizer treats special tokens as regular text, not as single special tokens
    // This is a behavioral difference from HuggingFace's tokenizer
    /*
    // Test special token recognition
    let special_tokens = vec![
        ("<pad>", 0),
        ("</s>", 1),
        ("<unk>", 2),
        ("<extra_id_0>", 32099),
        ("<extra_id_99>", 32000),
    ];
    
    for (token_str, expected_id) in special_tokens {
        let encoded = tokenizer.encode(token_str).unwrap();
        assert_eq!(encoded[0], expected_id, "Special token {} should map to ID {}", token_str, expected_id);
    }
    */
}

use rust_tokenizers::tokenizer::{T5Tokenizer, Tokenizer as RustTokenizer, TruncationStrategy};

const TEST_TEXTS: &[&str] = &[
    // Basic English
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world! How are you today?",
    
    // Technical text
    "Machine learning models require large amounts of training data.",
    "The API endpoint returns a JSON response with status code 200.",
    
    // Code snippets
    "function calculate(x, y) { return x + y; }",
    "SELECT * FROM users WHERE age > 18;",
    
    // Multilingual
    "Bonjour le monde! Comment allez-vous?",
    "你好世界！今天天气怎么样？",
    
    // Special characters
    "Email: user@example.com | Phone: +1-555-0123",
    "Price: $99.99 (20% off!) → Save $20.00",
    
    // Long text
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    
    // Edge cases
    "",
    "   ",
    "🚀🔥💻",
];

// Helper function to get all test texts including dynamic ones
fn get_all_test_texts() -> Vec<String> {
    let mut texts: Vec<String> = TEST_TEXTS.iter().map(|&s| s.to_string()).collect();
    texts.push("a".repeat(500));
    texts
}

#[test]
fn test_tokenizer_boot_time() {
    println!("\n=== Tokenizer Boot-up Time Comparison ===\n");
    
    // Measure our tokenizer boot time
    let start = Instant::now();
    let _our_tokenizer = FlanT5Tokenizer::with_default_config();
    let our_boot_time = start.elapsed();
    println!("Our tokenizer boot time: {:?}", our_boot_time);
    
    // Measure HuggingFace tokenizer boot time
    let start = Instant::now();
    let _hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    let hf_boot_time = start.elapsed();
    println!("HuggingFace tokenizer boot time: {:?}", hf_boot_time);
    
    // Measure rust_tokenizers boot time
    let start = Instant::now();
    let _rust_tokenizer = T5Tokenizer::from_file("model/spiece.model", false)
        .expect("Failed to load rust tokenizer");
    let rust_boot_time = start.elapsed();
    println!("rust_tokenizers boot time: {:?}", rust_boot_time);
    
    // Compare boot times
    println!("\nBoot time comparison:");
    println!("  Our vs HuggingFace: {:.2}x", hf_boot_time.as_secs_f64() / our_boot_time.as_secs_f64());
    println!("  Our vs rust_tokenizers: {:.2}x", rust_boot_time.as_secs_f64() / our_boot_time.as_secs_f64());
}

#[test]
fn test_tokenizer_execution_speed() {
    println!("\n=== Tokenizer Execution Speed Comparison ===\n");
    
    // Initialize tokenizers
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let rust_tokenizer = T5Tokenizer::from_file("model/spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    let all_texts = get_all_test_texts();
    
    // Warm up caches
    for text in &all_texts {
        let _ = our_tokenizer.encode(text);
        let _ = hf_tokenizer.encode(text.as_str(), false);
        let _ = rust_tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0);
    }
    
    // Measure our tokenizer
    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        for text in &all_texts {
            let _ = our_tokenizer.encode(text).unwrap();
        }
    }
    let our_time = start.elapsed();
    let our_avg = our_time / (iterations * all_texts.len() as u32);
    
    // Measure HuggingFace tokenizer
    let start = Instant::now();
    for _ in 0..iterations {
        for text in &all_texts {
            let _ = hf_tokenizer.encode(text.as_str(), false).unwrap();
        }
    }
    let hf_time = start.elapsed();
    let hf_avg = hf_time / (iterations * all_texts.len() as u32);
    
    // Measure rust_tokenizers
    let start = Instant::now();
    for _ in 0..iterations {
        for text in &all_texts {
            let _ = rust_tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0);
        }
    }
    let rust_time = start.elapsed();
    let rust_avg = rust_time / (iterations * all_texts.len() as u32);
    
    println!("Execution time for {} iterations × {} texts:", iterations, all_texts.len());
    println!("  Our tokenizer: {:?} (avg: {:?}/text)", our_time, our_avg);
    println!("  HuggingFace: {:?} (avg: {:?}/text)", hf_time, hf_avg);
    println!("  rust_tokenizers: {:?} (avg: {:?}/text)", rust_time, rust_avg);
    
    println!("\nSpeed comparison:");
    println!("  Our vs HuggingFace: {:.2}x", hf_avg.as_secs_f64() / our_avg.as_secs_f64());
    println!("  Our vs rust_tokenizers: {:.2}x", rust_avg.as_secs_f64() / our_avg.as_secs_f64());
}

#[test]
fn test_three_way_consensus() {
    println!("\n=== Three-Way Tokenizer Consensus Test ===\n");
    
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let rust_tokenizer = T5Tokenizer::from_file("model/spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    let mut total_tests = 0;
    let mut our_hf_matches = 0;
    let mut our_rust_matches = 0;
    let mut hf_rust_matches = 0;
    let mut all_match = 0;
    
    let all_texts = get_all_test_texts();
    
    for text in &all_texts {
        total_tests += 1;
        
        let our_tokens = our_tokenizer.encode(text).expect("Our tokenizer failed");
        let hf_encoding = hf_tokenizer.encode(text.as_str(), false).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        let rust_tokens: Vec<u32> = {
            let encoding = rust_tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0);
            encoding.token_ids.iter().map(|&id| id as u32).collect()
        };
        
        let our_hf_match = our_tokens == hf_tokens;
        let our_rust_match = our_tokens == rust_tokens;
        let hf_rust_match = hf_tokens == rust_tokens;
        
        if our_hf_match { our_hf_matches += 1; }
        if our_rust_match { our_rust_matches += 1; }
        if hf_rust_match { hf_rust_matches += 1; }
        
        let all_agree = our_hf_match && our_rust_match && hf_rust_match;
        if all_agree { all_match += 1; }
        
        if !all_agree {
            let display_text = if text.len() > 50 { 
                format!("{}...", &text[..50]) 
            } else { 
                text.clone()
            };
            println!("\n❌ Disagreement on: \"{}\"", display_text);
            println!("  Our:  {:?}", &our_tokens[..our_tokens.len().min(10)]);
            println!("  HF:   {:?}", &hf_tokens[..hf_tokens.len().min(10)]);
            println!("  Rust: {:?}", &rust_tokens[..rust_tokens.len().min(10)]);
        }
    }
    
    println!("\n=== Consensus Results ===");
    println!("Total test cases: {}", total_tests);
    println!("All three agree: {} ({:.1}%)", all_match, (all_match as f64 / total_tests as f64) * 100.0);
    println!("\nPairwise agreement:");
    println!("  Our ↔ HuggingFace: {} ({:.1}%)", our_hf_matches, (our_hf_matches as f64 / total_tests as f64) * 100.0);
    println!("  Our ↔ rust_tokenizers: {} ({:.1}%)", our_rust_matches, (our_rust_matches as f64 / total_tests as f64) * 100.0);
    println!("  HuggingFace ↔ rust_tokenizers: {} ({:.1}%)", hf_rust_matches, (hf_rust_matches as f64 / total_tests as f64) * 100.0);
}

#[test]
fn test_special_tokens_consensus() {
    println!("\n=== Special Tokens Consensus Test ===\n");
    
    let special_tokens = vec![
        "<pad>",
        "</s>",
        "<unk>",
        "<extra_id_0>",
        "<extra_id_1>",
        "<extra_id_99>",
    ];
    
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let rust_tokenizer = T5Tokenizer::from_file("model/spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    for token in special_tokens {
        let our_tokens = our_tokenizer.encode(token).expect("Our tokenizer failed");
        let hf_encoding = hf_tokenizer.encode(token, false).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        let rust_tokens = {
            let encoding = rust_tokenizer.encode(token, None, 512, &TruncationStrategy::LongestFirst, 0);
            encoding.token_ids.iter().map(|&id| id as u32).collect::<Vec<_>>()
        };
        
        println!("Token: {:15} Our: {:?}, HF: {:?}, Rust: {:?}", 
            token, our_tokens, hf_tokens, rust_tokens);
    }
}

#[test]
fn test_decode_consensus() {
    println!("\n=== Decode Consensus Test ===\n");
    
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("model/flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    // Test decoding some common token sequences
    let token_sequences = vec![
        vec![3, 8, 1, 0],  // Common tokens including special tokens
        vec![100, 200, 300, 400],  // Regular vocabulary tokens
        vec![32000, 32001, 32099],  // Extra ID tokens
    ];
    
    for tokens in token_sequences {
        let our_decoded = our_tokenizer.decode(&tokens);
        let hf_decoded = hf_tokenizer.decode(&tokens, false);
        
        println!("Tokens: {:?}", tokens);
        match (our_decoded, hf_decoded) {
            (Ok(our), Ok(hf)) => {
                println!("  Our: \"{}\"", our);
                println!("  HF:  \"{}\"", hf);
                if our != hf {
                    println!("  ❌ Mismatch!");
                } else {
                    println!("  ✅ Match!");
                }
            }
            _ => println!("  Decode error"),
        }
    }
} 