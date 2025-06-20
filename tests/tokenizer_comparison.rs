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
    
    // Try to initialize HuggingFace tokenizer for T5
    // Note: This requires the tokenizer file to be in the expected format
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    println!("Testing tokenization consistency across implementations...\n");
    
    let mut total_tests = 0;
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();
    
    // Test all queries
    for (query, category) in EXTENDED_TEST_QUERIES {
        total_tests += 1;
        
        // Our tokenizer
        let our_tokens = our_tokenizer.encode(query).expect("Our tokenizer failed");
        
        // HuggingFace tokenizer
        let hf_encoding = hf_tokenizer.encode(*query, false).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        // Compare results
        if our_tokens != hf_tokens {
            failed_tests.push((query, category, our_tokens.clone(), hf_tokens.clone()));
            println!("❌ MISMATCH in category '{}': \"{}\"", category, query);
            println!("   Our tokens: {:?}", our_tokens);
            println!("   HF tokens:  {:?}", hf_tokens);
            println!();
        } else {
            passed_tests += 1;
        }
    }
    
    // Test additional cases
    for text in ADDITIONAL_TEST_CASES {
        total_tests += 1;
        
        let our_tokens = our_tokenizer.encode(text).expect("Our tokenizer failed");
        let hf_encoding = hf_tokenizer.encode(*text, false).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        if our_tokens != hf_tokens {
            failed_tests.push((text, &"Additional", our_tokens.clone(), hf_tokens.clone()));
            println!("❌ MISMATCH: \"{}\"", text);
            println!("   Our tokens: {:?}", our_tokens);
            println!("   HF tokens:  {:?}", hf_tokens);
            println!();
        } else {
            passed_tests += 1;
        }
    }
    
    // Print summary
    println!("\n=== Tokenization Consistency Test Summary ===");
    println!("Total tests: {}", total_tests);
    println!("Passed: {} ({}%)", passed_tests, (passed_tests * 100) / total_tests);
    println!("Failed: {} ({}%)", failed_tests.len(), (failed_tests.len() * 100) / total_tests);
    
    if !failed_tests.is_empty() {
        println!("\nFailed test details:");
        for (text, category, our, hf) in &failed_tests[..5.min(failed_tests.len())] {
            println!("  Text: \"{}\" ({})", text, category);
            println!("  Our: {:?}", our);
            println!("  HF:  {:?}", hf);
        }
        if failed_tests.len() > 5 {
            println!("  ... and {} more failures", failed_tests.len() - 5);
        }
    }
    
    // Assert all tests pass
    assert_eq!(failed_tests.len(), 0, "Tokenization consistency tests failed");
}

#[test]
fn test_performance_comparison() {
    println!("\n=== Performance Comparison ===");
    
    // Initialize tokenizers
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
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
    
    let batch_sizes = vec![10, 50, 100, 500];
    
    for size in batch_sizes {
        let texts: Vec<&str> = EXTENDED_TEST_QUERIES.iter()
            .cycle()
            .take(size)
            .map(|(text, _)| *text)
            .collect();
        
        println!("\nBatch size: {}", size);
        
        // Our batch tokenizer
        let start = Instant::now();
        let _ = batch_tokenizer.encode_batch(&texts).unwrap();
        let our_batch_time = start.elapsed();
        
        // HuggingFace (sequential)
        let start = Instant::now();
        for text in &texts {
            let _ = hf_tokenizer.encode(*text, false).unwrap();
        }
        let hf_seq_time = start.elapsed();
        
        println!("  Our batch tokenizer: {:?} (avg: {:?})", our_batch_time, our_batch_time / size as u32);
        println!("  HF sequential:       {:?} (avg: {:?})", hf_seq_time, hf_seq_time / size as u32);
        println!("  Speedup: {:.2}x", hf_seq_time.as_secs_f64() / our_batch_time.as_secs_f64());
    }
}

#[test]
fn test_decode_consistency() {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
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
    config.add_eos = true;
    config.add_bos = false;
    config.pad_to_max_length = true;
    config.max_length = 20;
    
    let our_tokenizer = FlanT5Tokenizer::new(config);
    
    let test_text = "Test special tokens";
    let tokens = our_tokenizer.encode(test_text).unwrap();
    
    println!("Text: \"{}\"", test_text);
    println!("Tokens (padded to {}): {:?}", tokens.len(), tokens);
    
    // Check for EOS token
    assert!(tokens.contains(&1), "EOS token (1) should be present");
    
    // Check for padding
    assert_eq!(tokens.len(), 20, "Should be padded to max_length");
    assert!(tokens.contains(&0), "PAD token (0) should be present");
} 