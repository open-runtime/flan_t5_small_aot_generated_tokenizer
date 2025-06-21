use flan_t5_tokenizer::{FlanT5Tokenizer, TokenizerConfig, BatchTokenizer, BatchConfig};
use std::time::Instant;

// Add rust_tokenizers import
use rust_tokenizers::tokenizer::{T5Tokenizer, Tokenizer as RustTokenizer, TruncationStrategy};

// Extended real-world test queries from production data
const PRODUCTION_QUERIES: &[(&str, &str)] = &[
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

// Developer conversation examples
const DEVELOPER_CONVERSATIONS: &[(&str, &str)] = &[
    // Debugging & Errors
    ("TypeError: Cannot read property 'map' of undefined", "Debug"),
    ("why is my useState not updating the UI?", "Debug"),
    ("getting CORS error when calling my API endpoint", "Debug"),
    ("Segmentation fault (core dumped)", "Debug"),
    ("npm ERR! code ENOENT npm ERR! syscall open", "Debug"),
    ("Module not found: Can't resolve 'react-router-dom'", "Debug"),
    ("error TS2322: Type 'string' is not assignable to type 'number'", "Debug"),
    ("fatal: refusing to merge unrelated histories", "Debug"),
    ("docker: Error response from daemon: Conflict.", "Debug"),
    ("EADDRINUSE: address already in use :::3000", "Debug"),
    
    // Code Requests
    ("write a function to debounce API calls in React", "Code Request"),
    ("how do I implement binary search in Rust?", "Code Request"),
    ("create a Python decorator for timing function execution", "Code Request"),
    ("implement a custom hook for localStorage in React", "Code Request"),
    ("write a SQL query to find duplicate rows", "Code Request"),
    ("create a regex to validate email addresses", "Code Request"),
    ("implement quicksort algorithm in JavaScript", "Code Request"),
    ("write a bash script to backup MySQL database", "Code Request"),
    ("create a Dockerfile for a Node.js application", "Code Request"),
    ("implement a JWT authentication middleware", "Code Request"),
    
    // Architecture & Design
    ("should I use microservices or monolithic architecture?", "Architecture"),
    ("what's the best way to structure a React project?", "Architecture"),
    ("explain the difference between REST and GraphQL", "Architecture"),
    ("when should I use Redis vs MongoDB?", "Architecture"),
    ("what's the proper way to handle state in React Native?", "Architecture"),
    ("how do I implement clean architecture in Go?", "Architecture"),
    ("explain event-driven architecture with examples", "Architecture"),
    ("what's the best practice for API versioning?", "Architecture"),
    ("should I use serverless for this use case?", "Architecture"),
    ("how to implement CQRS pattern in .NET Core?", "Architecture"),
    
    // Performance & Optimization
    ("my React app is rendering too many times", "Performance"),
    ("how to optimize this SQL query that's taking 30 seconds", "Performance"),
    ("webpack bundle size is too large, how to reduce it?", "Performance"),
    ("implement lazy loading for images in Next.js", "Performance"),
    ("optimize this O(n²) algorithm", "Performance"),
    ("my Node.js API response time is slow", "Performance"),
    ("how to implement caching in Spring Boot", "Performance"),
    ("reduce Docker image size for production", "Performance"),
    ("optimize database indexing for better performance", "Performance"),
    ("implement pagination for large datasets", "Performance"),
    
    // Git & Version Control
    ("how do I undo the last commit?", "Git"),
    ("merge vs rebase, which should I use?", "Git"),
    ("how to cherry-pick a commit from another branch", "Git"),
    ("squash multiple commits into one", "Git"),
    ("resolve merge conflicts in package-lock.json", "Git"),
    ("how to revert a pushed commit", "Git"),
    ("create a git hook for pre-commit linting", "Git"),
    ("how to use git bisect to find a bug", "Git"),
    ("set up git flow for team collaboration", "Git"),
    ("recover deleted branch in git", "Git"),
];

// Digital worker conversation examples
const DIGITAL_WORKER_CONVERSATIONS: &[(&str, &str)] = &[
    // Data Analysis
    ("analyze the sales data from Q3 and identify trends", "Data Analysis"),
    ("create a pivot table showing revenue by region and product", "Data Analysis"),
    ("what's the correlation between marketing spend and conversions?", "Data Analysis"),
    ("generate a report on customer churn rate", "Data Analysis"),
    ("visualize the user engagement metrics from last month", "Data Analysis"),
    ("compare this year's performance with last year", "Data Analysis"),
    ("identify outliers in the expense reports", "Data Analysis"),
    ("calculate the ROI of our latest campaign", "Data Analysis"),
    ("forecast next quarter's revenue based on current trends", "Data Analysis"),
    ("segment customers based on purchasing behavior", "Data Analysis"),
    
    // Content Creation
    ("write a blog post about remote work best practices", "Content"),
    ("create social media captions for our product launch", "Content"),
    ("draft an email to customers about the new features", "Content"),
    ("generate SEO-optimized product descriptions", "Content"),
    ("write documentation for the API endpoints", "Content"),
    ("create a press release for the company announcement", "Content"),
    ("draft meeting notes from today's standup", "Content"),
    ("write a case study about our successful client project", "Content"),
    ("create FAQ content for the support page", "Content"),
    ("generate newsletter content for this month", "Content"),
    
    // Project Management
    ("update the sprint board with current task status", "Project Management"),
    ("create a project timeline for the new feature", "Project Management"),
    ("what tasks are blocking the release?", "Project Management"),
    ("generate a burndown chart for this sprint", "Project Management"),
    ("assign tasks to team members based on capacity", "Project Management"),
    ("create risk assessment for the project", "Project Management"),
    ("update stakeholders on project progress", "Project Management"),
    ("plan the next sprint based on velocity", "Project Management"),
    ("track dependencies between teams", "Project Management"),
    ("create a retrospective summary", "Project Management"),
    
    // Customer Support
    ("respond to customer complaint about shipping delay", "Support"),
    ("help user reset their password", "Support"),
    ("explain how to use the advanced features", "Support"),
    ("troubleshoot login issues for enterprise client", "Support"),
    ("process refund request for damaged product", "Support"),
    ("escalate technical issue to engineering team", "Support"),
    ("create ticket for bug reported by customer", "Support"),
    ("follow up on unresolved support tickets", "Support"),
    ("update knowledge base with common solutions", "Support"),
    ("draft apology email for service outage", "Support"),
];

// Complex multi-turn conversation examples
const MULTI_TURN_CONVERSATIONS: &[(&str, &str)] = &[
    // Contextual follow-ups
    ("I'm working on a React app", "Context"),
    ("it's throwing an error when I try to update state", "Context"),
    ("the error says 'Cannot update during an existing state transition'", "Context"),
    ("I'm calling setState inside render method", "Context"),
    ("oh wait, should I move it to useEffect?", "Context"),
    ("actually, let me show you the code", "Context"),
    ("```jsx\nconst [count, setCount] = useState(0);\nsetCount(count + 1); // This is in render\n```", "Context"),
    ("how do I fix this properly?", "Context"),
    
    // Clarifications and corrections
    ("can you help me with the database", "Clarification"),
    ("sorry I meant the database schema", "Clarification"),
    ("specifically for user authentication", "Clarification"),
    ("we're using PostgreSQL btw", "Clarification"),
    ("and we need to support OAuth", "Clarification"),
    ("actually, make that OAuth2 with Google and GitHub", "Clarification"),
    ("oh and also need to store refresh tokens", "Clarification"),
    ("wait, is it secure to store refresh tokens in the database?", "Clarification"),
    
    // Building on previous context
    ("implement a shopping cart in React", "Building"),
    ("now add ability to update quantities", "Building"),
    ("also need to calculate total price", "Building"),
    ("include tax calculation based on state", "Building"),
    ("add discount code functionality", "Building"),
    ("make it persist in localStorage", "Building"),
    ("sync with backend API", "Building"),
    ("handle offline mode gracefully", "Building"),
    ("add optimistic updates", "Building"),
    ("implement undo functionality", "Building"),
];

// Edge cases and stress tests - using Vec for dynamic strings
fn get_edge_cases() -> Vec<String> {
    let cases = vec![
        // Empty and whitespace
        "".to_string(),
        " ".to_string(),
        "   ".to_string(),
        "\t".to_string(),
        "\n".to_string(),
        "\r\n".to_string(),
        " \t \n ".to_string(),
        
        // Special characters
        "!@#$%^&*()_+-=[]{}|;':\",./<>?".to_string(),
        "\\\\\\\\".to_string(),
        "````````".to_string(),
        "////////".to_string(),
        "########".to_string(),
        "~~~~~~~~".to_string(),
        "||||||||".to_string(),
        
        // Unicode and emojis
        "🚀🔥💻🎉🎨🎯🎮🎭".to_string(),
        "你好世界".to_string(),
        "مرحبا بالعالم".to_string(),
        "Привет мир".to_string(),
        "こんにちは世界".to_string(),
        "🇺🇸🇬🇧🇯🇵🇩🇪🇫🇷".to_string(),
        "👨‍💻👩‍💻👨‍🔬👩‍🔬".to_string(),
        "❤️💔💕💖💗💘💙💚".to_string(),
        
        // Very long strings
        "a".repeat(1000),
        "word ".repeat(200),
        "The quick brown fox jumps over the lazy dog. ".repeat(50),
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(30),
        
        // Code snippets
        "function test() { return 'hello'; }".to_string(),
        "const arr = [1, 2, 3].map(x => x * 2);".to_string(),
        "public class Main { public static void main(String[] args) {} }".to_string(),
        "#include <iostream>\nint main() { std::cout << \"Hello\"; }".to_string(),
        "SELECT * FROM users WHERE id = ?;".to_string(),
        "docker run -p 8080:80 nginx:latest".to_string(),
        "git commit -m 'feat: add new feature'".to_string(),
        
        // Mixed languages
        "English 中文 Español Русский العربية".to_string(),
        "Hello世界Bonjour世界Hola".to_string(),
        "function 函数() { return 'مرحبا'; }".to_string(),
        
        // Malformed text
        "this is a test〉〉〉".to_string(),
        "incomplete markdown ```".to_string(),
        "broken <html> tags </div>".to_string(),
        "unclosed quotes 'hello".to_string(),
        "mismatched brackets {[(])}".to_string(),
        
        // Numbers and math
        "1234567890".to_string(),
        "3.14159265359".to_string(),
        "1e-10".to_string(),
        "∑∏∫∂∇±≈≠≤≥".to_string(),
        "x² + y² = r²".to_string(),
        "E = mc²".to_string(),
        
        // URLs and paths
        "https://www.example.com/path?query=value&foo=bar".to_string(),
        "http://localhost:3000/api/v1/users/123".to_string(),
        "file:///C:/Users/Name/Documents/file.txt".to_string(),
        "/usr/local/bin/python3".to_string(),
        "../../relative/path/to/file.js".to_string(),
        "C:\\Windows\\System32\\cmd.exe".to_string(),
    ];
    
    cases
}

#[test]
fn test_all_production_queries() {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let rust_tokenizer = T5Tokenizer::from_file("spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    println!("\n=== Testing Production Queries ===");
    
    let mut mismatches = 0;
    let mut rust_mismatches = 0;
    let total = PRODUCTION_QUERIES.len();
    
    for (query, category) in PRODUCTION_QUERIES {
        let our_tokens = our_tokenizer.encode(query).expect("Our tokenizer failed");
        let hf_encoding = hf_tokenizer.encode(*query, false).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        let rust_tokens = {
            let encoding = rust_tokenizer.encode(query, None, 512, &TruncationStrategy::LongestFirst, 0);
            encoding.token_ids.iter().map(|&id| id as u32).collect::<Vec<_>>()
        };
        
        let hf_match = our_tokens == hf_tokens;
        let rust_match = our_tokens == rust_tokens;
        
        if !hf_match {
            mismatches += 1;
        }
        if !rust_match {
            rust_mismatches += 1;
        }
        
        if !hf_match || !rust_match {
            println!("\n❌ Mismatch in {}: \"{}\"", category, query);
            println!("   Our:  {:?}", our_tokens);
            println!("   HF:   {:?}", hf_tokens);
            println!("   Rust: {:?}", rust_tokens);
        }
    }
    
    let hf_accuracy = ((total - mismatches) as f64 / total as f64) * 100.0;
    println!("\n✅ HuggingFace accuracy: {:.2}% ({}/{})", hf_accuracy, total - mismatches, total);
    
    let rust_accuracy = ((total - rust_mismatches) as f64 / total as f64) * 100.0;
    println!("✅ rust_tokenizers accuracy: {:.2}% ({}/{})", rust_accuracy, total - rust_mismatches, total);
}

#[test]
fn test_developer_conversations() {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let rust_tokenizer = T5Tokenizer::from_file("spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    println!("\n=== Testing Developer Conversations ===");
    
    let mut hf_category_accuracy: std::collections::HashMap<&str, (usize, usize)> = std::collections::HashMap::new();
    let mut rust_category_accuracy: std::collections::HashMap<&str, (usize, usize)> = std::collections::HashMap::new();
    
    for (query, category) in DEVELOPER_CONVERSATIONS {
        let our_tokens = our_tokenizer.encode(query).expect("Our tokenizer failed");
        let hf_encoding = hf_tokenizer.encode(*query, false).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        let rust_tokens = {
            let encoding = rust_tokenizer.encode(query, None, 512, &TruncationStrategy::LongestFirst, 0);
            encoding.token_ids.iter().map(|&id| id as u32).collect::<Vec<_>>()
        };
        
        let hf_entry = hf_category_accuracy.entry(category).or_insert((0, 0));
        hf_entry.1 += 1;
        if our_tokens == hf_tokens {
            hf_entry.0 += 1;
        }
        
        let rust_entry = rust_category_accuracy.entry(category).or_insert((0, 0));
        rust_entry.1 += 1;
        if our_tokens == rust_tokens {
            rust_entry.0 += 1;
        }
    }
    
    println!("\nHuggingFace comparison:");
    for (category, (correct, total)) in hf_category_accuracy {
        let accuracy = (correct as f64 / total as f64) * 100.0;
        println!("  {}: {:.2}% ({}/{})", category, accuracy, correct, total);
    }
    
    println!("\nrust_tokenizers comparison:");
    for (category, (correct, total)) in rust_category_accuracy {
        let accuracy = (correct as f64 / total as f64) * 100.0;
        println!("  {}: {:.2}% ({}/{})", category, accuracy, correct, total);
    }
}

#[test]
fn test_digital_worker_conversations() {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let rust_tokenizer = T5Tokenizer::from_file("spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    println!("\n=== Testing Digital Worker Conversations ===");
    
    let mut hf_category_accuracy: std::collections::HashMap<&str, (usize, usize)> = std::collections::HashMap::new();
    let mut rust_category_accuracy: std::collections::HashMap<&str, (usize, usize)> = std::collections::HashMap::new();
    
    for (query, category) in DIGITAL_WORKER_CONVERSATIONS {
        let our_tokens = our_tokenizer.encode(query).expect("Our tokenizer failed");
        let hf_encoding = hf_tokenizer.encode(*query, false).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        let rust_tokens = {
            let encoding = rust_tokenizer.encode(query, None, 512, &TruncationStrategy::LongestFirst, 0);
            encoding.token_ids.iter().map(|&id| id as u32).collect::<Vec<_>>()
        };
        
        let hf_entry = hf_category_accuracy.entry(category).or_insert((0, 0));
        hf_entry.1 += 1;
        if our_tokens == hf_tokens {
            hf_entry.0 += 1;
        }
        
        let rust_entry = rust_category_accuracy.entry(category).or_insert((0, 0));
        rust_entry.1 += 1;
        if our_tokens == rust_tokens {
            rust_entry.0 += 1;
        }
    }
    
    println!("\nHuggingFace comparison:");
    for (category, (correct, total)) in hf_category_accuracy {
        let accuracy = (correct as f64 / total as f64) * 100.0;
        println!("  {}: {:.2}% ({}/{})", category, accuracy, correct, total);
    }
    
    println!("\nrust_tokenizers comparison:");
    for (category, (correct, total)) in rust_category_accuracy {
        let accuracy = (correct as f64 / total as f64) * 100.0;
        println!("  {}: {:.2}% ({}/{})", category, accuracy, correct, total);
    }
}

#[test]
fn test_multi_turn_conversations() {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    
    let rust_tokenizer = T5Tokenizer::from_file("spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    println!("\n=== Testing Multi-turn Conversations ===");
    
    let mut hf_correct = 0;
    let mut rust_correct = 0;
    let total = MULTI_TURN_CONVERSATIONS.len();
    
    for (i, (turn, _context)) in MULTI_TURN_CONVERSATIONS.iter().enumerate() {
        let our_tokens = our_tokenizer.encode(turn).expect("Our tokenizer failed");
        let hf_encoding = hf_tokenizer.encode(*turn, false).expect("HF tokenizer failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        
        let rust_tokens = {
            let encoding = rust_tokenizer.encode(turn, None, 512, &TruncationStrategy::LongestFirst, 0);
            encoding.token_ids.iter().map(|&id| id as u32).collect::<Vec<_>>()
        };
        
        if our_tokens == hf_tokens {
            hf_correct += 1;
        }
        
        if our_tokens == rust_tokens {
            rust_correct += 1;
        }
        
        // Show first few mismatches
        if i < 3 {
            let hf_mismatch = our_tokens != hf_tokens;
            let rust_mismatch = our_tokens != rust_tokens;
            
            if hf_mismatch || rust_mismatch {
                println!("\nTurn {}: \"{}\"", i + 1, turn);
                println!("Our:  {:?}", our_tokens);
                println!("HF:   {:?}", hf_tokens);
                println!("Rust: {:?}", rust_tokens);
            }
        }
    }
    
    let hf_accuracy = (hf_correct as f64 / total as f64) * 100.0;
    println!("\nHuggingFace accuracy: {:.2}% ({}/{})", hf_accuracy, hf_correct, total);
    
    let rust_accuracy = (rust_correct as f64 / total as f64) * 100.0;
    println!("rust_tokenizers accuracy: {:.2}% ({}/{})", rust_accuracy, rust_correct, total);
}

#[test]
fn test_edge_cases() {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    let rust_tokenizer = T5Tokenizer::from_file("spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    println!("\n=== Testing Edge Cases ===");
    
    let edge_cases = get_edge_cases();
    let mut stats = ConsensusStats::default();
    let mut consensus_failures = Vec::new();
    
    for text in &edge_cases {
        let our_result = our_tokenizer.encode(text);
        let hf_result = hf_tokenizer.encode(text.as_str(), false);
        let rust_result = rust_tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0);
        
        match (our_result, hf_result) {
            (Ok(our_tokens), Ok(hf_encoding)) => {
                let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
                let rust_tokens: Vec<u32> = rust_result.token_ids.iter().map(|&id| id as u32).collect();
                
                let our_hf_match = our_tokens == hf_tokens;
                let our_rust_match = our_tokens == rust_tokens;
                let hf_rust_match = hf_tokens == rust_tokens;
                
                stats.total += 1;
                if our_hf_match && our_rust_match { stats.all_agree += 1; }
                if our_hf_match { stats.our_hf_agree += 1; }
                if our_rust_match { stats.our_rust_agree += 1; }
                if hf_rust_match { stats.hf_rust_agree += 1; }
                
                // Track cases where HF and rust agree but ours doesn't
                if hf_rust_match && !our_hf_match {
                    stats.consensus_against_ours += 1;
                    if text.len() < 50 {
                        consensus_failures.push((text.clone(), our_tokens.clone(), hf_tokens.clone(), rust_tokens.clone()));
                    }
                }
                
                if !our_hf_match || !our_rust_match {
                    if text.len() < 50 {
                        println!("\nEdge case: {:?}", text);
                        println!("Our:  {:?}", our_tokens);
                        println!("HF:   {:?}", hf_tokens);
                        println!("Rust: {:?}", rust_tokens);
                    }
                }
            }
            (Err(_), Err(_)) => stats.all_agree += 1,
            _ => {}
        }
    }
    
    // Print consensus failures
    if !consensus_failures.is_empty() {
        println!("\n⚠️  CONSENSUS FAILURES (HF and rust_tokenizers agree, but ours differs):");
        for (text, our, hf, rust) in consensus_failures {
            println!("\nText: {:?}", text);
            println!("  Our:  {:?}", our);
            println!("  HF:   {:?}", hf);
            println!("  Rust: {:?}", rust);
        }
    }
    
    stats.print_summary("Edge Cases");
}

// Helper struct for consensus statistics
#[derive(Default)]
struct ConsensusStats {
    total: usize,
    all_agree: usize,
    our_hf_agree: usize,
    our_rust_agree: usize,
    hf_rust_agree: usize,
    consensus_against_ours: usize,
}

impl ConsensusStats {
    fn print_summary(&self, test_name: &str) {
        println!("\n=== {} Summary ===", test_name);
        println!("Total tests: {}", self.total);
        println!("All three agree: {} ({:.1}%)", 
            self.all_agree, 
            (self.all_agree as f64 / self.total as f64) * 100.0
        );
        println!("\nPairwise agreement:");
        println!("Our ↔ HuggingFace: {} ({:.1}%)", 
            self.our_hf_agree, 
            (self.our_hf_agree as f64 / self.total as f64) * 100.0
        );
        println!("Our ↔ rust_tokenizers: {} ({:.1}%)", 
            self.our_rust_agree, 
            (self.our_rust_agree as f64 / self.total as f64) * 100.0
        );
        println!("HuggingFace ↔ rust_tokenizers: {} ({:.1}%)", 
            self.hf_rust_agree, 
            (self.hf_rust_agree as f64 / self.total as f64) * 100.0
        );
        
        if self.consensus_against_ours > 0 {
            println!("\n⚠️  IMPORTANT: {} cases where HF and rust_tokenizers agree but ours differs ({:.1}%)",
                self.consensus_against_ours,
                (self.consensus_against_ours as f64 / self.total as f64) * 100.0
            );
        }
    }
}

#[test]
fn test_batch_processing() {
    let config = BatchConfig {
        max_batch_size: 32,
        batch_timeout: std::time::Duration::from_millis(10),
        num_workers: 4,
    };
    
    let tokenizer = FlanT5Tokenizer::with_default_config();
    let batch_tokenizer = BatchTokenizer::new(tokenizer, config);
    
    // Combine various conversation types
    let mut all_texts = Vec::new();
    all_texts.extend(PRODUCTION_QUERIES.iter().map(|(q, _)| *q));
    all_texts.extend(DEVELOPER_CONVERSATIONS.iter().map(|(q, _)| *q));
    all_texts.extend(DIGITAL_WORKER_CONVERSATIONS.iter().map(|(q, _)| *q));
    
    println!("\n=== Batch Processing Test ===");
    println!("Processing {} texts in batches...", all_texts.len());
    
    let start = Instant::now();
    let results = batch_tokenizer.encode_batch(&all_texts).expect("Batch encoding failed");
    let duration = start.elapsed();
    
    assert_eq!(results.len(), all_texts.len());
    println!("✅ Batch processed {} texts in {:?}", all_texts.len(), duration);
    println!("   Average: {:?} per text", duration / all_texts.len() as u32);
}

#[test]
fn test_decode_consistency() {
    let our_tokenizer = FlanT5Tokenizer::with_default_config();
    let hf_tokenizer = tokenizers::Tokenizer::from_file("flan_t5_small_tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");
    let rust_tokenizer = T5Tokenizer::from_file("spiece.model", false)
        .expect("Failed to load rust tokenizer");
    
    println!("\n=== Testing Decode Consistency ===");
    
    let test_texts = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "🦀 Rust is awesome! 🚀",
        "function test() { return true; }",
        "Mixed English and 中文 text",
    ];
    
    for text in test_texts {
        // First encode with all three tokenizers
        let our_tokens = our_tokenizer.encode(text).unwrap();
        let hf_encoding = hf_tokenizer.encode(text, false).unwrap();
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();
        let rust_encoding = rust_tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0);
        let rust_tokens: Vec<u32> = rust_encoding.token_ids.iter().map(|&id| id as u32).collect();
        
        println!("\nOriginal: \"{}\"", text);
        println!("Our tokens:  {:?}", our_tokens);
        println!("HF tokens:   {:?}", hf_tokens);
        println!("Rust tokens: {:?}", rust_tokens);
        
        // Check if tokenization matches
        let _all_match = our_tokens == hf_tokens && our_tokens == rust_tokens;
        let hf_rust_match = hf_tokens == rust_tokens;
        
        if hf_rust_match && our_tokens != hf_tokens {
            println!("⚠️  CONSENSUS: HF and rust_tokenizers agree, but ours differs!");
        }
        
        // Decode with our tokenizer (using our tokens)
        let our_decoded = our_tokenizer.decode(&our_tokens).unwrap();
        
        // Decode with HuggingFace (using HF tokens)
        let hf_decoded = hf_tokenizer.decode(&hf_tokens, false).unwrap();
        
        // Note: rust_tokenizers doesn't provide a decode method in the same way
        // So we'll compare token-to-text mapping indirectly
        
        println!("Decoded (ours): \"{}\"", our_decoded);
        println!("Decoded (HF):   \"{}\"", hf_decoded);
        
        // The decoded text should be similar to original (may differ in whitespace and unknown tokens)
        let normalized_original = text.trim();
        let normalized_our = our_decoded.trim();
        
        // For texts with Unicode/emojis that become UNK tokens, we expect "<unk>" in the decoded output
        if our_tokens.contains(&2) {  // Contains UNK token
            // Just verify that decoding worked without panic
            println!("Note: Text contains unknown tokens, decoded with <unk> placeholders");
        } else {
            // For texts without UNK tokens, decoded should match original
            assert_eq!(
                normalized_our.replace(" ", ""), 
                normalized_original.replace(" ", ""),
                "Our decode mismatch: original='{}', decoded='{}'", 
                normalized_original, 
                normalized_our
            );
        }
    }
}

#[test]
fn test_special_tokens_handling() {
    println!("\n=== Testing Special Tokens ===");
    
    let mut config = TokenizerConfig::default();
    config.add_eos_token = true;
    
    let tokenizer = FlanT5Tokenizer::new(config);
    
    let test_cases = vec![
        "Short text",
        "This is a longer text that should be tokenized",
        "",
    ];
    
    for text in test_cases {
        let tokens = tokenizer.encode(text).unwrap();
        
        println!("\nText: \"{}\"", text);
        println!("Tokens (len={}): {:?}", tokens.len(), tokens);
        
        // Check EOS token (ID 1)
        if text.is_empty() {
            assert_eq!(tokens, vec![1], "Empty text should return just EOS token");
        } else {
            assert_eq!(*tokens.last().unwrap(), 1, "Missing EOS token");
        }
    }
}

#[test]
#[ignore = "Caching not implemented in main tokenization path"]
fn test_performance_characteristics() {
    let tokenizer = FlanT5Tokenizer::with_default_config();
    
    println!("\n=== Performance Characteristics ===");
    
    // Test cache effectiveness
    let cached_text = "This text will be cached for performance";
    
    // Warm up the cache
    tokenizer.encode(cached_text).unwrap();
    
    // Now test with cache hit
    let start = Instant::now();
    for _ in 0..1000 {
        tokenizer.encode(cached_text).unwrap();
    }
    let cached_duration = start.elapsed();
    
    // Clear cache
    tokenizer.clear_cache();
    
    // Test without cache (single unique text, but cache cleared each time)
    let start = Instant::now();
    for _ in 0..1000 {
        tokenizer.clear_cache();  // Clear cache to force recomputation
        tokenizer.encode(cached_text).unwrap();
    }
    let uncached_duration = start.elapsed();
    
    println!("Cached (1000 iterations): {:?}", cached_duration);
    println!("Uncached (1000 iterations): {:?}", uncached_duration);
    println!("Cache speedup: {:.2}x", uncached_duration.as_secs_f64() / cached_duration.as_secs_f64());
    
    // The cached version should be faster (at least 20% speedup)
    assert!(cached_duration < uncached_duration * 5 / 6, 
        "Cache not providing expected speedup: cached {:?} vs uncached {:?}", 
        cached_duration, uncached_duration);
} 