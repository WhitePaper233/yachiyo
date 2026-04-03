use agent_core::openai::OpenAiProvider;
use agent_core::{CompletionRequest, ModelProvider};
use futures::StreamExt;

#[tokio::test]
async fn test_streaming_completion() {
    // Create provider with customizable endpoint
    let provider = OpenAiProvider::with_base_url("api_key", "http://localhost:11434/v1");

    // Create a streaming completion request with thinking enabled
    let request = CompletionRequest::new("qwen3.5", "给我讲在如何用C++实现二分查找算法")
        .with_temperature(0.7)
        .with_max_tokens(100000)
        .with_streaming()
        .with_thinking();

    println!("=== Streaming tokens from OpenAI API ===\n");

    // Execute streaming request
    let mut stream = provider.complete_stream(request);
    let mut full_text = String::new();
    let mut thinking_text = String::new();

    while let Some(token_result) = stream.next().await {
        match token_result {
            Ok(token) => {
                if token.is_thinking {
                    print!("[THINK] {}", token.text);
                    thinking_text.push_str(&token.text);
                } else {
                    print!("{}", token.text);
                    full_text.push_str(&token.text);
                }

                // Check if this is the last token
                if let Some(reason) = &token.finish_reason {
                    println!("\n\n=== Finished (reason: {}) ===", reason);
                }
            }
            Err(e) => {
                eprintln!("\nError receiving token: {}", e);
                break;
            }
        }
    }

    println!("\n\n=== Thinking Process ===\n{}", thinking_text);
    println!("\n\n=== Full Response ===\n{}", full_text);
}
