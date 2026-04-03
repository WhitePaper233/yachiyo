//! Agent Core - A unified interface for AI model providers
//!
//! This crate provides a decoupled abstraction for interacting with various
//! AI model providers (OpenAI, Anthropic, DeepSeek, etc.) through a common interface.
//!
//! # Example
//!
//! ```no_run
//! use agent_core::{CompletionRequest, ModelProvider, openai::OpenAiProvider};
//! use futures::StreamExt;
//!
//! async fn example() {
//!     // Create an OpenAI provider
//!     let provider = OpenAiProvider::new("your-api-key");
//!
//!     // Create a completion request
//!     let request = CompletionRequest::new("gpt-4", "Hello, world!")
//!         .with_temperature(0.7)
//!         .with_max_tokens(100);
//!
//!     // Non-streaming completion
//!     let response = provider.complete(request.clone()).await.unwrap();
//!     println!("Response: {}", response.content);
//!
//!     // Streaming completion
//!     let mut stream = provider.complete_stream(request).await;
//!     while let Some(token) = stream.next().await {
//!         if let Ok(token) = token {
//!             print!("{}", token.text);
//!         }
//!     }
//! }
//! ```

pub mod openai;
pub mod provider;
pub mod types;

// Re-export main types for convenience
pub use provider::{ModelProvider, ProviderError, ProviderResult};
pub use types::{
    CompletionRequest, CompletionResponse, Message, MessageRole, SamplingMethod, StreamToken,
    UsageInfo,
};
