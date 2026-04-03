//! Provider trait for decoupling different model providers

use crate::types::{CompletionRequest, CompletionResponse, StreamToken};
use futures::Stream;
use std::pin::Pin;

/// Error type for provider operations
pub type ProviderResult<T> = Result<T, ProviderError>;

/// Error type for provider operations
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("API error: {status} - {message}")]
    ApiError { status: u16, message: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Stream error: {0}")]
    StreamError(String),

    #[error("Provider not configured: {0}")]
    NotConfigured(String),
}

/// Trait for AI model providers (OpenAI, Anthropic, DeepSeek, etc.)
///
/// This trait abstracts the differences between various AI model providers
/// and provides a unified interface for completion requests.
pub trait ModelProvider: Send + Sync {
    /// Execute a completion request and return the full response
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<CompletionResponse>> + Send + '_>>;

    /// Execute a streaming completion request and return a stream of tokens
    fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = ProviderResult<StreamToken>> + Send + '_>>;
}

// Helper to use Future without explicitly importing it
use std::future::Future;
