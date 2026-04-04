//! Provider trait for decoupling different embedding model providers

use crate::types::{EmbeddingRequest, EmbeddingResponse};
use std::future::Future;

/// Error type for embedding provider operations
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("API error: {status} - {message}")]
    ApiError { status: u16, message: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Provider not configured: {0}")]
    NotConfigured(String),
}

/// Result type for embedding operations
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;

/// Trait for embedding model providers (OpenAI, Ollama, local servers, etc.)
///
/// This trait abstracts the differences between various embedding providers
/// and provides a unified interface for embedding requests.
pub trait EmbeddingProvider: Send + Sync {
    /// Execute an embedding request and return the full response
    fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> Pin<Box<dyn Future<Output = EmbeddingResult<EmbeddingResponse>> + Send + '_>>;
}

// Required for Pin + Future without explicit import
use std::pin::Pin;
