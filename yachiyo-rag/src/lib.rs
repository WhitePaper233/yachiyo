//! RAG (Retrieval-Augmented Generation) module
//!
//! Provides embedding and vector storage capabilities for building RAG pipelines.

pub mod openai;
pub mod provider;
pub mod types;

// Re-export main types for convenience
pub use openai::OpenAiEmbeddingProvider;
pub use provider::{EmbeddingError, EmbeddingProvider, EmbeddingResult};
pub use types::{
    EmbeddingData, EmbeddingEncodingFormat, EmbeddingInput, EmbeddingRequest, EmbeddingResponse,
    EmbeddingUsage,
};
