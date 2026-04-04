//! Core types for embedding requests and responses

use serde::{Deserialize, Serialize};

/// Input for embedding: either a single string or an array of strings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Single text input
    Text(String),
    /// Multiple text inputs (batch)
    Texts(Vec<String>),
}

impl EmbeddingInput {
    /// Create from a single text
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Create from multiple texts
    pub fn texts(texts: Vec<String>) -> Self {
        Self::Texts(texts)
    }
}

impl From<String> for EmbeddingInput {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for EmbeddingInput {
    fn from(s: &str) -> Self {
        Self::Text(s.into())
    }
}

impl From<Vec<String>> for EmbeddingInput {
    fn from(v: Vec<String>) -> Self {
        Self::Texts(v)
    }
}

/// Embedding request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Model identifier (e.g., "text-embedding-3-small", "bge-m3")
    pub model: String,
    /// Input text(s) to embed
    pub input: EmbeddingInput,
    /// Optional dimensions for the output embedding (provider-dependent)
    pub dimensions: Option<usize>,
    /// Format for the output embeddings
    #[serde(default)]
    pub encoding_format: EmbeddingEncodingFormat,
}

/// Encoding format for embedding output
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingEncodingFormat {
    #[default]
    Float,
    Base64,
    Binary,
}

impl EmbeddingRequest {
    /// Create a new embedding request with a single text input
    pub fn new(model: impl Into<String>, input: impl Into<EmbeddingInput>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            dimensions: None,
            encoding_format: EmbeddingEncodingFormat::default(),
        }
    }

    /// Set the output dimensions
    pub fn with_dimensions(mut self, dim: usize) -> Self {
        self.dimensions = Some(dim);
        self
    }

    /// Set the encoding format
    pub fn with_encoding_format(mut self, format: EmbeddingEncodingFormat) -> Self {
        self.encoding_format = format;
        self
    }
}

/// A single embedding result with its metadata
#[derive(Debug, Clone)]
pub struct EmbeddingData {
    /// The embedding vector (list of floats)
    pub embedding: Vec<f32>,
    /// Index of this embedding in the original request (for batch requests)
    pub index: usize,
}

/// Usage information for the embedding request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    /// Number of tokens in the input
    pub prompt_tokens: u32,
    /// Total tokens processed
    pub total_tokens: u32,
}

/// Embedding response
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// List of embedding results
    pub data: Vec<EmbeddingData>,
    /// The model used for generating embeddings
    pub model: String,
    /// Token usage information
    pub usage: Option<EmbeddingUsage>,
}

impl EmbeddingResponse {
    /// Get the first embedding (convenience for single-input requests)
    pub fn first(&self) -> Option<&Vec<f32>> {
        self.data.first().map(|d| &d.embedding)
    }

    /// Get all embeddings as a 2D vec
    pub fn embeddings(&self) -> Vec<&Vec<f32>> {
        self.data.iter().map(|d| &d.embedding).collect()
    }
}
