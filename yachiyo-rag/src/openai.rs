//! OpenAI-compatible embedding provider
//!
//! Supports OpenAI API format and any provider that implements the `/v1/embeddings` endpoint
//! (e.g., OpenAI, Ollama, LiteLLM, One-API, etc.)

use reqwest::{Client, Url};
use std::pin::Pin;

use crate::provider::{EmbeddingError, EmbeddingProvider, EmbeddingResult};
use crate::types::{EmbeddingData, EmbeddingRequest, EmbeddingResponse};

/// Wire-format request sent to OpenAI-compatible APIs
#[derive(Debug, Clone, serde::Serialize)]
struct OpenAiEmbeddingRequest {
    model: String,
    input: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
}

/// Wire-format response from OpenAI-compatible APIs
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
struct OpenAiEmbeddingResponse {
    object: String,
    data: Vec<OpenAiEmbeddingData>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
struct OpenAiEmbeddingData {
    object: String,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

/// OpenAI-compatible embedding provider
pub struct OpenAiEmbeddingProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAiEmbeddingProvider {
    /// Create a new provider pointing to OpenAI's default endpoint
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".into(),
        }
    }

    /// Create a new provider with a custom base URL
    ///
    /// This works with any OpenAI-compatible server:
    /// - Ollama: `http://localhost:11434/v1`
    /// - LiteLLM: `http://localhost:4000`
    /// - One-API / New-API: custom proxy URLs
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    /// Build the full endpoint URL
    fn build_url(&self) -> Result<Url, EmbeddingError> {
        let url = Url::parse(&self.base_url)
            .map_err(|e| EmbeddingError::NotConfigured(format!("Invalid base URL: {e}")))?
            .join("/v1/embeddings")
            .map_err(|e| EmbeddingError::NotConfigured(format!("Invalid base URL: {e}")))?;
        Ok(url)
    }

    /// Convert internal request to wire format
    fn build_request(&self, request: &EmbeddingRequest) -> OpenAiEmbeddingRequest {
        // Serialize input to JSON value (OpenAI API expects "input" to be either
        // a string or an array of strings in JSON)
        let input_json = serde_json::to_value(&request.input)
            .expect("EmbeddingInput should always serialize to valid JSON");

        OpenAiEmbeddingRequest {
            model: request.model.clone(),
            input: input_json,
            dimensions: request.dimensions,
            encoding_format: match request.encoding_format {
                crate::types::EmbeddingEncodingFormat::Float => Some("float".into()),
                crate::types::EmbeddingEncodingFormat::Base64 => Some("base64".into()),
                crate::types::EmbeddingEncodingFormat::Binary => Some("binary".into()),
            },
        }
    }

    /// Parse wire-format response into internal format
    fn parse_response(
        &self,
        response: OpenAiEmbeddingResponse,
    ) -> EmbeddingResult<EmbeddingResponse> {
        let data: Vec<EmbeddingData> = response
            .data
            .into_iter()
            .map(|d| EmbeddingData {
                embedding: d.embedding,
                index: d.index,
            })
            .collect();

        let usage = response.usage.map(|u| crate::types::EmbeddingUsage {
            prompt_tokens: u.prompt_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(EmbeddingResponse {
            data,
            model: response.model,
            usage,
        })
    }
}

impl EmbeddingProvider for OpenAiEmbeddingProvider {
    fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> Pin<Box<dyn Future<Output = EmbeddingResult<EmbeddingResponse>> + Send + '_>> {
        Box::pin(async move {
            let url = self.build_url()?;
            let wire_request = self.build_request(&request);

            let response = self
                .client
                .post(url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&wire_request)
                .send()
                .await?;

            let status = response.status().as_u16();
            if !response.status().is_success() {
                let message = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".into());
                return Err(EmbeddingError::ApiError { status, message });
            }

            let body: OpenAiEmbeddingResponse = response.json().await?;
            self.parse_response(body)
        })
    }
}
