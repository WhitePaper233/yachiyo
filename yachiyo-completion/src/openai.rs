//! OpenAI v1 API provider implementation

use crate::provider::{ModelProvider, ProviderError, ProviderResult};
use crate::types::{
    CompletionRequest, CompletionResponse, MessageRole, SamplingMethod, StreamToken, UsageInfo,
};
use async_stream::try_stream;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;

/// OpenAI API provider configuration
pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    base_url: String,
    endpoint: String,
}

impl OpenAiProvider {
    /// Create a new OpenAI provider with the given API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            endpoint: "/chat/completions".to_string(),
        }
    }

    /// Create a new OpenAI provider with custom base URL (for compatible APIs)
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into(),
            endpoint: "/chat/completions".to_string(),
        }
    }

    /// Set a custom API endpoint path (default: /chat/completions)
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Get the full URL for API requests
    fn build_url(&self) -> String {
        format!("{}{}", self.base_url, self.endpoint)
    }

    /// Convert internal request to OpenAI API format
    fn build_request(&self, request: &CompletionRequest) -> OpenAiRequest {
        OpenAiRequest {
            model: request.model.clone(),
            messages: request
                .messages
                .iter()
                .map(|m| OpenAiMessage {
                    role: match m.role {
                        MessageRole::System => "system".to_string(),
                        MessageRole::User => "user".to_string(),
                        MessageRole::Assistant => "assistant".to_string(),
                        MessageRole::Tool => "tool".to_string(),
                    },
                    content: m.content.clone(),
                })
                .collect(),
            max_tokens: request.max_tokens,
            temperature: match &request.sampling {
                Some(SamplingMethod::Temperature(t)) => Some(*t),
                _ => None,
            },
            top_p: match &request.sampling {
                Some(SamplingMethod::TopP(p)) => Some(*p),
                _ => None,
            },
            stop: request.stop_sequences.clone(),
            stream: request.stream,
            thinking: request.thinking,
        }
    }

    /// Parse OpenAI response into internal format
    fn parse_response(&self, response: OpenAiResponse) -> CompletionResponse {
        let choice = &response.choices[0];
        CompletionResponse {
            content: choice.message.content.clone(),
            finish_reason: choice.finish_reason.clone(),
            usage: response.usage.map(|u| UsageInfo {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
        }
    }
}

impl ModelProvider for OpenAiProvider {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<CompletionResponse>> + Send + '_>> {
        Box::pin(async move {
            let openai_req = self.build_request(&request);
            let url = self.build_url();

            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&openai_req)
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let error_body = response.text().await.unwrap_or_default();
                return Err(ProviderError::ApiError {
                    status: status.as_u16(),
                    message: error_body,
                });
            }

            let openai_response: OpenAiResponse = response.json().await?;
            Ok(self.parse_response(openai_response))
        })
    }

    fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = ProviderResult<StreamToken>> + Send + '_>> {
        let mut stream_request = request.clone();
        stream_request.stream = true;
        let openai_req = self.build_request(&stream_request);
        let url = self.build_url();
        let client = self.client.clone();
        let api_key = self.api_key.clone();

        let stream = try_stream! {
            let response = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&openai_req)
                .send()
                .await
                .map_err(|e| ProviderError::Network(e))?;

            let status = response.status();
            if !status.is_success() {
                let error_body = response.text().await.unwrap_or_default();
                Err(ProviderError::ApiError {
                    status: status.as_u16(),
                    message: error_body,
                })?;
                return;
            }

            // Stream bytes and parse SSE events
            use futures::StreamExt;
            let mut event_stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = event_stream.next().await {
                let chunk = chunk_result.map_err(|e| ProviderError::Network(e))?;
                let chunk_str = String::from_utf8_lossy(&chunk);
                buffer.push_str(&chunk_str);

                // Process complete SSE events
                while let Some(pos) = buffer.find("\n\n") {
                    let event = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    for line in event.lines() {
                        if line.starts_with("data: ") {
                            let data = &line[6..];
                            if data == "[DONE]" {
                                return;
                            }

                            if let Ok(chunk_response) = serde_json::from_str::<OpenAiStreamChunk>(data) {
                                if let Some(choice) = chunk_response.choices.first() {
                                    // Emit thinking token if present
                                    if let Some(ref thinking) = choice.delta.reasoning_content {
                                        let thinking_token = StreamToken {
                                            text: thinking.clone(),
                                            is_thinking: true,
                                            finish_reason: choice.finish_reason.clone(),
                                        };
                                        yield thinking_token;
                                    }

                                    // Emit content token if present
                                    if let Some(ref content) = choice.delta.content {
                                        let token = StreamToken {
                                            text: content.clone(),
                                            is_thinking: false,
                                            finish_reason: choice.finish_reason.clone(),
                                        };
                                        yield token;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        Box::pin(stream)
    }
}

// OpenAI API request/response types

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<bool>,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamDelta {
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}
