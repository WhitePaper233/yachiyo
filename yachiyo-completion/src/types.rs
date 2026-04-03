//! Core types for completion requests and responses

use serde::{Deserialize, Serialize};

/// Role of a message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

/// Sampling method for token generation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SamplingMethod {
    Temperature(f32),
    TopP(f32),
    TopK(i32),
}

/// Completion request configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
    pub model: String,
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Sampling method
    pub sampling: Option<SamplingMethod>,
    /// Sequences where the API will stop generating further tokens
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to stream the response
    pub stream: bool,
    /// Enable thinking/reasoning tokens (for models that support it)
    pub thinking: Option<bool>,
}

/// A single token in the streaming response
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// The token text
    pub text: String,
    /// Whether this is a thinking/reasoning token
    pub is_thinking: bool,
    /// Optional finish reason (Some means this is the last token)
    pub finish_reason: Option<String>,
}

/// Complete (non-streaming) response
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// The generated text
    pub content: String,
    /// Reason the generation stopped
    pub finish_reason: Option<String>,
    /// Token usage information
    pub usage: Option<UsageInfo>,
}

/// Token usage information
#[derive(Debug, Clone)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl CompletionRequest {
    /// Create a new completion request with a single user message
    pub fn new(model: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: content.into(),
            }],
            max_tokens: None,
            sampling: None,
            stop_sequences: None,
            stream: false,
            thinking: None,
        }
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set temperature for sampling
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.sampling = Some(SamplingMethod::Temperature(temp));
        self
    }

    /// Enable streaming
    pub fn with_streaming(mut self) -> Self {
        self.stream = true;
        self
    }

    /// Add a system message
    pub fn with_system_message(mut self, content: impl Into<String>) -> Self {
        self.messages.insert(
            0,
            Message {
                role: MessageRole::System,
                content: content.into(),
            },
        );
        self
    }

    /// Add a user message
    pub fn with_user_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message {
            role: MessageRole::User,
            content: content.into(),
        });
        self
    }

    /// Add an assistant message
    pub fn with_assistant_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message {
            role: MessageRole::Assistant,
            content: content.into(),
        });
        self
    }

    /// Enable thinking/reasoning tokens
    pub fn with_thinking(mut self) -> Self {
        self.thinking = Some(true);
        self
    }
}
