//! Type definitions for the vLLM Chat Completions API.
//!
//! Reuses shared types from the OpenAI Chat Completions provider where possible,
//! but defines custom request/response types for vLLM-specific fields:
//! - Request: `chat_template_kwargs`, `include_reasoning`, `top_k`, `min_p`, `repetition_penalty`
//! - Response: `reasoning` field (vLLM uses `reasoning`, not `reasoning_content` like OpenAI)

use serde::{Deserialize, Serialize};

use crate::providers::openai_chat_completions::client::types::{
    ChatMessage, DeltaToolCall, LogProbs, ResponseFormat, Role, StopSequences, StreamOptions, Tool,
    ToolCall, ToolChoice, Usage,
};

// ============================================================================
// REQUEST TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct VllmChatCompletionsOptions {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequences>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    // vLLM-specific fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_reasoning: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
}

// ============================================================================
// RESPONSE TYPES
// ============================================================================

/// vLLM-specific chat message for responses.
/// Uses `reasoning` field instead of OpenAI's `reasoning_content`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VllmChatMessage {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VllmChoice {
    pub index: u32,
    pub message: VllmChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VllmChatCompletionsResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<VllmChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

// ============================================================================
// STREAMING TYPES
// ============================================================================

/// vLLM-specific delta for streaming responses.
/// Uses `reasoning` field instead of OpenAI's `reasoning_content`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct VllmDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct VllmStreamChoice {
    pub index: u32,
    pub delta: VllmDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct VllmChatCompletionsStreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<VllmStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum VllmStreamEvent {
    Chunk(VllmChatCompletionsStreamChunk),
    Done,
    Error(String),
    Open,
}
