//! Type definitions for vLLM-specific response types.
//!
//! Reuses shared types from the OpenAI Chat Completions provider where possible,
//! but defines custom response types because vLLM uses `reasoning` field
//! instead of `reasoning_content` like OpenAI.

use serde::{Deserialize, Serialize};

use crate::providers::openai_chat_completions::client::types::{
    DeltaToolCall, LogProbs, Role, ToolCall, Usage,
};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_usage_only_chunk() {
        let json = r#"{"id":"chatcmpl-af39d78a86c24247","object":"chat.completion.chunk","created":1773898751,"model":"Qwen3.5-122B-A10B-NVFP4","choices":[],"usage":{"prompt_tokens":12,"total_tokens":17,"completion_tokens":5}}"#;
        let chunk: VllmChatCompletionsStreamChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.choices.is_empty());
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 12);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 17);
    }
}
