//! Settings for the vLLM provider.

use derive_builder::Builder;

/// Settings for the vLLM provider.
///
/// vLLM is a high-throughput LLM serving engine that exposes an
/// OpenAI-compatible API with extensions for reasoning and sampling.
#[derive(Debug, Clone, Builder)]
#[builder(setter(into), default)]
pub struct VllmSettings {
    /// The name of the provider.
    pub provider_name: String,

    /// The API base URL.
    pub base_url: String,

    /// The API key for authentication. Optional for local deployments.
    pub api_key: String,

    /// Custom API path override. When set, this path is used instead of
    /// the default "chat/completions".
    pub path: Option<String>,

    /// Default `chat_template_kwargs` to send with every request.
    /// These are merged with reasoning-specific kwargs when reasoning is enabled.
    pub chat_template_kwargs: Option<serde_json::Value>,

    /// Whether to include reasoning output in responses.
    /// When `None`, defaults to `true` when reasoning effort is set.
    pub include_reasoning: Option<bool>,
}

impl Default for VllmSettings {
    fn default() -> Self {
        Self {
            provider_name: "vLLM".to_string(),
            base_url: "http://localhost:8000/v1".to_string(),
            api_key: std::env::var("VLLM_API_KEY").unwrap_or_default(),
            path: None,
            chat_template_kwargs: None,
            include_reasoning: None,
        }
    }
}
