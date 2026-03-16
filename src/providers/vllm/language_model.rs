//! Language model implementation for the vLLM provider.
//!
//! Similar to the OpenAI Chat Completions implementation but reads `reasoning`
//! instead of `reasoning_content` from responses and stream deltas, and uses
//! vLLM-specific request fields via `chat_template_kwargs`.

use crate::core::capabilities::ModelName;
use crate::core::client::LanguageModelClient;
use crate::core::language_model::{
    LanguageModel, LanguageModelOptions, LanguageModelResponse, LanguageModelResponseContentType,
    LanguageModelStreamChunk, LanguageModelStreamChunkType, ProviderStream, ReasoningEffort,
};
use crate::core::messages::AssistantMessage;
use crate::core::tools::ToolCallInfo;
use crate::error::Result;
use crate::providers::openai_chat_completions::client::ChatCompletionsOptions;
use crate::providers::vllm::Vllm;
use crate::providers::vllm::client::types;
use crate::providers::vllm::settings::VllmSettings;
use async_trait::async_trait;
use futures::StreamExt;

#[async_trait]
impl<M: ModelName> LanguageModel for Vllm<M> {
    fn name(&self) -> String {
        self.inner.options.model.clone()
    }

    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        let additional_headers = options.headers.clone();

        // Build vLLM-specific reasoning kwargs before converting options
        let (chat_template_kwargs, include_reasoning) =
            build_reasoning_kwargs(options.reasoning_effort.as_ref(), &self.settings);
        self.vllm_chat_template_kwargs = chat_template_kwargs;
        self.vllm_include_reasoning = include_reasoning;

        // Use standard From conversion
        let mut opts: ChatCompletionsOptions = options.into();
        opts.model = self.inner.options.model.clone();
        self.inner.options = opts;

        let response: types::VllmChatCompletionsResponse = self
            .send(&self.settings.base_url, additional_headers)
            .await?;

        let mut contents = Vec::new();

        for choice in response.choices {
            // Handle reasoning content (vLLM uses "reasoning" field)
            if let Some(reasoning) = choice.message.reasoning
                && !reasoning.is_empty()
            {
                contents.push(LanguageModelResponseContentType::Reasoning {
                    content: reasoning,
                    extensions: Default::default(),
                });
            }

            // Handle text content
            if let Some(text) = choice.message.content
                && !text.is_empty()
            {
                contents.push(LanguageModelResponseContentType::Text(text));
            }

            // Handle tool calls
            if let Some(tool_calls) = choice.message.tool_calls {
                for tool_call in tool_calls {
                    let mut tool_info = ToolCallInfo::new(tool_call.function.name);
                    tool_info.id(tool_call.id);
                    tool_info.input(
                        serde_json::from_str(&tool_call.function.arguments)
                            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
                    );
                    contents.push(LanguageModelResponseContentType::ToolCall(tool_info));
                }
            }
        }

        Ok(LanguageModelResponse {
            contents,
            usage: response.usage.map(|u| u.into()),
        })
    }

    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream> {
        let additional_headers = options.headers.clone();

        // Build vLLM-specific reasoning kwargs before converting options
        let (chat_template_kwargs, include_reasoning) =
            build_reasoning_kwargs(options.reasoning_effort.as_ref(), &self.settings);
        self.vllm_chat_template_kwargs = chat_template_kwargs;
        self.vllm_include_reasoning = include_reasoning;

        // Use standard From conversion
        let mut opts: ChatCompletionsOptions = options.into();
        opts.model = self.inner.options.model.clone();
        opts.stream = Some(true);
        self.inner.options = opts;

        let stream = self
            .send_and_stream(&self.settings.base_url, additional_headers)
            .await?;

        // State for accumulating tool calls across chunks
        use std::collections::HashMap;
        let mut accumulated_tool_calls: HashMap<u32, (String, String, String)> = HashMap::new();

        // Map stream events to SDK stream chunks
        let stream = stream.map(move |evt_res| match evt_res {
            Ok(types::VllmStreamEvent::Chunk(chunk)) => {
                let mut results = Vec::new();

                for choice in chunk.choices {
                    // Reasoning delta (vLLM uses "reasoning" field)
                    if let Some(reasoning) = choice.delta.reasoning
                        && !reasoning.is_empty()
                    {
                        results.push(LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::Reasoning(reasoning),
                        ));
                    }

                    // Text delta
                    if let Some(content) = choice.delta.content
                        && !content.is_empty()
                    {
                        results.push(LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::Text(content),
                        ));
                    }

                    // Accumulate tool call deltas
                    if let Some(tool_calls) = choice.delta.tool_calls {
                        for tool_call in tool_calls {
                            let tool_call_index = tool_call.index;
                            let entry = accumulated_tool_calls.entry(tool_call_index).or_insert((
                                String::new(),
                                String::new(),
                                String::new(),
                            ));

                            // Accumulate ID
                            if let Some(id) = tool_call.id {
                                entry.0 = id;
                            }

                            // Accumulate name and arguments
                            if let Some(function) = tool_call.function {
                                if let Some(name) = function.name {
                                    entry.1 = name;
                                }
                                if let Some(args) = function.arguments {
                                    entry.2.push_str(&args);
                                    let tool_call_id = if entry.0.is_empty() {
                                        format!("vllm-tool-call-{tool_call_index}")
                                    } else {
                                        entry.0.clone()
                                    };
                                    let tool_name = if entry.1.is_empty() {
                                        "unknown".to_string()
                                    } else {
                                        entry.1.clone()
                                    };
                                    results.push(LanguageModelStreamChunk::Delta(
                                        LanguageModelStreamChunkType::ToolCallDelta {
                                            tool_call_id,
                                            tool_name,
                                            delta: args,
                                        },
                                    ));
                                }
                            }
                        }
                    }

                    if let Some(finish_reason) = choice.finish_reason {
                        let usage = chunk.usage.clone().map(|u| u.into());

                        match finish_reason.as_str() {
                            "stop" | "length" => {
                                results.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::Text(String::new()),
                                    usage,
                                }));
                            }
                            "tool_calls" | "function_call" => {
                                // Send accumulated tool calls
                                for (index, (id, name, args)) in &accumulated_tool_calls {
                                    let resolved_id = if id.is_empty() {
                                        format!("vllm-tool-call-{index}")
                                    } else {
                                        id.clone()
                                    };
                                    let resolved_name = if name.is_empty() {
                                        "unknown".to_string()
                                    } else {
                                        name.clone()
                                    };

                                    let mut tool_info = ToolCallInfo::new(resolved_name);
                                    tool_info.id(resolved_id);
                                    tool_info.input(serde_json::from_str(args).unwrap_or_else(
                                        |_| serde_json::Value::Object(serde_json::Map::new()),
                                    ));
                                    results.push(LanguageModelStreamChunk::Done(
                                        AssistantMessage {
                                            content: LanguageModelResponseContentType::ToolCall(
                                                tool_info,
                                            ),
                                            usage: usage.clone(),
                                        },
                                    ));
                                }
                            }
                            "content_filter" => {
                                results.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::Text(String::new()),
                                    usage,
                                }));
                                results.push(LanguageModelStreamChunk::Delta(
                                    LanguageModelStreamChunkType::Failed(
                                        "Content filtered".to_string(),
                                    ),
                                ));
                            }
                            // For any unknown finish reason, treat as normal completion
                            _ => {
                                results.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::Text(String::new()),
                                    usage,
                                }));
                            }
                        }
                    }
                }

                Ok(results)
            }
            Ok(types::VllmStreamEvent::Open) => Ok(vec![]),
            Ok(types::VllmStreamEvent::Done) => Ok(vec![]),
            Ok(types::VllmStreamEvent::Error(e)) => Ok(vec![LanguageModelStreamChunk::Delta(
                LanguageModelStreamChunkType::Failed(e),
            )]),
            Err(e) => Err(e),
        });

        Ok(Box::pin(stream))
    }
}

// ============================================================================
// Reasoning effort -> chat_template_kwargs + include_reasoning
// ============================================================================

/// Builds vLLM-specific `chat_template_kwargs` and `include_reasoning` from
/// the SDK reasoning effort level and provider settings.
fn build_reasoning_kwargs(
    reasoning_effort: Option<&ReasoningEffort>,
    settings: &VllmSettings,
) -> (Option<serde_json::Value>, Option<bool>) {
    match reasoning_effort {
        Some(ReasoningEffort::None) => {
            // Explicitly disable reasoning
            let mut kwargs = settings
                .chat_template_kwargs
                .clone()
                .unwrap_or_else(|| serde_json::json!({}));
            // Ensure enable_thinking is explicitly false
            if let serde_json::Value::Object(ref mut map) = kwargs {
                map.insert(
                    "enable_thinking".to_string(),
                    serde_json::Value::Bool(false),
                );
            }
            (Some(kwargs), Some(false))
        }
        Some(effort) => {
            // Build reasoning kwargs
            let effort_str = match effort {
                ReasoningEffort::Low => "low",
                ReasoningEffort::Medium => "medium",
                ReasoningEffort::High => "high",
                ReasoningEffort::XHigh => "high", // vLLM doesn't support xhigh
                ReasoningEffort::None => unreachable!(),
            };

            let mut kwargs = serde_json::json!({
                "enable_thinking": true
            });

            // For Low/Medium/High/XHigh, also add reasoning_effort
            kwargs.as_object_mut().unwrap().insert(
                "reasoning_effort".to_string(),
                serde_json::json!(effort_str),
            );

            // Deep merge with settings defaults (reasoning kwargs take precedence)
            if let Some(default_kwargs) = &settings.chat_template_kwargs {
                let mut merged = default_kwargs.clone();
                merge_json(&mut merged, kwargs);
                (Some(merged), Some(true))
            } else {
                (Some(kwargs), Some(true))
            }
        }
        None => {
            // No reasoning effort set; use settings defaults if present
            let kwargs = settings.chat_template_kwargs.clone();
            let include_reasoning = settings.include_reasoning;
            (kwargs, include_reasoning)
        }
    }
}

/// Shallow-merges `overlay` into `base`. For object values, overlay keys
/// take precedence over base keys.
fn merge_json(base: &mut serde_json::Value, overlay: serde_json::Value) {
    if let (serde_json::Value::Object(base_map), serde_json::Value::Object(overlay_map)) =
        (base, overlay)
    {
        for (key, value) in overlay_map {
            base_map.insert(key, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn default_settings() -> VllmSettings {
        VllmSettings::default()
    }

    #[test]
    fn test_reasoning_effort_none_disables_reasoning() {
        let (kwargs, include_reasoning) =
            build_reasoning_kwargs(Some(&ReasoningEffort::None), &default_settings());
        assert_eq!(include_reasoning, Some(false));
        let kwargs = kwargs.unwrap();
        assert_eq!(kwargs["enable_thinking"], json!(false));
    }

    #[test]
    fn test_reasoning_effort_high_enables_thinking() {
        let (kwargs, include_reasoning) =
            build_reasoning_kwargs(Some(&ReasoningEffort::High), &default_settings());
        assert_eq!(include_reasoning, Some(true));

        let kwargs = kwargs.unwrap();
        assert_eq!(kwargs["enable_thinking"], json!(true));
        assert_eq!(kwargs["reasoning_effort"], json!("high"));
    }

    #[test]
    fn test_reasoning_effort_xhigh_maps_to_high() {
        let (kwargs, include_reasoning) =
            build_reasoning_kwargs(Some(&ReasoningEffort::XHigh), &default_settings());
        assert_eq!(include_reasoning, Some(true));

        let kwargs = kwargs.unwrap();
        assert_eq!(kwargs["enable_thinking"], json!(true));
        assert_eq!(kwargs["reasoning_effort"], json!("high"));
    }

    #[test]
    fn test_reasoning_effort_low() {
        let (kwargs, include_reasoning) =
            build_reasoning_kwargs(Some(&ReasoningEffort::Low), &default_settings());
        assert_eq!(include_reasoning, Some(true));

        let kwargs = kwargs.unwrap();
        assert_eq!(kwargs["enable_thinking"], json!(true));
        assert_eq!(kwargs["reasoning_effort"], json!("low"));
    }

    #[test]
    fn test_settings_kwargs_merged_with_reasoning() {
        let settings = VllmSettings {
            chat_template_kwargs: Some(json!({
                "custom_key": "custom_value",
                "enable_thinking": false
            })),
            ..Default::default()
        };

        let (kwargs, include_reasoning) =
            build_reasoning_kwargs(Some(&ReasoningEffort::Medium), &settings);
        assert_eq!(include_reasoning, Some(true));

        let kwargs = kwargs.unwrap();
        // Reasoning kwargs take precedence
        assert_eq!(kwargs["enable_thinking"], json!(true));
        assert_eq!(kwargs["reasoning_effort"], json!("medium"));
        // Settings defaults are preserved
        assert_eq!(kwargs["custom_key"], json!("custom_value"));
    }

    #[test]
    fn test_no_reasoning_uses_settings_defaults() {
        let settings = VllmSettings {
            chat_template_kwargs: Some(json!({"enable_thinking": true})),
            include_reasoning: Some(true),
            ..Default::default()
        };

        let (kwargs, include_reasoning) = build_reasoning_kwargs(None, &settings);
        assert_eq!(include_reasoning, Some(true));
        assert_eq!(kwargs, Some(json!({"enable_thinking": true})));
    }
}
