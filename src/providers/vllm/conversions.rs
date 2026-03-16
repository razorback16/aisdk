//! Helper functions and conversions for the vLLM provider.
//!
//! Reuses the existing `Message -> ChatMessage` and `SdkTool -> Tool` conversions
//! from the OpenAI Chat Completions provider, and adds vLLM-specific reasoning
//! effort mapping via `chat_template_kwargs`.

use crate::core::language_model::{LanguageModelOptions, ReasoningEffort};
use crate::providers::openai_chat_completions::client::types as oai_types;
use crate::providers::vllm::client::types as vllm_types;
use crate::providers::vllm::settings::VllmSettings;

// ============================================================================
// LanguageModelOptions -> VllmChatCompletionsOptions
// ============================================================================

/// Converts SDK language model options into vLLM-specific chat completions options.
///
/// This is a function rather than a `From` impl because it requires access to
/// `VllmSettings` for merging default `chat_template_kwargs` with reasoning kwargs.
pub(crate) fn convert_options(
    options: LanguageModelOptions,
    settings: &VllmSettings,
) -> vllm_types::VllmChatCompletionsOptions {
    // 1. Convert system prompt + messages to Vec<ChatMessage> (reuse oai conversion)
    let mut messages: Vec<oai_types::ChatMessage> = Vec::new();

    if let Some(system_prompt) = options.system {
        messages.push(oai_types::ChatMessage {
            role: oai_types::Role::System,
            content: Some(system_prompt),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    messages.extend(
        options
            .messages
            .into_iter()
            .map(|tagged| tagged.message.into()),
    );

    // 2. Convert tools via existing SdkTool -> oai_types::Tool conversion
    let tools: Option<Vec<oai_types::Tool>> = options.tools.map(|tool_list| {
        tool_list
            .tools
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|t| t.clone().into())
            .collect()
    });

    // 3. Convert schema to response_format (same as openai_chat_completions)
    let response_format = options.schema.map(|schema| {
        let mut json_value = serde_json::to_value(schema).unwrap();

        if let serde_json::Value::Object(ref mut obj) = json_value {
            obj.insert(
                "additionalProperties".to_string(),
                serde_json::Value::Bool(false),
            );
        }

        oai_types::ResponseFormat::JsonSchema {
            json_schema: oai_types::JsonSchemaDefinition {
                name: json_value
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Response")
                    .to_string(),
                schema: json_value.clone(),
                description: json_value
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                strict: Some(true),
            },
        }
    });

    // 4. Reasoning effort mapping — the key vLLM difference
    let (chat_template_kwargs, include_reasoning) =
        build_reasoning_kwargs(options.reasoning_effort.as_ref(), settings);

    // 5. tool_choice: "auto" if tools present
    let tool_choice = if tools.is_some() {
        Some(oai_types::ToolChoice::String("auto".to_string()))
    } else {
        None
    };

    // 6. parallel_tool_calls: true if tools present
    let parallel_tool_calls = if tools.is_some() { Some(true) } else { None };

    // 7. Temperature/top_p scaled from 0-100 to 0.0-1.0
    // 8. stop sequences (same logic as openai)
    vllm_types::VllmChatCompletionsOptions {
        model: String::new(),
        messages,
        frequency_penalty: options.frequency_penalty,
        max_completion_tokens: options.max_output_tokens,
        presence_penalty: options.presence_penalty,
        response_format,
        seed: options.seed,
        stop: options.stop_sequences.map(|seqs| {
            if seqs.len() == 1 {
                oai_types::StopSequences::Single(seqs[0].clone())
            } else {
                oai_types::StopSequences::Multiple(seqs.into_iter().take(4).collect())
            }
        }),
        stream: None,
        stream_options: None,
        temperature: options.temperature.map(|t| t as f32 / 100.0),
        top_p: options.top_p.map(|t| t as f32 / 100.0),
        tools,
        tool_choice,
        parallel_tool_calls,
        n: None,
        chat_template_kwargs,
        include_reasoning,
        top_k: None,
        min_p: None,
        repetition_penalty: None,
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
    use crate::core::language_model::LanguageModelOptions;
    use serde_json::json;

    fn default_settings() -> VllmSettings {
        VllmSettings::default()
    }

    #[test]
    fn test_reasoning_effort_none_disables_reasoning() {
        let options = LanguageModelOptions {
            reasoning_effort: Some(ReasoningEffort::None),
            ..Default::default()
        };

        let result = convert_options(options, &default_settings());
        assert_eq!(result.include_reasoning, Some(false));
        let kwargs = result.chat_template_kwargs.unwrap();
        assert_eq!(kwargs["enable_thinking"], json!(false));
    }

    #[test]
    fn test_reasoning_effort_high_enables_thinking() {
        let options = LanguageModelOptions {
            reasoning_effort: Some(ReasoningEffort::High),
            ..Default::default()
        };

        let result = convert_options(options, &default_settings());
        assert_eq!(result.include_reasoning, Some(true));

        let kwargs = result.chat_template_kwargs.unwrap();
        assert_eq!(kwargs["enable_thinking"], json!(true));
        assert_eq!(kwargs["reasoning_effort"], json!("high"));
    }

    #[test]
    fn test_reasoning_effort_xhigh_maps_to_high() {
        let options = LanguageModelOptions {
            reasoning_effort: Some(ReasoningEffort::XHigh),
            ..Default::default()
        };

        let result = convert_options(options, &default_settings());
        assert_eq!(result.include_reasoning, Some(true));

        let kwargs = result.chat_template_kwargs.unwrap();
        assert_eq!(kwargs["enable_thinking"], json!(true));
        assert_eq!(kwargs["reasoning_effort"], json!("high"));
    }

    #[test]
    fn test_reasoning_effort_low() {
        let options = LanguageModelOptions {
            reasoning_effort: Some(ReasoningEffort::Low),
            ..Default::default()
        };

        let result = convert_options(options, &default_settings());
        assert_eq!(result.include_reasoning, Some(true));

        let kwargs = result.chat_template_kwargs.unwrap();
        assert_eq!(kwargs["enable_thinking"], json!(true));
        assert_eq!(kwargs["reasoning_effort"], json!("low"));
    }

    #[test]
    fn test_settings_kwargs_merged_with_reasoning() {
        let options = LanguageModelOptions {
            reasoning_effort: Some(ReasoningEffort::Medium),
            ..Default::default()
        };

        let settings = VllmSettings {
            chat_template_kwargs: Some(json!({
                "custom_key": "custom_value",
                "enable_thinking": false
            })),
            ..Default::default()
        };

        let result = convert_options(options, &settings);
        let kwargs = result.chat_template_kwargs.unwrap();

        // Reasoning kwargs take precedence
        assert_eq!(kwargs["enable_thinking"], json!(true));
        assert_eq!(kwargs["reasoning_effort"], json!("medium"));
        // Settings defaults are preserved
        assert_eq!(kwargs["custom_key"], json!("custom_value"));
    }

    #[test]
    fn test_no_reasoning_uses_settings_defaults() {
        let options = LanguageModelOptions::default();

        let settings = VllmSettings {
            chat_template_kwargs: Some(json!({"enable_thinking": true})),
            include_reasoning: Some(true),
            ..Default::default()
        };

        let result = convert_options(options, &settings);
        assert_eq!(result.include_reasoning, Some(true));
        assert_eq!(
            result.chat_template_kwargs,
            Some(json!({"enable_thinking": true}))
        );
    }

    #[test]
    fn test_temperature_and_top_p_scaled() {
        let options = LanguageModelOptions {
            temperature: Some(70),
            top_p: Some(90),
            ..Default::default()
        };

        let result = convert_options(options, &default_settings());
        assert!((result.temperature.unwrap() - 0.7).abs() < f32::EPSILON);
        assert!((result.top_p.unwrap() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stop_sequences_single() {
        let options = LanguageModelOptions {
            stop_sequences: Some(vec!["STOP".to_string()]),
            ..Default::default()
        };

        let result = convert_options(options, &default_settings());
        assert!(matches!(
            result.stop,
            Some(oai_types::StopSequences::Single(_))
        ));
    }

    #[test]
    fn test_stop_sequences_multiple_truncated() {
        let options = LanguageModelOptions {
            stop_sequences: Some(vec![
                "S1".to_string(),
                "S2".to_string(),
                "S3".to_string(),
                "S4".to_string(),
                "S5".to_string(),
            ]),
            ..Default::default()
        };

        let result = convert_options(options, &default_settings());
        if let Some(oai_types::StopSequences::Multiple(seqs)) = result.stop {
            assert_eq!(seqs.len(), 4);
        } else {
            panic!("expected multiple stop sequences");
        }
    }
}
