use crate::core::Message;
use crate::core::language_model::{
    LanguageModelOptions, LanguageModelResponseContentType, ReasoningEffort, Usage,
};
use crate::providers::anthropic::client::{
    AnthropicAssistantMessageParamContent, AnthropicMessageDeltaUsage, AnthropicMessageParam,
    AnthropicOptions, AnthropicThinking, AnthropicTool, AnthropicUsage,
};
use crate::providers::anthropic::extensions;

impl From<LanguageModelOptions> for AnthropicOptions {
    fn from(options: LanguageModelOptions) -> Self {
        let mut messages = Vec::new();
        let mut request = AnthropicOptions::builder();
        request.model("");

        // Set max_tokens from the core options, defaulting to 64K.
        request.max_tokens(options.max_output_tokens.unwrap_or(64_000));

        if let Some(system) = options.system
            && !system.is_empty()
        {
            request.system(Some(system));
        } else {
            request.system(None);
        }

        // convert messages to anthropic messages
        for msg in options.messages {
            match msg.message {
                Message::System(s) => {
                    if !s.content.is_empty() {
                        request.system(Some(s.content));
                    }
                }
                Message::User(u) => {
                    messages.push(AnthropicMessageParam::User {
                        content:
                            crate::providers::anthropic::client::AnthropicUserMessageContent::Text(
                                u.content,
                            ),
                    });
                }
                Message::Assistant(a) => {
                    let block = match a.content {
                        LanguageModelResponseContentType::Text(text) => {
                            AnthropicAssistantMessageParamContent::Text { text }
                        }
                        LanguageModelResponseContentType::ToolCall(tool) => {
                            AnthropicAssistantMessageParamContent::ToolUse {
                                id: tool.tool.id,
                                input: tool.input,
                                name: tool.tool.name,
                            }
                        }
                        LanguageModelResponseContentType::Reasoning {
                            content,
                            extensions,
                        } => {
                            let signature = extensions
                                .get::<extensions::AnthropicThinkingMetadata>()
                                .signature
                                .clone()
                                .unwrap_or_else(|| content.clone());
                            AnthropicAssistantMessageParamContent::Thinking {
                                thinking: content.clone(),
                                signature,
                            }
                        }
                        LanguageModelResponseContentType::NotSupported(_) => continue,
                    };
                    // Merge consecutive assistant messages into a single message
                    // with multiple content blocks. This prevents 400 errors from
                    // Anthropic which requires strict user/assistant alternation.
                    if let Some(AnthropicMessageParam::Assistant { content }) = messages.last_mut()
                    {
                        content.push(block);
                    } else {
                        messages.push(AnthropicMessageParam::Assistant {
                            content: vec![block],
                        });
                    }
                }
                Message::Tool(tool) => {
                    messages.push(AnthropicMessageParam::User {
                        content: crate::providers::anthropic::client::AnthropicUserMessageContent::Blocks(vec![
                            crate::providers::anthropic::client::AnthropicUserMessageContentBlock::ToolResult {
                                tool_use_id: tool.tool.id,
                                content: tool.output.unwrap_or_default().to_string(),
                            },
                        ]),
                    });
                }
                Message::Developer(dev) => {
                    messages.push(AnthropicMessageParam::User {
                        content:
                            crate::providers::anthropic::client::AnthropicUserMessageContent::Text(
                                format!("<developer>\n{dev}\n</developer>"),
                            ),
                    });
                }
            }
        }
        // update messages
        request.messages(messages);

        // convert tools to anthropic tools
        if let Some(tools) = options.tools {
            request.tools(Some(
                tools
                    .tools
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .iter()
                    .map(|t| {
                        let tool = t.clone();
                        let mut tool_schema = tool.input_schema.to_value();
                        if let Some(schema) = tool_schema.as_object_mut() {
                            schema.remove("$schema");
                        };
                        AnthropicTool {
                            name: tool.name,
                            description: tool.description,
                            input_schema: tool_schema,
                        }
                    })
                    .collect(),
            ));
        }

        // convert reasoning to antropic thinking
        request.thinking(options.reasoning_effort.map(|effort| match effort {
            // Instant disables thinking entirely
            ReasoningEffort::Instant => AnthropicThinking::Disable,
            // Low: 8K budget tokens
            ReasoningEffort::Low => AnthropicThinking::Enable {
                budget_tokens: 8_000,
            },
            // Medium: 16K budget tokens
            ReasoningEffort::Medium => AnthropicThinking::Enable {
                budget_tokens: 16_000,
            },
            // High: 32K budget tokens
            ReasoningEffort::High => AnthropicThinking::Enable {
                budget_tokens: 32_000,
            },
        }));

        request.build().expect("Failed to build AntropicRequest")
    }
}

impl From<AnthropicUsage> for Usage {
    fn from(usage: AnthropicUsage) -> Self {
        Self {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            cached_tokens: Some(usage.cache_creation_input_tokens + usage.cache_read_input_tokens),
            reasoning_tokens: None,
        }
    }
}

impl From<AnthropicMessageDeltaUsage> for Usage {
    fn from(usage: AnthropicMessageDeltaUsage) -> Self {
        Self {
            input_tokens: Some(usage.input_tokens.unwrap_or(0)),
            output_tokens: Some(usage.output_tokens),
            cached_tokens: Some(
                usage.cache_creation_input_tokens.unwrap_or(0)
                    + usage.cache_read_input_tokens.unwrap_or(0),
            ),
            reasoning_tokens: None,
        }
    }
}
