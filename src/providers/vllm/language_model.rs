//! Language model implementation for the vLLM provider.
//!
//! Similar to the OpenAI Chat Completions implementation but reads `reasoning`
//! instead of `reasoning_content` from responses and stream deltas, and uses
//! vLLM-specific request types with `chat_template_kwargs`.

use crate::core::capabilities::ModelName;
use crate::core::client::LanguageModelClient;
use crate::core::language_model::{
    LanguageModel, LanguageModelOptions, LanguageModelResponse, LanguageModelResponseContentType,
    LanguageModelStreamChunk, LanguageModelStreamChunkType, ProviderStream,
};
use crate::core::messages::AssistantMessage;
use crate::core::tools::ToolCallInfo;
use crate::error::Result;
use crate::providers::vllm::Vllm;
use crate::providers::vllm::client::types;
use crate::providers::vllm::conversions;
use async_trait::async_trait;
use futures::StreamExt;

#[async_trait]
impl<M: ModelName> LanguageModel for Vllm<M> {
    fn name(&self) -> String {
        self.options.model.clone()
    }

    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        let additional_headers = options.headers.clone();
        let mut opts = conversions::convert_options(options, &self.settings);
        opts.model = self.options.model.clone();
        self.options = opts;

        let response: types::VllmChatCompletionsResponse = self
            .send(&self.settings.base_url, additional_headers)
            .await?;

        let mut contents = Vec::new();

        for choice in response.choices {
            // Handle reasoning content (vLLM uses "reasoning" field)
            if let Some(reasoning) = choice.message.reasoning {
                if !reasoning.is_empty() {
                    contents.push(LanguageModelResponseContentType::Reasoning {
                        content: reasoning,
                        extensions: Default::default(),
                    });
                }
            }

            // Handle text content
            if let Some(text) = choice.message.content {
                if !text.is_empty() {
                    contents.push(LanguageModelResponseContentType::Text(text));
                }
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
        let mut opts = conversions::convert_options(options, &self.settings);
        opts.model = self.options.model.clone();
        opts.stream = Some(true);
        self.options = opts;

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
