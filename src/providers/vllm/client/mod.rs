//! Client implementation for the vLLM provider.
//!
//! Implements `LanguageModelClient` for `Vllm<M>`, handling request construction,
//! authentication, and SSE stream parsing for vLLM's chat completions API.

pub(crate) mod types;

pub(crate) use types::VllmChatCompletionsOptions;

use crate::core::capabilities::ModelName;
use crate::core::client::LanguageModelClient;
use crate::error::Error;
use crate::providers::vllm::Vllm;
use reqwest::header::CONTENT_TYPE;
use reqwest_eventsource::Event;
use types::*;

impl<M: ModelName> LanguageModelClient for Vllm<M> {
    type Response = VllmChatCompletionsResponse;
    type StreamEvent = VllmStreamEvent;

    fn path(&self) -> String {
        self.settings
            .path
            .clone()
            .unwrap_or_else(|| "chat/completions".to_string())
    }

    fn method(&self) -> reqwest::Method {
        reqwest::Method::POST
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());

        // Only add auth header if api_key is non-empty.
        // vLLM is commonly deployed without authentication.
        if !self.settings.api_key.is_empty() {
            headers.insert(
                "Authorization",
                format!("Bearer {}", self.settings.api_key).parse().unwrap(),
            );
        }

        headers
    }

    fn query_params(&self) -> Vec<(&str, &str)> {
        Vec::new()
    }

    fn body(&self) -> reqwest::Body {
        let body = serde_json::to_string(&self.options).unwrap();
        reqwest::Body::from(body)
    }

    fn parse_stream_sse(
        event: std::result::Result<Event, reqwest_eventsource::Error>,
    ) -> crate::error::Result<Self::StreamEvent> {
        match event {
            Ok(event) => match event {
                Event::Open => Ok(VllmStreamEvent::Open),
                Event::Message(msg) => {
                    if msg.data.trim() == "[DONE]" || msg.data.is_empty() {
                        return Ok(VllmStreamEvent::Done);
                    }

                    let chunk: VllmChatCompletionsStreamChunk = serde_json::from_str(&msg.data)
                        .map_err(|e| Error::ApiError {
                            status_code: None,
                            details: format!("Invalid JSON in SSE: {e}"),
                        })?;

                    Ok(VllmStreamEvent::Chunk(chunk))
                }
            },
            Err(e) => {
                let status_code = match &e {
                    reqwest_eventsource::Error::InvalidStatusCode(status, _) => Some(*status),
                    _ => None,
                };

                Err(Error::ApiError {
                    status_code,
                    details: e.to_string(),
                })
            }
        }
    }

    fn end_stream(event: &Self::StreamEvent) -> bool {
        matches!(event, VllmStreamEvent::Done)
    }
}
