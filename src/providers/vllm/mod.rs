//! vLLM provider for the AI SDK.
//!
//! This module provides the vLLM provider, which supports reasoning via
//! `chat_template_kwargs` and uses the `reasoning` field in responses.

pub mod capabilities;
pub(crate) mod client;
pub(crate) mod conversions;
pub mod language_model;
pub mod settings;

use std::marker::PhantomData;

use crate::core::DynamicModel;
use crate::core::capabilities::ModelName;
use crate::core::utils::validate_base_url;
use crate::error::Result;
use client::VllmChatCompletionsOptions;
use settings::VllmSettings;

/// The vLLM provider.
#[derive(Debug, Clone)]
pub struct Vllm<M: ModelName> {
    /// Configuration settings for the vLLM provider.
    pub settings: VllmSettings,
    /// Request options for the API call.
    pub(crate) options: VllmChatCompletionsOptions,
    _phantom: PhantomData<M>,
}

impl<M: ModelName> Vllm<M> {
    /// Returns a builder for configuring the vLLM provider.
    pub fn builder() -> VllmBuilder<M> {
        VllmBuilder::default()
    }
}

impl<M: ModelName> Default for Vllm<M> {
    fn default() -> Self {
        let settings = VllmSettings::default();
        let options = VllmChatCompletionsOptions {
            model: M::MODEL_NAME.to_string(),
            ..Default::default()
        };

        Self {
            settings,
            options,
            _phantom: PhantomData,
        }
    }
}

impl Vllm<DynamicModel> {
    /// Creates a vLLM provider with a dynamic model name using default settings.
    ///
    /// This allows you to specify the model name as a string rather than
    /// using a statically-typed model.
    ///
    /// **WARNING**: when using `DynamicModel`, model capabilities are not validated.
    /// This means there is no compile-time guarantee that the model supports requested features.
    ///
    /// For custom configuration (base URL, API key, etc.), use the builder pattern:
    /// `Vllm::<DynamicModel>::builder().model_name(...).base_url(...).build()`
    ///
    /// # Parameters
    ///
    /// * `name` - The model identifier (e.g., "Qwen/Qwen3-32B")
    ///
    /// # Returns
    ///
    /// A configured `Vllm<DynamicModel>` provider instance with default settings.
    pub fn model_name(name: impl Into<String>) -> Self {
        let settings = VllmSettings::default();
        let options = VllmChatCompletionsOptions {
            model: name.into(),
            ..Default::default()
        };

        Self {
            settings,
            options,
            _phantom: PhantomData,
        }
    }
}

/// Builder for the vLLM provider.
pub struct VllmBuilder<M: ModelName> {
    settings: VllmSettings,
    options: VllmChatCompletionsOptions,
    _phantom: PhantomData<M>,
}

impl<M: ModelName> Default for VllmBuilder<M> {
    fn default() -> Self {
        let settings = VllmSettings::default();
        let options = VllmChatCompletionsOptions {
            model: M::MODEL_NAME.to_string(),
            ..Default::default()
        };

        Self {
            settings,
            options,
            _phantom: PhantomData,
        }
    }
}

impl VllmBuilder<DynamicModel> {
    /// Sets the model name from a string.
    ///
    /// **WARNING**: when using `DynamicModel`, model capabilities are not validated.
    /// This means there is no compile-time guarantee that the model supports requested features.
    ///
    /// # Parameters
    ///
    /// * `model_name` - The model identifier (e.g., "Qwen/Qwen3-32B")
    ///
    /// # Returns
    ///
    /// The builder with the model name set.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.options.model = model_name.into();
        self
    }
}

impl<M: ModelName> VllmBuilder<M> {
    /// Sets the provider name. Defaults to "vLLM".
    pub fn provider_name(mut self, provider_name: impl Into<String>) -> Self {
        self.settings.provider_name = provider_name.into();
        self
    }

    /// Sets the base URL for the vLLM API.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.settings.base_url = base_url.into();
        self
    }

    /// Sets the API key for the vLLM API.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.settings.api_key = api_key.into();
        self
    }

    /// Sets a custom API path, overriding the default.
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.settings.path = Some(path.into());
        self
    }

    /// Sets the default `chat_template_kwargs` on the provider settings.
    ///
    /// These kwargs are passed to vLLM's chat template engine and can be used
    /// to enable features like reasoning (e.g., `{"enable_thinking": true}`).
    pub fn chat_template_kwargs(mut self, kwargs: serde_json::Value) -> Self {
        self.settings.chat_template_kwargs = Some(kwargs);
        self
    }

    /// Sets whether to include reasoning output in responses.
    ///
    /// When enabled, the provider will request and parse the `reasoning` field
    /// from vLLM responses.
    pub fn include_reasoning(mut self, include: bool) -> Self {
        self.settings.include_reasoning = Some(include);
        self
    }

    /// Builds the vLLM provider.
    ///
    /// Validates the base URL. Unlike other providers, an API key is not required
    /// since vLLM is commonly deployed without authentication.
    ///
    /// # Returns
    ///
    /// A `Result` containing the configured `Vllm` provider or an `Error`.
    pub fn build(self) -> Result<Vllm<M>> {
        let base_url = validate_base_url(&self.settings.base_url)?;

        Ok(Vllm {
            settings: VllmSettings {
                base_url,
                ..self.settings
            },
            options: self.options,
            _phantom: PhantomData,
        })
    }
}

// Re-exports for convenience (empty since vLLM uses DynamicModel exclusively)
#[allow(unused_imports)]
pub use capabilities::*;
