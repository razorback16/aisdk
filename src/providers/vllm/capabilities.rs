//! Capabilities for vLLM models.
//!
//! vLLM serves user-deployed models, so no fixed model types are defined.
//! Users should use `DynamicModel` exclusively.

use crate::core::capabilities::*;
use crate::model_capabilities;
use crate::providers::vllm::Vllm;

model_capabilities! {
    provider: Vllm,
    models: {}
}
