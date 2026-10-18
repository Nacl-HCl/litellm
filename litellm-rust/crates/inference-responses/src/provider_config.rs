use litellm_core_utils::get_llm_provider_logic::{
    CustomLlmProvider, LlmProviders, get_custom_llm_provider,
};
use litellm_host::interceptors::ProviderIdentity;
use litellm_llms::{
    base_llm::responses::transformation::BaseResponsesApiConfig,
    openai::responses::transformation::OpenAiResponsesApiConfig,
    openrouter::responses::transformation::OPENROUTER_RESPONSES_CONFIG,
};

use crate::Error;

pub(crate) struct ResponsesProvider {
    pub identity: ProviderIdentity,
    pub config: &'static dyn BaseResponsesApiConfig,
}

/// Python's `get_llm_provider`: a bare model is OpenAI's, and with an explicit OpenRouter
/// provider a model without a vendor segment keeps the `openrouter/` prefix as its wire id.
pub(crate) fn resolve_provider_config(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<ResponsesProvider, Error> {
    let explicit = custom_llm_provider.filter(|provider| !provider.is_empty());
    let (provider, resolved) = match get_custom_llm_provider(model, explicit) {
        Some(CustomLlmProvider {
            model,
            custom_llm_provider,
        }) => (custom_llm_provider, model),
        None => ("openai", model),
    };
    if resolved.is_empty() {
        return Err(Error::InvalidProvider(model.into()));
    }
    let (config, wire_model): (&'static dyn BaseResponsesApiConfig, String) =
        match provider.parse::<LlmProviders>() {
            Ok(LlmProviders::Openai) if resolved.contains('/') => {
                return Err(Error::InvalidProvider(model.into()));
            }
            Ok(LlmProviders::Openai) => (&OpenAiResponsesApiConfig, resolved.into()),
            Ok(LlmProviders::Openrouter) if explicit.is_some() && !resolved.contains('/') => (
                &OPENROUTER_RESPONSES_CONFIG,
                format!("openrouter/{resolved}"),
            ),
            Ok(LlmProviders::Openrouter) => (&OPENROUTER_RESPONSES_CONFIG, resolved.into()),
            _ => return Err(Error::Unsupported("native HTTP responses provider")),
        };
    Ok(ResponsesProvider {
        identity: ProviderIdentity {
            model: wire_model,
            provider: provider.into(),
        },
        config,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Pinned against Python's `get_llm_provider` on 2026-10-11.
    #[rstest]
    #[case::bare("gpt-5", None, "openai", "gpt-5")]
    #[case::openai_prefix("openai/gpt-5", None, "openai", "gpt-5")]
    #[case::openai_explicit("gpt-5", Some("openai"), "openai", "gpt-5")]
    #[case::openrouter_prefix(
        "openrouter/anthropic/claude",
        None,
        "openrouter",
        "anthropic/claude"
    )]
    #[case::openrouter_explicit(
        "openrouter/anthropic/claude",
        Some("openrouter"),
        "openrouter",
        "anthropic/claude"
    )]
    #[case::openrouter_explicit_unprefixed(
        "anthropic/claude",
        Some("openrouter"),
        "openrouter",
        "anthropic/claude"
    )]
    #[case::openrouter_native_id(
        "openrouter/auto",
        Some("openrouter"),
        "openrouter",
        "openrouter/auto"
    )]
    #[case::openrouter_native_id_implicit("openrouter/auto", None, "openrouter", "auto")]
    fn providers_and_wire_models_match_python(
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] provider: &str,
        #[case] wire_model: &str,
    ) {
        let resolved = resolve_provider_config(model, custom_llm_provider).unwrap();
        assert_eq!(resolved.identity.provider, provider);
        assert_eq!(resolved.identity.model, wire_model);
    }

    #[rstest]
    #[case::other_prefix("other/model", None)]
    #[case::other_explicit("model", Some("other"))]
    #[case::openai_with_vendor_segment("openai/other/model", None)]
    #[case::empty_model("openai/", None)]
    #[case::conflicting_prefix("openrouter/anthropic/claude", Some("openai"))]
    fn everything_else_is_rejected(#[case] model: &str, #[case] custom_llm_provider: Option<&str>) {
        assert!(matches!(
            resolve_provider_config(model, custom_llm_provider),
            Err(Error::InvalidProvider(_) | Error::Unsupported(_))
        ));
    }
}
