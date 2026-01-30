"""Backend for OpenRouter API.

Supports:
- Standard chat completions
- Reasoning-enabled models (DeepSeek V3.2 Speciale, etc.)
- Provider routing preferences
"""

import logging
import os
import re
import time
from typing import Optional

from funcy import notnone, once, select_values
import openai

from .utils import FunctionSpec, OutputType, backoff_create

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

# Models that support reasoning/thinking tokens
# Note: deepseek-v3.2-speciale uses excessive reasoning tokens - prefer deepseek-v3.2 or deepseek-r1
REASONING_MODELS = [
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1-distill-llama-70b",
    "qwen/qwq-32b",
    "qwen/qwq-32b-preview",
]

# Recommended models for code generation (non-reasoning, more efficient)
RECOMMENDED_CODE_MODELS = [
    "deepseek/deepseek-v3.2",      # Best for code, fast
    "deepseek/deepseek-chat",       # Good alternative
    "deepseek/deepseek-coder",      # Specialized for code
]

# Models that support system messages properly
SYSTEM_MESSAGE_MODELS = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-v3",
    "deepseek/deepseek-v3.2-speciale",
    "anthropic/claude",
    "openai/gpt",
    "google/gemini",
]


def _supports_system_message(model: str) -> bool:
    """Check if model supports system messages."""
    model_lower = model.lower()
    for pattern in SYSTEM_MESSAGE_MODELS:
        if pattern in model_lower:
            return True
    return False


def _is_reasoning_model(model: str) -> bool:
    """Check if model supports reasoning tokens."""
    model_lower = model.lower()
    for pattern in REASONING_MODELS:
        if pattern in model_lower:
            return True
    # Also check for common reasoning model patterns
    if "deepseek-r1" in model_lower or "qwq" in model_lower:
        return True
    return False


@once
def _setup_openrouter_client():
    global _client
    _client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=0,
    )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    reasoning_effort: Optional[str] = None,
    enable_reasoning: Optional[bool] = None,
    provider: Optional[str] = None,
    provider_order: Optional[list] = None,
    provider_ignore: Optional[list] = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query OpenRouter API.

    Args:
        system_message: System prompt
        user_message: User message
        func_spec: Function specification (not supported yet)
        reasoning_effort: Reasoning effort level ("low", "medium", "high")
                         - For compatible models, enables reasoning tokens
        enable_reasoning: Explicitly enable/disable reasoning (overrides auto-detection)
        provider: Specific provider slug (e.g., "siliconflow/fp8", "deepinfra", "fireworks")
                  See https://openrouter.ai/docs/providers for available providers
        provider_order: List of preferred providers in order (e.g., ["siliconflow", "deepinfra"])
        provider_ignore: List of providers to ignore (e.g., ["together", "hyperbolic"])
        **model_kwargs: Additional model parameters (model, temperature, max_tokens, etc.)

    Returns:
        (output, request_time, input_tokens, output_tokens, info)
    """
    _setup_openrouter_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    if func_spec is not None:
        raise NotImplementedError(
            "We are not supporting function calling in OpenRouter for now."
        )

    model = filtered_kwargs.get("model", "")

    # Build messages based on model capabilities
    if _supports_system_message(model) and system_message:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if user_message:
            messages.append({"role": "user", "content": user_message})
        elif not user_message:
            # IMPORTANT: Always include a user message - models behave strangely
            # with only a system message (may output reasoning/continuation)
            messages.append({"role": "user", "content": "Please provide your response following the instructions above."})
    else:
        # Fallback: convert everything to user messages
        messages = [
            {"role": "user", "content": message}
            for message in [system_message, user_message]
            if message
        ]

    # Build extra_body for OpenRouter-specific features
    extra_body = {}

    # Provider routing configuration
    # See: https://openrouter.ai/docs/provider-routing
    provider_config = {}

    if provider:
        # Specific provider slug (e.g., "siliconflow/fp8", "deepinfra")
        # This forces routing to a specific provider
        provider_config["require_parameters"] = True
        # For provider-specific routing, append provider to model name
        # e.g., "deepseek/deepseek-v3.2:siliconflow/fp8"
        if "/" in provider and ":" not in filtered_kwargs.get("model", ""):
            filtered_kwargs["model"] = f"{filtered_kwargs.get('model', '')}:{provider}"
            logger.info(f"Using provider-specific model: {filtered_kwargs['model']}")

    if provider_order:
        provider_config["order"] = provider_order

    if provider_ignore:
        provider_config["ignore"] = provider_ignore

    if provider_config:
        extra_body["provider"] = provider_config
        logger.info(f"Provider config: {provider_config}")

    # Handle reasoning for supported models
    use_reasoning = enable_reasoning
    if use_reasoning is None:
        # Auto-detect based on model and reasoning_effort
        use_reasoning = _is_reasoning_model(model) or reasoning_effort is not None

    if use_reasoning:
        reasoning_config = {"enabled": True}

        # Map reasoning_effort to OpenRouter format if provided
        if reasoning_effort:
            # OpenRouter uses "effort" parameter for some models
            effort_map = {
                "low": "low",
                "medium": "medium",
                "high": "high",
            }
            if reasoning_effort.lower() in effort_map:
                reasoning_config["effort"] = effort_map[reasoning_effort.lower()]

        extra_body["reasoning"] = reasoning_config
        logger.info(f"Reasoning enabled for model {model}: {reasoning_config}")

    # Remove temperature for reasoning models (they often don't support it)
    if use_reasoning and "temperature" in filtered_kwargs:
        # Some reasoning models require temperature=1 or don't support it
        # DeepSeek R1 models typically work with temperature
        pass  # Keep temperature for now, remove if issues arise

    logger.info(f"OpenRouter API request: model={model}, system={system_message[:100] if system_message else None}...")

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        extra_body=extra_body if extra_body else None,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    # Extract response
    message = completion.choices[0].message
    output = message.content

    # Token counting
    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    # Check for reasoning tokens in usage
    reasoning_tokens = getattr(completion.usage, "reasoning_tokens", None)
    if reasoning_tokens is None:
        # Try alternative attribute names
        reasoning_tokens = getattr(completion.usage, "reasoningTokens", None)

    # Extract reasoning details if available
    reasoning_details = getattr(message, "reasoning_details", None)
    reasoning_content = getattr(message, "reasoning_content", None)

    # For reasoning models, check finish reason to understand response state
    finish_reason = completion.choices[0].finish_reason

    # If content is empty and finish_reason is 'length', the model was cut off
    # Log a warning - the response may be incomplete
    if (not output or output.strip() == "") and finish_reason == "length":
        logger.warning(
            "Response content is empty and finish_reason='length'. "
            "The model was cut off - consider increasing max_tokens."
        )
        # Try to extract from reasoning as fallback (might be incomplete)
        reasoning_text = getattr(message, "reasoning", None)
        if reasoning_text:
            output = reasoning_text
            logger.warning("Using 'reasoning' as fallback output (may be incomplete thinking, not final answer)")

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
        "finish_reason": finish_reason,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_details": reasoning_details,
        "reasoning_content": reasoning_content,
    }

    # Log with reasoning info if available
    token_info = f"in: {in_tokens}, out: {out_tokens}"
    if reasoning_tokens:
        token_info += f", reasoning: {reasoning_tokens}"

    logger.info(
        f"OpenRouter API call completed - {completion.model} - {req_time:.2f}s - "
        f"{in_tokens + out_tokens} tokens ({token_info})"
    )
    logger.debug(f"OpenRouter API response: {output[:500] if output else None}...")

    return output, req_time, in_tokens, out_tokens, info


def query_with_reasoning(
    system_message: str | None,
    user_message: str | None,
    model: str = "deepseek/deepseek-v3.2-speciale",
    reasoning_effort: str = "high",
    **model_kwargs,
) -> tuple[str, dict]:
    """
    Convenience function for querying reasoning-enabled models.

    Args:
        system_message: System prompt
        user_message: User message
        model: Model to use (default: deepseek-v3.2-speciale)
        reasoning_effort: "low", "medium", or "high"
        **model_kwargs: Additional parameters

    Returns:
        (output_text, full_info_dict)
    """
    output, req_time, in_tokens, out_tokens, info = query(
        system_message=system_message,
        user_message=user_message,
        model=model,
        reasoning_effort=reasoning_effort,
        enable_reasoning=True,
        **model_kwargs,
    )

    full_info = {
        **info,
        "request_time": req_time,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
    }

    return output, full_info
