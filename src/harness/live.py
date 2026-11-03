"""Explicitly configured, opt-in provider client. Importing this makes no API call."""

from agent.llm import AnthropicClient, to_anthropic_messages, to_anthropic_tools
from harness.limits import CostRates, ModelLimits, ResourceLimitedClient


def make_limited_anthropic_client(
    *,
    model: str,
    limits: ModelLimits,
    rates: CostRates,
    effort: str | None = None,
    thinking_display: str | None = None,
) -> ResourceLimitedClient:
    """Only call after model, rates and spending permission are agreed.

    The SDK reads its normal API-key environment variable; keys are never put in
    experiment metadata. There are no guessed default token prices. Token-count
    requests are estimates; provider/account limits are needed for a hard bill cap.
    """
    if limits.max_estimated_cost_usd is None:
        raise ValueError("Live sessions require an explicit estimated USD budget")
    if rates.input_per_million <= 0 or rates.output_per_million <= 0:
        raise ValueError("Live input/output token prices must be positive")
    client = AnthropicClient(model, limits.response_output_limit, effort, thinking_display)
    client._client = client._client.with_options(max_retries=0, timeout=60)

    def estimate(system, messages, tools):
        request = {"model": model, "system": system, "messages": to_anthropic_messages(messages)}
        if tools:
            request["tools"] = to_anthropic_tools(tools)
        return client._client.messages.count_tokens(**request).input_tokens

    return ResourceLimitedClient(client, limits, estimate, rates)
