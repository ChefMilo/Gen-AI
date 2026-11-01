import pytest

from agent.llm import LLMResponse, Usage
from harness.limits import CostRates, ModelLimits, ResourceLimitedClient, ResourceLimitReached


class CountingClient:
    max_tokens = 10
    model = "local-test"

    def __init__(self):
        self.calls = 0

    def complete(self, *args):
        self.calls += 1
        return LLMResponse(
            usage=Usage(
                input_tokens=5,
                output_tokens=3,
                cache_read_input_tokens=2,
                cache_creation_input_tokens=1,
            )
        )


RATES = CostRates(
    input_per_million=1,
    output_per_million=2,
    cache_read_per_million=0.1,
    cache_write_per_million=1.25,
)


def test_call_cap_blocks_before_an_extra_provider_request():
    client = CountingClient()
    wrapper = ResourceLimitedClient(
        client, ModelLimits(max_calls=1, response_output_limit=10), lambda *args: 8
    )
    wrapper.complete("", [], [])
    with pytest.raises(ResourceLimitReached, match="model_call_cap"):
        wrapper.complete("", [], [])
    assert client.calls == 1


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"max_input_tokens": 2}, "input_token_preflight_cap"),
        ({"max_output_tokens": 2}, "output_token_preflight_cap"),
        ({"max_estimated_cost_usd": 0.000001}, "estimated_cost_preflight_cap"),
    ],
)
def test_limits_reject_before_the_provider_is_called(overrides, reason):
    client = CountingClient()
    wrapper = ResourceLimitedClient(
        client, ModelLimits(response_output_limit=10, **overrides), lambda *args: 8, RATES
    )
    with pytest.raises(ResourceLimitReached, match=reason):
        wrapper.complete("", [], [])
    assert client.calls == 0


def test_cache_and_output_costs_are_accounted_separately():
    wrapper = ResourceLimitedClient(
        CountingClient(), ModelLimits(response_output_limit=10), lambda *args: 8, RATES
    )
    wrapper.complete("", [], [])
    assert wrapper.input_tokens == 8 and wrapper.output_tokens == 3
    assert wrapper.estimated_cost_usd == pytest.approx((5 + 6 + 0.2 + 1.25) / 1e6)


def test_unknown_prices_cannot_silently_disable_monetary_limit():
    with pytest.raises(ValueError, match="token prices"):
        ResourceLimitedClient(
            CountingClient(),
            ModelLimits(response_output_limit=10, max_estimated_cost_usd=1),
            lambda *args: 8,
        )


def test_failed_call_is_not_refunded_or_retried():
    client = CountingClient()

    def fail(*args):
        client.calls += 1
        raise RuntimeError("uncertain provider failure")

    client.complete = fail
    wrapper = ResourceLimitedClient(
        client,
        ModelLimits(response_output_limit=10, max_estimated_cost_usd=1),
        lambda *args: 8,
        RATES,
    )
    with pytest.raises(RuntimeError, match="uncertain provider failure"):
        wrapper.complete("", [], [])
    assert wrapper.uncertain_reservation_usd > 0
    with pytest.raises(ResourceLimitReached, match="model_call_error"):
        wrapper.complete("", [], [])
    assert client.calls == 1
