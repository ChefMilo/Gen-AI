"""Preflight model-call, token and explicitly priced cost limits.

Provider token counting and rate-based costs are estimates, not a billing-system
hard cap. Failed calls keep their reservation and stop further calls.
"""

import math
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from agent.llm import LLMResponse


class ResourceLimitReached(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class CostRates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)
    cache_read_per_million: float = Field(ge=0)
    cache_write_per_million: float = Field(ge=0)


class ModelLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_calls: int = Field(default=60, strict=True, ge=1)
    max_input_tokens: int = Field(default=100000, strict=True, ge=1)
    max_output_tokens: int = Field(default=60000, strict=True, ge=1)
    response_output_limit: int = Field(default=1000, strict=True, ge=1)
    max_estimated_cost_usd: float | None = Field(default=None, gt=0)
    input_safety_factor: float = Field(default=1.1, ge=1)


class ResourceLimitedClient:
    """Wrap a client with a trusted input-token estimator and response bound.

    A live caller must configure the underlying provider's max_tokens to the
    same response_output_limit. Cost rates must include the request's applicable
    tier/cache prices; missing rates cannot silently turn paid costs into zero.
    """

    def __init__(
        self,
        client,
        limits: ModelLimits,
        estimate_input: Callable,
        rates: CostRates | None = None,
        audit: Callable | None = None,
    ):
        if limits.max_estimated_cost_usd is not None and rates is None:
            raise ValueError("Explicit token prices are required for a monetary cap")
        if hasattr(client, "max_tokens") and client.max_tokens != limits.response_output_limit:
            raise ValueError("Provider max_tokens must match response_output_limit")
        self.client, self.limits, self.rates = client, limits, rates
        self.estimate_input, self.audit = estimate_input, audit
        self.model = getattr(client, "model", None)
        self.calls = self.input_tokens = self.output_tokens = 0
        self.estimated_cost_usd = 0.0
        self.uncertain_reservation_usd = 0.0
        self.stop_reason = None

    def _stop(self, reason):
        self.stop_reason = reason
        if self.audit:
            self.audit("model_call_rejected", reason=reason, **self.metadata())
        raise ResourceLimitReached(reason)

    def metadata(self):
        return {
            "model_calls": self.calls,
            "model_input_tokens": self.input_tokens,
            "model_output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "uncertain_reservation_usd": self.uncertain_reservation_usd,
            "resource_stop_reason": self.stop_reason,
        }

    def complete(self, system, messages, tools) -> LLMResponse:
        if self.stop_reason:
            self._stop(self.stop_reason)
        if self.calls >= self.limits.max_calls:
            self._stop("model_call_cap")
        estimate = self.estimate_input(system, messages, tools)
        if isinstance(estimate, bool) or not isinstance(estimate, int) or estimate < 0:
            raise ValueError("Input-token estimator must return a nonnegative integer")
        reserved_input = math.ceil(estimate * self.limits.input_safety_factor)
        reserved_output = self.limits.response_output_limit
        if self.input_tokens + reserved_input > self.limits.max_input_tokens:
            self._stop("input_token_preflight_cap")
        if self.output_tokens + reserved_output > self.limits.max_output_tokens:
            self._stop("output_token_preflight_cap")
        reserved_cost = 0.0
        if self.rates:
            input_rate = max(
                self.rates.input_per_million,
                self.rates.cache_read_per_million,
                self.rates.cache_write_per_million,
            )
            reserved_cost = (
                reserved_input * input_rate + reserved_output * self.rates.output_per_million
            ) / 1e6
        if (
            self.limits.max_estimated_cost_usd is not None
            and self.estimated_cost_usd + reserved_cost > self.limits.max_estimated_cost_usd
        ):
            self._stop("estimated_cost_preflight_cap")
        self.calls += 1
        try:
            response = self.client.complete(system, messages, tools)
        except Exception:
            self.uncertain_reservation_usd += reserved_cost
            self.stop_reason = "model_call_error"
            if self.audit:
                self.audit("model_call_error", **self.metadata())
            raise
        usage = response.usage
        counts = [
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
        ]
        if any(value < 0 for value in counts):
            self.stop_reason = "invalid_provider_usage"
            raise ValueError("Provider reported negative token usage")
        self.input_tokens += usage.total_input_tokens
        self.output_tokens += usage.output_tokens
        if self.rates:
            self.estimated_cost_usd += (
                usage.input_tokens * self.rates.input_per_million
                + usage.output_tokens * self.rates.output_per_million
                + usage.cache_read_input_tokens * self.rates.cache_read_per_million
                + usage.cache_creation_input_tokens * self.rates.cache_write_per_million
            ) / 1e6
        if self.input_tokens > self.limits.max_input_tokens:
            self.stop_reason = "input_token_usage_cap"
        elif self.output_tokens > self.limits.max_output_tokens:
            self.stop_reason = "output_token_usage_cap"
        elif (
            self.limits.max_estimated_cost_usd is not None
            and self.estimated_cost_usd > self.limits.max_estimated_cost_usd
        ):
            self.stop_reason = "estimated_cost_usage_cap"
        if self.audit:
            self.audit("model_call_completed", **self.metadata())
        return response
