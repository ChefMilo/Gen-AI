"""Research-period access, backtest budgets, and audit logging."""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.finding import EvidenceQuote, Finding, System
from backtest.engine import run_event_study
from dsl.signal import SignalSpec
from harness.corrections import holm_adjust
from store.prices import first_tradeable_date


class GuardrailError(RuntimeError):
    """A request violates a research-session rule."""


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    correction_family: Literal["budget", "attempted"] = "budget"
    mode: Literal["guarded", "unguarded"] = "guarded"

    research_start: date
    research_end: date
    budget: int = Field(default=30, strict=True, ge=0)
    min_events: int = Field(default=20, strict=True, ge=2)
    alpha: float = Field(default=0.05, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_dates(self) -> "HarnessConfig":
        if self.research_end < self.research_start:
            raise ValueError("research_end must not precede research_start")
        return self


def eastern_day(value) -> date:
    """Interpret naive acceptance timestamps as US Eastern."""
    timestamp = pd.Timestamp(value)

    if pd.isna(timestamp):
        raise ValueError("Acceptance timestamp must not be missing")

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("America/New_York")

    return timestamp.date()


class FindingClaim(BaseModel):
    """A claim submitted by the agent, without self-assigned statistics."""

    model_config = ConfigDict(extra="forbid")

    signal_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    step: int | None = Field(default=None, ge=0)


class FinalizeRequest(BaseModel):
    """Validated information supplied by the session controller."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    system: System
    claims: list[FindingClaim]


def validate_event_timing(
    events: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> None:
    """Verify trade dates for eligible research-period events."""
    eligible = events.loc[
        events["is_new_announcement"].eq(True) & events["extraction_status"].eq("passed")
    ]

    if eligible.empty:
        return

    spy = benchmarks.loc[benchmarks["ticker"].eq("SPY")]
    calendar = sorted(set(pd.to_datetime(spy["date"]).dt.date))
    if not calendar:
        raise ValueError("SPY trading calendar is required for timing validation")

    for event in eligible.to_dict(orient="records"):
        try:
            expected = first_tradeable_date(
                event["acceptance_datetime"],
                calendar=calendar,
            )
        except ValueError as error:
            raise GuardrailError(
                f"Cannot validate timing for {event['event_id']}: {error}"
            ) from error

        actual = pd.Timestamp(event["first_tradeable_date"]).date()

        if actual != expected:
            raise GuardrailError(
                f"Invalid first_tradeable_date for "
                f"{event['event_id']}: expected {expected}, "
                f"got {actual}"
            )


class ResearchBacktester:
    """The agent accesses this object through tool methods.

    This is an application boundary, not a sandbox for arbitrary Python.
    Raw data and internal state must never be exposed as agent tools.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        benchmarks: pd.DataFrame,
        events: pd.DataFrame,
        config: HarnessConfig,
        audit_path: Path,
    ):
        self._config = config
        self._specs: dict[str, SignalSpec] = {}
        self._results: dict[str, dict] = {}
        self._attempts: list[dict] = []
        self._audit_sequence = 0
        self._closed = False

        required = {
            "first_tradeable_date",
            "acceptance_datetime",
        }
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"Missing timing columns: {sorted(missing)}")

        start = config.research_start
        end = config.research_end

        event_days = pd.to_datetime(events["first_tradeable_date"], errors="raise").dt.date
        acceptance_days = events["acceptance_datetime"].map(eastern_day)

        visible = event_days.ge(start) & event_days.le(end) & acceptance_days.le(end)
        self._events = events.loc[visible].copy(deep=True)
        validate_event_timing(self._events, benchmarks)

        # Keep earlier price history, but never prices after the cutoff.
        price_days = pd.to_datetime(prices["date"]).dt.date
        benchmark_days = pd.to_datetime(benchmarks["date"]).dt.date

        self._prices = prices.loc[price_days.le(end)].copy(deep=True)
        self._benchmarks = benchmarks.loc[benchmark_days.le(end)].copy(deep=True)

        self._audit_path = Path(audit_path)
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Refuse to mix a new session with an existing audit file.
        with self._audit_path.open("x", encoding="utf-8"):
            pass

        self._record(
            "session_start",
            config=config.model_dump(mode="json"),
        )

    def _require_open(self) -> None:
        if self._closed:
            self._record(
                "request_rejected",
                reason="session_closed",
            )
            raise GuardrailError("Research session is closed")

    def _record(self, action: str, **details) -> None:
        entry = {
            "sequence": self._audit_sequence,
            "timestamp": datetime.now(UTC).isoformat(),
            "action": action,
            **details,
        }
        line = json.dumps(entry, allow_nan=False)

        with self._audit_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

        self._audit_sequence += 1

    def define(self, spec: dict) -> str:
        """Validate and register a signal without spending a backtest."""
        self._require_open()
        try:
            validated = SignalSpec.model_validate(spec)
        except ValueError as error:
            self._record(
                "definition_rejected",
                error=str(error),
            )
            raise

        signal_id = f"sig-{len(self._specs) + 1}"
        self._specs[signal_id] = validated.model_copy(deep=True)

        self._record(
            "signal_defined",
            signal_id=signal_id,
            spec=validated.model_dump(mode="json"),
        )
        return signal_id

    def budget_status(self) -> dict[str, int]:
        return {
            "budget": self._config.budget,
            "tests_used": len(self._attempts),
        }

    def run(self, signal_id: str) -> dict:
        """Every accepted execution attempt consumes one budget slot."""
        self._require_open()
        if signal_id not in self._specs:
            self._record(
                "run_rejected",
                signal_id=signal_id,
                reason="unknown_signal",
            )
            raise GuardrailError(f"Unknown signal: {signal_id}")

        if (
            self._config.mode == "guarded"
            and len(self._attempts) >= self._config.budget
        ):
            self._record(
                "run_rejected",
                signal_id=signal_id,
                reason="budget_exhausted",
            )
            raise GuardrailError("Backtest budget exhausted")

        # Reserve the slot BEFORE calculation. Errors do not refund it.
        attempt = {
            "attempt_id": len(self._attempts) + 1,
            "signal_id": signal_id,
            "status": "started",
            "p_value_raw": None,
        }
        self._attempts.append(attempt)

        self._record("backtest_started", **attempt)

        try:
            result = run_event_study(
                self._prices,
                self._benchmarks,
                self._events,
                self._specs[signal_id],
                signal_id,
                min_events=self._config.min_events,
            )
        except Exception as error:
            attempt["status"] = "error"
            self._record(
                "backtest_error",
                **attempt,
                error=str(error),
            )
            raise

        attempt["status"] = result.status
        attempt["p_value_raw"] = result.p_value_raw

        output = {
            **result.model_dump(mode="json"),
            "tests_used": len(self._attempts),
            "budget_remaining": (
                max(0, self._config.budget - len(self._attempts))
                if self._config.mode == "guarded"
                else None
            ),
            "budget_enforced": self._config.mode == "guarded",
        }
        self._results[signal_id] = output

        self._record(
            "backtest_completed",
            **attempt,
            result=output,
        )

        # Return a separate copy, so tool callers cannot mutate storage.
        return json.loads(json.dumps(output, allow_nan=False))

    def finalize(
        self,
        claims: list[dict],
        session_id: str,
        system: str,
    ) -> list[Finding]:
        """Close the session and score claims using all attempted tests.

        Called by the session controller after the agent finishes.
        It must not be exposed as an exploratory agent tool.
        """
        self._require_open()

        try:
            request = FinalizeRequest(
                session_id=session_id,
                system=system,
                claims=claims,
            )
            is_unguarded_system = request.system == "R3-U"
            is_unguarded_config = self._config.mode == "unguarded"

            if is_unguarded_system != is_unguarded_config:
                raise GuardrailError(
                    "R3-U requires unguarded mode; "
                    "other systems require guarded mode"
                )

            claimed_ids = [claim.signal_id for claim in request.claims]
            if len(claimed_ids) != len(set(claimed_ids)):
                raise GuardrailError("Duplicate claims are not allowed")

            for signal_id in claimed_ids:
                if signal_id not in self._results:
                    raise GuardrailError(f"Signal has no completed backtest: {signal_id}")
        except ValueError as error:
            self._record(
                "finalization_rejected",
                error=str(error),
            )
            raise
        except GuardrailError as error:
            self._record(
                "finalization_rejected",
                error=str(error),
            )
            raise

        # Failed and untestable attempts stay in the family with p=1.
        p_values = [
            attempt["p_value_raw"] if attempt["status"] == "tested" else 1.0
            for attempt in self._attempts
        ]
        if self._config.mode == "guarded":
            family_size = (
                self._config.budget
                if self._config.correction_family == "budget"
                else len(self._attempts)
            )
            adjusted = holm_adjust(
                p_values,
                family_size=family_size,
            )
        else:
            family_size = len(self._attempts)
            adjusted = list(p_values)

        findings = []

        for claim in request.claims:
            result = self._results[claim.signal_id]

            if result["status"] != "tested":
                self._record(
                    "claim_rejected",
                    signal_id=claim.signal_id,
                    reason=result["status"],
                )
                continue

            # tests_used identifies the attempt that produced this result.
            # If a signal was rerun, use its latest completed result.
            attempt_index = result["tests_used"] - 1
            corrected = adjusted[attempt_index]
            spec = self._specs[claim.signal_id]

            finding = Finding(
                finding_id=(f"{request.session_id}-f{len(findings) + 1}"),
                session_id=request.session_id,
                system=request.system,
                signal_id=claim.signal_id,
                signal_spec=spec.model_copy(deep=True),
                event_type=spec.event_type,
                event_ids=list(result["event_ids"]),
                n_events=result["n_events"],
                direction=spec.direction,
                effect_car=result["mean_car"],
                t_stat=result["t_stat"],
                p_value_raw=result["p_value_raw"],
                p_value_corrected=corrected,
                passed_correction=(corrected <= self._config.alpha),
                rationale=claim.rationale,
                evidence=claim.evidence,
            )
            findings.append(finding)

        self._closed = True
        self._record(
            "session_finalized",
            session_id=request.session_id,
            system=request.system,
            tests_used=len(self._attempts),
            correction_family=self._config.correction_family,
            family_size=family_size,
            findings=[finding.model_dump(mode="json") for finding in findings],
        )
        return findings

    def signal_options(self) -> dict:
        """Describe the supported signal language and breakdowns."""
        breakdowns = ["year"]
        breakdowns.extend(
            name for name in ("sector", "mcap_bucket") if name in self._events.columns
        )

        return {
            "event_types": ["dividend", "buyback"],
            "benchmarks": sorted(self._benchmarks["ticker"].dropna().unique().tolist()),
            "operators": {
                "number": [">", ">=", "<", "<=", "==", "!="],
                "category": ["==", "!="],
                "bool": ["==", "!="],
            },
            "fields": {
                "number": [
                    name
                    for name in (
                        "dividend_change_pct",
                        "dividend_per_share",
                        "buyback_amount_usd",
                        "buyback_pct_mcap",
                        "mcap_usd",
                    )
                    if name in self._events.columns
                ],
                "category": [
                    name
                    for name in ("mcap_bucket", "dividend_kind")
                    if name in self._events.columns
                ],
                "bool": [name for name in ("is_new_program",) if name in self._events.columns],
            },
            "holding_days": list(range(1, 61)),
            "directions": ["long", "short"],
            "entry": ["first_tradeable_date"],
            "breakdowns": breakdowns,
            "max_conditions": 5,
        }

    def breakdown(self, signal_id: str, by: str) -> dict:
        """Describe an already-run signal without another backtest.

        Subgroup significance tests are deliberately not provided.
        A new subgroup hypothesis must use define() and run().
        """
        if signal_id not in self._results:
            raise GuardrailError("Run the signal before requesting a breakdown")

        allowed = self.signal_options()["breakdowns"]
        if by not in allowed:
            raise GuardrailError(f"Unsupported breakdown {by!r}; use {allowed}")

        result = self._results[signal_id]
        rows = pd.DataFrame(result["event_returns"])
        groups = {}

        if not rows.empty:
            if by == "year":
                rows["group"] = pd.to_datetime(rows["first_tradeable_date"]).dt.year
            else:
                metadata = self._events[["event_id", by]]
                rows = rows.merge(
                    metadata,
                    on="event_id",
                    how="left",
                    validate="one_to_one",
                )
                rows["group"] = rows[by]

            sign = 1 if self._specs[signal_id].direction == "long" else -1

            for key, group in rows.groupby("group", dropna=False, sort=True):
                label = "missing" if pd.isna(key) else str(key)
                returns = group["car"].astype(float)

                groups[label] = {
                    "n_events": len(group),
                    "mean_car": float(returns.mean()),
                    "directional_mean_car": float(sign * returns.mean()),
                    "hit_rate": float((sign * returns > 0).mean()),
                }

        self._record(
            "breakdown_requested",
            signal_id=signal_id,
            by=by,
        )

        return {
            "signal_id": signal_id,
            "by": by,
            "descriptive_only": True,
            "groups": groups,
        }
