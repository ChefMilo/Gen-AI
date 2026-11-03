"""Controller-only fixed-signal holdout evaluation, once per named ledger.

No agent receives the holdout frames. The controller owns the canonical ledger
path; a Python caller with filesystem access is not sandboxed by this module.
"""

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.finding import Finding
from backtest.engine import run_event_study
from harness.backtester import eastern_day, validate_event_timing
from harness.corrections import holm_adjust


class HoldoutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    research_end: date
    holdout_start: date
    holdout_end: date
    min_events: int = Field(default=20, strict=True, ge=2)
    alpha: float = Field(default=0.05, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_split(self):
        if self.holdout_start <= self.research_end or self.holdout_end < self.holdout_start:
            raise ValueError("Holdout must follow research with no overlap")
        return self


def evaluate_holdout_once(
    findings: list[Finding],
    config: HoldoutConfig,
    ledger_path: Path,
    data_loader,
) -> dict:
    """Freeze all supplied discovery specs before calling the data loader.

    Failures retain the reservation; there is no automatic retry/reset pathway.
    A retry or a changed split is an explicit methodological decision for the team.
    """
    frozen = [finding.model_copy(deep=True) for finding in findings]
    if not frozen or any(not finding.passed_correction for finding in frozen):
        raise ValueError("Freeze a nonempty set of significant research findings")
    identifiers = [finding.signal_id for finding in frozen]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Holdout signal IDs must be unique")
    definitions = [finding.model_dump(mode="json") for finding in frozen]
    reservation = {
        "status": "reserved",
        "config": config.model_dump(mode="json"),
        "research_findings": definitions,
        "definition_sha256": hashlib.sha256(
            json.dumps(definitions, sort_keys=True).encode()
        ).hexdigest(),
        "reserved_at": datetime.now(UTC).isoformat(),
    }
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("x", encoding="utf-8") as stream:
        json.dump(reservation, stream, indent=2, allow_nan=False)
    try:
        prices, benchmarks, events = data_loader(config.holdout_start, config.holdout_end)
        event_days = pd.to_datetime(events["first_tradeable_date"]).dt.date
        accepted = events["acceptance_datetime"].map(eastern_day)
        events = events.loc[
            event_days.between(config.holdout_start, config.holdout_end)
            & accepted.le(config.holdout_end)
        ].copy()
        validate_event_timing(events, benchmarks)
        prices = prices.loc[
            pd.to_datetime(prices["date"]).dt.date.between(config.holdout_start, config.holdout_end)
        ].copy()
        benchmarks = benchmarks.loc[
            pd.to_datetime(benchmarks["date"]).dt.date.between(
                config.holdout_start, config.holdout_end
            )
        ].copy()
        results = [
            run_event_study(
                prices,
                benchmarks,
                events,
                finding.signal_spec,
                finding.signal_id,
                config.min_events,
            )
            for finding in frozen
        ]
        corrected = holm_adjust(
            [result.p_value_raw if result.status == "tested" else 1.0 for result in results]
        )
        rows = []
        for finding, result, p_value in zip(frozen, results, corrected, strict=True):
            mean = result.mean_car
            rows.append(
                {
                    "signal_id": finding.signal_id,
                    "research_effect_car": finding.effect_car,
                    "holdout_effect_car": mean,
                    "n_events": result.n_events,
                    "status": result.status,
                    "p_value_corrected": p_value,
                    "passed_correction": result.status == "tested" and p_value <= config.alpha,
                    "same_sign": bool(mean * finding.effect_car > 0) if mean is not None else None,
                    "retention_ratio": mean / finding.effect_car
                    if mean is not None and finding.effect_car
                    else None,
                }
            )
        report = {
            **reservation,
            "status": "completed",
            "results": rows,
            "completed_at": datetime.now(UTC).isoformat(),
        }
    except Exception as error:
        report = {**reservation, "status": "failed", "error": str(error)}
        ledger_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        raise
    ledger_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
