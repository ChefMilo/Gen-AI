"""INTERIM: stands in for Member 4's backtest engine and harness until they land.

The harness step at session end: the agent's claims become Finding objects, with the
multiple-testing correction applied across every test the session ran. Mirrors R0:

- The correction family is EVERY run_backtest call (the backtester's attempts, in
  order), including reruns and untestable signals, which enter at p = 1.
- Holm-adjusted p-values (holm_adjust, a copy of R0's). R3-U skips the correction:
  p_value_corrected = p_value_raw.
- passed_correction = status == "tested" and p_value_corrected <= alpha.
- A claim whose signal was not "tested" (too few measurable events, or all CARs equal)
  is not returned as a Finding, as R0 drops untestable attempts; it is listed in the
  report instead. Every other claim is returned, significant or not, with the flag.
- A claim refers to its signal's latest run.

Finding fields: effect_car is the RAW mean CAR (what the matcher compares with the
planted direction); direction is the claimed position; event_ids are the measurable
events (as R0). SignalSpec is still Member 4's placeholder, so it holds only the event
type and the default benchmark; the full spec goes into the report (SystemRun.metadata
and the trace).
"""

from typing import Any

from agent.finding import Finding, FindingClaim
from agent.interim.backtest import TESTED
from dsl.signal import SignalSpec


# TODO(member-3): dedupe with R0's holm_adjust (src/baselines/r0.py) once PR #17 merges.
def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-adjust p-values, preserving their original order (R0's holm_adjust)."""
    count = len(p_values)
    order = sorted(range(count), key=lambda i: p_values[i])
    adjusted = [1.0] * count
    previous = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        previous = max(previous, value)
        adjusted[index] = previous
    return adjusted


def corrected_p_values(attempts: list[dict[str, Any]], correct: bool) -> list[float]:
    """One corrected p per attempt; untestable attempts enter at p = 1, as in R0."""
    raw = [a["p_value_raw"] if a["status"] == TESTED else 1.0 for a in attempts]
    return holm_adjust(raw) if correct else raw


def claims_to_findings(
    claims: list[FindingClaim],
    attempts: list[dict[str, Any]],
    *,
    session_id: str,
    system: str,
    alpha: float,
    correct: bool = True,
) -> tuple[list[Finding], dict[str, Any]]:
    """(findings, report). report has the correction family and every claim's outcome."""
    corrected = corrected_p_values(attempts, correct)
    latest = {a["signal_id"]: i for i, a in enumerate(attempts)}
    findings, outcomes = [], []
    for claim in claims:
        i = latest.get(claim.signal_id)
        if i is None:  # report_finding only accepts backtested signals
            raise ValueError(f"claim on {claim.signal_id!r}, which was never backtested")
        a = attempts[i]
        outcome = {"signal_id": claim.signal_id, "test_index": a["test_index"],
                   "status": a["status"], "spec": a["spec"]}
        if a["status"] != TESTED:
            outcomes.append({**outcome, "returned": False})
            continue
        passed = corrected[i] <= alpha
        k = len(findings)
        findings.append(Finding(
            finding_id=f"{session_id}-f{k}",
            session_id=session_id,
            system=system,
            signal_id=f"{session_id}-{claim.signal_id}",
            signal_spec=SignalSpec(event_type=a["spec"]["event_type"]),
            event_type=a["spec"]["event_type"],
            event_ids=list(a["event_ids"]),
            n_events=a["n_events"],
            direction=a["spec"]["direction"],
            effect_car=a["mean_car"],
            t_stat=a["t_stat"],
            p_value_raw=a["p_value_raw"],
            p_value_corrected=corrected[i],
            passed_correction=passed,
            rationale=claim.rationale,
            evidence=claim.evidence,
        ))
        outcomes.append({**outcome, "returned": True, "finding_id": findings[-1].finding_id,
                         "p_value_corrected": corrected[i], "passed_correction": passed})
    report = {
        "correction": "holm" if correct else "none",
        "alpha": alpha,
        "tests_in_family": len(attempts),
        "attempts": [{**{k: v for k, v in a.items() if k != "event_ids"},
                      "n_event_ids": len(a["event_ids"]), "p_value_corrected": p}
                     for a, p in zip(attempts, corrected, strict=True)],
        "claims": outcomes,
    }
    return findings, report
