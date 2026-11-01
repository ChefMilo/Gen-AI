"""INTERIM: stands in for Member 4's backtest engine and harness until they land.

The harness step at session end: the agent's claims become Finding objects, with the
multiple-testing correction applied across every test the session ran. Mirrors R0:

- The correction family is EVERY counted run_backtest call (the backtester's attempts,
  in order), including untestable signals, which enter at p = 1. Duplicates (same
  events as an earlier test) are not counted; a claim on one refers to the original.
- Holm-adjusted p-values (holm_adjust, a copy of R0's). R3-U skips the correction:
  p_value_corrected = p_value_raw.
- passed_correction = status == "tested" and p_value_corrected <= alpha.
- A claim whose signal was not "tested" (too few measurable events, or all CARs equal)
  is not returned as a Finding, as R0 drops untestable attempts; it is listed in the
  report instead. Every other claim is returned, significant or not, with the flag.
- A claim refers to its signal's test (the original's, for a duplicate); its direction
  and spec are the claimed signal's own.
- Strict correction, measured not enforced (item 15): Holm over the family PLUS every
  breakdown group the agent viewed with inspect_results (each group with an n and a
  mean is one test; groups without a t-test enter at p = 1). Recorded per claim as
  p_value_corrected_strict / passed_correction_strict in the report (SystemRun.metadata);
  Finding is unchanged and passed_correction stays the official rule.

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
    aliases: dict[str, str] | None = None,
    specs: dict[str, dict] | None = None,
    viewed_groups: dict[tuple, float | None] | None = None,
) -> tuple[list[Finding], dict[str, Any]]:
    """(findings, report). report has the correction family and every claim's outcome.

    aliases: duplicate signal -> the signal it repeats; specs: signal -> its own spec
    (JSON), for the claimed direction; viewed_groups: breakdown groups seen (strict).
    """
    aliases, specs, viewed_groups = aliases or {}, specs or {}, viewed_groups or {}
    corrected = corrected_p_values(attempts, correct)
    raw = corrected_p_values(attempts, correct=False)
    group_ps = [1.0 if p is None else p for p in viewed_groups.values()]
    strict = holm_adjust(raw + group_ps)[: len(attempts)]
    latest = {a["signal_id"]: i for i, a in enumerate(attempts)}
    findings, outcomes = [], []
    for claim in claims:
        tested_as = aliases.get(claim.signal_id, claim.signal_id)
        i = latest.get(tested_as)
        if i is None:  # report_finding only accepts backtested signals
            raise ValueError(f"claim on {claim.signal_id!r}, which was never backtested")
        a = attempts[i]
        spec = specs.get(claim.signal_id, a["spec"])
        outcome = {"signal_id": claim.signal_id, "tested_as": tested_as,
                   "test_index": a["test_index"], "status": a["status"], "spec": spec}
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
            signal_spec=SignalSpec(event_type=spec["event_type"]),
            event_type=spec["event_type"],
            event_ids=list(a["event_ids"]),
            n_events=a["n_events"],
            direction=spec["direction"],
            effect_car=a["mean_car"],
            t_stat=a["t_stat"],
            p_value_raw=a["p_value_raw"],
            p_value_corrected=corrected[i],
            passed_correction=passed,
            rationale=claim.rationale,
            evidence=claim.evidence,
        ))
        outcomes.append({**outcome, "returned": True, "finding_id": findings[-1].finding_id,
                         "p_value_corrected": corrected[i], "passed_correction": passed,
                         "p_value_corrected_strict": strict[i],
                         "passed_correction_strict": strict[i] <= alpha})
    report = {
        "correction": "holm" if correct else "none",
        "alpha": alpha,
        "tests_in_family": len(attempts),
        "attempts": [{**{k: v for k, v in a.items() if k != "event_ids"},
                      "n_event_ids": len(a["event_ids"]), "p_value_corrected": p,
                      "p_value_corrected_strict": s}
                     for a, p, s in zip(attempts, corrected, strict, strict=True)],
        "claims": outcomes,
        "strict": {
            "family_size": len(attempts) + len(group_ps),
            "breakdown_groups": len(group_ps),
            "findings_passed": sum(o.get("passed_correction", False) for o in outcomes),
            "findings_passed_strict": sum(o.get("passed_correction_strict", False)
                                          for o in outcomes),
        },
    }
    return findings, report
