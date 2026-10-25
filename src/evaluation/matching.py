"""Matching findings to planted truths (docs/design.md section 3, "Scoring").

A finding matches a truth if all hold:
- same event_type;
- Jaccard(finding.event_ids, the truth's measurable targets) >= jaccard_min, where
  measurable = target_event_ids minus unmeasurable_event_ids;
- sign(effect_car) == the truth's direction (effect_car is read as the raw mean CAR
  of the events, not the CAR of the long/short position);
- passed_correction.

Each truth matches at most one finding and each finding at most one truth: among the
pairs that pass every test, the highest Jaccard is taken first, ties going to the
earlier finding (then the earlier truth).

Every finding gets an outcome. On a planted universe:
- matched;
- duplicate: passes every test against a truth another finding already matched;
- near_miss: same event_type and Jaccard high enough, but wrong sign and/or not
  significant (the reasons say which);
- unmatched: anything else. Its reasons describe the closest truth, if any.
On a placebo universe, a significant finding is a false_discovery and the rest are
not_significant. Findings on the real universe are unscored here (see
metrics.retention).
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from agent.finding import Finding
from evaluation.universe_record import UniverseRecord, UniverseType, measurable_target_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

# Float slack for Jaccard at the threshold, e.g. 6/10 against 0.6.
_JACCARD_TOL = 1e-12

Status = Literal[
    "matched", "duplicate", "near_miss", "unmatched", "false_discovery", "not_significant",
    "unscored",
]
Reason = Literal[
    "wrong_event_type", "low_jaccard", "wrong_sign", "not_significant", "truth_already_matched"
]
NEAR_MISS_REASONS = {"wrong_sign", "not_significant"}


def default_jaccard_min(path: Path = DEFAULT_CONFIG_PATH) -> float:
    with open(path, encoding="utf-8") as f:
        return float(yaml.safe_load(f)["jaccard_min"])


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


class Match(BaseModel):
    finding_id: str
    spec_id: str
    jaccard: float
    effect_car: float  # reported
    true_effect: float  # direction * effect
    effect_error: float  # effect_car - true_effect


class FindingOutcome(BaseModel):
    finding_id: str
    status: Status
    significant: bool  # passed_correction
    spec_id: str | None = None  # the matched truth, or the closest one
    jaccard: float | None = None  # against spec_id
    reasons: list[Reason] = []  # tests failed against spec_id


class MatchResult(BaseModel):
    universe_id: str
    universe_type: UniverseType
    jaccard_min: float
    n_truths: int
    matches: list[Match]
    outcomes: list[FindingOutcome]  # one per finding, in finding order
    missed_spec_ids: list[str]  # truths no finding matched

    def with_status(self, *statuses: Status) -> list[FindingOutcome]:
        return [o for o in self.outcomes if o.status in statuses]

    @property
    def n_significant(self) -> int:
        return sum(o.significant for o in self.outcomes)


def _failed_tests(finding: Finding, truth, measurable: set[str], jaccard_min: float):
    """(Jaccard, list of failed tests) for one finding against one truth."""
    j = jaccard(set(finding.event_ids), measurable)
    failed = []
    if finding.event_type != truth.spec.event_type:
        failed.append("wrong_event_type")
    if j < jaccard_min - _JACCARD_TOL:
        failed.append("low_jaccard")
    if sign(finding.effect_car) != truth.spec.direction:
        failed.append("wrong_sign")
    if not finding.passed_correction:
        failed.append("not_significant")
    return j, failed


def match_findings(
    findings: list[Finding], record: UniverseRecord, jaccard_min: float | None = None
) -> MatchResult:
    """Score findings against the universe's truths. jaccard_min defaults to the config."""
    if jaccard_min is None:
        jaccard_min = default_jaccard_min()
    ids = [f.finding_id for f in findings]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate finding_id in findings")

    if record.universe_type != "planted":
        outcomes = []
        for f in findings:
            if record.universe_type == "real":
                status = "unscored"
            else:
                status = "false_discovery" if f.passed_correction else "not_significant"
            outcomes.append(
                FindingOutcome(finding_id=f.finding_id, status=status,
                               significant=f.passed_correction)
            )
        return MatchResult(
            universe_id=record.universe_id, universe_type=record.universe_type,
            jaccard_min=jaccard_min, n_truths=0, matches=[], outcomes=outcomes,
            missed_spec_ids=[],
        )

    truths = record.truths
    measurable = [measurable_target_ids(t) for t in truths]
    tests = [
        [_failed_tests(f, t, m, jaccard_min) for t, m in zip(truths, measurable, strict=True)]
        for f in findings
    ]

    # Greedy one-to-one assignment over the pairs that pass every test.
    pairs = sorted(
        (-tests[i][k][0], i, k)
        for i in range(len(findings))
        for k in range(len(truths))
        if not tests[i][k][1]
    )
    finding_to_truth: dict[int, int] = {}
    taken: set[int] = set()
    for _, i, k in pairs:
        if i not in finding_to_truth and k not in taken:
            finding_to_truth[i] = k
            taken.add(k)

    matches, outcomes = [], []
    for i, f in enumerate(findings):
        significant = f.passed_correction
        if i in finding_to_truth:
            k = finding_to_truth[i]
            spec = truths[k].spec
            true_effect = spec.direction * spec.effect
            matches.append(Match(
                finding_id=f.finding_id, spec_id=spec.spec_id, jaccard=tests[i][k][0],
                effect_car=f.effect_car, true_effect=true_effect,
                effect_error=f.effect_car - true_effect,
            ))
            outcomes.append(FindingOutcome(
                finding_id=f.finding_id, status="matched", significant=significant,
                spec_id=spec.spec_id, jaccard=tests[i][k][0],
            ))
            continue
        if not truths:
            outcomes.append(
                FindingOutcome(finding_id=f.finding_id, status="unmatched", significant=significant)
            )
            continue

        # The closest truth: one passing every test, then one passing event type and
        # Jaccard, then the same event type, then the highest Jaccard.
        def closeness(k, i=i):
            j, failed = tests[i][k]
            structural = not {"wrong_event_type", "low_jaccard"} & set(failed)
            return (not failed, structural, "wrong_event_type" not in failed, j, -k)

        k = max(range(len(truths)), key=closeness)
        j, failed = tests[i][k]
        if not failed:
            status, reasons = "duplicate", ["truth_already_matched"]
        elif set(failed) <= NEAR_MISS_REASONS:
            status, reasons = "near_miss", failed
        else:
            status, reasons = "unmatched", failed
        outcomes.append(FindingOutcome(
            finding_id=f.finding_id, status=status, significant=significant,
            spec_id=truths[k].spec.spec_id, jaccard=j, reasons=reasons,
        ))

    return MatchResult(
        universe_id=record.universe_id, universe_type=record.universe_type,
        jaccard_min=jaccard_min, n_truths=len(truths), matches=matches, outcomes=outcomes,
        missed_spec_ids=[t.spec.spec_id for k, t in enumerate(truths) if k not in taken],
    )
