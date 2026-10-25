"""Hand-built truths, findings and records for the evaluation tests."""

from agent.finding import EvidenceQuote, Finding
from dsl.signal import SignalSpec
from evaluation.universe_record import UniverseRecord, planted_record
from universes.placebo import PlaceboInfo
from universes.planted import PlantSpec, PlantTruth

J = 0.6


def ids(prefix: str, n: int, start: int = 0) -> list[str]:
    return [f"{prefix}{i}" for i in range(start, start + n)]


def truth(
    spec_id: str = "t1",
    targets: list[str] | None = None,
    unmeasurable: list[str] | None = None,
    event_type: str = "dividend",
    direction: int = 1,
    effect: float = 0.02,
    strength: str = "moderate",
    complexity: str = "single",
    feature_type: str = "structured",
) -> PlantTruth:
    targets = ids("e", 10) if targets is None else targets
    spec = PlantSpec(
        spec_id=spec_id, event_type=event_type, condition="x > 0", effect=effect,
        direction=direction, strength=strength, complexity=complexity,
        feature_type=feature_type,
    )
    return PlantTruth(
        spec=spec, target_event_ids=targets, n_targets=len(targets), overlaps=[],
        unmeasurable_event_ids=unmeasurable or [],
    )


def finding(
    finding_id: str = "f1",
    event_ids: list[str] | None = None,
    event_type: str = "dividend",
    effect_car: float = 0.02,
    passed: bool = True,
    system: str = "R3",
) -> Finding:
    event_ids = ids("e", 10) if event_ids is None else event_ids
    return Finding(
        finding_id=finding_id, session_id="s1", system=system, signal_id=f"sig-{finding_id}",
        signal_spec=SignalSpec(event_type=event_type), event_type=event_type,
        event_ids=event_ids, n_events=len(event_ids),
        direction="long" if effect_car >= 0 else "short", effect_car=effect_car,
        t_stat=3.0, p_value_raw=0.001, p_value_corrected=0.01 if passed else 0.2,
        passed_correction=passed, rationale="test",
        evidence=[EvidenceQuote(event_id=event_ids[0] if event_ids else "e0", quote="q")],
    )


def planted(*truths: PlantTruth) -> UniverseRecord:
    return planted_record("u1", list(truths))


def placebo_info(seed: int = 0) -> PlaceboInfo:
    return PlaceboInfo(
        seed=seed, min_gap_days=10, holding_days=5, n_eligible=0, n_moved=0,
        original_first_tradeable_date={}, valid_days={}, dropped=[],
        shift_trading_days={}, date_adjustments=[],
    )
