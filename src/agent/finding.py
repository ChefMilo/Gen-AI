"""Finding format (interface 5 in docs/design.md section 7)."""

from typing import Literal

from pydantic import BaseModel, model_validator

from dsl.signal import SignalSpec
from store.schema import EventType

System = Literal["R0", "R1", "R2", "R3", "R3-U"]
Direction = Literal["long", "short"]


class EvidenceQuote(BaseModel):
    """An exact quote, tied to an event or to a retrieved document."""

    event_id: str | None = None
    doc_ref: str | None = None
    quote: str

    @model_validator(mode="after")
    def _needs_source(self) -> "EvidenceQuote":
        if self.event_id is None and self.doc_ref is None:
            raise ValueError("evidence needs an event_id or a doc_ref")
        return self


# TODO(member-3): owner confirms. Placeholder agreed with Member 5 (Evaluation),
# which matches findings to planted signals on event_ids.
class Finding(BaseModel):
    """One reported signal.

    effect_car is the raw mean event CAR over event_ids, signed as computed: negative
    when prices fell after the events. It is not the return of a long/short position
    (the harness's directional_mean_car is not used here). direction is a separate
    field: the position taken. The stub systems and R0 set it from the sign of
    effect_car; the agent harness copies it from the signal spec, so it can disagree
    with that sign. Evaluation (evaluation.matching) tests sign(effect_car) against the
    planted direction and ignores direction.
    """

    finding_id: str
    session_id: str
    system: System
    signal_id: str
    signal_spec: SignalSpec
    event_type: EventType
    event_ids: list[str]
    n_events: int
    direction: Direction
    effect_car: float
    t_stat: float
    p_value_raw: float
    p_value_corrected: float
    passed_correction: bool
    rationale: str
    evidence: list[EvidenceQuote]


# What the agent submits with report_finding. Per the team design (docs/design.md 3.3,
# step 7 of section 4), the harness turns claims into full Finding objects at session
# end: corrections need every test the session ran, so the agent cannot compute them.
class FindingClaim(BaseModel):
    signal_id: str
    rationale: str
    evidence: list[EvidenceQuote]
    step: int
