"""INTERIM: stands in for Member 4's backtest engine and harness until they land.

build_ports(view, ...) turns one universe view (evaluation.systems.UniverseView, or
anything with its attributes) into the agent's ports, and registers the matching tools.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from agent.fakes import EVENT_TYPES
from agent.interim.backtest import InterimBacktester
from agent.interim.data import (
    as_date,
    breakdowns,
    field_table,
    research_events,
    text_feature_table,
)
from agent.interim.events import InterimEventQuery
from agent.interim.retrieval import (
    RagFactory,
    make_retriever,
    retriever_cutoff,
    shared_index_retriever,
)
from agent.interim.spec import SpecModels, make_spec_models
from agent.interim.tools import register_interim_tools

NO_RESEARCH_END = "no_research_end"


@dataclass
class InterimPorts:
    backtester: InterimBacktester
    events: InterimEventQuery
    retriever: Any | None
    retriever_kind: str | None  # "fake", "rag" or None (R2)
    models: SpecModels
    research_start: date | None
    research_end: date | None
    retriever_cutoff: date | None
    warnings: list[str] = field(default_factory=list)


def build_ports(
    view: Any,
    *,
    budget: int,
    holding_days: int,
    min_events: int,
    with_retriever: bool,
    mask_dates: bool,
    rag_factory: RagFactory = shared_index_retriever,
) -> InterimPorts:
    start, end = as_date(view.research_start), as_date(view.research_end)
    events = research_events(view.events, start, end)
    fields = field_table(events)
    text = text_feature_table(events)
    models = make_spec_models(fields, text, EVENT_TYPES, (holding_days,))
    backtester = InterimBacktester(events, view.prices, view.benchmarks, budget, holding_days,
                                   min_events, models.signal_spec, research_end=end)
    retriever, kind = (make_retriever(events, end, mask_dates, rag_factory)
                       if with_retriever else (None, None))
    register_interim_tools(models, EVENT_TYPES, breakdowns(events), tuple(text),
                           mask_dates=mask_dates)
    return InterimPorts(
        backtester=backtester,
        events=InterimEventQuery(events, fields, models.signal_spec),
        retriever=retriever,
        retriever_kind=kind,
        models=models,
        research_start=start,
        research_end=end,
        retriever_cutoff=retriever_cutoff(events, end) if with_retriever else None,
        warnings=[] if end is not None else [NO_RESEARCH_END],
    )
