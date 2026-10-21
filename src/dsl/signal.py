"""Signal DSL schema (interface 2 in docs/design.md section 7)."""

from pydantic import BaseModel

from store.schema import DEFAULT_BENCHMARK, Benchmark, EventType


# TODO(member-4): placeholder only. Member 4 owns the full SignalSpec design
# (where clause, text filter, entry rule, holding window, direction, validator).
# Fixed by docs/data_schema.md: event types are dividend | buyback, the default
# benchmark is SPY, and entry is never before first_tradeable_date.
class SignalSpec(BaseModel):
    event_type: EventType
    benchmark: Benchmark = DEFAULT_BENCHMARK
