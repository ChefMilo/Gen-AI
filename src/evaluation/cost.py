"""Cost estimate for an experiment plan: LLM sessions per system x tokens x price.

Every system runs one session per universe in the plan. Systems that call no LLM
(LLM_FREE_SYSTEMS: the stubs and R0, whose fixed grid is pure pandas/scipy) cost
nothing and are left out.

No agent traces exist yet (no real-LLM pilot has been run), so tokens per session are
an input. The defaults are PLACEHOLDERS: input = agent.max_session_input_tokens from
configs/default.yaml (the per-session cap, so an upper bound) and a guessed output.
Replace them with measured means once traces exist. Prompt caching discounts are not
modelled, so cached input is priced at the full rate (an upper bound).
"""

from pathlib import Path

import pandas as pd
import yaml

from evaluation.plan import DEFAULT_CONFIG_PATH, ExperimentPlan

LLM_FREE_SYSTEMS = frozenset({"null", "random", "oracle", "r0"})

# USD per million tokens (input, output): Anthropic list prices, cached 2026-10-06.
MODEL_PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
PLACEHOLDER_OUTPUT_TOKENS = 40_000  # PLACEHOLDER: no measured value yet


def placeholder_input_tokens(path: Path = DEFAULT_CONFIG_PATH) -> int:
    """PLACEHOLDER: the per-session input cap, agent.max_session_input_tokens."""
    with open(path, encoding="utf-8") as f:
        return int(yaml.safe_load(f)["agent"]["max_session_input_tokens"])


def sessions_per_system(plan: ExperimentPlan, systems: list[str]) -> dict[str, int]:
    """One session per universe for every system."""
    return {name: len(plan.universes) for name in systems}


def cost_table(
    sessions: dict[str, int],
    tokens: dict[str, tuple[int, int]],
    input_price: float,
    output_price: float,
) -> pd.DataFrame:
    """Per system: sessions, input/output/total tokens and USD. tokens maps each system
    to (input, output) tokens per session; LLM-free systems are skipped."""
    rows = []
    for name, n in sessions.items():
        if name in LLM_FREE_SYSTEMS:
            continue
        tin, tout = tokens[name]
        rows.append({
            "system": name, "sessions": n,
            "input_tokens": n * tin, "output_tokens": n * tout,
            "tokens": n * (tin + tout),
            "usd": n * (tin * input_price + tout * output_price) / 1e6,
        })
    out = pd.DataFrame(rows, columns=["system", "sessions", "input_tokens", "output_tokens",
                                      "tokens", "usd"])
    if not out.empty:
        total = out.drop(columns="system").sum().to_dict()
        out = pd.concat([out, pd.DataFrame([{"system": "TOTAL", **total}])],
                        ignore_index=True)
    return out
