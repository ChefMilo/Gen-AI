"""Offline cost estimate for an experiment plan (evaluation.cost). No API calls.

    python scripts/estimate_cost.py                              # configs/final.yaml
    python scripts/estimate_cost.py --model claude-opus-5-5
    python scripts/estimate_cost.py --input-tokens 250000 --output-tokens 30000 \
        --tokens r1=20000,4000 --input-price 2 --output-price 10

Sessions per system come from the plan (the same planning as --dry-run; universes are
not built). Tokens per session default to PLACEHOLDERS (no agent traces exist yet).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.cost import (  # noqa: E402
    LLM_FREE_SYSTEMS,
    MODEL_PRICES,
    PLACEHOLDER_OUTPUT_TOKENS,
    cost_table,
    placeholder_input_tokens,
    sessions_per_system,
)
from evaluation.plan import REPO_ROOT, load_experiment_config, plan_experiment  # noqa: E402
from evaluation.run import load_base_data  # noqa: E402


def _override(text: str) -> tuple[str, tuple[int, int]]:
    name, _, pair = text.partition("=")
    tin, _, tout = pair.partition(",")
    return name.strip(), (int(tin), int(tout))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "final.yaml")
    parser.add_argument("--systems", help="comma-separated; default: the config's names")
    parser.add_argument("--model", default="claude-sonnet-5-5", choices=sorted(MODEL_PRICES),
                        help="list price to use (default: %(default)s)")
    parser.add_argument("--input-price", type=float, help="USD per Mtok; overrides --model")
    parser.add_argument("--output-price", type=float, help="USD per Mtok; overrides --model")
    parser.add_argument("--input-tokens", type=int, default=None,
                        help="input tokens per session (default: PLACEHOLDER = "
                             "agent.max_session_input_tokens, the per-session cap)")
    parser.add_argument("--output-tokens", type=int, default=PLACEHOLDER_OUTPUT_TOKENS,
                        help="output tokens per session (default: PLACEHOLDER %(default)s)")
    parser.add_argument("--tokens", action="append", default=[], metavar="SYSTEM=IN,OUT",
                        help="per-system tokens per session, e.g. r1=20000,4000")
    args = parser.parse_args(argv)

    cfg = load_experiment_config(args.config).resolved()
    names = [n.strip() for n in (args.systems.split(",") if args.systems
                                 else cfg.system_names) if n.strip()]
    data = load_base_data(cfg)
    plan = plan_experiment(data.prices, data.events, cfg, data.benchmarks)
    sessions = sessions_per_system(plan, names)

    placeholder = args.input_tokens is None
    tin = placeholder_input_tokens() if placeholder else args.input_tokens
    tokens = {name: (tin, args.output_tokens) for name in names}
    tokens.update(dict(_override(t) for t in args.tokens))
    pin, pout = MODEL_PRICES[args.model]
    pin = pin if args.input_price is None else args.input_price
    pout = pout if args.output_price is None else args.output_price

    print(f"plan {cfg.experiment_id}: {len(plan.universes)} universes "
          f"({len(plan.by_type('planted'))} planted, {len(plan.by_type('placebo'))} placebo, "
          f"{len(plan.by_type('real'))} real) = sessions per system")
    skipped = [n for n in names if n in LLM_FREE_SYSTEMS]
    if skipped:
        print(f"LLM-free, not costed: {', '.join(skipped)}")
    if placeholder or args.output_tokens == PLACEHOLDER_OUTPUT_TOKENS:
        print(f"PLACEHOLDER tokens/session: input {tin:,}, output {args.output_tokens:,} "
              "(no agent traces exist yet; pass measured values)")
    print(f"price: ${pin}/Mtok in, ${pout}/Mtok out ({args.model} list price unless "
          "overridden; no caching discount)")
    table = cost_table(sessions, tokens, pin, pout)
    print(table.to_string(index=False, formatters={
        "input_tokens": "{:,.0f}".format, "output_tokens": "{:,.0f}".format,
        "tokens": "{:,.0f}".format, "sessions": "{:,.0f}".format, "usd": "${:,.2f}".format}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
