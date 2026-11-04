"""Result figures from a summary. Reads summary/ only, plus an optional oracle.csv.

    python scripts/make_figures.py --results results/<experiment_id>
    python scripts/make_figures.py --results results/final-v1 \
        --oracle results/final-v1/oracle.csv --rule uniform

Writes <name>.png and <name>.svg to <results>/figures/ (<name>_uniform with --rule uniform).
--input-price and --output-price (USD per million tokens) are only used by cost_vs_recall,
for systems whose cost_usd is empty.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.figures import FIGURES, RULES, cost_vs_recall, load_figure_data  # noqa: E402

DPI = 200


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="results/<experiment_id>")
    parser.add_argument("--oracle", type=Path, help="oracle.csv (scripts/oracle_check.py)")
    parser.add_argument("--rule", choices=RULES, default="own")
    parser.add_argument("--input-price", type=float, help="USD per million input tokens")
    parser.add_argument("--output-price", type=float, help="USD per million output tokens")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if not (args.results / "summary").is_dir():
        parser.error(f"no summary/ in {args.results}; run python -m evaluation.summarize first")
    if args.oracle is not None and not args.oracle.exists():
        parser.error(f"no such oracle file: {args.oracle}")
    start = time.perf_counter()
    data = load_figure_data(args.results, args.oracle)
    out = args.results / "figures"
    out.mkdir(exist_ok=True)
    suffix = "" if args.rule == "own" else f"_{args.rule}"
    for name, make in FIGURES.items():
        options = {}
        if make is cost_vs_recall:
            options = {"input_price": args.input_price, "output_price": args.output_price}
        figure = make(data, rule=args.rule, **options)
        figure.savefig(out / f"{name}{suffix}.png", dpi=DPI)
        figure.savefig(out / f"{name}{suffix}.svg", metadata={"Date": None})
        print(f"{out / f'{name}{suffix}'}.png, .svg")
    print(f"{len(FIGURES)} figures in {time.perf_counter() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
