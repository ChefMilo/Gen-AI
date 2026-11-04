"""Re-score every system with runs on disk and rewrite summary/. No system is run.

    python -m evaluation.summarize --results results/<experiment_id>

Reads plan.json, universes/*/record.json and runs/ under --results; prices are not needed.
"""

import argparse
import logging
import time
from pathlib import Path

from evaluation.plan import load_plan
from evaluation.run import PLAN_FILE, summarize_results, systems_on_disk
from evaluation.summary import INDEX_FILE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.summarize", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="results/<experiment_id>")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not (args.results / PLAN_FILE).exists():
        parser.error(f"no {PLAN_FILE} in {args.results}")
    start = time.perf_counter()
    plan = load_plan(args.results / PLAN_FILE)
    _, files = summarize_results(plan, args.results)
    print(f"systems on disk: {', '.join(systems_on_disk(args.results)) or '(none)'}")
    print(f"{len(files)} tables and {INDEX_FILE} -> {args.results / 'summary'} "
          f"in {time.perf_counter() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
