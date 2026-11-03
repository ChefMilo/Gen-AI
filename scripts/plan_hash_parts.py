"""Build a plan and print its hash part by part, to find where two machines differ.

    python scripts/plan_hash_parts.py                      # configs/final.yaml
    python scripts/plan_hash_parts.py --out plan.json      # also write the canonical JSON

Prints plan_sha256, then a short hash of each top-level field of the canonical plan JSON
(and of each fingerprint field), so two runs can be compared line by line.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.plan import (  # noqa: E402
    REPO_ROOT,
    load_experiment_config,
    plan_canonical_json,
    plan_experiment,
    plan_sha256,
)
from evaluation.run import load_base_data  # noqa: E402


def _short(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "final.yaml")
    parser.add_argument("--out", type=Path, help="write the canonical plan JSON here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)

    cfg = load_experiment_config(args.config).resolved()
    data = load_base_data(cfg)
    plan = plan_experiment(data.prices, data.events, cfg, data.benchmarks)
    text = plan_canonical_json(plan)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    doc = json.loads(text)
    print(f"plan_sha256 {plan_sha256(plan)}")
    for key in sorted(doc):
        print(f"  {key:16s} {_short(doc[key])}")
    for key in sorted(doc["fingerprint"]):
        print(f"  fingerprint.{key:16s} {doc['fingerprint'][key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
