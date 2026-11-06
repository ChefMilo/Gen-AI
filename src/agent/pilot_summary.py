"""Summarise an experiment run's LLM systems under every correction rule (no API calls).

    python -m agent.pilot_summary results/dev_agent_v4
    python -m agent.pilot_summary results/final-v1

Reads plan.json, universes/<id>/record.json and runs/<system>/<id>/ (findings.jsonl and
metadata.json) as the runner writes them, and prints per system:
- recall, precision and mean placebo false discoveries under four rules:
  official (each finding's own passed_correction), uniform (Member 5's: Holm over
  max(budget, findings), evaluation.metrics.recorrect), attempted (Holm over the tests
  the session actually ran) and strict (attempted plus every breakdown group viewed);
- tokens by type, cost (agent.pricing) and backtests used;
- end reasons, and R1's parse statistics.
Recall = matched truths / truths and precision = matched / significant findings, both
pooled over the planted universes; false discoveries = mean per placebo universe.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from agent.config import load_config
from agent.finding import Finding
from agent.system import cost_usd
from evaluation.matching import match_findings
from evaluation.metrics import UniformCorrection, recorrect
from evaluation.plan import load_plan
from evaluation.universe_record import UniverseRecord

RULES = ("official", "uniform", "attempted", "strict")
TOKEN_FIELDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                "output_tokens")


def _flagged(findings: list[Finding], meta: dict, rule: str) -> list[Finding] | None:
    """Copies with passed_correction from the session's sensitivity values; None when the
    run predates them (older metadata), so the rule cannot be scored."""
    rows = {r["finding_id"]: r for r in meta.get("strict_correction", [])}
    if any(f"passed_correction_{rule}" not in rows.get(f.finding_id, {}) for f in findings):
        return None
    out = []
    for f in findings:
        row = rows.get(f.finding_id)
        passed = bool(row and row.get(f"passed_correction_{rule}"))
        out.append(f.model_copy(update={"passed_correction": passed}))
    return out


def load_pairs(out_dir: Path) -> tuple[list[dict], dict[str, UniverseRecord]]:
    plan = load_plan(out_dir / "plan.json")
    records = {u.universe_id: UniverseRecord.model_validate_json(
        (out_dir / "universes" / u.universe_id / "record.json").read_text(encoding="utf-8"))
        for u in plan.universes}
    pairs = []
    for meta_path in sorted((out_dir / "runs").glob("*/*/metadata.json")):
        run = json.loads(meta_path.read_text(encoding="utf-8"))
        if run.get("status") != "ok" or run["universe_id"] not in records:
            continue
        lines = (meta_path.parent / "findings.jsonl").read_text(encoding="utf-8").splitlines()
        pairs.append({"system": run["system"], "universe_id": run["universe_id"],
                      "meta": run.get("metadata", {}),
                      "findings": [Finding.model_validate_json(x) for x in lines if x.strip()]})
    return pairs, records


def _tokens(meta: dict) -> dict[str, int]:
    breakdown = meta.get("token_breakdown", {})
    return {f: int(breakdown.get(f, 0)) for f in TOKEN_FIELDS}


def summarize(out_dir: Path, budget: int | None = None, alpha: float = 0.05,
              jaccard_min: float | None = None) -> pd.DataFrame:
    """One row per (system, rule), plus the per-system cost and run columns. A rule a run
    has no values for (older metadata) is NaN; a missing cost is recomputed from the tokens
    with agent.pricing."""
    pricing = load_config()["agent"].get("pricing") or {}
    pairs, records = load_pairs(Path(out_dir))
    # Systems without an agent harness (R0, the stubs) record no budget: use the plan's.
    plan_budget = load_plan(Path(out_dir) / "plan.json").config.backtest_budget
    rows = []
    for system in sorted({p["system"] for p in pairs}):
        mine = [p for p in pairs if p["system"] == system]
        tokens = {f: sum(p["meta"].get("token_breakdown", {}).get(f, 0) for p in mine)
                  for f in TOKEN_FIELDS}
        costs = [p["meta"].get("cost_usd") if p["meta"].get("cost_usd") is not None
                 else cost_usd(p["meta"].get("model"), _tokens(p["meta"]), pricing)
                 for p in mine]
        base = {
            "system": system, "sessions": len(mine),
            "backtests_used": sum(p["meta"].get("backtests_used", 0) for p in mine),
            **tokens, "tokens": sum(tokens.values()),
            "cost_usd": sum(costs) if costs and all(c is not None for c in costs) else None,
            "end_reasons": dict(Counter(p["meta"].get("end_reason") for p in mine)),
        }
        if any("specs_proposed" in p["meta"] for p in mine):  # R1
            for key in ("specs_proposed", "specs_invalid", "specs_over_budget",
                        "parse_retries"):
                base[key] = sum(p["meta"].get(key, 0) for p in mine)
            base["parse_failed"] = sum(p["meta"].get("end_reason") == "parse_failed"
                                       for p in mine)
        for rule in RULES:
            truths = matched = significant = 0
            placebo, false_discoveries = 0, 0
            available = True
            for p in mine:
                record = records[p["universe_id"]]
                if rule == "official":
                    findings = p["findings"]
                elif rule == "uniform":
                    b = budget or p["meta"].get("budget") or plan_budget
                    findings = recorrect(p["findings"], UniformCorrection(budget=b, alpha=alpha))
                else:
                    findings = _flagged(p["findings"], p["meta"], rule)
                    if findings is None:
                        available = False
                        break
                match = match_findings(findings, record, jaccard_min)
                if record.universe_type == "planted":
                    truths += match.n_truths
                    matched += len(match.matches)
                    significant += match.n_significant
                elif record.universe_type == "placebo":
                    placebo += 1
                    false_discoveries += len(match.with_status("false_discovery"))
            if not available:
                rows.append({**base, "rule": rule, "recall": float("nan"),
                             "precision": float("nan"),
                             "placebo_false_discoveries": float("nan")})
                continue
            rows.append({**base, "rule": rule,
                         "recall": matched / truths if truths else float("nan"),
                         "precision": matched / significant if significant else float("nan"),
                         "placebo_false_discoveries": (false_discoveries / placebo
                                                       if placebo else float("nan"))})
    return pd.DataFrame(rows)


def report(table: pd.DataFrame) -> str:
    if table.empty:
        return "no finished pairs found"
    scores = table.pivot(index="system", columns="rule",
                         values=["recall", "precision", "placebo_false_discoveries"])
    scores = scores.reindex(columns=list(RULES), level=1)
    per_system = table.drop_duplicates("system").set_index("system")
    cols = ["sessions", "backtests_used", *TOKEN_FIELDS, "tokens", "cost_usd", "end_reasons"]
    r1_cols = [c for c in ("specs_proposed", "specs_invalid", "specs_over_budget",
                           "parse_retries", "parse_failed") if c in per_system]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        parts = ["Scores by correction rule (pooled over planted / mean over placebo):",
                 scores.round(3).to_string(), "", "Tokens and cost:",
                 per_system[cols].to_string()]
        if r1_cols:
            parts += ["", "R1 parsing:", per_system[r1_cols].dropna(how="all").to_string()]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.pilot_summary", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="the run's output folder")
    parser.add_argument("--budget", type=int,
                        help="uniform rule's budget (default: each session's, else the plan's)")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    print(report(summarize(args.results, args.budget, args.alpha)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
