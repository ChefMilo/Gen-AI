"""Offline paired guardrail stress test. Scripted clients, NOT LLM discovery evidence.

python -m harness.offline --config configs/person4_offline.yaml --out-dir results/person4-offline
"""

import argparse
import hashlib
import importlib.metadata
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.llm import FakeLLM, LLMResponse, ToolCall
from backtest.portfolio import PortfolioConfig
from backtest.selection import select_events
from dsl.signal import SignalSpec
from evaluation.metrics import evaluate, metrics_table
from evaluation.plan import UniverseEntry
from evaluation.stats import compare_systems, system_summary
from evaluation.systems import make_view
from evaluation.universe_store import BaseData, build_universe
from harness.experiment import AgentResearchSystem
from harness.limits import ModelLimits, ResourceLimitedClient
from rag.index import HybridIndex
from rag.retriever import Retriever
from universes.planted import PlantSpec
from universes.synthetic import make_synthetic_benchmarks, make_synthetic_prices

ROOT = Path(__file__).resolve().parents[2]


class OfflineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_kind: str = "offline_scripted_stress_test"
    research_start: date = date(2020, 1, 2)
    research_end: date = date(2020, 3, 31)
    n_planted: int = Field(default=10, ge=0)
    n_placebo: int = Field(default=10, ge=0)
    n_tickers: int = Field(default=80, ge=80, le=400)
    base_seed: int = 0
    backtest_budget: int = Field(default=3, ge=1, le=8)
    n_boot: int = Field(default=500, ge=1)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)

    @model_validator(mode="after")
    def validate_design(self):
        if self.experiment_kind != "offline_scripted_stress_test":
            raise ValueError("This runner is only an offline scripted stress test")
        if self.n_tickers % 4 or self.n_planted + self.n_placebo < 1:
            raise ValueError("Use a multiple of four tickers and at least one universe")
        if (self.research_end - self.research_start).days < 40:
            raise ValueError("The demonstration needs at least 40 calendar days")
        return self


class OfflineEmbedder:
    name = "offline-hashed-bag-of-words-v1"

    def vector(self, text):
        result = np.zeros(64, dtype=np.float32)
        for token in text.lower().split():
            result[int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 64] += 1
        return result / (np.linalg.norm(result) or 1)

    def encode_documents(self, texts):
        return np.stack([self.vector(text) for text in texts]) if texts else np.zeros((0, 64))

    def encode_query(self, text):
        return self.vector(text)


def demonstration_data(config: OfflineConfig, seed: int) -> BaseData:
    tickers = [f"D{i:03d}" for i in range(config.n_tickers)]
    prices = make_synthetic_prices(tickers, config.research_start, config.research_end, seed)
    # This is an explicitly clean synthetic fixture, not filtered real data.
    prices["is_bad_price"] = False
    prices["bad_price_reason"] = None
    benchmarks = make_synthetic_benchmarks(config.research_start, config.research_end, seed)
    days = pd.DatetimeIndex(benchmarks.loc[benchmarks["ticker"].eq("SPY"), "date"])
    event_day = days[10]
    rows = []
    quarter = config.n_tickers // 4
    for i, ticker in enumerate(tickers):
        dividend = i < config.n_tickers // 2
        small = i % (2 * quarter) < quarter
        event_type = "dividend" if dividend else "buyback"
        quote = (
            "OFFLINE SYNTHETIC: the board raised its regular dividend by 15 percent."
            if dividend
            else "OFFLINE SYNTHETIC: the board authorised a new share buyback."
        )
        feature = "dividend_framing" if dividend else "buyback_rationale"
        label = ("confidence" if small else "routine") if dividend else "excess_capital"
        rows.append(
            {
                "event_id": f"demo-{ticker}",
                "ticker": ticker,
                "event_type": event_type,
                "acceptance_datetime": event_day + pd.Timedelta(hours=9),
                "first_tradeable_date": event_day.date(),
                "announcement_date": event_day.date(),
                "record_date": None,
                "payable_date": None,
                "ex_date": None,
                "buyback_expiry": None,
                "is_new_announcement": True,
                "extraction_status": "passed",
                "dividend_change_pct": 15.0 if dividend else None,
                "dividend_per_share": 0.115 if dividend else None,
                "dividend_kind": "regular" if dividend else None,
                "buyback_pct_mcap": (8.0 if small else 3.0) if not dividend else None,
                "buyback_amount_usd": 40_000_000.0 if not dividend else None,
                "is_new_program": not dividend,
                "mcap_bucket": "small" if small else "mid",
                "mcap_usd": 500_000_000.0 if small else 5_000_000_000.0,
                "text_features": {feature: {"label": label, "quote": quote}},
                "evidence": {"announcement": quote},
            }
        )
    return BaseData(prices, benchmarks, pd.DataFrame(rows))


def signal_grid() -> list[dict]:
    return [
        {"event_type": "dividend"},
        {
            "event_type": "dividend",
            "conditions": [{"field": "dividend_change_pct", "op": ">", "value": 10}],
        },
        {
            "event_type": "dividend",
            "conditions": [{"field": "mcap_bucket", "op": "==", "value": "small"}],
        },
        {
            "event_type": "dividend",
            "text_filter": {"feature": "dividend_framing", "equals": "confidence"},
        },
        {"event_type": "buyback"},
        {
            "event_type": "buyback",
            "conditions": [{"field": "buyback_pct_mcap", "op": ">", "value": 5}],
        },
        {
            "event_type": "buyback",
            "conditions": [{"field": "is_new_program", "op": "==", "value": True}],
        },
        {
            "event_type": "dividend",
            "conditions": [{"field": "dividend_kind", "op": "==", "value": "special"}],
        },
    ]


def scripted_client(view, seed):
    del seed  # same fixed requests in both arms; no universe type/truth is available
    requests = [("search_announcements", {"query": "dividend announcement", "k": 3})]
    for i, spec in enumerate(signal_grid(), 1):
        selected = select_events(view.events, SignalSpec.model_validate(spec))
        evidence = []
        if not selected.empty:
            row = selected.iloc[0]
            evidence = [{"event_id": row["event_id"], "quote": row["evidence"]["announcement"]}]
        requests.extend(
            [
                (
                    "record_hypothesis",
                    {
                        "hypothesis": f"Predefined stress-test signal {i}",
                        "rationale": "Offline plumbing test, not an LLM-generated hypothesis",
                    },
                ),
                ("define_signal", {"spec": spec, "hypothesis_id": f"h{i}"}),
                ("run_backtest", {"signal_id": f"sig-{i}"}),
                (
                    "report_finding",
                    {
                        "signal_id": f"sig-{i}",
                        "rationale": "Report the predefined test result",
                        "evidence": evidence,
                    },
                ),
            ]
        )
    requests.append(("finish", {"reason": "Offline script finished"}))
    script = [
        LLMResponse(
            tool_calls=[ToolCall(id=f"call-{i}", name=name, arguments=arguments)],
            stop_reason="tool_use",
        )
        for i, (name, arguments) in enumerate(requests)
    ]
    return ResourceLimitedClient(FakeLLM(script), ModelLimits(), lambda *args: 0)


def offline_retriever(view, config):
    docs = []
    for row in view.events.to_dict(orient="records"):
        docs.append(
            {
                "doc_ref": f"{row['ticker']}/{row['event_id']}/offline",
                "ticker": row["ticker"],
                "company_name": row["ticker"],
                "accession_no": row["event_id"],
                "items": "8.01",
                "filing_date": pd.Timestamp(row["acceptance_datetime"]).normalize(),
                "is_div_candidate": row["event_type"] == "dividend",
                "is_buyback_candidate": row["event_type"] == "buyback",
                "text": row["evidence"]["announcement"],
            }
        )
    index = HybridIndex.build(
        pd.DataFrame(docs), OfflineEmbedder(), max_chars=500, candidates=100, rrf_k=60
    )
    return Retriever(index, [], cutoff=config.research_end)


def run_offline(config: OfflineConfig, output_dir: Path) -> dict:
    """Run paired arms on identical opaque universe views and seed values.

    Uses in-memory universe construction rather than evaluation.run's Git-based
    output check. It neither suppresses nor monkeypatches that existing check.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config_json = config.model_dump(mode="json")
    (output_dir / "config.json").write_text(json.dumps(config_json, indent=2), encoding="utf-8")
    kinds = ["planted"] * config.n_planted + ["placebo"] * config.n_placebo
    np.random.default_rng(config.base_seed).shuffle(kinds)
    scores, design = [], []
    systems = [
        AgentResearchSystem(
            name,
            scripted_client,
            output_dir / "runs",
            offline_retriever,
            portfolio=config.portfolio,
            client_kind="scripted",
        )
        for name in ("R3", "R3-U")
    ]
    for i, kind in enumerate(kinds):
        seed = config.base_seed + i
        base = demonstration_data(config, seed)
        spec = PlantSpec(
            spec_id="demo-small-dividend-effect",
            event_type="dividend",
            condition="mcap_bucket == 'small'",
            effect=0.02,
            direction=1,
            seed=seed,
            strength="strong",
            complexity="single",
            feature_type="structured",
        )
        entry = UniverseEntry(
            universe_id=f"offline-u{i:03d}",
            universe_type=kind,
            seed=seed,
            holding_days=5,
            specs=[spec] if kind == "planted" else [],
            min_events=20 if kind == "planted" else None,
            min_gap_days=5 if kind == "placebo" else None,
        )
        universe = build_universe(entry, base)
        view = make_view(
            entry.universe_id,
            universe.prices,
            universe.benchmarks,
            universe.events,
            config.research_start,
            config.research_end,
        )
        design.append(
            {
                "universe_id": entry.universe_id,
                "kind": kind,
                "seed": seed,
                "record": universe.record.model_dump(mode="json"),
            }
        )
        for system in systems:
            result = system.run(view, config.backtest_budget, seed)
            scores.append(
                evaluate(system.name, universe.record, result.findings, metadata=result.metadata)
            )
    tables = metrics_table(scores)
    summaries = system_summary(tables, n_boot=config.n_boot, seed=config.base_seed)
    comparisons = compare_systems(tables, n_boot=config.n_boot, seed=config.base_seed)
    tables.per_universe.to_csv(output_dir / "per_universe.csv", index=False)
    tables.breakdowns.to_csv(output_dir / "recall_breakdowns.csv", index=False)
    tables.truths.to_csv(output_dir / "truths.csv", index=False)
    summaries.to_csv(output_dir / "system_summary.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    (output_dir / "universe_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    sources = {}
    for pattern in (
        "src/backtest/*.py",
        "src/harness/*.py",
        "src/agent/loop.py",
        "src/agent/prompts.py",
    ):
        for path in sorted(ROOT.glob(pattern)):
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "experiment_kind": config.experiment_kind,
        "paid_model_calls": 0,
        "research_dates_are_demonstration_only": True,
        "paired_sessions": 2 * len(kinds),
        "universe_count": len(kinds),
        "created_at": datetime.now(UTC).isoformat(),
        "code_sha256": sources,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "pydantic")
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# Person 4 offline guardrail verification",
        "",
        "**Scripted stress test. This is not evidence of real LLM discovery performance.**",
        "",
        "Both arms used identical data, seeds and predefined requests. R3 enforced a budget "
        "and Holm correction; R3-U disabled those two controls. Both retained dates, "
        "minimum sample size, source checks, audit and model-resource limits.",
        "",
        "Near-duplicate hypotheses deliberately stress correction; this script is not "
        "a representative research agent. All company data and dates are synthetic.",
        "",
        "| System | Planted recall | Precision | Mean placebo false discoveries |",
        "|---|---:|---:|---:|",
    ]
    for row in summaries.to_dict(orient="records"):
        lines.append(
            f"| {row['system']} | {row['recall']:.3f} | "
            f"{row['precision']:.3f} | {row['mean_false_discoveries']:.3f} |"
        )
    lines += [
        "",
        f"Universes: {len(kinds)}. Sessions: {2 * len(kinds)}. Paid API calls: 0.",
        "",
        "Portfolio results, costs, daily NAV, trades and nominal-trial DSR are in each "
        "run.json and audit. Event CAR includes event day; portfolio exposure starts "
        "after entry close. DSR is diagnostic only and assumes the supplied trial "
        "dispersion/nominal count; it does not correct serial dependence.",
        "",
        "Next: agree the real research/holdout dates, configure the shared dataset/index, "
        "and explicitly approve a small live-model budget. Do not report this table "
        "as the project's empirical R3/R3-U result.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/person4_offline.yaml")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.out_dir.resolve()
    if (
        not output.is_relative_to((ROOT / "results").resolve())
        or output == (ROOT / "results").resolve()
    ):
        parser.error("Choose a NEW subdirectory of this repository's results folder")
    config = OfflineConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    manifest = run_offline(config, output)
    print(f"Completed {manifest['paired_sessions']} scripted sessions; paid API calls: 0")
    print(f"Report: {output / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
