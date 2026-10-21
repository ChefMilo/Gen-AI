## **Solution Architecture: Agentic Event-Driven Alpha Research with Planted-Signal Evaluation**

> **Note (21 Oct):** the data source is now SEC EDGAR 8-Ks for 180 US S&P 400/600 dividend payers (dividend and buyback events, SPY benchmark). Where this doc mentions SGX, rights issues or STI, `docs/data_schema.md` takes precedence.

### **1. Overview**

The system is an **LLM research agent** that studies company announcements (dividends and corporate actions) and searches for event-driven return patterns. It forms hypotheses, retrieves supporting evidence from an announcement knowledge base (RAG), defines trading signals, runs backtests through tools, and reports findings.

Because “the agent found alpha” cannot be verified directly, the system is evaluated on **constructed ground truth**:

- **Planted universes:** real prices with known injected effects. Does the agent find them? (recall)

- **Placebo universes:** event dates shuffled so no real effect exists. Does the agent claim false discoveries? (false discovery rate)

- **Real universe with a locked holdout:** does an in-sample finding survive on a later period the agent never saw? (out-of-sample retention)

**Research question:** Can an LLM research agent with retrieval discover genuine event-driven effects more reliably than a fixed search pipeline, without fooling itself through data snooping, and do guardrails on its testing behaviour reduce false discoveries?

### **2. High-Level Architecture**

```text
┌───────────────────────────── DATA LAYER ─────────────────────────────┐
│ Source adapter (SGX manual │ EDGAR) → PDF/text → A+ extraction │
│ (structured output + rule checks) → EVENT STORE (point-in-time) │
│ Price data + benchmark index → PRICE STORE │
└───────────────┬──────────────────────────────────────┬───────────────┘
▼ ▼
┌──────────── KNOWLEDGE LAYER (RAG) ─────────┐ ┌──── UNIVERSE BUILDER ────┐
│ chunk index (dense + BM25 hybrid) │ │ real (research period) │
│ metadata filters: company · type · date │ │ planted (inject effects) │
│ time-aware: date ≤ research cutoff │ │ placebo (shuffle dates) │
│ text features: LLM classify + evidence │ │ holdout (locked) │
│ (e.g. rights-issue purpose, div. framing) │ └────────────┬─────────────┘
└───────────────────┬────────────────────────┘ │
▼ ▼
┌──────────────────────── GUARDRAIL HARNESS ───────────────────────────┐
│ time lockout · testing budget (N backtests) · min events per signal · │
│ multiple-testing correction on reported findings · step/cost caps · │
│ announcement text = DATA · full audit log │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ RESEARCH AGENT (API LLM, ReAct loop + research notebook memory) │ │
│ │ hypothesise → retrieve → define signal → backtest → interpret → │ │
│ │ refine / abandon / report │ │
│ └──┬───────────┬────────────┬─────────────┬────────────┬──────────┘ │
│ ▼ ▼ ▼ ▼ ▼ │
│ search\_ query\_ get_text\_ define\_ run_backtest │
│ announce- events feature signal inspect_results │
│ ments (structured) (DSL) report_finding │
└──────────────────────────────┬───────────────────────────────────────┘
▼
┌──────────────────────── BACKTEST ENGINE ─────────────────────────────┐
│ deterministic · unit-tested · event study (abnormal returns vs │
│ benchmark) · event strategy P&L with costs · stats: mean CAR, t-stat, │
│ hit rate, Sharpe, n events │
└──────────────────────────────┬───────────────────────────────────────┘
▼
┌──────────────────────── EVALUATION LAYER ────────────────────────────┐
│ baselines R0 (grid search) · R1 (single-shot LLM) · R2 (agent, no RAG)│
│ planted recall · placebo false discoveries · holdout retention · │
│ effect-size error · RAG metrics · cost · agent behaviour · statistics │
└──────────────────────────────┬───────────────────────────────────────┘
▼
┌───────────────────────────────┐
│ DEMO (Streamlit): live agent │
│ trace · findings + evidence · │
│ planted vs recovered view │
└───────────────────────────────┘
```

### **3. Components**

#### **3.1 Data Layer**

**Source adapter.** Default: about 400+ hand-downloaded SGX announcements (dividends, rights issues, bonus issues, consolidations), 2015–2026. Alternative: SEC EDGAR 8-K filings within EDGAR’s access rules. Only this adapter changes between the two.

**Extraction (A+, non-agentic).** Structured-output LLM call, then Pydantic schema validation, rule checks (trading days, date ordering, positive values, evidence grounding), and one retry with the error message. Failures are flagged and excluded.

- Data quality is checked against a hand-labelled gold set of **60 announcements** and reported as field-level F1 and numbers/dates exact accuracy.

**Event store (point-in-time).** One row per event:

| **Field**                          | **Notes**                                                                                      |
|------------------------------------|------------------------------------------------------------------------------------------------|
| event_id, company, event_type      | —                                                                                              |
| announce_datetime                  | **When the information became public.** Signals may only trade after this                      |
| ex_date, record_date, payment_date | —                                                                                              |
| amount / ratio / price, currency   | —                                                                                              |
| derived fields                     | e.g. dividend change vs prior period, special-dividend flag, market-cap bucket at announcement |
| text features                      | From the knowledge layer (3.2), each with its evidence quote                                   |
| extraction_status                  | passed or flagged                                                                              |

**Price store.** Daily adjusted prices for the company universe, plus a benchmark (STI for SGX). Confirm that the source’s terms permit academic use.

#### **3.2 Knowledge Layer (RAG)**

- **Index:** announcement text chunked by section, stored in a dense vector index plus a BM25 keyword index for **hybrid retrieval**.

- **Metadata filters:** company, event type, date. **Time-aware retrieval** is enforced by the harness: during research, only documents dated on or before the research cutoff are retrievable.

- **Structured retrieval:** query_events filters the event store (e.g. “rights issues by small caps, 2016–2021”).

- **Text-derived features**, produced by LLM classification over retrieved passages, each with an evidence quote:

  - Rights-issue purpose: debt repayment, expansion, working capital, acquisition

  - Dividend framing: sustainable, one-off, or “subject to performance”

  - Management outlook tone in the announcement: positive, neutral, cautious

- Validated against human labels on the 60-announcement gold set.

#### **3.3 Research Agent**

- **Model:** API LLM in a ReAct-style loop.

- **Memory:** a **research notebook** logging every hypothesis, signal definition, backtest result and decision, so the agent builds on or abandons earlier lines of enquiry.

**Tools**

| **Tool**                                       | **Purpose**                                                           |
|------------------------------------------------|-----------------------------------------------------------------------|
| search_announcements(query, filters)           | Hybrid semantic + keyword search over announcement text               |
| query_events(filters)                          | Structured query over the event store                                 |
| get_text_feature(event_ids, feature)           | Text-derived features with evidence                                   |
| define_signal(spec)                            | Register a signal in the signal DSL (below); validated by the harness |
| run_backtest(signal_id)                        | Event study + strategy P&L; **counts against the testing budget**     |
| inspect_results(signal_id)                     | Breakdown by year, sector, market cap; event list                     |
| report_finding(signal_id, rationale, evidence) | Submit a final claim; corrections are applied by the harness          |

**Signal DSL.** The agent defines signals as constrained JSON rather than arbitrary code. That makes every test countable, reproducible and safe to execute. Example:

```json
{
"event_type": "dividend",
"where": "dividend_change_pct > 10 AND mcap_bucket = 'small'",
"text_filter": {"feature": "dividend_framing", "equals": "sustainable"},
"entry": "close_after_announce",
"holding_days": 5,
"benchmark": "STI",
"direction": "long"
}
```

#### **3.4 Backtest Engine (deterministic)**

- **Event study:** market-adjusted abnormal returns over the holding window. Reports cumulative abnormal return (CAR), t-statistic, hit rate and number of events.

- **Strategy view:** equal-weight event portfolio with estimated transaction costs and board-lot constraints. Reports Sharpe ratio and drawdown.

- **Point-in-time rules:** entry is never before announce_datetime; holdout prices are inaccessible during research.

- **Unit-tested** on synthetic series with known answers before any agent touches it.

#### **3.5 Guardrail Harness**

| **Guardrail**                                                                                                  | **What it prevents**                       |
|----------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| Time lockout (research period vs holdout)                                                                      | Look-ahead bias; peeking at test data      |
| Time-aware retrieval                                                                                           | Agent reading post-cutoff announcements    |
| Testing budget (e.g. 30 backtests per session)                                                                 | Unlimited data snooping                    |
| Multiple-testing correction on reported findings (Holm–Bonferroni across all tests run; deflated Sharpe ratio) | Chance results reported as discoveries     |
| Minimum events per signal (e.g. ≥20)                                                                           | Conclusions from tiny samples              |
| DSL validation                                                                                                 | Malformed or unsafe signal definitions     |
| Announcement text passed as data                                                                               | Prompt injection inside documents          |
| Step and cost caps                                                                                             | Runaway sessions                           |
| Audit log                                                                                                      | Untraceable claims; enables trace analysis |

#### **3.6 Evaluation Layer**

**Universe builder**

- **Planted universes:** real prices with an injected abnormal return on a defined event subset. Effects vary along three axes:

  - **Strength:** e.g. 0.5%, 1%, 2%, 4% abnormal return.

  - **Complexity:** a single field condition (dividend increase) vs a conjunction (dividend increase AND small cap).

  - **Feature type:** structured-field signals vs **text-feature signals** (e.g. rights issues for debt repayment). Text-feature signals are only discoverable through RAG, which is what separates R3 from R2.

- **Placebo universes:** event dates permuted within each company, which destroys any real event–return link. Every reported finding is false by construction.

- **Real universe:** research on the earlier period, final test once on the locked later period, ideally after the API model’s knowledge cutoff.

- **Repeats:** about 10 planted and 10 placebo universes per system, each with different random seeds and planted specs. **Statistical power comes from the number of universes, not from real alpha.**

**Research systems compared**

| **ID** | **System**                                                                                           | **Question**                                     |
|--------|------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| R0     | Fixed pipeline: grid search over predefined structured features, same budget and corrections, no LLM | Deterministic baseline                           |
| R1     | Single LLM call proposes hypotheses once; harness tests them; no feedback                            | Does feedback from results matter?               |
| R2     | Agent with backtest tools, structured data only, no RAG                                              | Does iterative agency help?                      |
| R3     | Agent with tools + agentic RAG + text features                                                       | Does retrieval add discoveries?                  |
| R3-U   | R3 without testing budget or corrections                                                             | How often does an unconstrained agent overclaim? |

**Scoring a planted signal as recovered.** A reported finding matches a planted signal if **all** of the following hold:

- same event type,

- the finding’s event set overlaps the planted event set with Jaccard ≥ 0.6,

- the effect sign is correct,

- it passes the corrected significance threshold.

**Metrics**

- **Discovery:** planted recall (overall, by strength, complexity and feature type), precision, false discovery rate on placebo, effect-size estimation error.

- **Generalisation:** in-sample vs holdout effect on the real universe; sign agreement; retention ratio.

- **Efficiency:** backtests used, tool calls, tokens, cost and time per session.

- **Agent behaviour:** hypotheses tried, abandonment rate, how often findings cite retrieved evidence, budget exhaustion.

- **RAG:**

  - Retrieval recall@k on 30 hand-labelled queries

  - Text-feature accuracy vs gold labels

  - Evidence faithfulness, via LLM-as-judge validated on 50 human-rated items

  - Single-pass vs agentic RAG on the same queries

- **Data quality:** extraction F1 on the 60-announcement gold set.

- **Statistics:** recall and false discovery rate with bootstrap confidence intervals across universes; paired comparisons between systems on identical universes.

**Headline figure:** a detection curve showing planted recall vs effect strength for R0, R1, R2 and R3, alongside false discovery rates on placebo.

### **4. Run Flow (one research session)**

1.  The harness loads a universe (planted, placebo or real-research) and sets the budget and cutoff.

2.  The agent reads a task brief: event types available, research period, budget.

3.  The agent forms a hypothesis, retrieves supporting evidence (announcements, event counts, text features) and logs it to the notebook.

4.  It defines a signal in the DSL; the harness validates it.

5.  It runs the backtest (budget decremented) and inspects the breakdowns.

6.  It refines, abandons or pursues a new hypothesis, repeating until the budget or step cap is reached.

7.  It reports findings with rationale and evidence; the harness applies corrections and the minimum-events rule.

8.  **Evaluation (outside the agent’s reach):** findings are scored against planted specs, the placebo null, or the holdout period.

### **5. Technology Stack**

| **Layer**                | **Tools**                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------|
| Ingestion and extraction | PyMuPDF; API model structured output; Pydantic                                                 |
| Storage                  | Parquet or SQLite for events and prices; JSONL for traces and notebooks                        |
| RAG                      | Sentence-embedding model, vector index (e.g. FAISS or Chroma), BM25, metadata filtering        |
| Agent                    | Python ReAct loop with function calling; one API model                                         |
| Backtesting              | pandas, NumPy; custom event-study engine with pytest unit tests                                |
| Statistics               | SciPy, statsmodels (Holm correction), bootstrap                                                |
| Demo                     | Streamlit                                                                                      |
| Reproducibility          | Configs, fixed seeds for universe generation, cached API outputs, URL manifest instead of PDFs |

### **6. Repository Structure**

```text
project/
├── configs/ # budgets, cutoffs, universe specs, agent limits
├── data/
│ ├── sources.csv # announcement URLs + metadata
│ ├── gold/ # 60-announcement extraction + text-feature labels
│ └── rag_queries/ # 30 labelled retrieval queries
├── src/
│ ├── ingest/ # source adapters, PDF→text, A+ extraction
│ ├── store/ # event store, price store, point-in-time access
│ ├── rag/ # index, hybrid retrieval, text features
│ ├── agent/ # research agent, tools, notebook memory
│ ├── dsl/ # signal DSL schema + validator
│ ├── backtest/ # event study, strategy P&L, unit tests
│ ├── harness/ # lockout, budget, corrections, audit log
│ ├── universes/ # planted + placebo generators
│ ├── baselines/ # R0 grid search, R1 single-shot
│ └── eval/ # scoring, matching, statistics, RAG eval
├── app/ # Streamlit demo
├── results/ # metrics, traces, figures
└── README.md
```

### **7. Work Split (five members)**

Each member owns one layer end to end and can explain it at the whiteboard without notes.

| **Member**                   | **Owns**                    | **Builds**                                                                                                                                    | **Experiments and metrics**                                                                                   | **Class topics**                               |
|------------------------------|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| **1. Data**                  | Data layer                  | Corpus (400+ announcements), source adapter, A+ extraction, rule checks, event store with point-in-time fields, price store                   | Extraction F1 on the gold set; data-quality report; event counts by type and year                             | Evaluation (dataset design); structured output |
| **2. Retrieval**             | Knowledge layer             | Chunking, hybrid index, time-aware filters, search_announcements, text-feature classifiers with evidence                                      | Recall@k on 30 queries; text-feature accuracy; faithfulness (validated LLM judge); single-pass vs agentic RAG | Agentic RAG; LLM-as-judge                      |
| **3. Agent**                 | Research agent              | ReAct loop, tool interfaces, notebook memory, prompts, R1 single-shot and R2 no-RAG variants                                                  | Agent behaviour metrics; cost and tokens; trace analysis of successful vs failed sessions                     | LLM agents and agentic AI                      |
| **4. Backtest + guardrails** | Backtest engine and harness | Signal DSL and validator, event study, strategy P&L, unit tests, time lockout, testing budget, corrections, minimum-events rule, audit log    | Engine correctness tests; R3 vs R3-U overclaiming study; injection checks on announcement text                | Harnesses, security and ethics                 |
| **5. Evaluation**            | Evaluation layer            | Planted and placebo universe generators, matching rule, R0 grid-search baseline, statistics pipeline, detection-curve figures, Streamlit demo | Planted recall, placebo FDR, holdout retention, system comparisons with CIs; qualitative failure review       | Evaluating LLMs                                |

**Shared tasks (week 1, everyone)**

- Label the 60-announcement gold set (12 each), covering extraction fields and text features.

- Write the 30 RAG evaluation queries with relevant documents marked (6 each).

**Interfaces to freeze by 20 Oct**, so members can build in parallel:

1.  **Event store schema** (Member 1 → everyone)

2.  **Signal DSL schema** (Member 4 → Members 3 and 5)

3.  **Backtest result schema** (Member 4 → Members 3 and 5)

4.  **Tool function signatures** (Member 3 ↔ Members 2 and 4)

5.  **Finding format** (Member 3 → Member 5)

**Key dependencies**

- Member 5’s universe generators need the event and price stores (Member 1) and a tested backtest engine (Member 4) by about 27 Oct.

- Member 3 can develop against a stub backtest and stub retrieval until then.

- Member 2’s text features must exist before Member 5 can plant text-feature signals.

### **8. Timeline**

| **Dates**          | **Milestones**                                                                                                                                                 | **Course load**                 |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
| 16–20 Oct          | Interfaces frozen; corpus download split across team; gold set and RAG queries labelled; backtest engine skeleton with first unit tests                        | —                               |
| 21–27 Oct          | A+ extraction run on full corpus; event and price stores; hybrid index and text features; DSL and harness; agent loop on stubs; planted and placebo generators | Quiz 6 (27/30 Oct)              |
| 28 Oct–3 Nov       | Integration; R0–R3 and R3-U run on all universes; RAG evaluation; holdout test on the real universe                                                            | Quiz 7 (3/6 Nov)                |
| 4–7 Nov            | Statistics, detection curves, failure review, demo, report draft                                                                                               | —                               |
| 8 Nov–presentation | Report finalised, README check, unaided rehearsal per member                                                                                                   | G1: Tue 10 Nov · G2: Fri 13 Nov |

### **9. Cut List (if time runs short)**

Cut in this order:

1.  Outlook-tone feature (keep rights-issue purpose and dividend framing)

2.  Strategy P&L view (keep the event study)

3.  Single-pass vs agentic RAG comparison

4.  R1 single-shot baseline

**Never cut:**

- planted and placebo evaluation,

- the R0 baseline (answers “is agency needed?”),

- R2 vs R3 (answers “does RAG help?”),

- R3 vs R3-U (the guardrail finding),

- the time lockout.
