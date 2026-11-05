"""INTERIM: stands in for Member 4's backtest engine and harness until they land.

Everything in this package is meant to be replaced:

    spec.py       signal spec models          ->  Member 4's SignalSpec (src/dsl/)
    data.py       usable events, period cut   ->  Member 4's harness (time lockout)
    events.py     InterimEventQuery           ->  Member 1's event store
    backtest.py   InterimBacktester, budget   ->  Member 4's backtest engine and harness
    harness.py    Holm, claims -> Findings    ->  Member 4's harness (corrections)
    retrieval.py  retriever selection, rag wrapper (kept until Member 2's contract settles)

The backtester mirrors R0 (src/baselines/r0.py on PR #17) exactly, so that R0 and the
agents are compared on the same returns, filters, minimum-events rule and correction.
"""
