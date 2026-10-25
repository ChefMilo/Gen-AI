"""Shared helpers for the universe tests (imported by name, like tests/rag/rag_helpers.py)."""


def zero_noise(prices, benchmarks):
    """Every stock's ret equals SPY's, so the reference AR is zero before planting."""
    spy = benchmarks[benchmarks["ticker"] == "SPY"].set_index("date")["ret"]
    out = prices.assign(ret=prices["date"].map(spy).to_numpy()).sort_values(["ticker", "date"])
    growth = (1 + out["ret"].fillna(0)).groupby(out["ticker"]).cumprod()
    first = out.groupby("ticker")["adj_close"].transform("first")
    return out.assign(adj_close=first * growth).sort_index()
