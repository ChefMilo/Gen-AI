"""Independent replicate draws of a synthetic price panel: fresh idiosyncratic noise.

Every universe built from one base panel otherwise shares its noise, so two universes
planting the same spec give near-identical statistics. resample_idiosyncratic splits
each ticker's daily return on SPY,

    ret = alpha + beta * ret_SPY + resid    (OLS per ticker)

and replaces resid with a fresh Gaussian draw of the same standard deviation, seeded by
the universe. alpha, beta and the market (SPY) path are kept, as are the events: like
planting, only `ret` and `adj_close` change (adj_close is rebuilt from the new ret), so
close, OHLC, raw_close, mcap and the events derived from them stay valid. The abnormal
returns of two draws still share the (beta - 1) * ret_SPY + alpha term, which is small
next to the noise (synthetic betas are 0.6-1.4, SPY's daily sd 1%, residual sd 1.2-2.5%).

Meant for synthetic panels, whose residuals are pure noise; on real prices it would
also erase genuine event effects (evaluation.plan only allows it for synthetic data).
Deterministic in (prices, benchmarks, seed); nothing needs to be stored.
"""

import numpy as np
import pandas as pd

from store.schema import DEFAULT_BENCHMARK

_NOISE_STREAM = 23


def resample_idiosyncratic(
    prices: pd.DataFrame, benchmarks: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """A copy of prices with each ticker's residual return redrawn (module docstring)."""
    out = prices.copy()
    spy = benchmarks.loc[benchmarks["ticker"] == DEFAULT_BENCHMARK].set_index("date")["ret"]
    market = pd.to_datetime(prices["date"]).map(
        spy.set_axis(pd.to_datetime(spy.index))).to_numpy(dtype=float)
    ret = prices["ret"].to_numpy(dtype=float)
    ok = np.isfinite(ret) & np.isfinite(market)

    frame = pd.DataFrame({"t": prices["ticker"].to_numpy(), "r": np.where(ok, ret, np.nan),
                          "m": np.where(ok, market, np.nan)})
    g = frame.groupby("t", sort=False)
    mean_r, mean_m = g["r"].transform("mean"), g["m"].transform("mean")
    dr, dm = frame["r"] - mean_r, frame["m"] - mean_m
    cov = (dr * dm).groupby(frame["t"], sort=False).transform("mean")
    var = (dm * dm).groupby(frame["t"], sort=False).transform("mean")
    beta = (cov / var.where(var > 0)).fillna(0.0).to_numpy()
    alpha = (mean_r - beta * mean_m).to_numpy()
    resid = frame["r"].to_numpy() - alpha - beta * frame["m"].to_numpy()
    sd = pd.Series(resid).groupby(frame["t"], sort=False).transform("std").to_numpy()

    rng = np.random.default_rng([seed, _NOISE_STREAM])
    fresh = rng.standard_normal(len(prices)) * np.nan_to_num(sd)
    new_ret = np.where(ok, alpha + beta * market + fresh, ret)

    # Rebuild adj_close per ticker in date order from its first row (unchanged).
    order = np.lexsort((pd.to_datetime(prices["date"]).to_numpy(), prices["ticker"].to_numpy()))
    adj = prices["adj_close"].to_numpy(dtype=float).copy()
    tickers = prices["ticker"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, tickers[1:] != tickers[:-1]])
    ends = np.r_[starts[1:], len(order)]
    growth = 1 + np.nan_to_num(new_ret[order])
    for s, e in zip(starts, ends, strict=True):
        rows = order[s:e]
        adj[rows[1:]] = adj[rows[0]] * np.cumprod(growth[s + 1:e])
    out["ret"] = new_ret
    out["adj_close"] = adj
    return out
