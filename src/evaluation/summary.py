"""Reading summary/ and describing it: the shared CSV reader, system order and environment.

Kept free of the runner so figures and the app can read a summary without importing it.
"""

import platform
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY_DIR = "summary"
INDEX_FILE = "index.json"
# The systems of the experiment, in reporting order. Others (stubs) follow alphabetically.
SYSTEM_ORDER = ["r0", "r1", "r2", "r3", "r3-u"]


def read_summary(path: str | Path, name: str) -> pd.DataFrame:
    """<path>/<name>.csv, with only empty cells as missing.

    pandas' defaults read the system name "null" (and "None", "NA") as missing.
    """
    return pd.read_csv(Path(path) / f"{name}.csv", keep_default_na=False, na_values=[""])


def ordered_systems(names: Iterable[str]) -> list[str]:
    """names in reporting order, without repeats: SYSTEM_ORDER, then the rest sorted.

    Comparisons are system_a minus system_b in this order, so their sign does not depend
    on which systems a run happened to list.
    """
    names = set(names)
    return [s for s in SYSTEM_ORDER if s in names] + sorted(names - set(SYSTEM_ORDER))


def environment() -> dict:
    """The versions and CPU the summary was computed on."""
    simd = np.show_config(mode="dicts")["SIMD Extensions"]
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cpu_simd": {"baseline": list(simd["baseline"]), "found": list(simd["found"])},
    }
