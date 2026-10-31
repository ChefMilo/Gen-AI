"""Mask dates in announcement text before the model sees it (agent.mask_dates_in_text).

Placebo universes move events to random dates but leave the announcement text alone, so
a passage saying "of record on January 22, 2015" would tell the agent the event's real
date (PR #6). The agent's retrieval layer masks dates in every passage and quote it
returns, on every universe type, so that R3 behaves the same everywhere. Evidence quotes
are masked the same way before grounding, so a quote of a masked passage still grounds.

Formats: "January 22, 2015", "Jan. 22, 2015", "22 January 2015", "2015-01-22",
"1/22/2015", a month next to a year ("January 2015"), and a month with a day and no
year ("January 22"). Month names must be capitalised, so the verb "may" is left alone.
"""

import re

MASK = "[DATE]"

_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|"
    r"December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\b\.?"
)
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_YEAR = r"\d{4}"

# Longest formats first, so "January 22, 2015" becomes one [DATE], not "[DATE], 2015".
DATE_PATTERNS = [
    rf"\b{_MONTH}\s+{_DAY},?\s+{_YEAR}\b",  # January 22, 2015 / Jan. 22, 2015
    rf"\b{_DAY}\s+{_MONTH},?\s+{_YEAR}\b",  # 22 January 2015
    r"\b\d{4}-\d{2}-\d{2}\b",  # 2015-01-22
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # 1/22/2015
    rf"\b{_MONTH},?\s+{_YEAR}\b",  # January 2015
    rf"\b{_MONTH}\s+{_DAY}\b(?!\s*%)",  # January 22 (not "May 5%")
]
_DATES = re.compile("|".join(f"(?:{p})" for p in DATE_PATTERNS))


def mask_dates(text: str) -> str:
    """text with every recognised date replaced by [DATE]."""
    return _DATES.sub(MASK, text)


def maybe_mask(text: str | None, enabled: bool) -> str | None:
    return mask_dates(text) if enabled and text else text
