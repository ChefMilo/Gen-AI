# Gold set: hand-labelled events

30 filings, labelled by hand, for measuring the LLM event extraction (`events.parquet`).
The same 30 compare candidate models (for example Haiku vs Sonnet) before the full run.

**Label by reading the filing yourself. Don't paste an LLM's answer in here**, or the
gold set ends up grading the model against itself.

| File | What it holds |
|---|---|
| `filings.csv` | The 30 filings: ids, links, why each was picked, and labelling status |
| `events.csv` | One row per event found in them (empty until labelled) |

`src/ingest/gold.py` loads both, checks them and scores extracted events against them.

## How the 30 were picked

From the dividend and buyback candidates in `index/filing_index.parquet`, keep and
keep_from companies only, one filing per company:

- **Cases:** plain, increased, special and first-ever dividends; new, increased,
  share-based and suspended buybacks; filings with both; earnings releases that only
  report past activity; and filings that match the keywords but announce nothing
  (covenants, charter wording, preferred stock terms).
- **Spread:** every year from 2016 to 2026 (7 filings in 2024 or later, the proposed
  holdout), 16 small and 14 mid companies, 8 sectors.

The `note` column says what makes each one tricky. It is a hint, not the answer.

## Labelling a filing

1. Open its text: `$GENAI_DATA_DIR/text/edgar/<ticker>/<accession_no>/*.txt`. If you need
   the original, `source_url` is the SEC folder.
2. Add one row to `events.csv` per **common-stock** dividend or buyback the filing
   mentions. Use its `gold_id` and `accession_no`. Leave fields blank when the filing
   doesn't state them.
3. In `filings.csv`, set `n_events` (0 if none), `status` to `labelled`, and
   `labelled_by` to your name.
4. Run `python -m ingest.gold`. It should print `OK`.
5. A second person reads the filing, fixes anything they disagree with, and sets
   `status` to `checked` and `checked_by`.

## Rules

- **Common stock only.** Preferred-stock dividends are not events.
- **is_new_announcement:** `True` if this filing announces the decision for the first
  time (the board declares or approves it now). `False` if the text says "previously
  announced", or only reports what already happened ("declared and paid during the
  quarter", "repurchased 1.2 million shares"). If an earnings release says a dividend was
  "recently declared", look for an earlier 8-K from the company in `filing_index`.
  Past-only events still get a row, so the extractor is scored on telling them apart.
- **dividend_per_share:** US dollars per share. "24.25 cents" is `0.2425`, "$.10" is `0.10`.
- **dividend_kind:** `regular`, `special` or `initial` (first ever). An increased regular
  dividend is `regular`. A special paid with the regular dividend is two rows.
- **buyback_amount_usd:** the amount newly authorised in this filing. For an increase,
  the added amount, not the new total (put the total in `notes`). `$300 million` is
  `300000000`.
- **buyback_shares:** only if the program is stated in shares.
- **is_new_program:** `True` for a new program (including one that replaces an old one),
  `False` for an increase or extension.
- **Dates:** `YYYY-MM-DD`. `announcement_date` is the date the text gives for the
  decision or announcement. `buyback_expiry` only if an end date is stated.
- **quote:** one sentence copied exactly from the text, the one that states the event.
  `python -m ingest.gold` checks it is in the filing when `GENAI_DATA_DIR` is set.
- **notes:** anything a second reader should know (ambiguity, the buyback total).

Example rows for two events in one filing:

```
gold_id,accession_no,event_type,is_new_announcement,announcement_date,dividend_per_share,dividend_kind,record_date,payable_date,buyback_amount_usd,buyback_shares,is_new_program,buyback_expiry,quote,notes
G99,0000000000-24-000001,dividend,True,2024-05-02,0.25,regular,2024-05-15,2024-06-01,,,,,"The Board declared a quarterly cash dividend of $0.25 per share, payable June 1, 2024 to holders of record on May 15, 2024.",
G99,0000000000-24-000001,buyback,True,2024-05-02,,,,,50000000,,False,,"The Board approved a $50 million increase to the existing share repurchase program.",new total $150 million
```

## Scoring an extraction run

```python
from ingest import gold

filings, labels = gold.load_filings(), gold.load_events()
result = gold.score(filings, labels, extracted_events)  # store.schema.Event rows
```

Only `labelled` and `checked` filings count. Events pair up within a filing and event type.
The result has event precision and recall, accuracy per field, and the share of filings
that are entirely right.
