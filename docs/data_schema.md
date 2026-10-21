All data lives in the shared Drive folder Gen AI/Data/. Filings and prices are ready; the event store is in progress.

## **Ground rules**

1.  **Use the helpers, not the raw files.** load_prices(), load_benchmarks() and first_tradeable_date() are in notebook 03, section 10. load_prices() already removes bad price days and dropped companies.

2.  **Trade timing:** an event can only be traded from its first_tradeable_date, never earlier.

3.  **Returns:** always use ret / adj_close, never close.

4.  **Which companies, from when:** follow universe_quality.csv (recommendation, usable_from).

5.  **Period:** 2016-01-01 to 2026 (partial year). Prices start in 2015 for look-back windows.

## **Companies: universe.csv and universe_quality.csv**

**universe.csv**: 180 companies, half S&P 400 mid caps and half S&P 600 small caps, all dividend payers.

| **Column**            | **Type**    | **Example** | **Meaning**                                  |
|-----------------------|-------------|-------------|----------------------------------------------|
| ticker                | text        | WSO         | stock symbol                                 |
| name                  | text        | Watsco      | company name                                 |
| sector                | text        | Industrials | GICS sector (Real Estate excluded)           |
| bucket                | small / mid | mid         | source index: S&P 600 = small, S&P 400 = mid |
| n_dividends_2016_2025 | int         | 40          | dividends actually paid (Yahoo)              |
| cik                   | text        | 0000105016  | SEC company ID                               |

**universe_quality.csv**: whether to use each company, and from when. 156 keep, 21 keep_from, 3 drop (OZK, PFBC, PNFP).

| **Column**                           | **Type**                | **Meaning**                                                                                                             |
|--------------------------------------|-------------------------|-------------------------------------------------------------------------------------------------------------------------|
| ticker                               | text                    |                                                                                                                         |
| recommendation                       | keep / keep_from / drop | drop = don't use; keep_from = use only from usable_from                                                                 |
| usable_from                          | date                    | first date this company's data is valid                                                                                 |
| flags                                | text                    | why: too_few_filings, filings_start_late (changed SEC ID), listed_after_2016, known_broken_history, mcap_mostly_missing |
| filings, first_filing, last_filing   | int, date               | 8-K coverage                                                                                                            |
| div_candidates, buyback_candidates   | int                     | keyword-flagged filings                                                                                                 |
| price_start, last_date, mcap_missing | date, date, share       | price coverage                                                                                                          |

## **Filings: filing_index and text/edgar**

**index/filing_index.csv** (also .parquet): one row per 8-K, about 19,500 rows. Only items 2.02, 7.01, 8.01 and 5.02 are kept.

| **Column**                                             | **Type**    | **Meaning**                                                           |
|--------------------------------------------------------|-------------|-----------------------------------------------------------------------|
| ticker, cik, company_name                              | text        |                                                                       |
| accession_no                                           | text        | unique SEC filing ID, the join key to the event store                 |
| form_type                                              | 8-K / 8-K/A |                                                                       |
| filing_date, year                                      | date, int   |                                                                       |
| acceptance_datetime_raw                                | text        | SEC acceptance timestamp (converted to US Eastern in the event store) |
| items                                                  | text        | 8-K items, e.g. 2.02,9.01                                             |
| source_url                                             | text        | filing folder on SEC                                                  |
| docs, n_ex99                                           | text, int   | documents downloaded: main 8-K + exhibit 99 press releases            |
| is_div_candidate, is_buyback_candidate, is_exec_change | bool        | loose keyword flags, not final events                                 |
| div_snippet, buyback_snippet                           | text        | matching text, for quick review                                       |
| n_chars, error                                         | int, text   | text length, download or parse errors                                 |

Candidates are not events: many 2.02 earnings releases only mention past dividends or buybacks. The event store decides which are real announcements.

**text/edgar/** (for Retrieval): one .txt per document, the main 8-K plus each exhibit 99.x press release. Join to the filing index on ticker + accession_no. The original HTML is in raw/edgar/ with the same layout.

`text/edgar/<TICKER>/<accession_no>/<document>.txt`

## **Prices: prices/ folder**

**prices/prices.parquet**: one row per ticker per trading day, 2015 to today, about 515,000 rows.

| **Column**                     | **Type**            | **Meaning**                                                        |
|--------------------------------|---------------------|--------------------------------------------------------------------|
| date, ticker                   | date, text          |                                                                    |
| open, high, low, close         | float               | split-adjusted prices                                              |
| adj_close                      | float               | split- and dividend-adjusted; use this for returns                 |
| ret                            | float               | daily return from adj_close                                        |
| volume                         | int                 |                                                                    |
| raw_close                      | float               | actual traded price that day (unadjusted)                          |
| split_factor                   | float               | splits after this date (raw = close × factor)                      |
| shares_outstanding             | float               | point-in-time share count                                          |
| shares_source                  | sec / yahoo         | Yahoo fallback for dual-class companies                            |
| mcap_usd                       | float               | market cap = raw_close × shares                                    |
| mcap_bucket                    | small / mid / large | thirds of the universe by market cap on that date                  |
| is_bad_price, bad_price_reason | bool, text          | skip these days: new_listing, extreme_return, known_broken_history |

**prices/benchmarks.parquet**: the same price columns (date, ticker, open, high, low, close, adj_close, volume, ret) for SPY (S&P 500, main benchmark for abnormal returns), IWM (Russell 2000, small caps) and MDY (S&P 400, mid caps).

**Other files in prices/**

| **File**                   | **Columns**                                   | **Use**                                           |
|----------------------------|-----------------------------------------------|---------------------------------------------------|
| trading_calendar.csv       | date                                          | NYSE trading days (from SPY)                      |
| dividends_yf.parquet       | ex_date, ticker, dividend                     | Yahoo dividend history, to cross-check extraction |
| splits_yf.parquet          | date, ticker, split_ratio                     | stock splits                                      |
| shares_outstanding.parquet | ticker, filed, end, val, form, source         | raw SEC share counts behind market cap            |
| price_coverage.csv         | per-ticker gaps, late starts, extreme returns | data-quality reporting                            |

## **Helpers**

Copy these from notebook 03, section 10, into src/store/ and use them instead of reading files directly.

```python
load_prices(tickers=None, start=None, end=None, drop_bad=True, usable_only=True) # -> DataFrame
load_benchmarks(ticker="SPY") # -> DataFrame
first_tradeable_date(acceptance_et) # SEC acceptance time (US Eastern) -> first date a trade is allowed
# before 16:00 ET on a trading day = same day, else next trading day
```

## **Event store: events.parquet (coming next)**

One row per dividend or buyback event, produced by LLM extraction from the candidate filings. One filing can produce several events. **Use only rows with is_new_announcement = True and extraction_status = "passed".** Until it's ready, Backtest and Evaluation can build against fake events with these columns.

| **Field**                                    | **Type**                    | **Example**                    | **Notes**                                                                                             |
|----------------------------------------------|-----------------------------|--------------------------------|-------------------------------------------------------------------------------------------------------|
| event_id                                     | text                        | 0001437749-25-015532-buyback-1 | accession no. + type + counter                                                                        |
| event_type                                   | dividend / buyback          | buyback                        |                                                                                                       |
| ticker, cik, company_name                    | text                        | HSTM                           |                                                                                                       |
| accession_no, form_type, items, source_url   | text                        |                                | links back to the filing index                                                                        |
| acceptance_datetime                          | datetime (US Eastern)       |                                | when SEC received the filing                                                                          |
| first_tradeable_date                         | date                        |                                | **the event date for backtests**; same day if before 16:00 ET on a trading day, else next trading day |
| announcement_date                            | date, optional              |                                | date stated in the text                                                                               |
| is_new_announcement                          | bool                        | True                           | False = only reports past activity                                                                    |
| dividend_per_share                           | number (USD)                | 0.10                           | "10 cents" becomes 0.10                                                                               |
| dividend_kind                                | regular / special / initial | special                        |                                                                                                       |
| record_date, payable_date                    | date, optional              |                                | usually stated                                                                                        |
| ex_date                                      | date, optional              |                                | rarely stated; not extracted                                                                          |
| prev_dividend_per_share, dividend_change_pct | number                      | 0.09, +11.1                    | derived in code                                                                                       |
| buyback_amount_usd                           | number                      | 25000000                       | "\$300 million" becomes 300000000                                                                     |
| buyback_shares                               | integer, optional           |                                | if authorised in shares                                                                               |
| is_new_program                               | bool                        | True                           | False = increase or extension                                                                         |
| buyback_expiry                               | date, optional              |                                |                                                                                                       |
| buyback_pct_mcap                             | number                      | 2.1                            | derived: amount ÷ market cap × 100                                                                    |
| mcap_usd, mcap_bucket                        | number, small / mid / large |                                | derived, day before first_tradeable_date                                                              |
| text_features                                | object                      |                                | filled by Retrieval, e.g. buyback_rationale, dividend_framing; each a label + exact quote             |
| evidence                                     | object                      |                                | exact quote behind each extracted field                                                               |
| extraction_status, flags                     | passed / flagged, list      |                                | why an event was flagged                                                                              |
| extracted_at, extractor_version              | datetime, text              |                                | reproducibility                                                                                       |

**Rules**

1.  The LLM extracts; code derives. Fields marked derived are never asked of the LLM.

2.  Every extracted value needs an exact quote in evidence, or the event is flagged.

3.  Sanity checks: dividend between \$0.001 and \$50 per share; buyback amount positive; record date on or before payable date.

Open questions: which LLM API to use, and the final text-feature labels from Retrieval.

## **Known limitations (for the report)**

- **Survivorship bias:** the universe is today's S&P 400/600 members, so companies delisted since 2016 are missing.

- **Coverage gaps:** some companies announce dividends only through newswires, not 8-Ks (see index/dividend_coverage.csv). Banks that file with the FDIC are dropped.

- **Yahoo data:** free and not survivorship-free. About 8% of share counts come from Yahoo (dual-class companies).
