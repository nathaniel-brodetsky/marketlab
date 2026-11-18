# marketlab

Research on order-book data recorded by
[marketrec](https://github.com/nathaniel-brodetsky/marketrec): it reads the recorder's
tables, leaves out every interval in which the data cannot be trusted, and builds bars and
microstructure features, with a test that none of them uses data from after its own time.

Status: the first stage of the research side: the contract reader, bars and the first
layer of features. The execution simulator, the validation protocol (purged
cross-validation, surrogate data, PBO), Hawkes processes and the viewer come next. There are
no research results yet. This file lists only what is implemented and tested.

The data contract is [`SCHEMA.md`](SCHEMA.md), a copy of marketrec's, version 1. marketrec
only writes those files and marketlab only reads them; the two repositories share nothing
else.

## Run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]' && pytest
marketlab day data/binance/BTCUSDT/2026-11-21
```

`marketlab day` needs a day recorded by marketrec. Without one, `marketlab synth /tmp/synth`
writes a synthetic day: a toy market, consistent the way the recorder's output is, with a
hole and a late interval in it, and nothing to say about real markets.

```text
$ marketlab day /tmp/synth/binance/BTCUSDT/2026-11-21
day         /tmp/synth/binance/BTCUSDT/2026-11-21
symbol      BTCUSDT, price step 0.01, qty step 0.00001, feed json
run         synthetic, 10:00:00-10:10:00 UTC, book synced after the start, derivation 1
rows        depth_diff 23308, book_ticker 9671, trades 3647, book_l20 5892, quality 7
usable      592.0 s of 600.0 s from the first row to the last (98.67%), in 3 segments
excluded    5.0 s without a valid book, 3.0 s late
longest     237.0 s from 10:06:03 UTC
```

| command | what it does |
|---------|--------------|
| `marketlab day DIR` | runs, rows, how much of the time with a connection open is usable, in how many segments, and what was left out |
| `marketlab bars DIR [--every 1s]` | time bars against tick, volume and dollar bars: return distribution and dependence |
| `marketlab features DIR [--train DIR ...] [--out FILE]` | every feature on the day, summarized; fitted features need training days |
| `marketlab synth ROOT` | write a synthetic day to try the commands |

`--late` keeps the intervals in which the data arrived late (see below).

## Architecture

```text
marketrec (Rust)                         marketlab (Python)
  WebSocket + REST                         io/        contract checks, valid intervals, segments
  -> frames (primary record)               bars/      time, tick, volume, dollar
  -> Parquet tables + quality journal  ->  features/  layer 1 on the grid of valid book rows
     (data/binance/<symbol>/<date>/)       registry   every method by name
                                           next:      execution/, validation/, events/, app/
```

Prices and quantities stay integers (steps of the symbol) until a feature turns them into
a ratio or a price; every time is the recorder's receive time in nanoseconds.

## Where the data can be used

The rules come from the contract. The book is valid from a `synced` or `resync` event in
the quality journal until the next `gap`, `invalid_book`, `reconnect`, `overflow` or
`disconnected` with reason `stopped`. A connection replaced by its successor (`handover`)
leaves it valid. A run that goes on over midnight starts the day with a valid book
(`book_at_start` in the manifest), and each run's data ends with the run.

Late data (more than 1 s behind the exchange, between `late` and `on_time`, or until the
connection ends) is left out as well unless `--late` is given: the book is right there,
but it shows the market as it was seconds before, and the receive times bunch up when the
backlog arrives.

Each usable interval is a separate **segment**. Nothing reaches across a hole: a window
over the past is reported only once it lies wholly inside its segment, so after every hole
the features warm up again; a flow or a return is never taken between the last row before
a hole and the first after it; bars start again in each segment.

The usable time computed this way agrees with the recorder's own count (`synced_ns` in the
manifest) on the days in `tests/data`: two implementations of the same rules.

Checks on reading: each part's columns, schema version, table, symbol and steps against
the contract and the manifest; a part the manifest lists must exist; unfinished parts
(`.part-NNN.parquet.tmp`) are never read. A day still being written has no manifest yet; it
is read from its parts, with the book taken as unknown until the first `synced`.

## Bars

- **time**: aligned to multiples of the length, kept only where the whole bar lies in one
  segment; bars without trades are kept.
- **tick, volume, dollar**: a bar closes at the trade that brings its count of trades,
  base quantity or notional to the threshold; trades are not split, the next bar starts
  from zero, and the unfinished last bar of a segment is dropped.

A bar's price is the mid at its close from the best bid and ask, not the last trade, so the
bounce between bid and ask is not counted as a return. `marketlab bars` compares the types
on a day: standard deviation, skew, excess kurtosis, Jarque-Bera, first autocorrelation,
Ljung-Box over 10 lags and the share of zero returns, with activity thresholds set to give
about as many bars as the time bars. The claim that activity bars give returns closer to
normal and less dependent is to be measured on recorded days, not assumed.

## Features (layer 1)

All features are reported on one grid: the `book_l20` rows inside segments, about every
100 ms while the book is valid. Windows are `(t - w, t]`.

| method | columns | from | notes |
|--------|---------|------|-------|
| `book` | `mid`, `spread_ticks`, `spread_bps`, `wmid_dev_bps`, `imb_1/5/20`, `depth_<x>bp`, `reach_bid_bps`, `reach_ask_bps` | `book_l20` | depth within x bp is null where the 20 recorded levels do not reach that far; `reach_*` says how far they do |
| `ofi` | `ofi_<w>` | `book_ticker` | order-flow imbalance at the best level, Cont, Kukanov, Stoikov (2014), from every best-price update |
| `mlofi` | `ofi_l<m>_<w>`, `mlofi_<w>` | `book_l20` | per-level OFI and their combination with least-squares weights fitted on training days, Xu, Gould, Howison (2019) |
| `trades` | `trade_imb_<w>`, `volume_<w>` | `trades` | aggressive buying minus selling; the aggressor's side is exact |
| `vpin` | `vpin` | `trades` | Easley, Lopez de Prado, O'Hara (2012); bucket size fitted on training days |
| `intensity` | `trade_rate_<w>`, `ticker_rate_<w>`, `book_rate_<w>` | `trades`, `book_ticker` | events per second; book updates counted by how far the update ID moves |
| `cancels` | `cancel_share_<w>` | `book_l20`, `trades` | share of the quantity that left a level without a trade at its price to explain it |
| `vol` | `rv_<w>`, `range_<w>` | `book_l20` | realized volatility and range of the mid, in bp |
| `microprice` | `microprice`, `microprice_dev_bps` | `book_l20` | Stoikov (2018), fitted on training days |

Every method has `fit(train_days)` and `transform(day)`. A method is one class and one line
in `marketlab/registry.py`; the tests then cover it.

## No look-ahead

The value at time t uses only data received at or before t. `tests/test_lookahead.py` cuts
synthetic days with holes, a reconnect and late data at random times and checks that no
feature and no bar before the cut changes, for every registered method; fitted methods are
fitted once on another day. A deliberately peeking method is in the same file to show the
check fails on it. The check found one real difference: a matrix product summed in a
different order for a different number of rows, so the same row gave a value differing in
the last bits; the combination is now summed level by level.

`python scripts/mutants.py` injects 29 one-line faults one at a time (a window that reads
200 ms ahead, an interval that includes its end, a flow or a return across a hole, a VPIN
bucket that closes at the next trade, a bar that keeps its unfinished tail, an unchecked
part, and others) and runs the tests on each: all 29 are caught.

## Numbers

None yet. Measured numbers go here once a full recorded day is in: `marketlab day`, `bars`
and `features` on it, with the machine and the date.

`python scripts/timing.py` gives a rough size of the work: a synthetic hour with the rates
of a recorded BTCUSDT hour (about 436 thousand best-price updates, 165 thousand trades,
36 thousand book rows). On the development container all features took 0.6 s and the bar
comparison 0.2 s. The market there is a toy, so this is not a result.

## Limitations

- **Twenty levels every 100 ms.** The tables keep the top 20 levels only at those times.
  For BTCUSDT that is cents, a few hundredths of a basis point, so depth bands are small
  and `reach_*` shows it; multi-level OFI and cancel share see only changes that survive
  100 ms.
- **Cancel share is an estimate.** Aggregated L2 data cannot tell a cancel from a fill;
  fills are taken from trades at the level's price, and a level that leaves the top 20
  without crossing the best price may only have been pushed down, so it is not counted.
- **VPIN** with the canonical 50 buckets a day over 50 buckets needs a day of unbroken
  data before its first value; `VPIN(buckets_per_day=..., window=...)` sets shorter ones.
  It goes through the same tests as any feature: Andersen and Bondarenko found its
  forecasting power for volatility comes from its link to volume.
- **Days are read one at a time.** A segment ends at midnight even when the recording goes
  on, so windows warm up again after it. Reading consecutive days as one is next.
- **Fitted methods need training days.** `marketlab features` skips them without
  `--train`; fitting on the day being described would leak.
- **Receive time.** Every time is the recording machine's; the lag behind the exchange is
  in the manifest, measured with that machine's clock.

## Tests

```bash
pytest                     # contract, intervals, bars, features, look-ahead
python scripts/mutants.py  # the faults above, each must make the tests fail
ruff check . && ruff format --check .
```

`tests/data/contract` holds three small days written by marketrec itself
(see [`tests/data/README.md`](tests/data/README.md)); the reader is tested against them,
and the synthetic writer's parts against their columns.
