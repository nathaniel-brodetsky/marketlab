# marketlab

Research on order-book data recorded by
[marketrec](https://github.com/nathaniel-brodetsky/marketrec): it reads the recorder's
tables, leaves out every interval in which the data cannot be trusted, builds bars and
microstructure features with a test that none of them uses data from after its own time,
simulates fills by queue position to measure what trading would have cost, and runs each
hypothesis through one validation protocol from a file committed before the run.

Status: the contract reader, bars, the first layer of features, the fill simulator with
markouts, and the validation protocol with its first experiment written down
(`experiments/001-mlofi.yaml`). Its result goes into `results/` once it has run on the
recorded days, whatever the verdict. Hawkes processes and the viewer come next. This file
lists only what is implemented and tested.

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
problems    1 gap, 1 resync
usable      592.0 s of 600.0 s from the first row to the last (98.67%), in 3 segments
excluded    5.0 s without a valid book, 3.0 s late
longest     237.0 s from 10:06:03 UTC
```

| command | what it does |
|---------|--------------|
| `marketlab day DIR` | runs, rows, the quality events that are not routine (gaps, undecodable messages, overflows), how much of the time with a connection open is usable, in how many segments, and what was left out |
| `marketlab bars DIR [--every 1s]` | time bars against tick, volume and dollar bars: return distribution and dependence |
| `marketlab features DIR [--train DIR ...] [--out FILE]` | every feature on the day, summarized; fitted features need training days |
| `marketlab execution DIR [--every 10s] [--size 0.001] [--cancel-after 10s] [--latency 300ms] [--aggressive]` | probe orders through the fill simulator in each queue mode: what became of them, fill rate, wait, markouts |
| `marketlab strategy --data ROOT [--offsets 0 1 2] [--signal 1s] [--threshold 1] [--vol-window 1m] [--vol-quantiles none 0.8] [--life 10s] [--max-position 5] [--modes ...] [--fees ...] [--by-day] [--workers N] [--out FILE]` | a market-making rule quoting both sides through the fill simulator, day by day: P&L before fees, what the fills captured against the mid and what the position lost, the maker fee at which it breaks even, net P&L at given fees |
| `marketlab experiment run SPEC --data ROOT [--workers N] [--final]` | run a committed hypothesis (a forecast, a `maker` rule or a `volatility` forecast) through the protocol and write its result to `results/`; `--final` opens the held-out days, once |
| `marketlab experiment list` | every hypothesis, run or only written, with its verdict and its p-value adjusted for all of them as they stand |
| `marketlab archive check [ROOT]` | integrity of Binance trade archives downloaded with `scripts/fetch_binance.py`: missing ids and trades, times, gaps between days |
| `marketlab synth ROOT` | write a synthetic day to try the commands |

`--late` keeps the intervals in which the data arrived late (see below).

## App

```bash
pip install -e '.[app]'
streamlit run app/main.py                                   # data in ../marketrec/data
streamlit run app/main.py -- --data ~/projects/marketrec/data
```

One Streamlit application with Plotly charts, in Russian. It reads the data root (the
`--data` argument, else `MARKETLAB_DATA`, else `marketrec/data` next to this repository)
and computes nothing heavy. It has one screen so far, the market:

- a stretch of a day, 1 to 60 minutes, picked by its UTC start: the mid, the mid weighted by
  the best levels' sizes, and trades as triangles by the side that took liquidity; the
  book's 20 levels a side as a heat map over time; the spread; the depth of each side;
  and, if asked, level-1 order-flow imbalance over the last second with the next second's
  return under it;
- a replay of 10 to 120 seconds of it, a frame every 100 ms: the book as it had arrived at
  that moment, as bars by ticks from the mid, with the trades of the last 100 ms, next to
  the mid with a line at the moment; play, pause and a slider, in the browser.

Every chart shades in grey the time without a valid book and in orange the time the data
was more than a second late, so that a hole never looks like a quiet market; lines break at
holes. Long stretches are thinned to at most 1200 time bins (each shows the last book row
in it) and 30,000 trades (the largest) before drawing. A stretch is read alone: only the
parts and row groups that hold its minutes, plus two seconds either side, which the order
flow of the last second and the return of the next one need (`Day.open(..., between=)`).
The chart of order flow and returns shows what happened; whether one predicts the other is
for the experiments, with their surrogates. The screens for data quality, latency,
features, point processes, execution, regimes, dynamics and experiments come next.

## Architecture

```text
marketrec (Rust)                         marketlab (Python)
  WebSocket + REST                         io/        contract checks, valid intervals, segments
  -> frames (primary record)               bars/      time, tick, volume, dollar
  -> Parquet tables + quality journal  ->  features/  layer 1 on the grid of valid book rows
     (data/binance/<symbol>/<date>/)       execution/ queue-position fill simulator, markouts
                                           targets/   future returns and volatility
                                           validation/ splits, IC, surrogates, PBO, FDR, DSR
                                           experiment hypothesis file -> verdict
                                           experiments/ one YAML per hypothesis, committed first
                                           results/   one JSON per run, failures included
                                           registry   every method by name
                                           view/      what the app's screens show, tested
                                           app/       the Streamlit application
                                           events/    market orders as a Hawkes process
                                           volatility HAR and a feature against the next seconds
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
Ljung-Box over 10 lags and the share of zero returns, with activity thresholds found by
counting the bars they make, so that each kind gives as many bars as the time bars. (Total
volume divided by the number wanted gives too few: a large trade passes the threshold by
far and its excess leaves with it. On a recorded day that halved the volume bars.) The
claim that activity bars give returns closer to normal and less dependent is to be
measured on recorded days, not assumed.

## Features (layer 1)

All features are reported on one grid: the `book_l20` rows inside segments, about every
100 ms while the book is valid. Windows are `(t - w, t]`.

| method | columns | from | notes |
|--------|---------|------|-------|
| `book` | `mid`, `spread_ticks`, `spread_bps`, `wmid_dev_bps`, `imb_1/5/20`, `depth_<x>bp`, `reach_bid_bps`, `reach_ask_bps` | `book_l20` | depth within x bp (0.1 and 0.25 by default) is null where the 20 recorded levels do not reach that far; `reach_*` says how far they do: a median of 0.5 bp for BTCUSDT on 2026-11-01 |
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

## Point processes (layer 2)

`marketlab/events/hawkes.py`: aggressive buying and selling as a two-dimensional Hawkes
process with an exponential kernel, each market order raising for a while the chance of
the next ones on both sides:

    lambda_i(t) = mu_i + sum_j alpha_ij beta sum_{t_k^j < t} exp(-beta (t - t_k^j))

- Events are market orders, not trades: trades with one exchange time and one aggressor
  side are one order sweeping several levels, and counting them apart would plant
  simultaneous events the model cannot have.
- Maximum likelihood on a window, written out with its gradient in one pass over the
  events (numba): the decayed sums S_j and their derivatives in beta D_j are carried
  forward, D_j(t + dt) = exp(-beta dt) (D_j(t) + dt S_j(t)). Parameters on the log scale,
  L-BFGS-B. The tests compare the likelihood and the gradient with direct double sums and
  finite differences, and recover the parameters of simulated data (Ogata's thinning).
- The branching ratio is the spectral radius of alpha: the share of orders caused by
  earlier ones.
- The time-rescaling check: between a side's events the compensator grows by an
  exponential with mean 1 if the model is right; each window gets a Kolmogorov-Smirnov
  p-value per side, and the share of windows that pass is reported. On simulated data the
  right model passes and a Poisson model of the same events fails.
- Rolling: a fit every `step` on the `window` of events before it, only where the window
  lies inside one valid interval; the intensity at a time comes from the last fit made at
  or before it, never one made later. On data with the rates of a BTCUSDT day (27 market
  orders a second) a day of fits every 5 minutes on 10-minute windows takes seconds.

## Execution

**This simulator cannot be calibrated against reality**: there are no fills of our own to
compare it with. It is conservative by construction instead: conclusions come from the
pessimistic queue mode, a passive order fills only when recorded trades reach it, and
nothing is assumed about the book beyond the recorded levels. Our orders also change
nothing in the recorded market: no one reacts to them, and the liquidity a market order
takes is there again for the next one.

Every time is the recorder's receive time. `latency` is how long after a decision an order
takes effect on that clock: half a round trip to reach the exchange, then one feed lag
until what happened there is received, so latency = RTT / 2 + lag. `marketlab execution`
takes both medians from the day's manifest unless `--latency` is given. A cancel takes the
same time, so an order can fill while its cancel is on the way.

- **Market orders** take liquidity from the book seen when they arrive, level by level;
  the book is the last `book_l20` row with the depth updates received after it applied.
- **Passive orders** are post-only: one that would cross the book when it arrives is
  rejected. Otherwise it joins the end of its price level's queue, behind the level's
  quantity less the trades at that price already received that the book does not show
  yet. A trade at its price that hits its side uses up the quantity ahead, then fills the
  order; a trade at a worse price means the level was taken out and fills it whole.
  Quantity added to the level joins behind it.
- **A price deeper than the book shows** (past its 20 levels, with no depth update there
  since the snapshot) has a queue that cannot be known: the outcome is `unknown`. Asked
  to (`unseen="behind"`, as `marketlab strategy` does), the order rests anyway behind
  everything there: trades at its price do not reach it until a depth update shows the
  level, and all of that level is taken to be ahead of it, in every mode. A trade at a
  worse price still fills it.
- **Cancels in the queue.** When a depth update shows the level smaller than the trades
  since the last update explain, the rest was cancelled, ahead of the order or behind it:
  the data cannot say. `pessimistic`: all behind; `proportional`: the share ahead is the
  share of the level ahead of the order; `optimistic`: all ahead. In every mode no more can
  be ahead than the level holds.
- Trades are matched to depth updates by exchange time, so a trade received just after the
  depth update it is part of is neither counted twice nor taken for a cancel. This uses a
  trade received after the update to reconstruct what the exchange had already done; it
  never feeds a decision.
- **Markouts**: for each fill, the mid's change after 100 ms, 1 s and 10 s relative to the
  fill price, in basis points, signed so that positive is good for us. A passive fill
  starts with half the spread in its favour; a negative markout is adverse selection. Not
  computed when the horizon falls outside the fill's segment.

Outcomes: `filled`, `partial` (part filled, the rest cancelled), `cancelled`, `rejected`,
`unknown` (the queue at the price could not be known), `hole` (the segment ended while the
order rested), `open` (still resting when the data ends). Fill rates count only the first
three.

`tests/test_execution.py` builds queues by hand where the answer is known: trades using up
the queue, cancels in each mode, the cap at the level, a trade through the price, a fill
while the cancel is on the way, trades seen before arrival, a trade received after its
depth update, a crossing order, a level the book does not show, a hole, the end of the data, a market order walking the
book, signed markouts and markouts stopped by a hole.

## Strategy

`marketlab strategy` (`marketlab/strategy.py`) runs a market-making rule through the
simulator above, one day at a time, and answers one question: at what maker fee would it
break even?

- One post-only order of `--size` on each side at the best price, or, with `--offsets`,
  that many basis points from the mid (rounded away from it, never inside the best
  prices); each offset is a run of its own. An order rests until it fills or for
  `--life`; the next is priced from the best prices of the moment the last one's outcome
  is known: its last fill, its cancel taking effect, its arrival if it was rejected.
- The position is limited to `--max-position` orders either way; at the limit only the
  side that reduces it is quoted.
- With `--signal W`, an order is withdrawn, and none is placed, while the level-1
  order-flow imbalance over `W`, divided by its spread over the hour before (hypothesis
  002), points against its side beyond `--threshold`.
- With `--vol-window W`, a volatility gate: on both sides an order is withdrawn, and none
  is placed, while the realized volatility of the trade price over `W` (one-second log
  changes of the last trade price) is above its quantile over the `--vol-lookback` before
  (default 60m), not counting the second compared. `--vol-quantiles` takes several, a run
  each, and `none` for a run without the gate: `--vol-window 1m --vol-quantiles none 0.5
  0.8`. Where the gate is not defined (until the window and half the lookback fit in a
  segment), orders go out as without it. Hypothesis 006 found this volatility to forecast
  the size of the next 30 s move, not its direction; the gate tests whether a maker gains
  by stepping aside then.
- A day starts flat; what is held at its end is valued at its last mid.

Each order goes through the simulator alone, so the market does not react to it (see
above), but the orders follow one another and the position ties the sides together: a
decision sees only the fills received by its time. Gross P&L splits into `captured`, each
fill against the mid at its time, and `inventory`, what the position then made or lost.
The break-even fee is gross P&L per traded notional, in basis points: a maker fee below it
leaves a profit, a negative one is a rebate. Conclusions from the pessimistic queue;
`--modes` shows the others for sensitivity. An order away from the best prices may rest at
a level the book does not show; it then waits behind all of it (see Execution), and
`unseen` reports the share of such orders. The command prints one row per offset, gate
and queue mode, then net P&L a day at each of `--fees`; `--by-day` adds a table of the days
for each run.

What it gave on the 21 free days of Tardis.dev's recording of BTCUSDT (the first of each
month, 2025-03-01 .. 2026-11-01), 0.001 a side, life 10 s, at most five orders held,
latency 300 ms unless said: a backtest, and exploration, not a hypothesis written before
it was run.

| rule | break-even maker fee, bp (mean of days, standard error) |
|------|------|
| at the best prices, pessimistic queue | -1.417 (0.021) |
| the same, proportional queue | -1.334 (0.019) |
| the same, optimistic queue | -1.044 (0.022) |
| latency 5 ms / 50 ms | -1.431 / -1.417 (0.019) |
| OFI over 1 s beyond 1 | -1.428 (0.023) |
| volatility gate over 1 m, quantile 0.5 / 0.8 of the hour before | -1.452 (0.034) / -1.433 (0.022) |
| 0.5 / 1 / 2 bp from the mid | -1.402 (0.033) / -1.340 (0.043) / -1.208 (0.096) |

No rule made money on a single day before fees. Almost all of the loss is what the
position lost after its fills (`inventory`, about -1.3 bp of the traded notional in every
rule): the price went on through the level that filled the order. The orders the gate
kept out, in the busiest minutes, lost at the same rate as those it let through, so the
gate traded less, not better. Orders away from the mid rested mostly at levels the book did
not show (`unseen` 0.48, 0.85, 0.98), behind all of them, so those three rows are the
pessimistic side; the gap between the queue modes at the best prices, 0.38 bp, is a guide
to how far they may be. At a positive maker fee, the rule loses.

`tests/test_strategy.py`: every order's fills are what the simulator gives it alone, at
the price the rule gives from the best prices last seen, at them and away from them; the
position never passes its limit and every decision knew it; the accounts add up; a day in
pieces equals the whole day, also for orders resting long past their piece; a day cut
short changes no decision before the cut; the signal, the gate and both together keep
orders out while against them and withdraw them at the first row against; the gate worked
out by hand, high only above its limit, reading no trade after its time; a threshold no
value reaches is no signal, and a gate never defined is no gate.

## Validation protocol

A result counts only if it passes all six steps, the same for every method, so results can
be compared with each other.

1. **The hypothesis before the run.** `experiments/NNN-name.yaml` fixes the data, the
   feature and its parameters, the target and horizon, the variants, the protocol's
   settings and the bar to clear. `marketlab experiment run` refuses a file that is not
   committed or has changed since its commit, and writes the file's commit and the code's
   commit into the result; code with uncommitted changes is allowed but recorded.
2. **Splits by time.** Out-of-sample predictions come from purged k-fold over days: a
   training sample whose span (from the start of its feature window to the end of its
   target) meets a test day's span, widened by the embargo, is dropped. The estimate is a
   distribution over the paths of combinatorial purged cross-validation, not one number.
   The last days are held out; `--final` opens them once and a second time is refused.
3. **Surrogate data.** The out-of-sample prediction and the target are misaligned three
   ways, at least 200 times each: minute blocks of the prediction shuffled, the prediction
   replaced by an IAAFT surrogate of itself (same values, nearly the same spectrum), the
   target shifted by at least ten horizons. The real IC must beat the 95th percentile of
   each. The random numbers depend only on the seed, so the surrogates come out the same
   on any number of processes.
4. **Multiple tests.** Benjamini-Hochberg over every hypothesis in `results/`, failed ones
   included, and every one written in `experiments/` but not run yet, at p = 1: a verdict
   cannot gain from being run before the others. The probability of backtest overfitting
   (CSCV) over the hypothesis's variants; the deflated Sharpe ratio for trading rules.
5. **Costs.** A trading rule is judged through the fill simulator, with the pessimistic
   queue. (No trading rule has been written down yet.)
6. **Stability.** The IC by day, by six hours of the day and by the volatility of the
   minute before: a signal that works one day in seven is not a signal.

The metric for a forecast of returns is the rank IC by day, with Newey-West standard errors
because samples overlap when the horizon is longer than their spacing; days are taken as
independent. Each run ends in `results/NNN-name.json` with one of three verdicts:
`passed`, `failed`, `insufficient data`, and the reasons. Days with fewer usable hours than
the hypothesis asks are left out and named, with their hours, so that a short sample has a
visible cause.

`tests/test_validation.py` checks each statistic against an answer worked out in advance:
purging and embargo sample by sample, the CPCV paths, Newey-West against the plain error,
the IC against SciPy's Spearman, surrogates that keep the values and lose the order, PBO
of zero for a real edge and one for a trap, Benjamini-Hochberg by hand, and the deflated
Sharpe ratio paying for trials. `tests/test_experiment.py` runs the protocol on data with a
planted signal (it passes every step) and without one (it fails on the surrogates), and
checks the commit rule and the one-time held-out days.

**The first experiment** (`experiments/001-mlofi.yaml`): multi-level OFI of the last second
over the best ten levels, weights fitted on the training days, against the mid's return
over the next second; six variants for PBO; 4 of the 21 free days of Tardis.dev held out.
On the 17 development days it passed (rank IC 0.29, positive on every day, surrogates at
most 0.002, PBO 0.20); the move it forecasts is about 0.3 bp a second on average and 1 bp
at most, far below a taker's fees.

**Hypotheses about trading rules** have a `kind`. `kind: maker` (`marketlab/maker.py`)
asks whether a signal cuts what passive orders lose to adverse selection: probe orders at
the best prices go through the fill simulator once per day, and a rule withdraws an order
when the signal turns against it. Withdrawing changes nothing before it takes effect, so
every rule and every surrogate follows from that one simulation (an order withdrawn at `t`
keeps the fills received by `t + latency`; `tests/test_maker.py` checks this against the
simulator itself). The metric is the fills' quantity-weighted mean markout with the rule
minus without it, by day; with no fitted part the days are scored directly. Controls: the
signal rolled within its day by at least ten minutes (same withdrawals, wrong times), PBO
over the variants, Benjamini-Hochberg over the registry, the pessimistic queue for the
verdict and the other two for sensitivity. The report gives the mean markout with the rule
as the maker fee at which it breaks even. A day goes through the simulator an hour at a
time, with a minute of data before each hour and as much after as an order's cancel,
latency and longest markout reach: a busy day's 50 million depth updates do not fit in
memory four times over, an hour's do, and every order sees what it sees in the whole day
(the tests compare the two). `experiments/002-maker-ofi-pull.yaml` is the
first: level-1 OFI of the last second, scaled by its spread over the hour before, beyond
one unit against the order, 10-second markouts, the same held-out days as 001.

The held-out days are opened only after every hypothesis meant for them is committed, so
that none is written having seen them.

`kind: volatility` (`marketlab/volatility.py`) asks whether a feature forecasts the next
seconds' volatility better than volatility does. The target is log(0.1 bp + the mid's
realized volatility over the next 10 s), every 10 s, so targets do not overlap. Baselines,
as the specification asks: the next window as the current one, and HAR (the log realized
volatility of the last 10 s, 60 s and 300 s, by least squares). The candidate is HAR with
the feature's columns. The metric is the gain over HAR by day, 1 - MSE(candidate) /
MSE(HAR), out of sample on a purged k-fold over days; the control rolls the feature's
columns within each day by at least ten minutes and fits again.
`experiments/003-hawkes-vol.yaml`: the Hawkes log intensity, its change over the last
minute and the branching ratio, from fits every five minutes on the ten before.
`experiments/004-vpin-vol.yaml`: VPIN, with its bucket volume set from the development
days' volume, and Andersen and Bondarenko's critique written into the hypothesis: HAR
already holds recent volatility, so VPIN is expected to add nothing.

## No look-ahead

The value at time t uses only data received at or before t. `tests/test_lookahead.py` cuts
synthetic days with holes, a reconnect and late data at random times and checks that no
feature and no bar before the cut changes, for every registered method; fitted methods are
fitted once on another day. A deliberately peeking method is in the same file to show the
check fails on it. The check found one real difference: a matrix product summed in a
different order for a different number of rows, so the same row gave a value differing in
the last bits; the combination is now summed level by level.

For the simulator the same file checks that each probe decision equals the one made on the
day cut at the decision's own time, and that an order's outcome, once final, and a
markout, once its horizon has passed, do not change when the data after them is removed.
It found one real error: an order was reported as caught in a hole from the last
nanosecond of its segment, before the event that ended the segment had been received; a
hole now counts from that event.

`python scripts/mutants.py` injects 45 one-line faults one at a time (a window that reads
200 ms ahead, an interval that includes its end, a flow or a return across a hole, a VPIN
bucket that closes at the next trade, a bar that keeps its unfinished tail, an order that
arrives or cancels without latency, cancels that may push the queue past the level, trades
counted twice, a probe that prices from 300 ms ahead, a markout across a hole, and others)
and runs the tests on each: all 45 are caught. For the protocol it adds 13 more: no purging,
no embargo, every CPCV path through the same split, Newey-West without its lags, the IC on
values instead of ranks, the held-out days used in the run, a shift surrogate that does not
shift, PBO that picks with hindsight, Bonferroni in place of Benjamini-Hochberg, and others;
and one for the bars, thresholds taken from totals instead of counted: 59. The later
stages (point processes, the maker probe, the strategy and its gate, the archives, the
trade-flow, screen and edge kinds) add their own, among them a gate that counts the second
it is compared with and a taker filled with no latency: 156 in all, each caught.

## Numbers

A day of Tardis.dev's recording of BTCUSDT (2026-11-01, imported with `marketrec import
tardis`: 10.2 million depth rows, 3.4 million best bid and ask updates, 1.1 million trades,
0.86 million book rows), on the author's laptop (Fedora, 32 threads), 2026-11-18:

| command | wall time |
|---------|-----------|
| `marketlab day DIR` | 2.0 s |
| `marketlab features DIR --train DIR` (every feature, fitted ones trained on 2026-10-01) | 16.0 s |
| `marketlab execution DIR --latency 300ms --aggressive` (17,278 orders in each of the three queue modes, and market orders) | 51.3 s |

Results of hypotheses are not here but in `results/`, written by the protocol.

`python scripts/timing.py` gives a rough size of the work: a synthetic hour with the rates
of a recorded BTCUSDT hour (about 436 thousand best-price updates, 165 thousand trades,
36 thousand book rows). On the development container all features took 0.6 s, the bar
comparison 0.2 s, and 2,880 simulated orders (one per side every 10 s in three queue modes,
plus market orders) 3.2 s. The market there is a toy, so this is not a result.

## Limitations

- **Twenty levels every 100 ms.** The tables keep the top 20 levels only at those times.
  For BTCUSDT that is cents, a few hundredths of a basis point, so depth bands are small
  and `reach_*` shows it; multi-level OFI and cancel share see only changes that survive
  100 ms.
- **Cancel share is an estimate.** Aggregated L2 data cannot tell a cancel from a fill;
  fills are taken from trades at the level's price, and a level that leaves the top 20
  without crossing the best price may only have been pushed down, so it is not counted.
- **VPIN** uses 1000 buckets a day over 50 by default, so its first value comes about an
  hour into a segment. The paper's 50 over 50 needs a day of unbroken data, and on a
  quieter day than the training ones gives none: so it was on 2026-11-01.
  It goes through the same tests as any feature: Andersen and Bondarenko found its
  forecasting power for volatility comes from its link to volume.
- **Days are read one at a time.** A segment ends at midnight even when the recording goes
  on, so windows warm up again after it. Reading consecutive days as one is next.
- **Fitted methods need training days.** `marketlab features` skips them without
  `--train`; fitting on the day being described would leak.
- **Receive time.** Every time is the recording machine's; the lag behind the exchange is
  in the manifest, measured with that machine's clock.
- **The simulator sees the book in 100 ms steps.** Depth updates come every 100 ms, so the
  queue at arrival and the cancels are known at that resolution; trades are exact.
- **Memory.** The simulator holds a day's depth updates in memory: for BTCUSDT that is tens
  of millions of rows, a gigabyte or two. Reading them hour by hour is next.
- **The surrogates test the alignment, not the fit.** Multi-level OFI is fitted on the mid's
  change over the same window, not on the target, so the surrogates misalign the finished
  out-of-sample prediction with the target instead of refitting on each surrogate.
- **Days as independent units.** The recorded days are a month apart, so purging and the
  embargo remove nothing between them; they matter once consecutive days are recorded.
- **Experiment files** take `mlofi` as the feature and `return` as the target so far; other
  methods are added to the runner as they get experiments.

## Binance public archives

`python scripts/fetch_binance.py --start 2026-09-01 --end 2026-11-18` downloads Binance's
daily trade archives from data.binance.vision (`aggTrades` by default, or `--kind trades`;
spot, or `--market um`/`cm` futures) into `binance-data/`, each file checked against the
SHA-256 checksum Binance publishes beside it. Run again, it fetches only what is missing
or failed its check; a day not published yet is skipped. Standard library only, so it runs
on any machine with Python 3.9 or newer. The archives hold trades without the order book:
enough for hypotheses on trades alone (intensity, VPIN, volatility, trade clocks), not for
order-flow imbalance or the fill simulator. Spot timestamps are in microseconds from
2025-01-01, in milliseconds before. `--kind klines --interval 1h --monthly` fetches price
bars a month a file (`binance-data/<market>/klines/<SYMBOL>/1h/`).

`python scripts/fetch_funding.py --start 2019-09-01 --end 2026-10-31` writes the funding
history of a USD-M perpetual to `binance-data/um/fundingRate/<SYMBOL>/<SYMBOL>-fundingRate.csv`,
from Binance's public market-data endpoint `/fapi/v1/fundingRate` (no key, no account), a
thousand settlements a request; run again, it asks only for what comes after its last row.
`tests/test_fetch.py` runs both scripts against a local server.

`marketlab archive check [binance-data]` then checks the downloaded days before any
research touches them, without looking at returns: aggregate ids that skip, raw trade ids
an aggregate range does not reach, times out of order or outside their day, empty rows,
the longest silence, dates missing from the range and ids that do not continue from one
day to the next. Exit code 1 when anything is wrong. `marketlab/archive.py` reads the zips
(with or without a header line, either time unit) into frames with times in nanoseconds.

### Hypotheses on the archives: `kind: tradeflow`

`experiments/005-tradeflow-100t.yaml` is the first: does signed taker volume, aggressive
buys minus sells over the last 100 aggregate trades, forecast the trade price's return over
the next 30 seconds? `marketlab experiment run experiments/005-tradeflow-100t.yaml --data
binance-data` runs it on the development days the spec names; `--final` opens its held-out
days, once. The windows are counts of aggregate trades (`100t`) or times (`5s`); the grid
is every `sampling` between each day's first and last trade. The feature has no fitted
parameters and a sign fixed by the hypothesis, so every development day is out of sample
for it; the choice between windows is what PBO measures. Otherwise the protocol is the
forecast one: rank IC by day with Newey-West errors inside the day, three kinds of
surrogates, Benjamini-Hochberg over every hypothesis written, stability by time of day and
by the last minute's move. `tests/test_tradeflow.py` works the flow and the target out by
hand, cuts a day to show the flow reads nothing after its time, and finds a planted effect
and not its absence.

### Many signals as one hypothesis: `kind: screen`

`experiments/006-screen-trades.yaml` tries 44 combinations of a trade-only signal and a
target at once (signed volume and order count, size skew; trade intensity, volume rate,
realized volatility, Hawkes intensity; returns and their size over 5 s to 5 min) and
counts them as one hypothesis. The development run scores every combination, measures
PBO over the whole grid, and applies the rule written in the spec blindly: the best by
mean IC that are positive on enough days and beat their surrogates, until it has the
number it should take. Its verdict is then `screened` (or `failed`); the claim itself is
decided by `--final`, which scores only the chosen ones on the held-out days, once, with
Benjamini-Hochberg over them. Until then the screen counts at p = 1 in the family.
`tests/test_screen.py` works every signal out by hand, cuts days to show none reads a
trade after its time (Hawkes included), takes a planted signal, refuses noise, and passes
over a signal its surrogates refuse.

`marketlab experiment list` shows, for each hypothesis, what its held-out days said:
`confirmed k/n` for a screen or an edge, `opened` for the others, `-` before.

### Does it pay what it costs: `kind: edge`

A forecast can be real and still too small to trade. `experiments/007-flow-beats-cost.yaml`
asks whether the signed flow points to moves larger than what a taker pays for them. The
rule: at each time of the 5 s grid, with no position open, buy if the flow is above its
95th percentile over the two hours before, sell if below its 5th; filled at the last
trade price 300 ms later, closed at the last trade price a horizon later (1 to 60 min),
one position at a time, none that would close after midnight. The mean over days of each
day's mean signed return is the round-trip cost at which the rule breaks even, and the
hypothesis names the cost it must beat: 9 bp, two taker fees of a regular Binance USD-M
futures account paying in BNB. The development run scores all 32 combinations of flow and
horizon, measures PBO over them, and takes, blindly, the one with the strongest evidence of
beating the cost (the smallest one-sided p of its mean against it), if it beats the cost on
enough days and its surrogates (the flow moved in time within its day by a random offset).
Only that one goes to the held-out days, eight months of 2026 that no trade signal has read
yet (less the first of each month, read before as Tardis.dev's spot days); if none is
taken, they stay closed. `tests/test_edge.py` works the returns and the rule out by hand,
cuts days to show no decision reads a trade after its time, finds a planted edge, does not
find one where there is none or where the cost is above it, and passes over a combination
its surrogates refuse.

### Funding carry: `kind: carry`

`experiments/008-funding-carry.yaml` leaves the short horizons: it holds BTC on spot
against an equal short in the perpetual and collects the funding, a position held for days
so that its fees, 24 bp in and out at a regular account's taker rates, are spread over many
settlements. The rule decides only at a settlement: hold over the next period while the
mean rate of the last 21 settlements is above zero, opening and closing at the open of the
hour bar that starts then. The return, in bp of the notional at the start of each period,
is the funding received (at Binance's mark price), the change of the basis between the two
legs, and the fees; by UTC day, tested for a mean above zero with Newey-West errors, PBO
over twelve rules by month, Benjamini-Hochberg over every hypothesis written. Development
January 2020 .. December 2023; held out, once, January 2024 .. October 2026. Sharpe ratios
are from monthly sums: daily ones overstate it, the funding staying high or low for months.

`marketlab carry-report SPEC --data binance-data --riskfree binance-data/fred/GS3M.csv`
describes a carry once its verdict is recorded, the held-out period only once it has been
opened: by year, the primary rule and holding always in % a year of the notional; the
three-month Treasury bill's rate (`scripts/fetch_fred.py` downloads FRED's GS3M); the
primary's return on the capital it needs, the spot leg and the short's margin at
`--leverage`, and that over the bills; Sharpe from months, also over the bills; the worst
months; and the largest rise of the perpetual within 8 h, a day and a week while the
position was held, with the leverage at which the short leg survives it if its margin is
topped up that often. Not modelled: the exchange's own risk, auto-deleveraging, slippage,
and the moving of margin between the legs.
`tests/test_carry.py` reads the archives in either time unit, works the money of a rule
out by hand, holds a year while the price triples, shows no decision changes with what
comes after it, finds a planted carry and none where the rate averages nil, opens the
held-out period once, and works the report's capital, bills, highs and rises out by hand.

## What has been found

Backtests on public data; each verdict as the protocol recorded it in `results/`.

| hypothesis | verdict | in short |
|------------|---------|----------|
| 001 multi-level OFI over 1 s | passed | |
| 002 maker withdrawing on OFI | failed | p 0.99 |
| 003 Hawkes intensity for volatility | failed | p 0.066 after Benjamini-Hochberg |
| 004 VPIN for volatility | failed | |
| 005 trade flow over 100 trades, next 30 s | passed; held out opened | IC 0.091 developing, 0.057 held out, positive on all 21 held-out days |
| 006 screen of 44 trade-only signals | screened, held out confirmed 3/3 | realized volatility over 5 and 1 min, intensity over 5 min, against the size of the next 30 s: IC 0.32 to 0.34 held out |
| 007 trade flow against a taker's 9 bp | failed on development, held-out months not opened | best 3.1 +- 1.0 bp a trade (60 min) |
| 008 funding carry, spot against the perpetual | passed, held out confirmed | 13.9% a year of the notional 2020 .. 2023 (16.6, 29.8, 2.1, 6.9 by year), 6.0% over January 2024 .. October 2026; holding always 6.9% |

The order flow forecasts the next seconds and volatility forecasts the size of the next
moves, both out of sample; neither pays its fees, the maker's (Strategy above) or the
taker's (007). The carry is paid after fees, but since 2022 at about the rate of US
Treasury bills (`carry-report`), with the exchange's risk and the short leg's margin on top.

Erratum: the comment in `experiments/007-flow-beats-cost.yaml` calls the days of
Tardis.dev's recording futures days. They are Binance spot BTCUSDT. The hypothesis is left
as committed; leaving those days out of its held-out months holds either way.

## Where to look: the cost wall

`marketlab costwall --data binance-data` asks, before any signal is built, at what horizon
one could pay for itself. A rule that trades its signal's top and bottom `q` (5% by
default) and holds for a horizon earns about `IC x sigma x phi(z_q) / q` a trade before
costs, `sigma` being the standard deviation of the return over the horizon; so a round
trip of `c` bp needs `IC = c / (sigma x phi(z_q) / q)`. The command measures `sigma` on the
data (the trade archives under an hour, hourly klines from an hour to a week) and prints
the IC needed against each of `--costs` (15 bp: a spot taker; 9: a USD-M futures taker;
3.6: USD-M maker fees alone, before the adverse selection measured in Strategy above).
Above 1, no signal can pay. Returns and signal are taken as jointly normal: a wall for
deciding where to look, not a verdict on a signal; the protocol still decides those.
`tests/test_costwall.py` works the needed IC out by hand and checks it on simulated
signals, and works out the returns on a grid of trades and between hourly closes.

## Many coins, a day at a time: a rough look

`python scripts/fetch_universe.py --start 2020-01-01 --end 2026-10-31` downloads, from
Binance's public endpoints, every USD-M perpetual quoted in USDT that Binance lists, of any
status (`binance-data/um/universe.csv`), its daily bars (`um/klines1d/<SYMBOL>.csv`) and
its funding history (as `fetch_funding.py` writes it), spacing its requests inside
Binance's limits; run again, it asks only for what is missing.

`marketlab xsection --data binance-data` then looks, cheaply, at four cross-sectional
signals: momentum over 7 and 30 days, the last day's reversal, and funding carry (minus the
mean rate of the last 7 days). Each day the universe is the 50 perpetuals with the most
quote volume over 30 days; the portfolio is long those strictly above the signal's 80th
percentile and short those strictly below its 20th (coins tied on a signal, as funding
rates often are, are not ranked by name), half the capital a side, rebalanced every 1 or 7
days, a round trip of 9 bp on every unit traded, the funding paid or received in the P&L.
It reports each signal's daily rank IC and the portfolio's gross, costs and net a year,
Sharpe from months and drawdown. It is not a hypothesis: nothing in it is written before it
runs, and it reads nothing after 2023-12-31, so that 2024 .. 2026 stay held out for the
hypothesis written about whatever it finds. The coins are those Binance still lists;
coins gone from its list are missing, which flatters a rule that buys losers. Five
contracts that are not a coin's own price are left out: USDC, the BTC-dominance and DeFi
indices, and two gold tokens. A perpetual that settled funding on a day for which Binance
serves no daily bar (an earlier listing of a coin listed again, for one) is missing from
the universe on that day; the command says how many such days there are, and for which
coins. Every file is read as text and then cast, since a column whose first rows look
whole (a volume of 100, then 1.5) would otherwise fail the read.
`tests/test_xsection.py` works the universe, the portfolio, its costs and its funding out
by hand, finds a planted momentum and nothing where there is none, and shows the held-out
years are neither asked for nor read; `tests/test_fetch.py` runs the download against a
local server.

## Tests

```bash
pytest                     # contract, intervals, bars, features, execution, look-ahead,
                           # validation statistics, the experiment runner, the app's data
                           # (and the app itself opening, with the app extra installed)
python scripts/mutants.py  # the faults above, each must make the tests fail
ruff check . && ruff format --check .
```

`tests/data/contract` holds three small days written by marketrec itself
(see [`tests/data/README.md`](tests/data/README.md)); the reader is tested against them,
and the synthetic writer's parts against their columns.
