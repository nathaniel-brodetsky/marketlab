"""A market-making strategy through the fill simulator: what it would have earned before the
maker fee, and the fee at which it breaks even.

The rule, on each side separately:

- one limit order of `size` on each side, post-only: at the best price on its side, or,
  with `offset_bp`, that far from the mid (a bid at or below the mid less `offset_bp`, an
  ask at or above the mid plus it), never inside the best prices;
- it rests until it fills or for `life`, whichever comes first; the next one is priced
  from the best prices of the moment the last one's outcome is known: at once after a fill, after a
  cancel once the cancel has taken effect (`latency` later), after a post-only rejection
  when the order would have reached the book;
- the position is limited: no order goes out that could take it beyond `max_position`
  orders on one side; at the limit only the side that reduces it is quoted;
- with a signal (level-1 order-flow imbalance over `window` divided by its spread over the
  hour before, as in hypothesis 002): an order is withdrawn, and none is placed, while the
  signal points against its side beyond `threshold` (below `-threshold` for a bid, above
  it for an ask). Where the signal is not defined (the first hour of a segment) the orders
  go out as without one.

Each order goes through the simulator alone (`execution.simulate`): the market does not
react to it, and its fills depend only on when it arrived and when it left. An order at a
price deeper than the book shows rests behind whatever is there (`unseen="behind"`): it
fills only once a depth update has shown its level and all of that level has traded, or
when a trade goes through its price; `unseen` in the results is the share of such orders.
The orders of both sides follow one another in time, since the position limit ties them: a
decision sees the fills received by its time and nothing later. A day starts flat; what is
held at its end is valued at the day's last mid. The day goes through the simulator an hour
at a time, as in `maker`, so that a busy day fits in memory; `tests/test_strategy.py`
checks that the pieces give what the whole day gives.

Gross P&L is in quote currency, before fees; the fee is charged on the traded notional, a
negative fee being a rebate. It splits into what the fills captured against the mid at the
time (`captured`, half the spread when the mid holds) and what the position then made or
lost as the mid moved (`inventory`). The break-even fee is gross P&L per traded notional:
a maker fee below it leaves a profit.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl

from .execution import Market, Mode, Order, Outcome, simulate
from .execution.market import ASK, BID
from .features import duration_ns
from .io.day import Day
from .maker import BEFORE, CHUNK, signal

NEVER = np.iinfo(np.int64).max
RETRY = 100_000_000  # with no best price seen yet in the segment, look again 100 ms later
WAIT = 1_000_000_000  # at the position limit, look again after the next fill or a second
SEC = 1_000_000_000


@dataclass(frozen=True)
class Quoting:
    size: Decimal = Decimal("0.001")
    life: str = "10s"
    max_position: int = 5  # in orders of `size`, on either side
    latency: str = "300ms"
    window: str | None = None  # the signal's window; None: no signal
    threshold: float = 1.0
    offset_bp: float = 0.0  # distance from the mid; 0: at the best prices

    def __post_init__(self) -> None:
        if self.max_position < 1:
            raise ValueError("max_position must be at least one order")
        if self.threshold < 0:
            raise ValueError("threshold must not be negative")
        if not 0 <= self.offset_bp < 1e4:
            raise ValueError("offset_bp must be from 0 to below 10000")
        for name in ("life", "latency") + (("window",) if self.window else ()):
            if duration_ns(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def name(self) -> str:
        sig = f"OFI {self.window} beyond {self.threshold:g}" if self.window else "no signal"
        where = f"{self.offset_bp:g} bp from the mid" if self.offset_bp else "at the best prices"
        return (
            f"{sig}; {self.size} a side {where}, life {self.life}, at most "
            f"{self.max_position} orders held, latency {self.latency}"
        )


ORDER_SCHEMA = {
    "id": pl.Int64, "side": pl.Int8, "decided": pl.Int64, "price": pl.Int64,
    "cancel": pl.Int64, "outcome": pl.String, "filled": pl.Int64, "unseen": pl.Boolean,
}  # fmt: skip
FILL_SCHEMA = {
    "id": pl.Int64, "side": pl.Int8, "ts": pl.Int64, "price": pl.Float64,
    "qty": pl.Float64, "mid": pl.Float64,
}  # fmt: skip


@dataclass
class DayResult:
    date: str
    mode: str
    #: Our orders: id, side, decided, price (steps), cancel (decision), outcome, filled
    #: (steps), unseen (its level deeper than the book showed at its arrival).
    orders: pl.DataFrame
    #: Our fills: id, side, ts, price and mid at the fill (quote currency), qty (base units).
    fills: pl.DataFrame
    gross: float  # quote currency, before fees, with the end position at the last mid
    captured: float  # against the mid at each fill
    notional: float  # traded, quote currency
    end_position: float  # base units
    max_position: float  # largest absolute position, base units
    last_mid: float
    quoted: dict[str, float] = field(default_factory=dict)  # share of valid time with an order

    @property
    def inventory(self) -> float:
        return self.gross - self.captured

    @property
    def unseen(self) -> float:
        """Share of the orders that rested at a level the book did not show."""
        return float(self.orders["unseen"].mean()) if len(self.orders) else float("nan")

    @property
    def per_notional_bp(self) -> float:
        """Gross P&L per traded notional, bp: the maker fee at which the day breaks even."""
        return self.gross / self.notional * 1e4 if self.notional > 0 else float("nan")

    def net(self, fee_bp: float) -> float:
        return self.gross - fee_bp / 1e4 * self.notional

    def row(self) -> dict:
        return {
            "date": self.date,
            "mode": self.mode,
            "orders": len(self.orders),
            "fills": len(self.fills),
            "notional": self.notional,
            "gross": self.gross,
            "captured": self.captured,
            "inventory": self.inventory,
            "break_even_bp": self.per_notional_bp,
            "end_position": self.end_position,
            "max_position": self.max_position,
            "quoted_bid": self.quoted.get("bid", float("nan")),
            "quoted_ask": self.quoted.get("ask", float("nan")),
            "unseen": self.unseen,
        }


class _Signal:
    """The signal on the day's grid, read as it was known at a time."""

    def __init__(self, sig: pl.DataFrame, window: str, threshold: float):
        self.ts = sig["ts"].to_numpy()
        self.seg_end = sig["seg_end"].to_numpy()
        z = sig[f"z_{window}"].to_numpy().astype(float)  # None -> NaN
        n = len(z)
        idx = np.arange(n)
        self.against: dict[int, np.ndarray] = {}
        self.next_against: dict[int, np.ndarray] = {}
        self.next_clear: dict[int, np.ndarray] = {}
        with np.errstate(invalid="ignore"):
            conds = {BID: z < -threshold, ASK: z > threshold}
        for side, a in conds.items():
            self.against[side] = a
            self.next_against[side] = _next_true(a, idx, n)
            self.next_clear[side] = _next_true(~a, idx, n)

    def row(self, t: int) -> int:
        """The last grid row at or before `t` if `t` is still in that row's segment, else -1."""
        i = int(np.searchsorted(self.ts, t, side="right")) - 1
        return i if i >= 0 and t < self.seg_end[i] else -1

    def blocked_until(self, side: int, t: int) -> int | None:
        """If the signal is against `side` at `t`, the time of the row where it stops."""
        i = self.row(t)
        if i < 0 or not self.against[side][i]:
            return None
        j = int(self.next_clear[side][i])
        return int(self.ts[j]) if j < len(self.ts) else NEVER

    def first_against(self, side: int, after: int, until: int) -> int | None:
        """The time of the first grid row in (after, until] where the signal is against."""
        j = int(self.next_against[side][int(np.searchsorted(self.ts, after, side="right"))])
        if j < len(self.ts) and self.ts[j] <= until:
            return int(self.ts[j])
        return None


def _next_true(a: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    """For each row, the first row at or after it where `a` holds (n if none); length n+1."""
    if n == 0:
        return np.array([0])
    return np.concatenate([np.minimum.accumulate(np.where(a, idx, n)[::-1])[::-1], [n]])


def _price(m: Market, side: int, t: int, seg_start: int, offset_bp: float) -> int | None:
    """The order's price in steps from the best prices last seen at or before `t` in the
    segment: `offset_bp` from their mid, rounded away from it, and never inside them."""
    r = m.ticker["recv_ts"]
    i = int(np.searchsorted(r, t, side="right")) - 1
    if i < 0 or r[i] < seg_start:
        return None
    bid, ask = int(m.ticker["bid_px"][i]), int(m.ticker["ask_px"][i])
    return quote_price(side, bid, ask, offset_bp)


def quote_price(side: int, bid: int, ask: int, offset_bp: float) -> int:
    mid = (bid + ask) / 2
    if side == BID:
        return min(bid, int(np.floor(mid * (1 - offset_bp / 1e4))))
    return max(ask, int(np.ceil(mid * (1 + offset_bp / 1e4))))


def simulate_day(
    path: Path, q: Quoting, mode: Mode = Mode.PESSIMISTIC, chunk: str | None = CHUNK
) -> DayResult:
    """A recorded day, `chunk` at a time (None: whole)."""

    def piece(lo: int, hi: int) -> Day:
        return Day.open(path, between=(lo, hi))

    return _quote(Day.open(path), q, mode, chunk, piece, lambda: Day.open(path))


def simulate_in(day: Day, q: Quoting, mode: Mode = Mode.PESSIMISTIC) -> DayResult:
    """A day already open (a synthetic one, or a day cut short), whole."""
    return _quote(day, q, mode, None, None, lambda: day)


def _quote(
    whole: Day,
    q: Quoting,
    mode: Mode,
    chunk: str | None,
    piece: Callable[[int, int], Day] | None,
    fresh: Callable[[], Day],
) -> DayResult:
    qty = int(q.size / whole.qty_step)
    if qty < 1 or Decimal(qty) * whole.qty_step != q.size:
        raise ValueError(f"{whole.date}: size {q.size} is not whole quantity steps")
    tick, unit = float(whole.price_step), float(whole.qty_step)
    lat, life = duration_ns(q.latency), duration_ns(q.life)
    intervals = whole.intervals
    starts = np.array([s for s, _ in intervals], dtype=np.int64)
    # the signal from a day of its own, let go once computed: a whole day's book is large
    sig = _Signal(signal(fresh(), [q.window]), q.window, q.threshold) if q.window else None
    end = max(r.last_recv_ts for r in whole.runs) + 1
    if chunk is None or piece is None:
        pieces: list[tuple[int | None, int]] = [(None, end)]
    else:
        step = duration_ns(chunk)
        first = min(r.first_recv_ts for r in whole.runs)
        pieces = [(lo, lo + step) for lo in range(first // step * step, end, step)]
    # data read past a piece: an order decided at its end rests for `life` at most and its
    # cancel takes effect `latency` later
    reach = life + 2 * lat + 2 * SEC

    nxt = {BID: int(starts[0]) if len(starts) else NEVER}
    nxt[ASK] = nxt[BID]
    pending: list[tuple[int, int]] = []  # (time, signed steps) of fills not received yet
    position = max_abs = 0
    resting = {BID: 0, ASK: 0}  # time with an order of ours in the book
    order_rows: list[tuple] = []
    fill_rows: list[tuple] = []
    for lo, hi in pieces:
        m: Market | None = None
        while min(nxt.values()) < min(hi, end):
            side = BID if nxt[BID] <= nxt[ASK] else ASK
            t = nxt[side]
            while pending and pending[0][0] <= t:
                position += heapq.heappop(pending)[1]
                max_abs = max(max_abs, abs(position))
            k = int(np.searchsorted(starts, t, side="right")) - 1
            if k < 0 or t >= intervals[k][1]:  # in a hole: wait for the next valid interval
                nxt[side] = int(starts[k + 1]) if k + 1 < len(starts) else NEVER
                continue
            if sig is not None:
                until = sig.blocked_until(side, t)
                if until is not None:
                    nxt[side] = max(until, t + 1)
                    continue
            if side * position + qty > q.max_position * qty:
                soon = t + WAIT
                if pending:
                    soon = min(soon, pending[0][0])
                nxt[side] = max(soon, t + 1)
                continue
            if m is None:
                m = Market.of(whole if lo is None else piece(lo - BEFORE, hi + reach))
            price = _price(m, side, t, intervals[k][0], q.offset_bp)
            if price is None:
                nxt[side] = t + RETRY
                continue
            cancel = t + life
            if sig is not None:
                pull = sig.first_against(side, t, cancel)
                if pull is not None:
                    cancel = pull
            o = Order(len(order_rows), side, qty, t, True, price, cancel)
            r = simulate(m, [o], lat, mode, unseen="behind")[0]
            filled = sum(fq for _, _, fq in r.fills)
            rested = r.outcome not in (Outcome.REJECTED, Outcome.UNKNOWN) and r.final > r.arrival
            hidden = rested and r.queue_at_arrival is None
            order_rows.append((o.id, side, t, price, cancel, str(r.outcome), filled, hidden))
            if r.fills:
                ts = np.array([f[0] for f in r.fills], dtype=np.int64)
                mids = m.mid_at(ts, np.full(len(ts), k))
                for (fts, p, fq), mid in zip(r.fills, mids, strict=True):
                    fill_rows.append((o.id, side, fts, p * tick, fq * unit, mid * tick))
                    heapq.heappush(pending, (fts, side * fq))
            # the next order is decided when this one's outcome is known: at its last fill,
            # when its cancel takes effect, at its arrival if it was refused
            if r.outcome in (Outcome.HOLE, Outcome.OPEN):  # its segment ended, or the data
                nxt[side] = r.final + 1
            else:
                nxt[side] = r.final
            if r.outcome not in (Outcome.REJECTED, Outcome.UNKNOWN):
                resting[side] += max(r.final - r.arrival, 0)
            nxt[side] = max(nxt[side], t + 1)
    while pending:
        position += heapq.heappop(pending)[1]
        max_abs = max(max_abs, abs(position))

    fills = pl.DataFrame(fill_rows, schema=FILL_SCHEMA, orient="row").sort("ts", "id")
    orders = pl.DataFrame(order_rows, schema=ORDER_SCHEMA, orient="row")
    last_mid = _last_mid(whole, piece, tick)
    signed = fills["side"].cast(pl.Float64) * fills["qty"]
    cash = -float((signed * fills["price"]).sum())
    end_position = position * unit
    valid = sum(e - s for s, e in intervals)
    return DayResult(
        date=whole.date,
        mode=str(mode),
        orders=orders,
        fills=fills,
        gross=cash + end_position * last_mid,
        captured=float((signed * (fills["mid"] - fills["price"])).sum()),
        notional=float((fills["price"] * fills["qty"]).sum()),
        end_position=end_position,
        max_position=max_abs * unit,
        last_mid=last_mid,
        quoted={
            n: resting[s] / valid if valid else float("nan")
            for n, s in (("bid", BID), ("ask", ASK))
        },
    )


def _last_mid(whole: Day, piece: Callable[[int, int], Day] | None, tick: float) -> float:
    """The last mid of the day's valid data, in quote currency; from its last minutes when
    the day is read in pieces."""
    intervals = whole.intervals
    if not intervals:
        return float("nan")
    lo, hi = intervals[-1]
    days = [piece(s, hi) for s in (max(lo, hi - 600 * SEC), lo)] if piece else [whole]
    for day in days:
        b = day.valid("book_ticker", ["recv_ts", "bid_px", "ask_px"])
        if not b.is_empty():
            last = b.sort("recv_ts").tail(1)
            return float((last["bid_px"][0] + last["ask_px"][0]) / 2 * tick)
    return float("nan")


# --- many days --------------------------------------------------------------------------


def _one(task: tuple[Path, Quoting, str, str | None]) -> DayResult:
    path, q, mode, chunk = task
    return simulate_day(path, q, Mode(mode), chunk)


def run(
    paths: Sequence[Path],
    q: Quoting,
    mode: Mode = Mode.PESSIMISTIC,
    workers: int = 1,
    chunk: str | None = CHUNK,
) -> list[DayResult]:
    """Every day in `paths` through the rule, in their order."""
    tasks = [(Path(p), q, str(mode), chunk) for p in paths]
    if workers <= 1 or len(tasks) <= 1:
        return [_one(t) for t in tasks]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            return list(pool.map(_one, tasks))
    except BrokenProcessPool as e:
        raise RuntimeError(
            "a worker was stopped from outside, most likely for want of memory: "
            f"run again with fewer than {workers} --workers"
        ) from e


@dataclass(frozen=True)
class Summary:
    days: int
    fills_per_day: float
    notional_per_day: float  # quote currency
    gross_per_day: float
    captured_bp: float  # per traded notional, over all days
    inventory_bp: float
    break_even_bp: float  # mean over days of gross per notional
    break_even_se: float  # its standard error over days
    positive_days: float  # share of days with a positive gross P&L

    def net_per_day(self, fee_bp: float) -> float:
        return self.gross_per_day - fee_bp / 1e4 * self.notional_per_day


def summarize(results: Sequence[DayResult]) -> Summary:
    nan = float("nan")
    if not results:
        return Summary(0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan)
    be = np.array([r.per_notional_bp for r in results])
    be = be[np.isfinite(be)]
    n = len(be)
    notional = sum(r.notional for r in results)
    return Summary(
        days=len(results),
        fills_per_day=float(np.mean([len(r.fills) for r in results])),
        notional_per_day=notional / len(results),
        gross_per_day=float(np.mean([r.gross for r in results])),
        captured_bp=sum(r.captured for r in results) / notional * 1e4 if notional else nan,
        inventory_bp=sum(r.inventory for r in results) / notional * 1e4 if notional else nan,
        break_even_bp=float(be.mean()) if n else nan,
        break_even_se=float(be.std(ddof=1) / np.sqrt(n)) if n > 1 else nan,
        positive_days=float(np.mean([r.gross > 0 for r in results])),
    )


def table(results: Sequence[DayResult]) -> pl.DataFrame:
    return pl.DataFrame([r.row() for r in results])
