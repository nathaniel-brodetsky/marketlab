"""How long features, bars and the fill simulator take on a synthetic hour with the rates
of a recorded BTCUSDT hour. A rough size of the work, not a result: the market is a toy.

    python scripts/timing.py
"""

import tempfile
import time

from marketlab import bars, execution, synth
from marketlab.features import grid
from marketlab.io import Day
from marketlab.registry import FITTED, default_methods


def main() -> None:
    root = tempfile.mkdtemp()
    s = synth.Scenario(seed=9, seconds=3600, book_rate=300, trade_rate=46, gaps=[(1000, 5)])
    day = Day.open(synth.write(s, root))
    rows = {t: day.raw(t, ["recv_ts"]).height for t in ("book_ticker", "trades", "book_l20")}
    print("rows       ", ", ".join(f"{t} {n}" for t, n in rows.items()))
    grid(day)  # read the tables once, as any first method would
    methods = default_methods()
    for m in methods:
        if m.name in FITTED:
            m.fit([day])
    total = 0.0
    for m in methods:
        t = time.perf_counter()
        m.transform(day)
        dt = time.perf_counter() - t
        total += dt
        print(f"{m.name:<11} {dt:.3f} s")
    print(f"features    {total:.3f} s in all")
    t = time.perf_counter()
    bars.compare(day, "1s")
    print(f"bars        {time.perf_counter() - t:.3f} s for the comparison")
    t = time.perf_counter()
    orders, _ = execution.run(day, every="10s", qty=100, cancel_after="10s", aggressive=True)
    print(
        f"execution   {time.perf_counter() - t:.3f} s for {orders.height} simulated orders "
        "(one per side every 10 s, three queue modes, and market orders)"
    )


if __name__ == "__main__":
    main()
