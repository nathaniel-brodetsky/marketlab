"""marketlab's app: `streamlit run app/main.py`, or with another data root
`streamlit run app/main.py -- --data ~/projects/marketrec/data`.

It reads the data root and results/ only, and computes nothing heavy: the experiments do
that. This version has the Market screen: a stretch of a day and a replay of its book.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from marketlab.io.day import Day
from marketlab.io.days import find_days
from marketlab.view import market as mk

NS = 1_000_000_000
LENGTHS = {"1 мин": 60, "5 мин": 300, "15 мин": 900, "30 мин": 1800, "60 мин": 3600}
REPLAY = {"10 с": 10, "30 с": 30, "60 с": 60, "120 с": 120}
SPEED = {"×1": 1.0, "×2": 2.0, "×4": 4.0, "×0.5": 0.5}
HOLE = "rgba(110,110,110,0.45)"
LATE = "rgba(255,150,0,0.30)"
BUY, SELL = "#1a9850", "#d73027"


def default_root() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data")
    known, _ = p.parse_known_args(sys.argv[1:])
    if known.data:
        return known.data
    if os.environ.get("MARKETLAB_DATA"):
        return os.environ["MARKETLAB_DATA"]
    return str(Path(__file__).resolve().parents[2] / "marketrec" / "data")


@st.cache_data(max_entries=64)
def day_info(path: str) -> tuple[list[tuple[int, int]], int, int]:
    """Valid intervals (late ones included) and the span of the day's runs."""
    d = Day.open(path, include_late=True)
    first = min(r.first_recv_ts for r in d.runs)
    last = max(r.last_recv_ts for r in d.runs)
    return d.intervals, first, last


@st.cache_data(max_entries=8, show_spinner="Читаю отрезок дня…")
def load(path: str, lo: int, hi: int) -> mk.Stretch:
    return mk.stretch(path, lo, hi)


def times(ns: np.ndarray) -> np.ndarray:
    return np.asarray(ns, dtype="datetime64[ns]")


def stamp(ns: int) -> str:
    """A time as plotly reads it in shapes: naive UTC to the millisecond."""
    t = dt.datetime.fromtimestamp(int(ns) // 1000 / 1e6, dt.UTC)
    return t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def clock(ns: int) -> str:
    t = dt.datetime.fromtimestamp(ns / NS, dt.UTC)
    return t.strftime("%H:%M:%S.") + f"{(ns // 100_000_000) % 10}"


def parse_clock(text: str, date: str) -> int | None:
    try:
        t = dt.datetime.strptime(f"{date} {text.strip()}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return int(t.replace(tzinfo=dt.UTC).timestamp()) * NS


def shade(fig: go.Figure, s: mk.Stretch, rows: int) -> None:
    for spans, color in ((s.holes, HOLE), (s.late, LATE)):
        for a, b in spans:
            for r in range(1, rows + 1):
                fig.add_vrect(
                    x0=stamp(a),
                    x1=stamp(b),
                    fillcolor=color,
                    line_width=0,
                    layer="above",
                    row=r,
                    col=1,
                )


def market_figure(s: mk.Stretch, flow: bool) -> tuple[go.Figure, int, int]:
    edges, rows = mk.bins(s)
    x = times(edges[:-1] + (edges[1:] - edges[:-1]) // 2)
    ln = mk.lines(s, rows)
    n = 6 if flow else 4
    titles = [
        "Середина, взвешенная середина и сделки",
        "Стакан: объём на уровнях 1–20 (сверху ask, снизу bid), log10",
        "Спред, тиков",
        "Глубина 20 уровней, базовой валюты",
    ]
    heights = [0.38, 0.30, 0.12, 0.20]
    if flow:
        titles += [
            "Поток ордеров уровня 1 за прошедшую секунду",
            "Доходность следующей секунды, bp",
        ]
        heights = [0.30, 0.24, 0.09, 0.13, 0.12, 0.12]
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
        subplot_titles=titles,
    )
    fig.add_trace(go.Scattergl(x=x, y=ln["mid"], name="середина", line={"width": 1.2}), 1, 1)
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=ln["weighted_mid"],
            name="взвешенная середина",
            line={"width": 1, "dash": "dot"},
        ),
        1,
        1,
    )
    shown, total = mk.thin_trades(s.trades)
    for side, color, symbol, name in (
        (True, BUY, "triangle-up", "сделки: покупатель взял"),
        (False, SELL, "triangle-down", "сделки: продавец взял"),
    ):
        k = shown.buy == side
        fig.add_trace(
            go.Scattergl(
                x=times(shown.ts[k]),
                y=shown.price[k],
                mode="markers",
                name=name,
                marker={
                    "color": color,
                    "symbol": symbol,
                    "size": np.clip(
                        4 + 3 * np.sqrt(shown.qty[k] / max(np.median(shown.qty), 1e-12)), 4, 16
                    ),
                    "opacity": 0.6,
                },
                customdata=shown.qty[k],
                hovertemplate="%{y:.2f}, %{customdata:.5f}<extra></extra>",
            ),
            1,
            1,
        )
    z = mk.heat(s, rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        logz = np.log10(z)
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=list(range(-20, 0)) + list(range(1, 21)),
            z=logz,
            colorscale="Viridis",
            showscale=False,
            customdata=z,
            hovertemplate="уровень %{y}: %{customdata:.5f}<extra></extra>",
        ),
        2,
        1,
    )
    fig.add_trace(go.Scattergl(x=x, y=ln["spread"], name="спред", showlegend=False), 3, 1)
    fig.add_trace(
        go.Scattergl(x=x, y=ln["bid_depth"], name="глубина bid", line={"color": BUY}), 4, 1
    )
    fig.add_trace(
        go.Scattergl(x=x, y=ln["ask_depth"], name="глубина ask", line={"color": SELL}), 4, 1
    )
    if flow:
        fig.add_trace(
            go.Bar(x=x, y=ln["ofi"], name="OFI уровня 1, 1 с", marker_color="#4575b4"), 5, 1
        )
        fig.add_trace(
            go.Scattergl(
                x=x, y=ln["next_return"], name="доходность +1 с", line={"color": "#636363"}
            ),
            6,
            1,
        )
    shade(fig, s, n)
    fig.update_layout(
        height=280 * n // 2 + 300,
        margin={"l": 70, "r": 20, "t": 90, "b": 30},
        legend={"orientation": "h", "y": 1.0, "yanchor": "top", "x": 0, "yref": "container"},
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=",.2f", row=1, col=1)
    return fig, len(shown), total


def replay_figure(s: mk.Stretch, frames: list[mk.Frame], speed: float) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        subplot_titles=["Середина и сделки", "Стакан: тики от середины, объём"],
        horizontal_spacing=0.08,
    )
    lo, hi = frames[0].ts, frames[-1].ts + 100_000_000
    ts, mid = mk.path(s, lo, hi)
    fig.add_trace(go.Scatter(x=times(ts), y=mid, line={"width": 1.2}, showlegend=False), 1, 1)
    tk = (s.trades.ts >= lo) & (s.trades.ts < hi)
    for side, color in ((True, BUY), (False, SELL)):
        k = tk & (s.trades.buy == side)
        fig.add_trace(
            go.Scatter(
                x=times(s.trades.ts[k]),
                y=s.trades.price[k],
                mode="markers",
                marker={"color": color, "size": 5, "opacity": 0.6},
                showlegend=False,
            ),
            1,
            1,
        )
    seen = mid[np.isfinite(mid)]
    y_lo = float(seen.min()) if len(seen) else 0.0
    y_hi = float(seen.max()) if len(seen) else 1.0
    pad = max((y_hi - y_lo) * 0.1, 5 * s.tick)
    y_range = [y_lo - pad, y_hi + pad]

    def now(f: mk.Frame) -> go.Scatter:
        x = times(np.array([f.ts, f.ts]))
        return go.Scatter(
            x=x, y=y_range, mode="lines", line={"color": "black", "width": 1}, showlegend=False
        )

    def bars(f: mk.Frame) -> list[go.BaseTraceType]:
        return [
            go.Bar(
                x=-f.bid_qty,
                y=f.bid_ticks,
                orientation="h",
                marker_color=BUY,
                width=0.8,
                customdata=f.bid_qty,
                hovertemplate="bid %{y} тиков: %{customdata:.5f}<extra></extra>",
                showlegend=False,
            ),
            go.Bar(
                x=f.ask_qty,
                y=f.ask_ticks,
                orientation="h",
                marker_color=SELL,
                width=0.8,
                customdata=f.ask_qty,
                hovertemplate="ask %{y} тиков: %{customdata:.5f}<extra></extra>",
                showlegend=False,
            ),
            go.Scatter(
                x=np.where(f.trade_buy, f.trade_qty, -f.trade_qty),
                y=f.trade_ticks,
                mode="markers",
                marker={
                    "symbol": "diamond-open",
                    "size": 12,
                    "color": np.where(f.trade_buy, BUY, SELL),
                    "line": {"width": 2},
                },
                showlegend=False,
                hovertemplate="сделка %{y} тиков: %{x:.5f}<extra></extra>",
            ),
        ]

    def title(f: mk.Frame) -> str:
        if f.row < 0:
            return f"{clock(f.ts)} UTC · нет валидного стакана"
        spread = (
            (f.ask_ticks[0] - f.bid_ticks[0]) if len(f.ask_ticks) and len(f.bid_ticks) else np.nan
        )
        return f"{clock(f.ts)} UTC · середина {f.mid:,.2f} · спред {spread:.0f} тик."

    first = frames[0]
    fig.add_trace(now(first), 1, 1)
    for tr in bars(first):
        fig.add_trace(tr, 1, 2)
    qty = np.concatenate([np.concatenate([f.bid_qty, f.ask_qty]) for f in frames] or [np.zeros(1)])
    x_max = float(np.percentile(qty, 99)) * 1.15 if len(qty) else 1.0
    depth = np.concatenate(
        [np.abs(np.concatenate([f.bid_ticks, f.ask_ticks])) for f in frames] or [np.zeros(1)]
    )
    y_max = float(np.percentile(depth, 99)) + 1 if len(depth) else 21.0
    duration = int(100 / speed)
    fig.frames = [
        go.Frame(
            data=[now(f), *bars(f)],
            traces=[3, 4, 5, 6],
            name=str(k),
            layout={"title": {"text": title(f)}},
        )
        for k, f in enumerate(frames)
    ]
    every = max(1, len(frames) // 60)
    fig.update_layout(
        height=600,
        title={"text": title(first)},
        margin={"l": 70, "r": 20, "t": 70, "b": 90},
        barmode="overlay",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.14,
                "xanchor": "left",
                "buttons": [
                    {
                        "label": "▶ играть",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": duration, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "❚❚ пауза",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "x": 0.2,
                "len": 0.8,
                "y": -0.1,
                "currentvalue": {"visible": False},
                "steps": [
                    {
                        "label": clock(f.ts)[3:] if k % every == 0 else "",
                        "method": "animate",
                        "args": [
                            [str(k)],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                        ],
                    }
                    for k, f in enumerate(frames)
                ],
            }
        ],
    )
    fig.update_yaxes(range=y_range, tickformat=",.2f", row=1, col=1)
    fig.update_xaxes(range=[-x_max, x_max], row=1, col=2, title_text="← bid · ask →")
    fig.update_yaxes(range=[-y_max, y_max], row=1, col=2)
    for a, b2 in s.holes:
        if b2 > lo and a < hi:
            fig.add_vrect(
                x0=stamp(max(a, lo)),
                x1=stamp(min(b2, hi)),
                fillcolor=HOLE,
                line_width=0,
                layer="above",
                row=1,
                col=1,
            )
    for a, b2 in s.late:
        if b2 > lo and a < hi:
            fig.add_vrect(
                x0=stamp(max(a, lo)),
                x1=stamp(min(b2, hi)),
                fillcolor=LATE,
                line_width=0,
                layer="above",
                row=1,
                col=1,
            )
    return fig


def main() -> None:
    st.set_page_config(page_title="marketlab · рынок", layout="wide")
    side = st.sidebar
    root = Path(side.text_input("Данные (корень marketrec)", default_root())).expanduser()
    base = root / "binance"
    symbols = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    if not symbols:
        st.error(f"В {root} нет данных: ожидается {base}/<символ>/<дата>")
        return
    symbol = side.selectbox("Символ", symbols)
    days = [p for p in find_days(root, symbol)]
    if not days:
        st.error(f"Для {symbol} нет дней с manifest.json")
        return
    path = side.selectbox("День", days, index=len(days) - 1, format_func=lambda p: p.name)
    intervals, _, last = day_info(str(path))
    if not intervals:
        st.warning("В этом дне нет ни одной секунды с валидным стаканом.")
        return
    default = clock(-(-intervals[0][0] // NS) * NS)[:8]
    start_text = side.text_input("Начало, UTC (ЧЧ:ММ:СС)", default)
    lo = parse_clock(start_text, path.name)
    if lo is None:
        side.error("Время в виде ЧЧ:ММ:СС")
        return
    length = LENGTHS[side.selectbox("Длина", list(LENGTHS), index=1)]
    hi = min(lo + length * NS, last + 1)
    if hi <= lo:
        side.error("Это время после конца записи дня")
        return
    flow = side.checkbox("Поток ордеров и доходность следующей секунды", value=True)

    s = load(str(path), lo, hi)
    st.title(f"Рынок · {symbol} · {path.name}")
    holes = sum(b - a for a, b in s.holes) / NS
    late = sum(b - a for a, b in s.late) / NS
    st.caption(
        f"{clock(lo)}–{clock(hi)} UTC: {len(s.book):,} снимков стакана, {len(s.trades):,} сделок; "
        f"без валидного стакана {holes:.1f} с (серым), с отставанием больше секунды {late:.1f} с "
        "(оранжевым: стакан верный, но запоздавший). Цены в валюте котировки, объёмы в базовой."
    )
    if len(s.book) == 0:
        st.warning("В этом отрезке нет валидного стакана.")
        return
    fig, shown, total = market_figure(s, flow)
    st.plotly_chart(fig)
    if shown < total:
        st.caption(f"Показаны {shown:,} крупнейших сделок из {total:,}.")
    if flow:
        st.caption(
            "Здесь только то, что было. Связь потока ордеров с будущей доходностью проверяется "
            "в экспериментах, вместе с суррогатами: results/001-mlofi-1s.json."
        )

    st.subheader("Реплей стакана")
    c1, c2, c3 = st.columns([3, 1, 1])
    span = (hi - lo) // NS
    offset = c1.slider("Начало реплея, секунд от начала отрезка", 0, max(0, int(span) - 1), 0)
    seconds = REPLAY[c2.selectbox("Длина реплея", list(REPLAY), index=1)]
    speed = SPEED[c3.selectbox("Скорость", list(SPEED))]
    frames = mk.replay(s, lo + offset * NS, seconds)
    if not frames:
        st.info("Реплей начинается после конца отрезка.")
        return
    st.plotly_chart(replay_figure(s, frames, speed))
    st.caption(
        "Кадр каждые 100 мс: стакан, каким он пришёл к этому моменту, и сделки за последние "
        "100 мс (ромбы). Слева середина за весь реплей и линия текущего момента."
    )


main()
