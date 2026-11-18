"""Every method by name. A new method is one class in `features/` (later `events/`,
`methods/`) and one line here; the look-ahead test then covers it automatically."""

from __future__ import annotations

from .features import (
    VPIN,
    BookShape,
    CancelShare,
    Intensity,
    Method,
    MicroPrice,
    MultiLevelOFI,
    OrderFlowImbalance,
    TradeFlow,
    Volatility,
)

METHODS: dict[str, type[Method]] = {
    cls.name: cls
    for cls in (
        BookShape,
        OrderFlowImbalance,
        MultiLevelOFI,
        TradeFlow,
        VPIN,
        Intensity,
        CancelShare,
        Volatility,
        MicroPrice,
    )
}

#: Methods that learn something in `fit` and refuse to `transform` before it.
FITTED = frozenset({"mlofi", "vpin", "microprice"})


def default_methods() -> list[Method]:
    return [cls() for cls in METHODS.values()]
