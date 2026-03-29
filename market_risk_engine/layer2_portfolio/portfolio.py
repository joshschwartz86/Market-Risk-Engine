"""Portfolio-level utilities: filtering, grouping, summary."""
from __future__ import annotations

from typing import Dict, List, Type

from .models import (
    CapFloor, CommodityFuturesOption, CommoditySwap,
    FXForward, FXOption, IRS, Portfolio, Swaption, TradeUnion,
)


def group_by_type(portfolio: Portfolio) -> Dict[str, List[TradeUnion]]:
    """Return trades grouped by their class name."""
    result: Dict[str, List[TradeUnion]] = {}
    for trade in portfolio.trades:
        key = type(trade).__name__
        result.setdefault(key, []).append(trade)
    return result


def filter_by_currency(portfolio: Portfolio, currency: str) -> List[TradeUnion]:
    """Return all trades denominated in or referencing a given currency."""
    out = []
    for trade in portfolio.trades:
        ccy = getattr(trade, "currency", None) or getattr(trade, "base_currency", None)
        if ccy == currency:
            out.append(trade)
    return out


def summary(portfolio: Portfolio) -> Dict[str, int]:
    """Return a count of trades by type."""
    counts: Dict[str, int] = {}
    for trade in portfolio.trades:
        key = type(trade).__name__
        counts[key] = counts.get(key, 0) + 1
    return counts
