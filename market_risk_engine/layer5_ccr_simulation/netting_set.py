"""Netting set aggregation for exposure calculation."""
from __future__ import annotations

from typing import List

from ..layer2_portfolio.models import Portfolio, TradeUnion


def get_netting_set_trades(portfolio: Portfolio, ns_id: str) -> List[TradeUnion]:
    """Return all trades belonging to a netting set."""
    return portfolio.trades_in_netting_set(ns_id)


def apply_netting(trade_mtms: List[float]) -> float:
    """
    Apply netting: exposure is max(sum(MtMs), 0).
    Under a netting agreement, negative MtMs offset positive ones.
    """
    return max(sum(trade_mtms), 0.0)


def all_netting_set_ids(portfolio: Portfolio) -> List[str]:
    """Return all distinct netting set IDs in the portfolio."""
    return list(portfolio.netting_sets.keys())
