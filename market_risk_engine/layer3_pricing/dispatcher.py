"""Route trades to their appropriate pricing engine."""
from __future__ import annotations

from typing import Dict, List, Type

from ..layer2_portfolio.models import (
    AmortizingIRS, CapFloor, CommodityFuturesOption, CommoditySwap,
    FloatFloatSwap, FXForward, FXOption, IRS, Portfolio, Swaption, TradeUnion,
)
from .base import MarketSnapshot, PricingEngine, PricingResult
from .cap_floor_pricer import CapFloorPricer
from .commodity_option_pricer import CommodityFuturesOptionPricer
from .commodity_swap_pricer import CommoditySwapPricer
from .fx_forward_pricer import FXForwardPricer
from .fx_option_pricer import FXOptionPricer
from .irs_pricer import AmortizingIRSPricer, FloatFloatSwapPricer, IRSPricer
from .swaption_pricer import SwaptionPricer


class PricingDispatcher:
    """
    Central dispatch hub: given a trade and a MarketSnapshot,
    select the correct pricer and return a PricingResult.
    """

    def __init__(self) -> None:
        self._engines: Dict[Type, PricingEngine] = {
            IRS: IRSPricer(),
            AmortizingIRS: AmortizingIRSPricer(),
            FloatFloatSwap: FloatFloatSwapPricer(),
            CapFloor: CapFloorPricer(),
            Swaption: SwaptionPricer(),
            FXForward: FXForwardPricer(),
            FXOption: FXOptionPricer(),
            CommoditySwap: CommoditySwapPricer(),
            CommodityFuturesOption: CommodityFuturesOptionPricer(),
        }

    def price_trade(self, trade: TradeUnion,
                    market: MarketSnapshot) -> PricingResult:
        engine = self._engines.get(type(trade))
        if engine is None:
            return PricingResult(
                trade_id=getattr(trade, "trade_id", "UNKNOWN"),
                npv=float("nan"),
                currency="N/A",
                error=f"No pricer registered for type {type(trade).__name__}",
            )
        return engine.price(trade, market)  # type: ignore[arg-type]

    def price_portfolio(
        self, portfolio: Portfolio, market: MarketSnapshot
    ) -> List[PricingResult]:
        return [self.price_trade(t, market) for t in portfolio.trades]

    def price_netting_set(
        self, portfolio: Portfolio, ns_id: str, market: MarketSnapshot
    ) -> List[PricingResult]:
        trades = portfolio.trades_in_netting_set(ns_id)
        return [self.price_trade(t, market) for t in trades]

    def netting_set_npv(
        self, portfolio: Portfolio, ns_id: str, market: MarketSnapshot
    ) -> float:
        """Sum of NPVs across all trades in a netting set (applying netting)."""
        results = self.price_netting_set(portfolio, ns_id, market)
        return sum(r.npv for r in results if r.error is None)
