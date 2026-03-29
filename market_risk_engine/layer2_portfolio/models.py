"""Trade dataclasses for all supported instrument types."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Union

from ..common.enums import OptionType, PayReceive


# ---------------------------------------------------------------------------
# Interest Rate Derivatives
# ---------------------------------------------------------------------------

@dataclass
class IRS:
    """Vanilla interest rate swap (fixed vs floating)."""
    trade_id: str
    currency: str
    notional: float
    effective_date: date
    maturity_date: date
    fixed_rate: float
    fixed_day_count: str                # e.g. "30360"
    float_index: str                    # e.g. "SOFR", "EURIBOR_6M"
    float_day_count: str
    payment_frequency: str              # "QUARTERLY", "SEMIANNUAL", "ANNUAL"
    pay_receive: PayReceive             # PAY fixed or RECEIVE fixed
    discount_curve_id: str
    forward_curve_id: str
    netting_set_id: Optional[str] = None


@dataclass
class CapFloor:
    """Interest rate cap or floor (strip of caplets/floorlets)."""
    trade_id: str
    currency: str
    notional: float
    effective_date: date
    maturity_date: date
    strike: float
    option_type: OptionType             # CAP or FLOOR
    float_index: str
    day_count: str
    payment_frequency: str
    vol_surface_id: str
    discount_curve_id: str
    forward_curve_id: str
    netting_set_id: Optional[str] = None


@dataclass
class Swaption:
    """European swaption (option to enter a swap)."""
    trade_id: str
    currency: str
    notional: float
    option_expiry: date
    underlying_start: date
    underlying_maturity: date
    strike: float                       # Fixed rate of the underlying swap
    option_type: OptionType             # PAYER or RECEIVER
    vol_model: str                      # "black" or "bachelier"
    vol_surface_id: str
    discount_curve_id: str
    forward_curve_id: str
    payment_frequency: str = "SEMIANNUAL"
    netting_set_id: Optional[str] = None


# ---------------------------------------------------------------------------
# FX Derivatives
# ---------------------------------------------------------------------------

@dataclass
class FXForward:
    """FX forward contract."""
    trade_id: str
    base_currency: str
    quote_currency: str
    notional_base: float
    delivery_date: date
    forward_rate_contractual: float     # Agreed forward rate (quote per base)
    pay_receive: PayReceive             # PAY base, RECEIVE quote (or vice versa)
    base_discount_curve_id: str
    quote_discount_curve_id: str
    fx_rate_id: str                     # e.g. "EURUSD"
    netting_set_id: Optional[str] = None


@dataclass
class FXOption:
    """Vanilla FX option (Garman-Kohlhagen)."""
    trade_id: str
    base_currency: str
    quote_currency: str
    notional_base: float
    expiry_date: date
    delivery_date: date
    strike: float                       # Strike in quote currency per unit base
    option_type: OptionType             # CALL or PUT on base currency
    vol_surface_id: str
    base_discount_curve_id: str
    quote_discount_curve_id: str
    fx_rate_id: str
    netting_set_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Commodity Derivatives
# ---------------------------------------------------------------------------

@dataclass
class CommoditySwap:
    """Fixed-for-floating commodity price swap."""
    trade_id: str
    commodity_id: str
    notional_quantity: float            # in commodity units (bbl, MMBtu, troy oz)
    effective_date: date
    maturity_date: date
    fixed_price: float                  # fixed leg price in USD per unit
    pay_receive: PayReceive             # PAY fixed or RECEIVE fixed
    payment_frequency: str
    commodity_curve_id: str
    discount_curve_id: str
    netting_set_id: Optional[str] = None


@dataclass
class CommodityFuturesOption:
    """Option on a commodity futures contract (Black-76)."""
    trade_id: str
    commodity_id: str
    notional_quantity: float
    futures_maturity: date
    option_expiry: date
    strike: float
    option_type: OptionType             # CALL or PUT
    vol_surface_id: str
    discount_curve_id: str
    commodity_curve_id: str
    netting_set_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Union type for dispatcher routing
# ---------------------------------------------------------------------------

TradeUnion = Union[
    IRS, CapFloor, Swaption,
    FXForward, FXOption,
    CommoditySwap, CommodityFuturesOption,
]


# ---------------------------------------------------------------------------
# Portfolio container
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    portfolio_id: str
    as_of_date: date
    trades: List[TradeUnion] = field(default_factory=list)
    netting_sets: dict = field(default_factory=dict)  # ns_id -> [trade_id, ...]

    def add_trade(self, trade: TradeUnion) -> None:
        self.trades.append(trade)
        ns = getattr(trade, "netting_set_id", None)
        if ns:
            self.netting_sets.setdefault(ns, []).append(trade.trade_id)

    def get_trade(self, trade_id: str) -> Optional[TradeUnion]:
        for t in self.trades:
            if t.trade_id == trade_id:
                return t
        return None

    def trades_in_netting_set(self, ns_id: str) -> List[TradeUnion]:
        ids = self.netting_sets.get(ns_id, [])
        return [t for t in self.trades if t.trade_id in ids]
