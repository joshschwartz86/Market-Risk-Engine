"""Trade dataclasses for all supported instrument types."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple, Union

from ..common.enums import AveragingType, AsianPayoffType, BusinessDayConvention, OptionType, PayReceive


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
    payment_frequency: str              # default frequency for both legs
    pay_receive: PayReceive             # PAY fixed or RECEIVE fixed
    discount_curve_id: str
    forward_curve_id: str
    netting_set_id: Optional[str] = None
    calendar_name: Optional[str] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    # Per-leg frequency overrides (default to payment_frequency when None)
    fixed_payment_frequency: Optional[str] = None
    float_payment_frequency: Optional[str] = None
    # Additive spread on the discount curve zero rate (decimal, e.g. 0.0010 = 1 bp)
    discount_spread: float = 0.0
    # Additive spread on every float period forward rate (decimal)
    forward_spread: float = 0.0


@dataclass
class AmortizingIRS:
    """
    Fixed-for-floating interest rate swap with a step-down (or step-up) notional.

    ``notional_schedule`` is a list of ``(payment_date, notional)`` pairs that maps
    each payment date in the generated schedule to the outstanding notional for that
    accrual period.  Any payment date not present in the schedule falls back to
    ``initial_notional``.
    """
    trade_id: str
    currency: str
    initial_notional: float
    notional_schedule: List[Tuple[date, float]]   # (payment_date, notional)
    effective_date: date
    maturity_date: date
    fixed_rate: float
    fixed_day_count: str
    float_index: str
    float_day_count: str
    payment_frequency: str
    pay_receive: PayReceive
    discount_curve_id: str
    forward_curve_id: str
    netting_set_id: Optional[str] = None
    calendar_name: Optional[str] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    fixed_payment_frequency: Optional[str] = None
    float_payment_frequency: Optional[str] = None
    discount_spread: float = 0.0
    forward_spread: float = 0.0

    def notional_at(self, payment_date: date) -> float:
        """Return the notional for the period ending on ``payment_date``."""
        lookup: Dict[date, float] = dict(self.notional_schedule)
        return lookup.get(payment_date, self.initial_notional)


@dataclass
class FloatFloatSwap:
    """
    Floating-for-floating (basis) swap: two floating legs referencing different
    indices (e.g. SOFR 3M vs EURIBOR 6M, or SOFR vs Fed Funds OIS).

    ``pay_receive`` applies to Leg 1: PAY means pay Leg 1 and receive Leg 2.
    Each leg may carry an additive spread (default 0).
    """
    trade_id: str
    currency: str
    notional: float
    effective_date: date
    maturity_date: date
    # Leg 1
    leg1_index: str
    leg1_day_count: str
    leg1_frequency: str
    leg1_forward_curve_id: str
    leg1_spread: float = 0.0            # additive spread on top of the floating rate
    # Leg 2
    leg2_index: str = ""
    leg2_day_count: str = "ACT360"
    leg2_frequency: str = "QUARTERLY"
    leg2_forward_curve_id: str = ""
    leg2_spread: float = 0.0
    # Common
    pay_receive: PayReceive = PayReceive.PAY
    discount_curve_id: str = ""
    discount_spread: float = 0.0
    netting_set_id: Optional[str] = None
    calendar_name: Optional[str] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING


@dataclass
class AmortizingFloatFloatSwap:
    """
    Floating-for-floating (basis) swap with a step-down or step-up notional.

    Mirrors ``AmortizingIRS`` but both legs are floating.  ``pay_receive``
    applies to Leg 1: PAY means pay Leg 1 and receive Leg 2.  The notional
    schedule maps each payment date to the outstanding notional for that period;
    dates absent from the schedule fall back to ``initial_notional``.
    """
    trade_id: str
    currency: str
    initial_notional: float
    notional_schedule: List[Tuple[date, float]]   # (payment_date, notional)
    effective_date: date
    maturity_date: date
    leg1_index: str
    leg1_day_count: str
    leg1_frequency: str
    leg1_forward_curve_id: str
    leg1_spread: float = 0.0
    leg2_index: str = ""
    leg2_day_count: str = "ACT360"
    leg2_frequency: str = "QUARTERLY"
    leg2_forward_curve_id: str = ""
    leg2_spread: float = 0.0
    pay_receive: PayReceive = PayReceive.PAY
    discount_curve_id: str = ""
    discount_spread: float = 0.0
    netting_set_id: Optional[str] = None
    calendar_name: Optional[str] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING

    def notional_at(self, payment_date: date) -> float:
        """Return the notional for the period ending on ``payment_date``."""
        lookup: Dict[date, float] = dict(self.notional_schedule)
        return lookup.get(payment_date, self.initial_notional)


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
    calendar_name: Optional[str] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    vol_model: str = "black"            # "black" (lognormal) or "bachelier" (normal)
    discount_spread: float = 0.0
    forward_spread: float = 0.0


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
    calendar_name: Optional[str] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    # Fixed leg frequency for the underlying swap (defaults to payment_frequency)
    fixed_payment_frequency: Optional[str] = None
    # Float leg frequency for the underlying swap (informational; FSR uses fixed leg)
    float_payment_frequency: Optional[str] = None
    discount_spread: float = 0.0
    forward_spread: float = 0.0


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
    calendar_name: Optional[str] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING


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
    calendar_name: Optional[str] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING


@dataclass
class AsianFXOption:
    """Asian (average rate) FX option — arithmetic or geometric averaging.

    Supports:
    - AVERAGE_PRICE: payoff vs a fixed strike K (avg(S) - K for CALL)
    - AVERAGE_STRIKE: payoff vs the average itself as the strike (S_final - avg(S) for CALL)

    Fixing schedule: supply either ``explicit_fixing_dates`` (takes priority) or
    ``fixing_frequency`` (used with effective_date/maturity_date via generate_schedule).
    Past fixings already observed must be provided in ``past_fixings``.
    """
    trade_id: str
    base_currency: str
    quote_currency: str
    notional_base: float
    effective_date: date                 # start of observation window
    maturity_date: date                  # final fixing date / option expiry
    delivery_date: date                  # cash settlement date
    strike: float                        # fixed strike (AVERAGE_PRICE only)
    option_type: OptionType              # CALL or PUT
    payoff_type: AsianPayoffType
    averaging_type: AveragingType
    vol_surface_id: str
    base_discount_curve_id: str
    quote_discount_curve_id: str
    fx_rate_id: str
    explicit_fixing_dates: Optional[List[date]] = None   # takes priority over frequency
    fixing_frequency: Optional[str] = None               # e.g. "MONTHLY"
    past_fixings: Dict[date, float] = field(default_factory=dict)
    netting_set_id: Optional[str] = None
    calendar_name: Optional[str] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING


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
    calendar_name: Optional[str] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING


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
    calendar_name: Optional[str] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING


@dataclass
class BermudanSwaption:
    """
    Bermudan swaption: option to enter a swap on any of a set of exercise dates.

    The pricer calibrates a Hull-White 1-factor model to the basket of
    coterminal European swaptions (each sharing ``underlying_maturity``) using
    the Bachelier (normal) implied vols from ``vol_surface_id``, then prices
    via a trinomial tree with backward induction.
    """
    trade_id: str
    currency: str
    notional: float
    exercise_dates: List[date]          # sorted list; at least one date
    underlying_start: date              # start of the underlying swap
    underlying_maturity: date           # final maturity of the underlying swap
    strike: float                       # fixed rate of the underlying swap
    option_type: OptionType             # PAYER or RECEIVER
    vol_surface_id: str                 # Bachelier (normal) vol surface id
    discount_curve_id: str
    forward_curve_id: str
    payment_frequency: str = "SEMIANNUAL"
    day_count: str = "ACT365"
    n_tree_steps: int = 100             # number of trinomial-tree time steps
    netting_set_id: Optional[str] = None
    calendar_name: Optional[str] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING


# ---------------------------------------------------------------------------
# Union type for dispatcher routing
# ---------------------------------------------------------------------------

TradeUnion = Union[
    IRS, AmortizingIRS, FloatFloatSwap, AmortizingFloatFloatSwap,
    CapFloor, Swaption, BermudanSwaption,
    FXForward, FXOption, AsianFXOption,
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
