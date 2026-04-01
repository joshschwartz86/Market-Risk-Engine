"""Asian (average rate) FX option pricer.

Supports:
- AVERAGE_PRICE (fixed strike): payoff = max(avg(S) - K, 0) for CALL
- AVERAGE_STRIKE: payoff = max(S_final - avg(S), 0) for CALL
- Arithmetic averaging via Turnbull-Wakeman moment-matching approximation
- Geometric averaging via exact log-normal closed-form
"""
from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional, Tuple

import scipy.stats as st

from ..common.date_utils import generate_schedule, year_fraction
from ..common.enums import AsianPayoffType, AveragingType, OptionType
from ..common.exceptions import PricingError
from ..layer1_market_data.vol_surface import VolSurfaceInterpolator
from ..layer1_market_data.yield_curve import YieldCurveInterpolator
from ..layer2_portfolio.models import AsianFXOption
from .base import MarketSnapshot, PricingEngine, PricingResult


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _resolve_fixing_dates(trade: AsianFXOption, cal) -> List[date]:
    """Return sorted list of all fixing dates (explicit list takes priority)."""
    if trade.explicit_fixing_dates:
        return sorted(trade.explicit_fixing_dates)
    if trade.fixing_frequency:
        return generate_schedule(
            trade.effective_date,
            trade.maturity_date,
            trade.fixing_frequency,
            calendar=cal,
            convention=trade.business_day_convention,
        )
    raise PricingError(
        f"Trade {trade.trade_id}: neither explicit_fixing_dates nor "
        f"fixing_frequency specified."
    )


def _partition_fixings(
    all_dates: List[date],
    today: date,
    past_fixings: Dict[date, float],
) -> Tuple[List[float], List[date]]:
    """Split fixing dates into (known_rates, future_dates).

    Raises PricingError if a past date is missing from past_fixings.
    """
    known: List[float] = []
    future: List[date] = []
    for d in sorted(all_dates):
        if d <= today:
            if d not in past_fixings:
                raise PricingError(
                    f"Fixing date {d} is in the past but no rate supplied in past_fixings."
                )
            known.append(past_fixings[d])
        else:
            future.append(d)
    return known, future


def _forward_fx(
    S: float,
    T: float,
    base_disc: YieldCurveInterpolator,
    quote_disc: YieldCurveInterpolator,
) -> float:
    """Forward FX rate at time T via covered-interest parity: F = S * DF_base / DF_quote."""
    if T <= 0:
        return S
    return S * base_disc.discount_factor(T) / quote_disc.discount_factor(T)


def _black76_option(F: float, K: float, sigma: float, T: float,
                    df: float, opt_type: OptionType) -> float:
    """Black-76 call/put on a forward F with strike K, vol sigma, time T, discount df."""
    if T <= 0 or sigma <= 0 or K <= 0:
        intrinsic = max(F - K, 0.0) if opt_type == OptionType.CALL else max(K - F, 0.0)
        return df * intrinsic
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if opt_type == OptionType.CALL:
        return df * (F * st.norm.cdf(d1) - K * st.norm.cdf(d2))
    else:
        return df * (K * st.norm.cdf(-d2) - F * st.norm.cdf(-d1))


# ---------------------------------------------------------------------------
# Pricer
# ---------------------------------------------------------------------------

class AsianFXOptionPricer(PricingEngine):
    """Price Asian (average rate) FX options.

    Arithmetic averaging: Turnbull-Wakeman moment-matching approximation.
    Geometric averaging: exact log-normal closed-form.
    """

    def price(self, trade: AsianFXOption, market: MarketSnapshot) -> PricingResult:  # type: ignore[override]
        try:
            return self._price(trade, market)
        except Exception as exc:
            return PricingResult(
                trade_id=trade.trade_id, npv=float("nan"),
                currency=trade.quote_currency, error=str(exc),
            )

    def _price(self, trade: AsianFXOption, market: MarketSnapshot) -> PricingResult:
        pair = f"{trade.base_currency}{trade.quote_currency}"
        if pair not in market.fx_rates:
            raise PricingError(f"FX rate '{pair}' not in market snapshot.")
        if trade.vol_surface_id not in market.vol_surfaces:
            raise PricingError(f"Missing vol surface '{trade.vol_surface_id}'.")
        if trade.base_discount_curve_id not in market.yield_curves:
            raise PricingError(f"Missing curve '{trade.base_discount_curve_id}'.")
        if trade.quote_discount_curve_id not in market.yield_curves:
            raise PricingError(f"Missing curve '{trade.quote_discount_curve_id}'.")

        base_disc = YieldCurveInterpolator(market.yield_curves[trade.base_discount_curve_id])
        quote_disc = YieldCurveInterpolator(market.yield_curves[trade.quote_discount_curve_id])
        vol_interp = VolSurfaceInterpolator(market.vol_surfaces[trade.vol_surface_id])
        S = market.fx_rates[pair].spot
        today = market.as_of_date

        cal = market.calendars.get(trade.calendar_name) if trade.calendar_name else None
        all_dates = _resolve_fixing_dates(trade, cal)
        known_rates, future_dates = _partition_fixings(all_dates, today, trade.past_fixings)
        N_total = len(all_dates)

        T_del = year_fraction(today, trade.delivery_date, "ACT365")
        df_del = quote_disc.discount_factor(max(T_del, 0.0))

        # --- degenerate: all fixings already observed ---
        if not future_dates:
            npv = self._all_past_npv(known_rates, trade, df_del)
            return PricingResult(
                trade_id=trade.trade_id, npv=npv,
                currency=trade.quote_currency, delta=0.0, vega=0.0,
            )

        # Forward FX rates and vols for each future fixing
        future_times = [year_fraction(today, d, "ACT365") for d in future_dates]
        forwards = [_forward_fx(S, T, base_disc, quote_disc) for T in future_times]

        vol_strike = trade.strike if trade.payoff_type == AsianPayoffType.AVERAGE_PRICE else None
        vols = [
            vol_interp.get_vol(
                max(T, 1e-4),
                vol_strike if vol_strike is not None else F,
            )
            for T, F in zip(future_times, forwards)
        ]

        # Dispatch to pricing formula
        if trade.averaging_type == AveragingType.ARITHMETIC:
            unit_npv = self._price_arithmetic_tw(
                known_rates, forwards, future_times, vols,
                trade.strike, N_total, trade.option_type, trade.payoff_type, df_del,
                forwards[-1],
            )
        else:  # GEOMETRIC
            unit_npv = self._price_geometric(
                known_rates, forwards, future_times, vols,
                trade.strike, N_total, trade.option_type, trade.payoff_type, df_del,
                forwards[-1],
            )

        npv = unit_npv * trade.notional_base

        # Vega: +1% absolute vol bump across all fixings
        vols_bumped = [v + 0.01 for v in vols]
        if trade.averaging_type == AveragingType.ARITHMETIC:
            unit_bumped = self._price_arithmetic_tw(
                known_rates, forwards, future_times, vols_bumped,
                trade.strike, N_total, trade.option_type, trade.payoff_type, df_del,
                forwards[-1],
            )
        else:
            unit_bumped = self._price_geometric(
                known_rates, forwards, future_times, vols_bumped,
                trade.strike, N_total, trade.option_type, trade.payoff_type, df_del,
                forwards[-1],
            )
        vega = (unit_bumped - unit_npv) * trade.notional_base

        # Delta: 1% spot bump via finite difference
        S_bump = S * 1.01
        fwds_bumped = [_forward_fx(S_bump, T, base_disc, quote_disc) for T in future_times]
        if trade.averaging_type == AveragingType.ARITHMETIC:
            unit_delta = self._price_arithmetic_tw(
                known_rates, fwds_bumped, future_times, vols,
                trade.strike, N_total, trade.option_type, trade.payoff_type, df_del,
                fwds_bumped[-1],
            )
        else:
            unit_delta = self._price_geometric(
                known_rates, fwds_bumped, future_times, vols,
                trade.strike, N_total, trade.option_type, trade.payoff_type, df_del,
                fwds_bumped[-1],
            )
        delta = (unit_delta - unit_npv) / (0.01 * S) * trade.notional_base

        return PricingResult(
            trade_id=trade.trade_id, npv=npv,
            currency=trade.quote_currency, delta=delta, vega=vega,
        )

    # ------------------------------------------------------------------
    # Degenerate case
    # ------------------------------------------------------------------

    def _all_past_npv(
        self, known_rates: List[float], trade: AsianFXOption, df_del: float,
    ) -> float:
        N = len(known_rates)
        avg = sum(known_rates) / N
        if trade.payoff_type == AsianPayoffType.AVERAGE_PRICE:
            intrinsic = max(avg - trade.strike, 0.0) if trade.option_type == OptionType.CALL \
                        else max(trade.strike - avg, 0.0)
        else:  # AVERAGE_STRIKE
            s_final = known_rates[-1]
            intrinsic = max(s_final - avg, 0.0) if trade.option_type == OptionType.CALL \
                        else max(avg - s_final, 0.0)
        return intrinsic * trade.notional_base * df_del

    # ------------------------------------------------------------------
    # Turnbull-Wakeman (arithmetic)
    # ------------------------------------------------------------------

    def _price_arithmetic_tw(
        self,
        known_rates: List[float],
        future_forwards: List[float],
        future_times: List[float],
        future_vols: List[float],
        K: float,
        N_total: int,
        opt_type: OptionType,
        payoff_type: AsianPayoffType,
        df_del: float,
        F_final: float,
    ) -> float:
        """Turnbull-Wakeman approximation for arithmetic average-rate FX options.

        Returns unit NPV (per unit of notional_base).
        """
        N_f = len(future_forwards)
        N_k = len(known_rates)

        # --- AVERAGE_PRICE: adjust strike for known fixings ---
        if payoff_type == AsianPayoffType.AVERAGE_PRICE:
            known_contribution = (sum(known_rates) / N_total) if N_k > 0 else 0.0
            K_adj = K - known_contribution
            # If K_adj <= 0, the call is already fully in the money from past fixings
            if K_adj <= 0.0 and opt_type == OptionType.CALL:
                intrinsic = (known_contribution - K) if known_contribution > K else 0.0
                return intrinsic * df_del
            if K_adj >= 0.0 and opt_type == OptionType.PUT:
                # deep OTM put after accounting for past fixings
                pass  # fall through to normal pricing with K_adj
        else:
            K_adj = None  # will use effective strike from average in AVERAGE_STRIKE branch

        # First moment of arithmetic average of future fixings
        M1 = sum(future_forwards) / N_f

        # Second moment
        # M2 = (1/N_f^2) * [sum_i F_i^2 * exp(sigma_i^2 * T_i)
        #                   + 2 * sum_{i<j} F_i * F_j * exp(sigma_i * sigma_j * T_i)]
        # using min(T_i, T_j) = T_i for i < j
        sum2 = 0.0
        for i, (F_i, T_i, sig_i) in enumerate(zip(future_forwards, future_times, future_vols)):
            # diagonal term
            sum2 += F_i ** 2 * math.exp(sig_i ** 2 * T_i)
            # off-diagonal terms (i < j), min(T_i, T_j) = T_i
            for j in range(i + 1, N_f):
                F_j = future_forwards[j]
                sig_j = future_vols[j]
                sum2 += 2.0 * F_i * F_j * math.exp(sig_i * sig_j * T_i)
        M2 = sum2 / (N_f ** 2)

        # Guard against M1=0 (shouldn't happen for FX forwards but be safe)
        if M1 <= 0 or M2 <= 0:
            return 0.0

        T1 = future_times[0]  # time to first future fixing

        log_ratio = math.log(M2 / M1 ** 2)
        if log_ratio <= 0:
            # Variance is negligible — treat as deterministic
            sigma_tw = 0.0
        else:
            sigma_tw = math.sqrt(log_ratio / max(T1, 1e-8))

        F_TW = M1  # effective forward = arithmetic mean of individual forwards
        weight = N_f / N_total  # scale by fraction of future fixings

        if payoff_type == AsianPayoffType.AVERAGE_PRICE:
            return weight * _black76_option(F_TW, K_adj, sigma_tw, T1, df_del, opt_type)
        else:
            # AVERAGE_STRIKE: treat F_final as the asset, F_TW as the effective strike
            # Effective vol is spread vol of S_T vs arithmetic average
            # Approximation: sigma_combined^2 = sigma_N^2 + sigma_TW^2 - 2*rho*sigma_N*sigma_TW
            # where rho = mean(T_i / T_N) (ratio of average time to final time)
            T_N = future_times[-1]
            sig_N = future_vols[-1]
            rho_avg = (sum(future_times) / N_f) / T_N if T_N > 0 else 0.0
            sigma_combined_sq = (sig_N ** 2 + sigma_tw ** 2
                                  - 2.0 * rho_avg * sig_N * sigma_tw)
            sigma_combined = math.sqrt(max(sigma_combined_sq, 1e-8))
            return weight * _black76_option(F_final, F_TW, sigma_combined, T_N, df_del, opt_type)

    # ------------------------------------------------------------------
    # Geometric exact closed-form
    # ------------------------------------------------------------------

    def _price_geometric(
        self,
        known_rates: List[float],
        future_forwards: List[float],
        future_times: List[float],
        future_vols: List[float],
        K: float,
        N_total: int,
        opt_type: OptionType,
        payoff_type: AsianPayoffType,
        df_del: float,
        F_final: float,
    ) -> float:
        """Exact geometric-average closed-form for Asian FX options.

        Returns unit NPV (per unit of notional_base).
        """
        N_f = len(future_forwards)
        N_k = len(known_rates)

        # Variance of geometric average: V_geo = (1/N_f^2) * sum_i sum_j sigma_i * sigma_j * min(T_i,T_j)
        V_geo = 0.0
        for i, (T_i, sig_i) in enumerate(zip(future_times, future_vols)):
            for j, (T_j, sig_j) in enumerate(zip(future_times, future_vols)):
                V_geo += sig_i * sig_j * min(T_i, T_j)
        V_geo /= N_f ** 2

        T_N = future_times[-1]
        sigma_geo = math.sqrt(V_geo / max(T_N, 1e-8))

        # Effective forward = geometric mean of individual forwards
        log_sum = sum(math.log(max(F, 1e-12)) for F in future_forwards)
        F_geo = math.exp(log_sum / N_f)

        weight = N_f / N_total

        if payoff_type == AsianPayoffType.AVERAGE_PRICE:
            known_contribution = (sum(known_rates) / N_total) if N_k > 0 else 0.0
            K_adj = K - known_contribution
            if K_adj <= 0.0 and opt_type == OptionType.CALL:
                return (known_contribution - K) * df_del
            return weight * _black76_option(F_geo, K_adj, sigma_geo, T_N, df_del, opt_type)
        else:
            # AVERAGE_STRIKE: asset = F_final, effective strike = F_geo
            sig_N = future_vols[-1]
            # Residual variance: var(S_T / A_geo) = sigma_N^2*T_N - V_geo
            residual_var = sig_N ** 2 * T_N - V_geo
            sigma_as = math.sqrt(max(residual_var, 1e-8))
            return weight * _black76_option(F_final, F_geo, sigma_as, T_N, df_del, opt_type)
