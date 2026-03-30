"""Hull-White 1-factor model calibration and Bermudan swaption trinomial-tree pricer."""
from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.optimize as sopt
import scipy.stats as st

from ..common.date_utils import year_fraction
from ..common.enums import OptionType
from ..common.exceptions import PricingError
from ..layer1_market_data.vol_surface import VolSurfaceInterpolator
from ..layer1_market_data.yield_curve import YieldCurveInterpolator
from ..layer2_portfolio.models import BermudanSwaption
from .base import MarketSnapshot, PricingEngine, PricingResult


# ---------------------------------------------------------------------------
# Hull-White 1-Factor Model
# ---------------------------------------------------------------------------

class HullWhiteModel:
    """
    Time-homogeneous Hull-White 1F short-rate model.

        dr(t) = [theta(t) - a * r(t)] dt + sigma * dW(t)

    theta(t) is chosen to fit the initial discount curve exactly.

    Parameters
    ----------
    a     : mean-reversion speed  (> 0)
    sigma : short-rate volatility (> 0)
    disc  : interpolator for the initial discount curve P(0, T)
    """

    def __init__(self, a: float, sigma: float,
                 disc: YieldCurveInterpolator) -> None:
        self.a = float(a)
        self.sigma = float(sigma)
        self._disc = disc

    # ------------------------------------------------------------------
    # Core HW building blocks
    # ------------------------------------------------------------------

    def B(self, t: float, T: float) -> float:
        """B(t,T) = (1 - exp(-a*(T-t))) / a.  Limit a->0 gives T-t."""
        tau = T - t
        if tau <= 0.0:
            return 0.0
        a = self.a
        if a < 1e-9:
            return tau
        return (1.0 - math.exp(-a * tau)) / a

    def _inst_fwd(self, t: float, eps: float = 1e-4) -> float:
        """Instantaneous forward rate f(0,t) via central finite difference."""
        t1 = max(t - eps, 1e-6)
        t2 = t + eps
        df1 = self._disc.discount_factor(t1)
        df2 = self._disc.discount_factor(t2)
        if df1 <= 0.0 or df2 <= 0.0:
            return self._disc.zero_rate(max(t, 1e-6))
        return -math.log(df2 / df1) / (t2 - t1)

    def _var_x(self, t: float) -> float:
        """Var[x(t)] = sigma^2 * (1 - exp(-2*a*t)) / (2*a)."""
        a, s = self.a, self.sigma
        if a < 1e-9:
            return s * s * t
        return s * s * (1.0 - math.exp(-2.0 * a * t)) / (2.0 * a)

    def _log_A(self, t: float, T: float) -> float:
        """
        log A(t,T) where P(t,T;r) = A(t,T) * exp(-B(t,T)*r).

        log A = log(P(0,T)/P(0,t)) + B(t,T)*f(0,t) - 0.5*B(t,T)^2 * Var[x(t)]
        """
        P0t = self._disc.discount_factor(t)
        P0T = self._disc.discount_factor(T)
        if P0t <= 0.0:
            return math.log(max(P0T, 1e-15))
        Bval = self.B(t, T)
        f0t = self._inst_fwd(t)
        var_t = self._var_x(t)
        return math.log(P0T / P0t) + Bval * f0t - 0.5 * Bval * Bval * var_t

    def bond_price(self, t: float, T: float, r: float) -> float:
        """Analytical zero-coupon bond price P(t,T) given short rate r(t)=r."""
        if T <= t:
            return 1.0
        return math.exp(self._log_A(t, T) - self.B(t, T) * r)

    def alpha(self, t: float) -> float:
        """
        Drift correction alpha(t) so that the tree fits the initial term structure.

            alpha(t) = f(0,t) + sigma^2/(2*a^2) * (1 - exp(-a*t))^2
        """
        a, s = self.a, self.sigma
        f = self._inst_fwd(t)
        if a < 1e-9:
            return f + 0.5 * s * s * t * t
        return f + s * s / (2.0 * a * a) * (1.0 - math.exp(-a * t)) ** 2

    # ------------------------------------------------------------------
    # European swaption via Jamshidian decomposition
    # ------------------------------------------------------------------

    def european_swaption_price(
        self,
        T0: float,
        payment_times: List[float],
        coupon_amounts: List[float],
        notional: float,
        opt_type: OptionType,
    ) -> float:
        """
        Analytical European swaption price using Jamshidian's decomposition.

        Parameters
        ----------
        T0             : option expiry (year fraction from today)
        payment_times  : coupon/principal cash-flow dates (year fractions > T0)
        coupon_amounts : c_i = strike*dt for intermediate; strike*dt+1 for final
        notional       : trade notional
        opt_type       : PAYER (pay fixed) or RECEIVER (receive fixed)

        Returns NPV in same currency units as notional.
        """
        if T0 <= 0.0 or not payment_times:
            return 0.0

        a, s = self.a, self.sigma

        # --- Find r* such that sum(c_i * P(T0, T_i; r*)) = 1 ---
        def swap_val(r_star: float) -> float:
            return (sum(ci * self.bond_price(T0, ti, r_star)
                        for ci, ti in zip(coupon_amounts, payment_times)) - 1.0)

        try:
            r_star = sopt.brentq(swap_val, -1.0, 5.0, xtol=1e-12)
        except ValueError:
            # Deep ITM or OTM: return approximate intrinsic
            pv0 = swap_val(0.0) + 1.0
            if opt_type == OptionType.PAYER:
                return notional * max(0.0, 1.0 - pv0)
            else:
                return notional * max(0.0, pv0 - 1.0)

        # sqrt((1-e^{-2aT0})/(2a)) — common factor for all sigma_p_i
        if a < 1e-9:
            sqrt_var = s * math.sqrt(T0)
        else:
            sqrt_var = s * math.sqrt((1.0 - math.exp(-2.0 * a * T0)) / (2.0 * a))

        P0_T0 = self._disc.discount_factor(T0)
        price = 0.0

        for ci, ti in zip(coupon_amounts, payment_times):
            if ti <= T0:
                continue
            Ki = self.bond_price(T0, ti, r_star)
            sigma_pi = sqrt_var * self.B(T0, ti)
            if sigma_pi <= 0.0:
                continue
            P0_Ti = self._disc.discount_factor(ti)
            h_i = math.log(P0_Ti / (Ki * P0_T0)) / sigma_pi + sigma_pi / 2.0

            if opt_type == OptionType.PAYER:
                # Sum of bond puts: Ki*P(0,T0)*N(sigma_pi - h_i) - P(0,Ti)*N(-h_i)
                bond_opt = (Ki * P0_T0 * st.norm.cdf(sigma_pi - h_i)
                            - P0_Ti * st.norm.cdf(-h_i))
            else:
                # Sum of bond calls: P(0,Ti)*N(h_i) - Ki*P(0,T0)*N(h_i - sigma_pi)
                bond_opt = (P0_Ti * st.norm.cdf(h_i)
                            - Ki * P0_T0 * st.norm.cdf(h_i - sigma_pi))

            price += ci * bond_opt

        return notional * price

    def european_swaption_normal_vol(
        self,
        T0: float,
        payment_times: List[float],
        coupon_amounts: List[float],
        notional: float,
        opt_type: OptionType,
        annuity: float,
        fsr: float,
        strike: float,
    ) -> float:
        """
        Invert the Bachelier (normal) formula to extract the implied vol
        that matches the HW analytical European swaption price.
        """
        hw_price = self.european_swaption_price(
            T0, payment_times, coupon_amounts, notional, opt_type
        )
        denom = notional * annuity
        if denom <= 0.0:
            return 0.0
        unit_price = hw_price / denom

        F, K = fsr, strike
        intrinsic = (max(F - K, 0.0) if opt_type == OptionType.PAYER
                     else max(K - F, 0.0))
        if unit_price <= intrinsic + 1e-14:
            return 0.0

        sqrt_T = math.sqrt(max(T0, 1e-8))

        def bachelier(sig: float) -> float:
            if sig <= 0.0:
                return intrinsic
            d = (F - K) / (sig * sqrt_T)
            if opt_type == OptionType.PAYER:
                return (F - K) * st.norm.cdf(d) + sig * sqrt_T * st.norm.pdf(d)
            else:
                return (K - F) * st.norm.cdf(-d) + sig * sqrt_T * st.norm.pdf(d)

        try:
            return sopt.brentq(lambda s: bachelier(s) - unit_price,
                               1e-8, 0.5, xtol=1e-9)
        except ValueError:
            return 0.0


# ---------------------------------------------------------------------------
# Helper: build coterminal swaption data
# ---------------------------------------------------------------------------

def _coterminal_data(
    exercise_dates: List[date],
    swap_maturity: date,
    strike: float,
    payment_frequency: str,
    day_count: str,
    today: date,
    disc: YieldCurveInterpolator,
) -> List[dict]:
    """
    For each exercise date, build the payment schedule of the coterminal swap
    (same maturity, exercised at that date) and compute its annuity and FSR.
    """
    freq_map = {"MONTHLY": 12, "QUARTERLY": 4, "SEMIANNUAL": 2, "ANNUAL": 1}
    dt_swap = 1.0 / freq_map[payment_frequency.upper()]
    t_N = year_fraction(today, swap_maturity, day_count)

    entries: List[dict] = []
    for ex_date in exercise_dates:
        t0 = year_fraction(today, ex_date, day_count)
        if t0 <= 0.0 or t0 >= t_N - 1e-6:
            continue

        payment_times: List[float] = []
        coupon_amounts: List[float] = []
        t = t0 + dt_swap
        while t <= t_N + 1e-9:
            is_last = (t + dt_swap > t_N + 1e-9)
            payment_times.append(t)
            coupon_amounts.append(strike * dt_swap + (1.0 if is_last else 0.0))
            t += dt_swap

        if not payment_times:
            continue

        annuity = sum(dt_swap * disc.discount_factor(ti) for ti in payment_times)
        df_t0 = disc.discount_factor(t0)
        df_tN = disc.discount_factor(t_N)
        fsr = (df_t0 - df_tN) / annuity if annuity > 1e-12 else strike

        entries.append({
            "t0": t0,
            "payment_times": payment_times,
            "coupon_amounts": coupon_amounts,
            "annuity": annuity,
            "fsr": fsr,
        })

    return entries


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_hull_white(
    exercise_dates: List[date],
    swap_maturity: date,
    strike: float,
    payment_frequency: str,
    day_count: str,
    opt_type: OptionType,
    today: date,
    disc: YieldCurveInterpolator,
    vol_interp: VolSurfaceInterpolator,
    notional: float = 1.0,
) -> Tuple[float, float]:
    """
    Calibrate Hull-White (a, sigma) to coterminal European swaption normal vols.

    Minimises the sum of squared differences between HW-implied and
    market Bachelier vols across all coterminal swaptions.

    Returns
    -------
    (a, sigma) : calibrated mean-reversion and short-rate vol.
    """
    entries = _coterminal_data(
        exercise_dates, swap_maturity, strike, payment_frequency,
        day_count, today, disc,
    )
    if not entries:
        raise PricingError("No valid coterminal swaptions for HW calibration.")

    market_vols = [
        vol_interp.get_vol(e["t0"], strike, forward=e["fsr"])
        for e in entries
    ]

    def objective(log_params: np.ndarray) -> float:
        a = math.exp(log_params[0])
        sigma = math.exp(log_params[1])
        try:
            hw = HullWhiteModel(a, sigma, disc)
        except Exception:
            return 1e10
        err = 0.0
        for e, sv_mkt in zip(entries, market_vols):
            try:
                sv_hw = hw.european_swaption_normal_vol(
                    e["t0"], e["payment_times"], e["coupon_amounts"],
                    notional, opt_type, e["annuity"], e["fsr"], strike,
                )
                err += (sv_hw - sv_mkt) ** 2
            except Exception:
                err += 1.0
        return err

    # --- Grid search for a robust initial guess ---
    sigma0 = float(np.mean(market_vols)) if market_vols else 0.01
    best_x0 = np.array([math.log(0.05), math.log(max(sigma0, 1e-5))])
    best_err = objective(best_x0)

    for a_try in (0.01, 0.03, 0.05, 0.10, 0.20, 0.50):
        x_try = np.array([math.log(a_try), math.log(max(sigma0, 1e-5))])
        err = objective(x_try)
        if err < best_err:
            best_err = err
            best_x0 = x_try

    result = sopt.minimize(
        objective,
        x0=best_x0,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 3000, "adaptive": True},
    )

    a_cal = float(np.clip(math.exp(result.x[0]), 0.001, 2.0))
    s_cal = float(np.clip(math.exp(result.x[1]), 1e-5, 0.20))
    return a_cal, s_cal


# ---------------------------------------------------------------------------
# Trinomial tree — backward induction
# ---------------------------------------------------------------------------

def _price_bermudan_tree(
    hw: HullWhiteModel,
    T_N: float,
    n_steps: int,
    exercise_steps: Set[int],
    step_to_paydata: Dict[int, Tuple[List[float], List[float]]],
    notional: float,
    opt_type: OptionType,
) -> float:
    """
    Hull-White trinomial tree backward induction.

    Tree state variable:  x(t) = r(t) - alpha(t)
    Short rate at node (step, j):  r = alpha(t_i) + j * dx

    Branching (Hull & White 1994):
      Normal nodes   |j| < jmax : up/mid/down to j+1, j, j-1
      Upper boundary j == jmax  : downward shift  to j, j-1, j-2
      Lower boundary j == -jmax : upward shift    to j+2, j+1, j

    Returns the option value at t=0, x=0 (tree root).
    """
    a = hw.a
    sigma = hw.sigma
    dt = T_N / n_steps
    dx = sigma * math.sqrt(3.0 * dt)

    # jmax: largest j for which normal branching gives valid probabilities
    # Normal pm = 2/3 - eta^2 >= 0  =>  |eta| = a*|j|*dt <= sqrt(2/3) ~ 0.816
    # Standard practice caps at 0.184 / (a*dt) for all probs to be comfortably positive
    if a < 1e-9:
        jmax = 50
    else:
        jmax = max(int(math.ceil(0.184 / (a * dt))) + 1, 3)
    jmax = min(jmax, 300)

    n_states = 2 * jmax + 1

    # Pre-compute alpha at each time step
    alphas = [hw.alpha(i * dt) for i in range(n_steps)]

    def _probs(j: int) -> Tuple[float, float, float, int, int, int]:
        """Return (pu, pm, pd, j_up, j_mid, j_dn)."""
        eta = -a * j * dt  # mean of Δj in dx-units
        if j >= jmax:
            # Upper boundary: branches → j, j-1, j-2
            pu = 7.0 / 6.0 + (3.0 * eta + eta * eta) / 2.0
            pm = -1.0 / 3.0 - 2.0 * eta - eta * eta
            pd = 1.0 / 6.0 + (eta + eta * eta) / 2.0
            return pu, pm, pd, j, j - 1, j - 2
        elif j <= -jmax:
            # Lower boundary: branches → j+2, j+1, j
            pu = 1.0 / 6.0 + (eta * eta - eta) / 2.0
            pm = -1.0 / 3.0 + 2.0 * eta - eta * eta
            pd = 7.0 / 6.0 + (eta * eta - 3.0 * eta) / 2.0
            return pu, pm, pd, j + 2, j + 1, j
        else:
            # Normal branching: branches → j+1, j, j-1
            pu = 1.0 / 6.0 + (eta * eta + eta) / 2.0
            pm = 2.0 / 3.0 - eta * eta
            pd = 1.0 / 6.0 + (eta * eta - eta) / 2.0
            return pu, pm, pd, j + 1, j, j - 1

    # Initialise terminal values (expired, worth 0)
    V = np.zeros(n_states)

    for step in range(n_steps - 1, -1, -1):
        t_curr = step * dt
        alpha_t = alphas[step]
        V_new = np.zeros(n_states)

        is_exercise = step in exercise_steps
        if is_exercise:
            pt, cas = step_to_paydata[step]

        for j in range(-jmax, jmax + 1):
            idx = j + jmax
            r = alpha_t + j * dx
            df = math.exp(-r * dt)

            pu, pm, pd, ju, jm, jd = _probs(j)

            # Clamp successor indices (safety net at boundaries)
            ju = max(-jmax, min(jmax, ju))
            jm = max(-jmax, min(jmax, jm))
            jd = max(-jmax, min(jmax, jd))

            cont = df * (pu * V[ju + jmax] + pm * V[jm + jmax] + pd * V[jd + jmax])

            if is_exercise:
                # Intrinsic = max(0, swap PV at this node)
                pv_bonds = sum(
                    ci * hw.bond_price(t_curr, ti, r)
                    for ci, ti in zip(cas, pt)
                )
                if opt_type == OptionType.PAYER:
                    intrinsic = notional * max(0.0, 1.0 - pv_bonds)
                else:
                    intrinsic = notional * max(0.0, pv_bonds - 1.0)
                V_new[idx] = max(cont, intrinsic)
            else:
                V_new[idx] = cont

        V = V_new

    return float(V[jmax])  # root node: t=0, j=0


# ---------------------------------------------------------------------------
# Public pricer
# ---------------------------------------------------------------------------

class BermudanSwaptionPricer(PricingEngine):
    """
    Bermudan swaption pricer: Hull-White 1F trinomial tree.

    Workflow
    --------
    1. Collect coterminal European swaption vols from the normal vol surface.
    2. Calibrate HW (a, sigma) by minimising squared vol errors.
    3. Build a uniform trinomial tree with n_tree_steps steps to T_N.
    4. Backward-induct, applying early-exercise at each exercise date.
    5. Return NPV; vega estimated via a +1bp sigma bump.
    """

    def price(self, trade: BermudanSwaption,  # type: ignore[override]
              market: MarketSnapshot) -> PricingResult:
        try:
            return self._price(trade, market)
        except Exception as exc:
            return PricingResult(
                trade_id=trade.trade_id, npv=float("nan"),
                currency=trade.currency, error=str(exc),
            )

    def _price(self, trade: BermudanSwaption,
               market: MarketSnapshot) -> PricingResult:
        # ---- validate market data ----
        for cid in (trade.discount_curve_id, trade.forward_curve_id):
            if cid not in market.yield_curves:
                raise PricingError(f"Missing curve '{cid}'.")
        if trade.vol_surface_id not in market.vol_surfaces:
            raise PricingError(f"Missing vol surface '{trade.vol_surface_id}'.")

        today = market.as_of_date
        disc = YieldCurveInterpolator(market.yield_curves[trade.discount_curve_id])
        vol_interp = VolSurfaceInterpolator(market.vol_surfaces[trade.vol_surface_id])
        day_count = trade.day_count

        # ---- calendar-adjust exercise dates ----
        cal = market.calendars.get(trade.calendar_name) if trade.calendar_name else None
        ex_dates = sorted(
            (cal.adjust(d, trade.business_day_convention) if cal else d)
            for d in trade.exercise_dates
        )
        ex_dates = [d for d in ex_dates if d > today]

        if not ex_dates:
            return PricingResult(trade_id=trade.trade_id, npv=0.0,
                                 currency=trade.currency)

        t_N = year_fraction(today, trade.underlying_maturity, day_count)
        if t_N <= 0.0:
            return PricingResult(trade_id=trade.trade_id, npv=0.0,
                                 currency=trade.currency)

        # ---- calibrate HW to coterminal swaptions ----
        a, sigma = calibrate_hull_white(
            ex_dates, trade.underlying_maturity, trade.strike,
            trade.payment_frequency, day_count, trade.option_type,
            today, disc, vol_interp, notional=1.0,
        )
        hw = HullWhiteModel(a, sigma, disc)

        # ---- build exercise-step map ----
        n_steps = trade.n_tree_steps
        dt = t_N / n_steps

        freq_map = {"MONTHLY": 12, "QUARTERLY": 4, "SEMIANNUAL": 2, "ANNUAL": 1}
        dt_swap = 1.0 / freq_map[trade.payment_frequency.upper()]

        exercise_steps: Set[int] = set()
        step_to_paydata: Dict[int, Tuple[List[float], List[float]]] = {}

        for ex_date in ex_dates:
            t0 = year_fraction(today, ex_date, day_count)
            step = int(round(t0 / dt))
            step = max(0, min(step, n_steps - 1))

            payment_times: List[float] = []
            coupon_amounts: List[float] = []
            t = t0 + dt_swap
            while t <= t_N + 1e-9:
                is_last = (t + dt_swap > t_N + 1e-9)
                payment_times.append(t)
                coupon_amounts.append(
                    trade.strike * dt_swap + (1.0 if is_last else 0.0)
                )
                t += dt_swap

            if payment_times:
                exercise_steps.add(step)
                step_to_paydata[step] = (payment_times, coupon_amounts)

        if not exercise_steps:
            return PricingResult(trade_id=trade.trade_id, npv=0.0,
                                 currency=trade.currency)

        # ---- price via backward induction ----
        npv = _price_bermudan_tree(
            hw, t_N, n_steps, exercise_steps, step_to_paydata,
            trade.notional, trade.option_type,
        )

        # ---- vega: bump sigma by +1bp and reprice (no recalibration) ----
        hw_bumped = HullWhiteModel(a, sigma + 0.0001, disc)
        npv_bumped = _price_bermudan_tree(
            hw_bumped, t_N, n_steps, exercise_steps, step_to_paydata,
            trade.notional, trade.option_type,
        )
        vega = npv_bumped - npv

        return PricingResult(
            trade_id=trade.trade_id,
            npv=npv,
            currency=trade.currency,
            vega=vega,
        )
