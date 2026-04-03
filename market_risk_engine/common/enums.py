from enum import Enum


class DayCount(str, Enum):
    ACT360 = "ACT360"
    ACT365 = "ACT365"
    DC30_360 = "30360"
    ACT_ACT = "ACTACT"


_DAYCOUNT_ALIASES: dict[str, str] = {
    "dayCount_Act_360": "ACT360",
    "dayCount_Act_365": "ACT365",
    "dayCount_30_360":  "30360",
    "dayCount_Act_Act": "ACTACT",
}


def normalise_day_count(s: str) -> str:
    """Map verbose incoming day count format to the canonical DayCount string.

    Accepts both the compact canonical form (e.g. ``"ACT360"``) and the
    verbose form supplied by upstream systems (e.g. ``"dayCount_Act_360"``).
    Unknown strings are returned unchanged so that ``DayCount(...)`` can
    raise its own ``ValueError`` with a meaningful message.
    """
    return _DAYCOUNT_ALIASES.get(s, s)


class Compounding(str, Enum):
    CONTINUOUS = "CONTINUOUS"
    ANNUAL = "ANNUAL"
    SEMIANNUAL = "SEMIANNUAL"
    QUARTERLY = "QUARTERLY"


class Frequency(str, Enum):
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMIANNUAL = "SEMIANNUAL"
    ANNUAL = "ANNUAL"


class PayReceive(str, Enum):
    PAY = "PAY"
    RECEIVE = "RECEIVE"


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"
    CAP = "CAP"
    FLOOR = "FLOOR"
    PAYER = "PAYER"
    RECEIVER = "RECEIVER"


class VolType(str, Enum):
    LOGNORMAL = "lognormal"
    NORMAL = "normal"


class StochasticProcess(str, Enum):
    GBM = "gbm"
    VASICEK = "vasicek"
    CIR = "cir"
    HULL_WHITE = "hullwhite"


class FactorType(str, Enum):
    YIELD_CURVE = "yield_curve"
    FX_SPOT = "fx_spot"
    COMMODITY = "commodity"
    VOL = "vol"


class BusinessDayConvention(str, Enum):
    FOLLOWING = "FOLLOWING"
    MODIFIED_FOLLOWING = "MODIFIED_FOLLOWING"
    PRECEDING = "PRECEDING"


class AveragingType(str, Enum):
    ARITHMETIC = "ARITHMETIC"
    GEOMETRIC = "GEOMETRIC"


class AsianPayoffType(str, Enum):
    AVERAGE_PRICE = "AVERAGE_PRICE"    # payoff vs fixed strike K
    AVERAGE_STRIKE = "AVERAGE_STRIKE"  # payoff vs avg(S) as strike
