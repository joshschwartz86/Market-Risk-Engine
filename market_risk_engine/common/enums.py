from enum import Enum


class DayCount(str, Enum):
    ACT360 = "ACT360"
    ACT365 = "ACT365"
    DC30_360 = "30360"
    ACT_ACT = "ACTACT"


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
