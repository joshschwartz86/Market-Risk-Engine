"""Global configuration defaults for the Market Risk Engine."""

DEFAULT_DAY_COUNT = "ACT360"
DEFAULT_INTERPOLATION = "cubic_spline"
DEFAULT_COMPOUNDING = "CONTINUOUS"
DEFAULT_LOOKBACK_DAYS = 250
DEFAULT_VAR_CONFIDENCE_LEVELS = [0.95, 0.99]

# CCR simulation defaults
DEFAULT_N_PATHS = 10_000
DEFAULT_N_TIME_STEPS = 50
DEFAULT_TIME_HORIZON = 10.0  # years

# Peak exposure percentile
PEAK_EXPOSURE_PERCENTILE = 0.95

# SABR calibration defaults
SABR_DEFAULT_BETA = 0.5
SABR_MAX_ITER = 1000
SABR_TOL = 1e-8

# Nearest PSD regularisation
PSD_EPSILON = 1e-8
