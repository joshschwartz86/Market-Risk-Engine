class MarketRiskEngineError(Exception):
    """Base exception for the market risk engine."""


class MarketDataError(MarketRiskEngineError):
    """Raised when market data is missing, malformed, or inconsistent."""


class PortfolioParseError(MarketRiskEngineError):
    """Raised when the portfolio XML cannot be parsed or validated."""


class PricingError(MarketRiskEngineError):
    """Raised when a pricing calculation fails."""


class CalibrationError(MarketRiskEngineError):
    """Raised when a model calibration fails to converge."""


class SimulationError(MarketRiskEngineError):
    """Raised when a Monte Carlo simulation encounters an error."""


class CorrelationMatrixError(MarketRiskEngineError):
    """Raised when a correlation matrix is not valid (not PSD, not symmetric, etc.)."""
