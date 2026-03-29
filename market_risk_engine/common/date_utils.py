from datetime import date
from .enums import DayCount


def year_fraction(start: date, end: date, day_count: str) -> float:
    """Compute the year fraction between two dates under the given day count convention."""
    dc = DayCount(day_count) if not isinstance(day_count, DayCount) else day_count

    if dc == DayCount.ACT360:
        return (end - start).days / 360.0

    if dc == DayCount.ACT365:
        return (end - start).days / 365.0

    if dc == DayCount.ACT_ACT:
        return (end - start).days / 365.25

    if dc == DayCount.DC30_360:
        d1, m1, y1 = start.day, start.month, start.year
        d2, m2, y2 = end.day, end.month, end.year
        d1 = min(d1, 30)
        if d1 == 30:
            d2 = min(d2, 30)
        days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        return days / 360.0

    raise ValueError(f"Unsupported day count convention: {day_count}")


def add_tenor(base: date, tenor_str: str) -> date:
    """Add a tenor string like '3M', '1Y', '6M' to a base date."""
    import calendar

    tenor_str = tenor_str.upper().strip()
    if tenor_str.endswith("D"):
        from datetime import timedelta
        return base + timedelta(days=int(tenor_str[:-1]))
    if tenor_str.endswith("W"):
        from datetime import timedelta
        return base + timedelta(weeks=int(tenor_str[:-1]))
    if tenor_str.endswith("M"):
        months = int(tenor_str[:-1])
        m = base.month - 1 + months
        year = base.year + m // 12
        month = m % 12 + 1
        day = min(base.day, calendar.monthrange(year, month)[1])
        return date(year, month, day)
    if tenor_str.endswith("Y"):
        years = int(tenor_str[:-1])
        try:
            return base.replace(year=base.year + years)
        except ValueError:
            return base.replace(year=base.year + years, day=28)

    raise ValueError(f"Cannot parse tenor string: {tenor_str}")


def frequency_to_period(frequency: str) -> float:
    """Return the year fraction between payments for a given frequency string."""
    freq_map = {
        "MONTHLY": 1 / 12,
        "QUARTERLY": 0.25,
        "SEMIANNUAL": 0.5,
        "ANNUAL": 1.0,
    }
    key = frequency.upper()
    if key not in freq_map:
        raise ValueError(f"Unknown frequency: {frequency}")
    return freq_map[key]


def generate_schedule(effective: date, maturity: date, frequency: str) -> list[date]:
    """Generate a list of payment dates from effective to maturity at the given frequency."""
    from datetime import timedelta
    import calendar

    period_months = {
        "MONTHLY": 1,
        "QUARTERLY": 3,
        "SEMIANNUAL": 6,
        "ANNUAL": 12,
    }[frequency.upper()]

    dates = []
    current = maturity
    while current > effective:
        dates.append(current)
        m = current.month - 1 - period_months
        years_back = 0
        while m < 0:
            m += 12
            years_back += 1
        year = current.year - years_back
        month = m + 1
        day = min(current.day, calendar.monthrange(year, month)[1])
        current = date(year, month, day)

    dates.reverse()
    return dates
