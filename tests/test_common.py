"""Unit tests for common utilities: Calendar, generate_schedule, year_fraction."""
from datetime import date

import pytest

from market_risk_engine.common.calendar import Calendar
from market_risk_engine.common.date_utils import generate_schedule, year_fraction
from market_risk_engine.common.enums import BusinessDayConvention, normalise_day_count


# ---------------------------------------------------------------------------
# Day count normalisation
# ---------------------------------------------------------------------------

_D1 = date(2024, 1, 15)
_D2 = date(2025, 1, 15)

@pytest.mark.parametrize("verbose,canonical", [
    ("dayCount_Act_360", "ACT360"),
    ("dayCount_Act_365", "ACT365"),
    ("dayCount_30_360",  "30360"),
    ("dayCount_Act_Act", "ACTACT"),
])
def test_normalise_day_count_maps_verbose_to_canonical(verbose, canonical):
    assert normalise_day_count(verbose) == canonical


def test_normalise_day_count_passes_through_canonical():
    """Canonical strings are returned unchanged."""
    for s in ("ACT360", "ACT365", "30360", "ACTACT"):
        assert normalise_day_count(s) == s


@pytest.mark.parametrize("verbose,canonical", [
    ("dayCount_Act_360", "ACT360"),
    ("dayCount_Act_365", "ACT365"),
    ("dayCount_30_360",  "30360"),
    ("dayCount_Act_Act", "ACTACT"),
])
def test_year_fraction_verbose_equals_canonical(verbose, canonical):
    """year_fraction with verbose format returns the same value as canonical."""
    assert year_fraction(_D1, _D2, verbose) == year_fraction(_D1, _D2, canonical)


def test_year_fraction_unknown_day_count_raises():
    with pytest.raises(ValueError):
        year_fraction(_D1, _D2, "UNKNOWN_FORMAT")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cal(*holidays: date) -> Calendar:
    return Calendar(name="TEST", holidays=set(holidays))


# ---------------------------------------------------------------------------
# Calendar.is_business_day
# ---------------------------------------------------------------------------

def test_weekday_no_holiday_is_business_day():
    cal = _cal()
    assert cal.is_business_day(date(2024, 1, 15))   # Monday


def test_saturday_not_business_day():
    cal = _cal()
    assert not cal.is_business_day(date(2024, 1, 13))  # Saturday


def test_sunday_not_business_day():
    cal = _cal()
    assert not cal.is_business_day(date(2024, 1, 14))  # Sunday


def test_holiday_weekday_not_business_day():
    holiday = date(2024, 1, 15)  # Monday
    cal = _cal(holiday)
    assert not cal.is_business_day(holiday)


# ---------------------------------------------------------------------------
# Calendar.adjust — FOLLOWING
# ---------------------------------------------------------------------------

def test_following_weekday_unchanged():
    cal = _cal()
    d = date(2024, 1, 15)  # Monday
    assert cal.adjust(d, BusinessDayConvention.FOLLOWING) == d


def test_following_saturday_rolls_to_monday():
    cal = _cal()
    sat = date(2024, 1, 13)   # Saturday
    assert cal.adjust(sat, BusinessDayConvention.FOLLOWING) == date(2024, 1, 15)


def test_following_sunday_rolls_to_monday():
    cal = _cal()
    sun = date(2024, 1, 14)   # Sunday
    assert cal.adjust(sun, BusinessDayConvention.FOLLOWING) == date(2024, 1, 15)


def test_following_holiday_rolls_forward():
    # Wednesday is a holiday → rolls to Thursday
    holiday = date(2024, 1, 17)  # Wednesday
    cal = _cal(holiday)
    assert cal.adjust(holiday, BusinessDayConvention.FOLLOWING) == date(2024, 1, 18)


def test_following_consecutive_holidays_roll_to_next_business_day():
    thu = date(2024, 1, 18)
    fri = date(2024, 1, 19)
    cal = _cal(thu, fri)
    # Thursday + Friday both holidays → rolls to Monday
    assert cal.adjust(thu, BusinessDayConvention.FOLLOWING) == date(2024, 1, 22)


# ---------------------------------------------------------------------------
# Calendar.adjust — PRECEDING
# ---------------------------------------------------------------------------

def test_preceding_weekday_unchanged():
    cal = _cal()
    d = date(2024, 1, 15)  # Monday
    assert cal.adjust(d, BusinessDayConvention.PRECEDING) == d


def test_preceding_saturday_rolls_to_friday():
    cal = _cal()
    sat = date(2024, 1, 13)   # Saturday
    assert cal.adjust(sat, BusinessDayConvention.PRECEDING) == date(2024, 1, 12)


def test_preceding_sunday_rolls_to_friday():
    cal = _cal()
    sun = date(2024, 1, 14)   # Sunday
    assert cal.adjust(sun, BusinessDayConvention.PRECEDING) == date(2024, 1, 12)


def test_preceding_holiday_rolls_backward():
    holiday = date(2024, 1, 17)  # Wednesday
    cal = _cal(holiday)
    assert cal.adjust(holiday, BusinessDayConvention.PRECEDING) == date(2024, 1, 16)


# ---------------------------------------------------------------------------
# Calendar.adjust — MODIFIED_FOLLOWING
# ---------------------------------------------------------------------------

def test_modified_following_mid_month_rolls_forward():
    cal = _cal()
    sat = date(2024, 1, 13)   # Saturday, mid-month
    # Forward roll stays in January → same as FOLLOWING
    assert cal.adjust(sat, BusinessDayConvention.MODIFIED_FOLLOWING) == date(2024, 1, 15)


def test_modified_following_month_end_rolls_backward():
    cal = _cal()
    # Saturday 2024-02-03 is mid-month; test a true month-end case:
    # 2024-03-31 is a Sunday → forward would go to 2024-04-01 (April) → use preceding
    sun_month_end = date(2024, 3, 31)  # Sunday, last day of March
    result = cal.adjust(sun_month_end, BusinessDayConvention.MODIFIED_FOLLOWING)
    assert result == date(2024, 3, 29)   # Friday in March


def test_modified_following_holiday_at_month_end_rolls_backward():
    # Mark Friday 2024-03-29 as a holiday too, so Saturday + Friday are both non-business
    friday_before = date(2024, 3, 29)
    cal = _cal(friday_before)
    sun_month_end = date(2024, 3, 31)
    result = cal.adjust(sun_month_end, BusinessDayConvention.MODIFIED_FOLLOWING)
    assert result == date(2024, 3, 28)  # Thursday in March


# ---------------------------------------------------------------------------
# generate_schedule with calendar
# ---------------------------------------------------------------------------

def test_generate_schedule_no_calendar_returns_raw_dates():
    dates = generate_schedule(date(2024, 1, 15), date(2024, 7, 15), "SEMIANNUAL")
    assert dates == [date(2024, 7, 15)]


def test_generate_schedule_calendar_adjusts_weekend():
    # 2024-07-15 is a Monday — no adjustment needed
    # Use a date that falls on a weekend: maturity 2024-07-13 (Saturday)
    # With FOLLOWING it should roll to 2024-07-15 (Monday)
    cal = _cal()
    dates = generate_schedule(
        date(2024, 1, 13), date(2024, 7, 13), "SEMIANNUAL",
        calendar=cal, convention=BusinessDayConvention.FOLLOWING,
    )
    assert dates == [date(2024, 7, 15)]


def test_generate_schedule_holiday_rolls_forward():
    holiday = date(2024, 4, 15)  # Monday — mark as holiday
    cal = _cal(holiday)
    dates = generate_schedule(
        date(2024, 1, 15), date(2024, 4, 15), "QUARTERLY",
        calendar=cal, convention=BusinessDayConvention.FOLLOWING,
    )
    # Should roll forward to Tuesday 2024-04-16
    assert dates == [date(2024, 4, 16)]


def test_generate_schedule_without_calendar_unchanged():
    """Omitting calendar leaves dates unadjusted (backward-compatible)."""
    dates_plain = generate_schedule(date(2024, 1, 15), date(2025, 1, 15), "QUARTERLY")
    dates_no_cal = generate_schedule(
        date(2024, 1, 15), date(2025, 1, 15), "QUARTERLY",
        calendar=None,
    )
    assert dates_plain == dates_no_cal
