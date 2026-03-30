"""Business calendar: holiday set and business-day adjustment."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Set

from .enums import BusinessDayConvention


@dataclass
class Calendar:
    """A named set of holidays used to adjust dates to valid business days."""

    name: str
    holidays: Set[date] = field(default_factory=set)

    def is_business_day(self, d: date) -> bool:
        """Return True if *d* is a weekday that is not in the holiday set."""
        return d.weekday() < 5 and d not in self.holidays

    def adjust(self, d: date, convention: BusinessDayConvention) -> date:
        """Roll *d* to the nearest business day under *convention*."""
        if self.is_business_day(d):
            return d

        if convention == BusinessDayConvention.FOLLOWING:
            candidate = d
            while not self.is_business_day(candidate):
                candidate += timedelta(days=1)
            return candidate

        if convention == BusinessDayConvention.PRECEDING:
            candidate = d
            while not self.is_business_day(candidate):
                candidate -= timedelta(days=1)
            return candidate

        if convention == BusinessDayConvention.MODIFIED_FOLLOWING:
            # Roll forward first
            candidate = d
            while not self.is_business_day(candidate):
                candidate += timedelta(days=1)
            if candidate.month == d.month:
                return candidate
            # Forward roll crossed a month boundary — use preceding instead
            candidate = d
            while not self.is_business_day(candidate):
                candidate -= timedelta(days=1)
            return candidate

        raise ValueError(f"Unknown BusinessDayConvention: {convention}")
