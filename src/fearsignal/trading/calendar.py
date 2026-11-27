"""NYSE sessions after the last bar, for dating live entries and exits.

Past sessions come from the price index. Future ones are weekdays minus the
exchange's full-day holidays (NYSE's published 2026-2027 schedule). Early
closes are still sessions.
"""

from __future__ import annotations

import pandas as pd

NYSE_HOLIDAYS = pd.to_datetime([
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
])


def sessions(known: pd.DatetimeIndex, n_future: int = 80) -> pd.DatetimeIndex:
    """Known sessions followed by the next ``n_future`` exchange sessions."""
    last = known[-1]
    days = pd.bdate_range(last + pd.Timedelta(days=1), periods=n_future * 2)
    fut = days[~days.isin(NYSE_HOLIDAYS)][:n_future]
    return known.append(fut)


def offset(cal: pd.DatetimeIndex, day: pd.Timestamp, k: int) -> pd.Timestamp:
    """The session k sessions after ``day`` on calendar ``cal``."""
    i = int(cal.get_loc(pd.Timestamp(day)))
    return cal[i + k]
