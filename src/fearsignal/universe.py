"""The AI-stock universe.

Inclusion rule
--------------
A US-listed common stock or ADR is included when a material part of its
business during 2023-2026 sits in the AI value chain, in one of eleven
segments: accelerated compute silicon, foundry/memory, semiconductor capital
equipment, EDA/IP, networking and optics, AI systems, data-center power and
cooling, data-center and GPU-cloud operators, AI platforms, AI/data software,
and AI-native applications.

Exclusions are deliberate: bitcoin miners whose AI-hosting pivot came late in
the sample (their 2023-24 risk is crypto risk), quantum-computing names, and
companies whose history the data vendor no longer serves under the ticker
(ANSS, JNPR, INFN, SPLK, CFLT, PSTG).

Later listings (IPOs, spin-offs, de-SPACs, relistings) are included and become
eligible automatically once they have ``SIGMA_WINDOW`` sessions of returns
inside the era. ``trade_start`` overrides the vendor's first date where the
vendor history begins with a pre-merger SPAC shell or a pre-relisting stub,
which would otherwise produce a meaningless scale estimate.

Known limitation: the list is drawn up with 2026 knowledge (survivorship and
"AI-winner" selection). The index is a contemporaneous thermometer, so this
shifts *which* stocks it averages, not *when* it reads high; it matters more
for any forward-return test, which is reported with that caveat.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Member:
    ticker: str
    segment: str
    trade_start: str | None = None  # ISO date; first session of the operating company


_SEGMENTS: dict[str, tuple[str, ...]] = {
    "compute": (
        "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MRVL", "ARM", "ALAB", "CRDO",
        "MPWR", "LSCC", "AMBA", "SITM", "NVTS", "MTSI", "SMTC", "HIMX",
    ),
    "foundry_memory": (
        "TSM", "MU", "WDC", "STX", "SNDK", "AMKR", "ASX", "GFS", "TSEM", "SIMO",
    ),
    "semicap": (
        "ASML", "AMAT", "LRCX", "KLAC", "TER", "ONTO", "CAMT", "NVMI", "ACMR",
        "AEHR", "FORM", "KLIC", "ENTG", "MKSI", "UCTT", "ICHR", "VECO", "PLAB",
        "PDFS",
    ),
    "eda_ip": ("SNPS", "CDNS", "RMBS", "CEVA"),
    "networking_optics": (
        "ANET", "CSCO", "CIEN", "LITE", "COHR", "AAOI", "FN", "APH", "GLW", "KEYS",
    ),
    "systems": ("SMCI", "DELL", "HPE", "CLS", "NTAP", "JBL", "FLEX"),
    "power_cooling": (
        "VRT", "ETN", "GEV", "PWR", "NVT", "MOD", "CEG", "VST", "TLN", "NRG",
        "OKLO", "SMR", "BE", "POWL", "EME", "FIX", "STRL",
    ),
    "datacenter_cloud": ("EQIX", "DLR", "CRWV", "NBIS", "APLD"),
    "platforms": (
        "MSFT", "GOOGL", "AMZN", "META", "AAPL", "ORCL", "IBM", "TSLA", "BABA", "BIDU",
    ),
    "software": (
        "PLTR", "SNOW", "MDB", "DDOG", "NET", "NOW", "CRM", "ADBE", "INTU", "WDAY",
        "APP", "ESTC", "GTLB", "DOCN", "CRWD", "PANW", "ZS", "S", "AI",
        "PATH", "SOUN", "BBAI", "DT",
    ),
    "applications": (
        "TEM", "RXRX", "SDGR", "MBLY", "AUR", "PONY", "WRD", "SERV", "SYM", "INOD",
        "UPST", "CRNC",
    ),
}

# Vendor histories that start before the operating company traded.
# Each date was set by inspecting the vendor series (see docs/DATA.md).
_TRADE_START: dict[str, str] = {
    "OKLO": "2024-05-10",  # de-SPAC close; earlier rows are the AltC shell
    "TLN": "2024-07-10",   # Nasdaq listing; earlier rows are thin OTC prints
    "SERV": "2024-04-18",  # Nasdaq uplisting; earlier rows are OTC
    "GEV": "2024-04-02",   # regular-way trading after the GE spin-off
    "SNDK": "2025-02-24",  # regular-way trading after the WDC spin-off
}


def members() -> list[Member]:
    out: list[Member] = []
    for segment, tickers in _SEGMENTS.items():
        for t in tickers:
            out.append(Member(t, segment, _TRADE_START.get(t)))
    tickers = [m.ticker for m in out]
    dupes = {t for t in tickers if tickers.count(t) > 1}
    if dupes:
        raise ValueError(f"duplicate tickers in universe: {sorted(dupes)}")
    return out


def tickers() -> list[str]:
    return [m.ticker for m in members()]


def table() -> pd.DataFrame:
    return pd.DataFrame([m.__dict__ for m in members()]).set_index("ticker")
