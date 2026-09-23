"""
Macro Factor Loader
===================
Downloads 5 macro time series from FRED (no API key required) and saves
them as quarterly panel data to data/raw/macro/macro_factors.parquet.

Series downloaded
-----------------
vix          : VIXCLS          - CBOE VIX daily close -> quarterly mean
hy_spread    : BAMLH0A0HYM2    - BofA HY OAS (%) daily -> quarterly mean
fed_funds    : FEDFUNDS         - Effective Fed Funds Rate monthly -> quarterly mean
unemployment : UNRATE           - US unemployment rate monthly -> quarterly mean
gdp_growth   : A191RL1Q225SBEA - Real GDP growth QoQ (%) quarterly

Usage
-----
    python -m src.data.macro_loader --data_dir /path/to/ST-GNN/data
"""

import argparse
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

FRED_SERIES = {
    "vix":          "VIXCLS",
    "hy_spread":    "BAMLH0A0HYM2",
    "fed_funds":    "FEDFUNDS",
    "unemployment": "UNRATE",
    "gdp_growth":   "A191RL1Q225SBEA",
}

MACRO_COLS = list(FRED_SERIES.keys())   # exported for graph_builder


def _fetch_fred(series_id: str) -> pd.Series:
    """Download a FRED series as a pandas Series indexed by date."""
    url = f"{FRED_BASE}?id={series_id}"
    logger.info("Fetching %s from FRED ...", series_id)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # FRED uses 'observation_date' or 'DATE' depending on endpoint version
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    s = df.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")   # '.' -> NaN
    s.name = series_id
    return s


def download_macro_factors(
    data_dir: str,
    start: str = "2000-01-01",
    end:   str = "2025-03-31",
) -> pd.DataFrame:
    """
    Download macro series, resample to quarter-end, and save parquet.

    Parameters
    ----------
    data_dir : str
        Root data directory (parent of raw/).
    start, end : str
        Date range for the download.

    Returns
    -------
    macro : pd.DataFrame
        Columns = MACRO_COLS, index = quarter-end dates.
    """
    frames = {}
    for name, fred_id in FRED_SERIES.items():
        try:
            s = _fetch_fred(fred_id)
            s = s.loc[start:end]
            # Resample to quarter-end mean (handles daily, monthly, quarterly)
            s_q = s.resample("QE").mean()
            frames[name] = s_q
        except Exception as exc:
            logger.error("Failed to fetch %s (%s): %s", name, fred_id, exc)
            frames[name] = pd.Series(dtype=float, name=name)

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "date"

    # Forward-fill then backward-fill for any missing quarter-ends (e.g. GDP lag)
    macro = macro.ffill().bfill()

    # Save
    out_path = Path(data_dir) / "raw" / "macro" / "macro_factors.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    macro.to_parquet(out_path)
    print(f"Saved {len(macro)} quarterly macro observations to {out_path}")
    print(macro.tail(4).to_string())
    return macro


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Root data directory (parent of raw/)")
    args = parser.parse_args()
    download_macro_factors(args.data_dir)
