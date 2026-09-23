"""
Temporal Graph Builder for Corporate Default Prediction
=======================================================
Builds PyTorch Geometric HeteroData snapshots from WRDS parquet files.

Graph structure per quarterly snapshot
---------------------------------------
Node type  : 'company'  (active S&P-rated US non-financial firms)
Edge types :
  (company, customer_of,  company)  -- supply chain (supplier -> customer)
  (company, supplier_of,  company)  -- supply chain reverse
  (company, subsidiary_of, company) -- parent holds subsidiary
  (company, parent_of,    company)  -- subsidiary reverse

Node features (NODE_FEATURE_COLS, z-score normalised on train set):
  Financial : de_ratio, intcov, curr_ratio, debt_ebitda, roa, roe, npm
  Market    : log_mktcap, ret_12m, vol_12m
  Rating    : rating_numeric  (0=AAA ... 21=D/SD, 22=NR/withdrawn)

Label:
  y = 1 if the company has ANY default event within the next
      HORIZON_QUARTERS quarters; y = 0 otherwise.

Temporal split (time-based, no look-ahead leakage):
  Train : 2001-Q1 - 2016-Q4
  Val   : 2017-Q1 - 2019-Q4
  Test  : 2020-Q1 - 2024-Q4
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RATING_MAP: Dict[str, int] = {
    "AAA": 0,  "AA+": 1,  "AA": 2,  "AA-": 3,
    "A+":  4,  "A":   5,  "A-":  6,
    "BBB+": 7, "BBB": 8,  "BBB-": 9,
    "BB+": 10, "BB":  11, "BB-": 12,
    "B+":  13, "B":   14, "B-":  15,
    "CCC+": 16, "CCC": 17, "CCC-": 18,
    "CC":  19, "C":   20,
    "D":   21, "SD":  21,
    "NR":  22, "WD":  22, "NM":  22, "PI": 22,
}

GSECTOR_MAP: Dict[str, int] = {
    '10': 1, '15': 2, '20': 3, '25': 4, '30': 5,
    '35': 6, '40': 7, '45': 8, '50': 9, '55': 10, '60': 11,
}  # 0 reserved for unknown/NR; mirrors FiLMBlock.GSECTOR_MAP

RATIO_COLS = ["de_ratio", "intcov", "curr_ratio", "debt_ebitda",
              "roa", "roe", "npm"]
MACRO_COLS = ["vix", "hy_spread", "fed_funds", "unemployment", "gdp_growth"]
NODE_FEATURE_COLS = RATIO_COLS + ["log_mktcap", "ret_12m", "vol_12m",
                                   "rating_numeric"] + MACRO_COLS

HORIZON_QUARTERS = 4   # 1-year default prediction horizon
N_SURVIVAL_BINS  = 52  # weekly bins for DeepHit survival curve (52 weeks = 1 year)

TRAIN_END = pd.Timestamp("2016-12-31")
VAL_END   = pd.Timestamp("2019-12-31")


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

class GraphDataLoader:
    """Load and preprocess all raw parquet files from BASE_DIR/raw/."""

    def __init__(self, base_dir: str) -> None:
        self.base = Path(base_dir)
        self._load_all()

    def _load_all(self) -> None:
        raw = self.base / "raw"

        # Rated universe
        self.rated_universe = pd.read_parquet(
            raw / "company/rated_universe.parquet"
        )

        # Default labels
        defaults = pd.read_parquet(
            raw / "defaults/compustat_sp_defaults_fallback.parquet"
        )
        defaults["default_date"] = pd.to_datetime(defaults["default_date"])
        self._default_dates: Dict[str, List[pd.Timestamp]] = (
            defaults.groupby("gvkey")["default_date"]
            .apply(sorted)
            .to_dict()
        )

        # S&P ratings: carry-forward last known rating per company
        sp = pd.read_parquet(raw / "ratings/sp_ratings_history.parquet")
        sp["rating_date"] = pd.to_datetime(sp["rating_date"])
        sp["rating_numeric"] = (
            sp["rating"].map(RATING_MAP).fillna(22).astype(int)
        )
        if "is_nr" not in sp.columns:
            sp["is_nr"] = sp["rating_numeric"] >= 22
        self.sp_ratings = sp.sort_values(["gvkey", "rating_date"])

        # Financial ratios (quarterly)
        ratios = pd.read_parquet(raw / "financials/ratios_all.parquet")
        ratios["qdate"] = pd.to_datetime(ratios["qdate"])
        self.ratios = ratios

        # CRSP monthly -> trailing 12m return and volatility
        crsp = pd.read_parquet(raw / "stock/crsp_monthly.parquet")
        crsp["date"] = pd.to_datetime(crsp["date"])
        ccm  = pd.read_parquet(raw / "stock/ccm_link.parquet")
        crsp["permno"] = crsp["permno"].astype("Int64")
        ccm["permno"]  = ccm["permno"].astype("Int64")
        crsp = (crsp
                .merge(ccm[["gvkey", "permno"]], on="permno", how="inner")
                .sort_values(["gvkey", "date"]))
        crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
        crsp["ret_12m"] = (
            crsp.groupby("gvkey")["ret"]
            .transform(lambda x: np.exp(np.log1p(x).rolling(12, min_periods=6).sum()) - 1)
        )
        crsp["vol_12m"] = (
            crsp.groupby("gvkey")["ret"]
            .transform(lambda x: x.rolling(12, min_periods=6).std())
        )
        crsp["log_mktcap"] = np.log1p(crsp["mktcap"].clip(lower=0))
        self.crsp = crsp[["gvkey", "date", "log_mktcap",
                           "ret_12m", "vol_12m"]].dropna(subset=["log_mktcap"])

        # Supply chain edges (supplier gvkey -> customer_gvkey)
        # Prefer revere_supply_chain_final.parquet (hybrid edge_weight: revenue_percent
        # where available, 1/out_degree otherwise). Falls back to raw revere parquet
        # (binary weight=1) then legacy Compustat schema.
        final_path  = raw / "edges/revere_supply_chain_final.parquet"
        revere_path = raw / "edges/revere_supply_chain.parquet"
        legacy_path = raw / "edges/customer_supplier_mapped.parquet"
        if final_path.exists():
            sc = pd.read_parquet(final_path)
            sc["start_date"] = pd.to_datetime(sc["start_date"])
            sc["end_date"]   = pd.to_datetime(sc["end_date"])
            if "edge_weight" not in sc.columns:
                sc["edge_weight"] = 1.0
            self._sc_schema = "revere"
        elif revere_path.exists():
            sc = pd.read_parquet(revere_path)
            sc["start_date"] = pd.to_datetime(sc["start_date"])
            sc["end_date"]   = pd.to_datetime(sc["end_date"])
            if "gvkey" in sc.columns and "supplier_gvkey" not in sc.columns:
                sc = sc.rename(columns={"gvkey": "supplier_gvkey"})
            sc["edge_weight"] = 1.0
            self._sc_schema = "revere"
        elif legacy_path.exists():
            sc = pd.read_parquet(legacy_path)
            sc["datadate"] = pd.to_datetime(sc["datadate"])
            if "gvkey" in sc.columns and "supplier_gvkey" not in sc.columns:
                sc = sc.rename(columns={"gvkey": "supplier_gvkey"})
            sc["edge_weight"] = 1.0
            self._sc_schema = "legacy"
        else:
            sc = pd.DataFrame(columns=["supplier_gvkey", "customer_gvkey", "edge_weight"])
            self._sc_schema = "empty"
        self.supply_chain = sc[sc["customer_gvkey"].notna()].copy()

        # Parent-subsidiary edges
        ps = pd.read_parquet(raw / "edges/parent_subsidiary_edges.parquet")
        ps["rdate"] = pd.to_datetime(ps["rdate"])
        self.parent_sub = ps[ps["sub_gvkey"].notna()].copy()

        # Common institutional blockholder edges (Cell 17 output)
        block_path = raw / "edges/common_blockholder_edges.parquet"
        if block_path.exists():
            bh = pd.read_parquet(block_path)
            bh["quarter"] = pd.to_datetime(bh["quarter"])
            self.blockholders = bh[
                bh["gvkey_a"].notna() & bh["gvkey_b"].notna()
            ].copy()
            self._has_blockholders = True
        else:
            self.blockholders = pd.DataFrame(
                columns=["gvkey_a", "gvkey_b", "quarter"])
            self._has_blockholders = False

        # FactSet Revere extended relationships: COMPETITOR, PARTNER, etc. (Cell 18)
        revere_ext_path = raw / "edges/revere_extended_relationships.parquet"
        if revere_ext_path.exists():
            re = pd.read_parquet(revere_ext_path)
            re["start_date"] = pd.to_datetime(re["start_date"])
            re["end_date"]   = pd.to_datetime(re["end_date"])
            self.revere_ext = re[
                re["source_gvkey"].notna() & re["target_gvkey"].notna()
            ].copy()
            self._has_revere_ext = True
        else:
            self.revere_ext = pd.DataFrame(
                columns=["source_gvkey", "target_gvkey", "rel_type",
                         "start_date", "end_date"])
            self._has_revere_ext = False

        # Macro factors (optional - omitted if parquet not yet downloaded)
        macro_path = raw / "macro/macro_factors.parquet"
        if macro_path.exists():
            self.macro = pd.read_parquet(macro_path)
            self.macro.index = pd.to_datetime(self.macro.index)
            self._has_macro = True
            logger.info("Loaded macro factors: %d quarters, cols=%s",
                        len(self.macro), list(self.macro.columns))
        else:
            self.macro = None
            self._has_macro = False
            logger.info("Macro factors not found - run src/data/macro_loader.py")

        logger.info(
            "Loaded: %d rated firms | %d rating rows | %d ratio rows | "
            "%d CRSP rows | %d supply chain (%s) | %d parent-sub | "
            "%d blockholder edges | %d revere-ext edges",
            len(self.rated_universe),
            len(self.sp_ratings),
            len(self.ratios),
            len(self.crsp),
            len(self.supply_chain),
            self._sc_schema,
            len(self.parent_sub),
            len(self.blockholders),
            len(self.revere_ext),
        )

    def get_quarter_ends(self, start: str = "2001-03-31",
                         end: str = "2024-12-31") -> List[pd.Timestamp]:
        """Return end-of-quarter dates in [start, end]."""
        return list(pd.date_range(start=start, end=end, freq="QE"))

    def get_weekly_dates(self, start: str = "2001-01-05",
                         end: str = "2024-12-31") -> List[pd.Timestamp]:
        """Return weekly Friday dates in [start, end] for micro-bucket snapshots."""
        return list(pd.date_range(start=start, end=end, freq="W-FRI"))

    def has_default_in_window(self, gvkey: str, t: pd.Timestamp,
                               horizon: int = HORIZON_QUARTERS,
                               freq: str = 'Q') -> int:
        """Return 1 if gvkey defaults in (t, t + horizon periods].

        Parameters
        ----------
        horizon : int
            Number of periods forward. Quarters when freq='Q'; weeks when freq='W'.
        freq : 'Q' | 'W'
            'Q' -> DateOffset(months=3*horizon); 'W' -> DateOffset(weeks=horizon).
        """
        dates = self._default_dates.get(gvkey, [])
        if not dates:
            return 0
        if freq == 'W':
            t_end = t + pd.DateOffset(weeks=horizon)
        else:
            t_end = t + pd.DateOffset(months=3 * horizon)
        return int(any(t < d <= t_end for d in dates))

    def survival_label(self, gvkey: str, t: pd.Timestamp,
                       n_bins: int = N_SURVIVAL_BINS) -> Tuple[int, int]:
        """Return (bin_idx, event) for 52-bin DeepHit survival loss.

        bin_idx : int in [0, n_bins-1]
            Which weekly bin the first post-t default falls in (floor(days/7)).
            Capped at n_bins-1. Censored firms also get bin_idx = n_bins-1.
        event : int 0 or 1
            1 if a default occurred within n_bins weeks, else 0.
        """
        dates = self._default_dates.get(gvkey, [])
        future = [d for d in dates if d > t]
        if future:
            days = (min(future) - t).days
            bin_idx = min(int(days // 7), n_bins - 1)
            return bin_idx, 1
        return n_bins - 1, 0


# ---------------------------------------------------------------------------
# Snapshot Builder
# ---------------------------------------------------------------------------

class SnapshotBuilder:
    """Build a single quarterly HeteroData snapshot."""

    def __init__(self, loader: GraphDataLoader) -> None:
        self.L = loader

    def _active_ratings(self, t: pd.Timestamp) -> pd.DataFrame:
        """Last known S&P rating per company at or before t, excluding NR."""
        past = self.L.sp_ratings[self.L.sp_ratings["rating_date"] <= t]
        last = (past.sort_values("rating_date")
                    .groupby("gvkey").last()
                    .reset_index())
        # Exclude firms whose last rating is a withdrawal / NR
        last = last[~last["is_nr"]]
        return last[["gvkey", "rating", "rating_numeric"]]

    def _ratios_at(self, t: pd.Timestamp,
                   gvkeys: set) -> pd.DataFrame:
        """Most recent quarterly ratios within 3 quarters of t."""
        t_lo = t - pd.DateOffset(months=9)
        window = self.L.ratios[
            self.L.ratios["gvkey"].isin(gvkeys)
            & (self.L.ratios["qdate"] >= t_lo)
            & (self.L.ratios["qdate"] <= t)
        ]
        return (window.sort_values("qdate")
                      .groupby("gvkey").last()
                      .reset_index()[["gvkey"] + RATIO_COLS + ["mktcap"]])

    def _crsp_at(self, t: pd.Timestamp, gvkeys: set) -> pd.DataFrame:
        """Most recent CRSP row within 3 months of t."""
        t_lo = t - pd.DateOffset(months=3)
        window = self.L.crsp[
            self.L.crsp["gvkey"].isin(gvkeys)
            & (self.L.crsp["date"] >= t_lo)
            & (self.L.crsp["date"] <= t)
        ]
        return (window.sort_values("date")
                      .groupby("gvkey").last()
                      .reset_index()[["gvkey", "log_mktcap",
                                      "ret_12m", "vol_12m"]])

    def _supply_chain_edges(
        self, t: pd.Timestamp, gvkey_idx: Dict[str, int]
    ) -> Optional[tuple]:
        """Supply chain edges (supplier->customer) active at snapshot t.

        Returns (edge_index [2, E], edge_attr [E, 1]) where edge_attr contains
        hybrid weights: revenue_percent/100 where disclosed, 1/out_degree otherwise.
        """
        sc = self.L.supply_chain
        if self.L._sc_schema == "revere":
            mask = (sc["start_date"] <= t) & (
                sc["end_date"].isna() | (sc["end_date"] > t)
            )
            sc = sc[mask]
        else:
            if "datadate" in sc.columns:
                sc = sc[sc["datadate"] <= t]
        valid = sc[
            sc["supplier_gvkey"].isin(gvkey_idx)
            & sc["customer_gvkey"].isin(gvkey_idx)
        ].drop_duplicates(subset=["supplier_gvkey", "customer_gvkey"])
        if len(valid) == 0:
            return None
        src = [gvkey_idx[g] for g in valid["supplier_gvkey"]]
        dst = [gvkey_idx[g] for g in valid["customer_gvkey"]]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        if "edge_weight" in valid.columns:
            weights = torch.tensor(
                valid["edge_weight"].fillna(1.0).clip(0.01, 1.0).values,
                dtype=torch.float32
            ).unsqueeze(1)                        # [E, 1]
        else:
            weights = torch.ones(edge_index.shape[1], 1)
        return edge_index, weights

    def _macro_at(self, t: pd.Timestamp) -> Optional[pd.Series]:
        """Return macro factor values for the most recent quarter at or before t."""
        if not self.L._has_macro:
            return None
        valid_idx = self.L.macro.index[self.L.macro.index <= t]
        if len(valid_idx) == 0:
            return None
        return self.L.macro.loc[valid_idx[-1]]

    def _parent_sub_edges(
        self, t: pd.Timestamp, gvkey_idx: Dict[str, int]
    ) -> Optional[torch.Tensor]:
        """Parent-subsidiary edges reported at or before t."""
        ps = self.L.parent_sub[self.L.parent_sub["rdate"] <= t]
        valid = ps[
            ps["gvkey"].isin(gvkey_idx)
            & ps["sub_gvkey"].isin(gvkey_idx)
        ].drop_duplicates(subset=["gvkey", "sub_gvkey"])
        if len(valid) == 0:
            return None
        src = [gvkey_idx[g] for g in valid["gvkey"]]
        dst = [gvkey_idx[g] for g in valid["sub_gvkey"]]
        return torch.tensor([src, dst], dtype=torch.long)

    def _blockholder_edges(
        self, t: pd.Timestamp, gvkey_idx: Dict[str, int]
    ) -> Optional[torch.Tensor]:
        """Common institutional blockholder edges active at snapshot t.

        Uses the most recent quarter-end snapshot at or before t.
        Returns undirected edges as (gvkey_a -> gvkey_b) only
        (co-ownership is symmetric; caller adds reverse if needed).
        """
        if not self.L._has_blockholders:
            return None
        bh = self.L.blockholders
        # Most recent quarter at or before t
        valid_qtrs = bh["quarter"][bh["quarter"] <= t]
        if len(valid_qtrs) == 0:
            return None
        latest_qtr = valid_qtrs.max()
        bh_t = bh[bh["quarter"] == latest_qtr]
        valid = bh_t[
            bh_t["gvkey_a"].isin(gvkey_idx)
            & bh_t["gvkey_b"].isin(gvkey_idx)
        ].drop_duplicates(subset=["gvkey_a", "gvkey_b"])
        if len(valid) == 0:
            return None
        src = [gvkey_idx[g] for g in valid["gvkey_a"]]
        dst = [gvkey_idx[g] for g in valid["gvkey_b"]]
        return torch.tensor([src, dst], dtype=torch.long)

    def _revere_ext_edges(
        self, t: pd.Timestamp, gvkey_idx: Dict[str, int],
        rel_type: str
    ) -> Optional[torch.Tensor]:
        """FactSet Revere extended relationship edges of a given rel_type active at t.

        Active if start_date <= t AND (end_date > t OR end_date is NaT).
        """
        if not self.L._has_revere_ext:
            return None
        re = self.L.revere_ext
        re_t = re[re["rel_type"] == rel_type]
        mask = (re_t["start_date"] <= t) & (
            re_t["end_date"].isna() | (re_t["end_date"] > t)
        )
        valid = re_t[mask][
            re_t[mask]["source_gvkey"].isin(gvkey_idx)
            & re_t[mask]["target_gvkey"].isin(gvkey_idx)
        ].drop_duplicates(subset=["source_gvkey", "target_gvkey"])
        if len(valid) == 0:
            return None
        src = [gvkey_idx[g] for g in valid["source_gvkey"]]
        dst = [gvkey_idx[g] for g in valid["target_gvkey"]]
        return torch.tensor([src, dst], dtype=torch.long)

    def build(
        self,
        t: pd.Timestamp,
        scaler_stats: Optional[Dict] = None,
        horizon: int = HORIZON_QUARTERS,
        min_nodes: int = 100,
        freq: str = 'Q',
        horizon_short: int = 1,
        horizon_mid: int = 2,
    ) -> Optional[HeteroData]:
        """
        Build a single snapshot graph at date t.

        Parameters
        ----------
        freq : 'Q' | 'W'
            Snapshot cadence; governs how horizon periods are interpreted
            ('Q' -> quarters, 'W' -> weeks).
        horizon_short, horizon_mid : int
            Short and mid prediction horizons in `freq` units.
            Primary horizon is `horizon` (same units).

        Returns None if fewer than min_nodes active companies.
        """
        # 1. Active companies at t
        active = self._active_ratings(t)
        if len(active) < min_nodes:
            return None
        gvkeys = set(active["gvkey"])

        # 2. Features
        ratios = self._ratios_at(t, gvkeys)
        crsp   = self._crsp_at(t, gvkeys)

        df = (active
              .merge(ratios, on="gvkey", how="inner")
              .merge(crsp,   on="gvkey", how="inner"))
        if len(df) < min_nodes:
            return None

        # 3. Default labels (multi-horizon)
        # When freq='Q': horizons in quarters (1q/2q/4q).
        # When freq='W': horizons in weeks (e.g. 4w/8w/52w).
        df = df.copy()
        df["y_1q"] = df["gvkey"].apply(
            lambda g: self.L.has_default_in_window(g, t, horizon_short, freq=freq)
        )
        df["y_2q"] = df["gvkey"].apply(
            lambda g: self.L.has_default_in_window(g, t, horizon_mid, freq=freq)
        )
        df["y"] = df["gvkey"].apply(
            lambda g: self.L.has_default_in_window(g, t, horizon, freq=freq)
        )

        # Survival labels: 52-bin (y_bin, y_event) for full DeepHit
        surv = df["gvkey"].apply(lambda g: self.L.survival_label(g, t))
        df["y_bin"]   = surv.apply(lambda x: x[0])
        df["y_event"] = surv.apply(lambda x: x[1])

        # 4a. Append macro features (same value broadcast to all companies)
        macro_row = self._macro_at(t)
        if macro_row is not None:
            for col in MACRO_COLS:
                df[col] = float(macro_row.get(col, 0.0))

        # 4b. Feature matrix -- clip ratio outliers, impute, normalise
        feat_cols = [c for c in NODE_FEATURE_COLS if c in df.columns]
        for col in RATIO_COLS:
            if col in df.columns:
                lo = df[col].quantile(0.01)
                hi = df[col].quantile(0.99)
                df[col] = df[col].clip(lo, hi)
        df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())

        X = df[feat_cols].values.astype(np.float32)
        if scaler_stats is not None:
            mean = np.array([scaler_stats["mean"].get(c, 0.0) for c in feat_cols])
            std  = np.array([scaler_stats["std"].get(c, 1.0)  for c in feat_cols])
        else:
            mean = X.mean(axis=0)
            std  = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - mean) / std

        # 5. Assemble HeteroData
        data = HeteroData()
        gvkey_list = df["gvkey"].tolist()
        gvkey_idx  = {g: i for i, g in enumerate(gvkey_list)}

        data["company"].x        = torch.tensor(X, dtype=torch.float)
        data["company"].y        = torch.tensor(df["y"].values,    dtype=torch.long)
        data["company"].y_1q     = torch.tensor(df["y_1q"].values, dtype=torch.long)
        data["company"].y_2q     = torch.tensor(df["y_2q"].values, dtype=torch.long)
        data["company"].y_bin    = torch.tensor(df["y_bin"].values,   dtype=torch.long)
        data["company"].y_event  = torch.tensor(df["y_event"].values, dtype=torch.long)
        data["company"].gvkey       = gvkey_list
        data["company"].company_ids = torch.tensor(
            [int(g) for g in gvkey_list], dtype=torch.long
        )
        data["company"].t        = t
        data["company"].num_feat = len(feat_cols)
        data["company"].feat_cols = feat_cols
        # Raw (un-normalised) rating for tier-recall metric (0=AAA ... 20=C, 21=SD, 22=NR)
        data["company"].rating_numeric_raw = torch.tensor(
            df["rating_numeric"].fillna(22).values.astype(int), dtype=torch.long
        )

        # GICS sector index for FiLM layer (0 = unknown/NR, 1-11 = GICS sectors)
        gsector_lookup = (self.L.rated_universe
                          .set_index("gvkey")["gsector"]
                          .to_dict())
        sector_ints = [
            GSECTOR_MAP.get(str(gsector_lookup.get(g, "")), 0)
            for g in gvkey_list
        ]
        data["company"].sector_idx = torch.tensor(sector_ints, dtype=torch.long)

        # 6. Edges
        sc_result = self._supply_chain_edges(t, gvkey_idx)
        if sc_result is not None:
            sc_ei, sc_attr = sc_result   # sc_attr: [E, 1] hybrid weights
            data[("company", "customer_of",  "company")].edge_index = sc_ei
            data[("company", "customer_of",  "company")].edge_attr  = sc_attr
            data[("company", "supplier_of",  "company")].edge_index = sc_ei.flip(0)
            data[("company", "supplier_of",  "company")].edge_attr  = sc_attr

        ps_ei = self._parent_sub_edges(t, gvkey_idx)
        if ps_ei is not None:
            data[("company", "subsidiary_of", "company")].edge_index = ps_ei
            data[("company", "parent_of",     "company")].edge_index = ps_ei.flip(0)

        # Common institutional blockholder edges (symmetric -- add both directions)
        bh_ei = self._blockholder_edges(t, gvkey_idx)
        if bh_ei is not None:
            data[("company", "co_owned_by",  "company")].edge_index = bh_ei
            data[("company", "co_owner_of",  "company")].edge_index = bh_ei.flip(0)

        # FactSet Revere extended: competitor and partner edges (symmetric)
        comp_ei = self._revere_ext_edges(t, gvkey_idx, "COMPETITOR")
        if comp_ei is not None:
            data[("company", "competitor_of", "company")].edge_index = comp_ei
            data[("company", "competitor_of_rev", "company")].edge_index = comp_ei.flip(0)

        supp_ext_ei = self._revere_ext_edges(t, gvkey_idx, "SUPPLIER")
        if supp_ext_ei is not None:
            n_se = supp_ext_ei.shape[1]
            se_attr = torch.ones(n_se, 1)
            data[("company", "supplier_ext_of",  "company")].edge_index = supp_ext_ei
            data[("company", "supplier_ext_of",  "company")].edge_attr  = se_attr
            data[("company", "customer_ext_of",  "company")].edge_index = supp_ext_ei.flip(0)
            data[("company", "customer_ext_of",  "company")].edge_attr  = se_attr

        part_ei = self._revere_ext_edges(t, gvkey_idx, "PARTNER")
        if part_ei is not None:
            data[("company", "partner_of",    "company")].edge_index = part_ei
            data[("company", "partner_of_rev", "company")].edge_index = part_ei.flip(0)

        # Ensure at least one edge type exists (required by PyG conv layers)
        if len(data.edge_types) == 0:
            n = len(gvkey_list)
            self_loops = torch.arange(n).unsqueeze(0).repeat(2, 1)
            data[("company", "same_company", "company")].edge_index = self_loops

        return data


# ---------------------------------------------------------------------------
# Scaler stats (fit on training snapshots)
# ---------------------------------------------------------------------------

def _compute_scaler_stats(
    loader: GraphDataLoader,
    builder: SnapshotBuilder,
    train_dates: List[pd.Timestamp],
    sample_every: int = 4,
) -> Dict:
    """Compute median and std for each feature across sampled training snapshots."""
    frames = []
    for t in train_dates[::sample_every]:
        active = builder._active_ratings(t)
        ratios = builder._ratios_at(t, set(active["gvkey"]))
        crsp   = builder._crsp_at(t, set(active["gvkey"]))
        df = (active
              .merge(ratios, on="gvkey", how="inner")
              .merge(crsp,   on="gvkey", how="inner"))
        if len(df) > 0:
            macro_row = builder._macro_at(t)
            if macro_row is not None:
                for col in MACRO_COLS:
                    df[col] = float(macro_row.get(col, 0.0))
            frames.append(df)

    if not frames:
        logger.warning("No training frames -- scaler stats unavailable.")
        return {}

    pool = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in NODE_FEATURE_COLS if c in pool.columns]
    return {
        "mean": pool[feat_cols].median().to_dict(),   # median more robust
        "std":  pool[feat_cols].std().to_dict(),
        "feat_cols": feat_cols,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_temporal_graphs(
    base_dir: str,
    start_date: Optional[str] = None,
    end_date:   str = "2024-12-31",
    horizon_quarters: int = HORIZON_QUARTERS,
    min_nodes: int = 100,
    freq: str = 'Q',
) -> Tuple[List[HeteroData], List[HeteroData], List[HeteroData]]:
    """
    Build train / val / test graph snapshot sequences from WRDS parquet files.

    Parameters
    ----------
    base_dir : str
        Root data directory (must contain raw/ subdirectory with parquet files).
    start_date : str
        First snapshot date string.
        Quarterly default '2001-03-31'; weekly default '2001-01-05'.
    end_date : str
        Last snapshot (default '2024-12-31').
    horizon_quarters : int
        Prediction horizon in quarters (used when freq='Q', default 4 = 1 year).
        Ignored when freq='W' (fixed at 52 weeks = ~1 year).
    min_nodes : int
        Skip snapshots with fewer active companies (default 100).
    freq : 'Q' | 'W'
        Snapshot cadence. 'Q' = quarterly (default); 'W' = weekly micro-buckets.
        Weekly mode: short=4w, mid=8w, primary=52w; T_LOOKBACK should be set
        to 52 in TemporalTrainer for 1-year weekly history.

    Returns
    -------
    train_graphs, val_graphs, test_graphs : List[HeteroData]
        Time-ordered snapshot lists for each split.
    """
    loader  = GraphDataLoader(base_dir)
    builder = SnapshotBuilder(loader)

    # Resolve per-frequency start: the weekly panel begins earlier (2001-01-05)
    # than the quarter-end grid (2001-03-31). Matches the notebook graph cache.
    if start_date is None:
        start_date = "2001-01-05" if freq == 'W' else "2001-03-31"

    if freq == 'W':
        dates = loader.get_weekly_dates(start_date, end_date)
        h_short, h_mid, h_long = 4, 8, 52   # weeks: ~1m, ~2m, ~1yr
        sample_every = 52                    # fit scaler on ~monthly samples
    else:
        dates = loader.get_quarter_ends(start_date, end_date)
        h_short, h_mid, h_long = 1, 2, horizon_quarters
        sample_every = 4                     # fit scaler on every 4th quarter

    train_dates = [d for d in dates if d <= TRAIN_END]
    logger.info("Computing scaler stats from %d training snapshots (freq=%s) ...",
                len(train_dates), freq)
    scaler = _compute_scaler_stats(loader, builder, train_dates,
                                   sample_every=sample_every)

    train_graphs: List[HeteroData] = []
    val_graphs:   List[HeteroData] = []
    test_graphs:  List[HeteroData] = []

    for t in dates:
        g = builder.build(
            t, scaler_stats=scaler,
            horizon=h_long, min_nodes=min_nodes,
            freq=freq, horizon_short=h_short, horizon_mid=h_mid,
        )
        if g is None:
            continue
        n  = len(g["company"].gvkey)
        nd = int(g["company"].y.sum())
        logger.debug("  %s: %d nodes, %d defaults (%.1f%%)",
                     t.date(), n, nd, 100 * nd / n)
        if t <= TRAIN_END:
            train_graphs.append(g)
        elif t <= VAL_END:
            val_graphs.append(g)
        else:
            test_graphs.append(g)

    logger.info("Built: %d train | %d val | %d test snapshots",
                len(train_graphs), len(val_graphs), len(test_graphs))
    return train_graphs, val_graphs, test_graphs


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_graph_summary(graphs: List[HeteroData],
                        split_name: str = "") -> None:
    """Print key statistics for a list of snapshot graphs."""
    if not graphs:
        print(f"{split_name}: empty")
        return

    g0 = graphs[0]
    t_min = min(g["company"].t for g in graphs)
    t_max = max(g["company"].t for g in graphs)
    node_counts   = [len(g["company"].gvkey) for g in graphs]
    default_rates = [g["company"].y.float().mean().item() for g in graphs]

    print(f"\n{'='*60}")
    print(f"GRAPH SUMMARY -- {split_name}  ({len(graphs)} snapshots)")
    print(f"{'='*60}")
    print(f"Period        : {t_min.date()} -> {t_max.date()}")
    print(f"Node features : {g0['company'].num_feat}  "
          f"({', '.join(g0['company'].feat_cols)})")
    print(f"Edge types    : {[et[1] for et in g0.edge_types]}")
    print(f"Nodes/snapshot: min={min(node_counts):,}  "
          f"mean={np.mean(node_counts):,.0f}  "
          f"max={max(node_counts):,}")
    print(f"Default rate  : min={min(default_rates):.1%}  "
          f"mean={np.mean(default_rates):.1%}  "
          f"max={max(default_rates):.1%}")


def save_graphs(train_graphs: List[HeteroData],
                val_graphs:   List[HeteroData],
                test_graphs:  List[HeteroData],
                base_dir: str) -> None:
    """Pickle all snapshot lists to BASE_DIR/processed/temporal_graphs.pkl."""
    out = Path(base_dir) / "processed"
    out.mkdir(exist_ok=True)
    path = out / "temporal_graphs.pkl"
    with open(path, "wb") as f:
        pickle.dump({"train": train_graphs,
                     "val":   val_graphs,
                     "test":  test_graphs}, f)
    n = len(train_graphs) + len(val_graphs) + len(test_graphs)
    logger.info("Saved %d snapshots -> %s", n, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    BASE_DIR = (sys.argv[1]
                if len(sys.argv) > 1
                else "./data/")

    train_g, val_g, test_g = build_temporal_graphs(BASE_DIR)

    print_graph_summary(train_g, "TRAIN")
    print_graph_summary(val_g,   "VAL")
    print_graph_summary(test_g,  "TEST")

    save_graphs(train_g, val_g, test_g, BASE_DIR)
