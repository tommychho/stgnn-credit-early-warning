"""
Historical Data Simulator for GNN Early Warning System

PURPOSE
-------
The paper's quantitative results are derived from WRDS Compustat, a commercial
subscription dataset that cannot be redistributed under its licence. This module
provides a synthetic substitute so the training and evaluation pipeline can be
run and verified end-to-end WITHOUT WRDS access.

It reproduces the STRUCTURE of the real panel (temporal snapshots, rating
migrations, borrower entry/exit, sector stress, economic-event regimes) from
documented economic dynamics. It does NOT copy, sample, or approximate any
individual WRDS record, and synthetic data cannot reproduce the paper's reported
numbers; it exists only to exercise the code path for reproducibility checking.

Generates 15 years of monthly snapshots with:
- Economic events (COVID, financial crisis, sector shocks)
- Rating migrations over time
- Borrower entry/exit (varying history lengths)
- Sector-specific stress periods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


class EconomicEvent:
    """Represents a major economic event affecting portfolio."""

    def __init__(self, name: str, start_month: int, duration_months: int,
                 affected_sectors: List[str], rating_impact: float,
                 default_multiplier: float):
        """
        Parameters
        ----------
        name : str
            Event name (e.g., "COVID-19", "Financial Crisis")
        start_month : int
            Month index when event starts (0 = Jan 2010)
        duration_months : int
            How many months the event lasts
        affected_sectors : List[str]
            Industry codes affected ("IND001", "IND002", etc.)
        rating_impact : float
            Average rating deterioration (-2 = downgrade by 2 notches)
        default_multiplier : float
            Multiplier on base default rate (2.0 = double defaults)
        """
        self.name = name
        self.start_month = start_month
        self.end_month = start_month + duration_months
        self.affected_sectors = affected_sectors
        self.rating_impact = rating_impact
        self.default_multiplier = default_multiplier

    def is_active(self, month: int) -> bool:
        """Check if event is active in given month."""
        return self.start_month <= month < self.end_month

    def get_impact(self, month: int, sector: str) -> Tuple[float, float]:
        """
        Get event impact for given month and sector.

        Returns
        -------
        Tuple[float, float]
            (rating_impact, default_multiplier)
        """
        if not self.is_active(month):
            return 0.0, 1.0

        if sector not in self.affected_sectors:
            return 0.0, 1.0

        # Impact ramps up and down
        months_since_start = month - self.start_month
        duration = self.end_month - self.start_month

        # Bell curve: weak at start, peak at middle, weak at end
        progress = months_since_start / duration
        intensity = 4 * progress * (1 - progress)  # Peak at 0.5

        rating_impact = self.rating_impact * intensity
        default_mult = 1.0 + (self.default_multiplier - 1.0) * intensity

        return rating_impact, default_mult


# Define major economic events (2010-2024)
ECONOMIC_EVENTS = [
    # European Debt Crisis (2011-2012)
    EconomicEvent(
        name="European Debt Crisis",
        start_month=12,  # Jan 2011
        duration_months=24,
        affected_sectors=["IND002"],  # Financial services
        rating_impact=-1.5,
        default_multiplier=1.8
    ),

    # Oil Price Collapse (2014-2016)
    EconomicEvent(
        name="Oil Price Collapse",
        start_month=48,  # Jan 2014
        duration_months=30,
        affected_sectors=["IND001"],  # Oil & Gas
        rating_impact=-2.5,
        default_multiplier=3.5
    ),

    # COVID-19 Pandemic (2020-2021)
    EconomicEvent(
        name="COVID-19 Pandemic",
        start_month=120,  # Jan 2020
        duration_months=18,
        affected_sectors=["IND003", "IND004", "IND005"],  # Services, Retail, Hospitality
        rating_impact=-2.0,
        default_multiplier=4.0
    ),

    # Interest Rate Hikes (2022-2023)
    EconomicEvent(
        name="Aggressive Rate Hikes",
        start_month=144,  # Jan 2022
        duration_months=18,
        affected_sectors=["IND002", "IND004"],  # Financial, Real Estate
        rating_impact=-1.0,
        default_multiplier=1.6
    ),
]


def generate_borrower_lifecycle(
    borrower_id: str,
    entry_month: int,
    exit_month: Optional[int],
    initial_rating: int,
    sector: str,
    base_migration_prob: float = 0.15,
    rng: np.random.Generator = None
) -> pd.DataFrame:
    """
    Generate rating history for a single borrower.

    Parameters
    ----------
    borrower_id : str
        Unique borrower identifier
    entry_month : int
        Month when borrower joined (0-179 for 15 years)
    exit_month : Optional[int]
        Month when borrower exited (None if still active)
    initial_rating : int
        Starting internal rating (1-24)
    sector : str
        Industry code
    base_migration_prob : float
        Monthly probability of rating change
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    pd.DataFrame
        Columns: month, borrower_id, internal_rating, external_rating,
                 is_defaulted, sector
    """
    if rng is None:
        rng = np.random.default_rng()

    # Determine actual exit month
    max_months = 180  # 15 years
    actual_exit = exit_month if exit_month is not None else max_months

    history = []
    current_rating = initial_rating
    is_defaulted = False

    for month in range(entry_month, actual_exit):
        # Check for economic event impacts
        rating_shock = 0.0
        default_mult = 1.0

        for event in ECONOMIC_EVENTS:
            if event.is_active(month):
                r_impact, d_mult = event.get_impact(month, sector)
                rating_shock += r_impact
                default_mult *= d_mult

        # Apply rating shock (deterioration)
        if rating_shock < 0:
            downgrade = int(np.ceil(abs(rating_shock)))
            current_rating = min(21, current_rating + downgrade)  # Max at C (not D yet)

        # Natural rating migration (random walk)
        if rng.random() < base_migration_prob:
            # Tendency to revert to mean (rating 12 = BB)
            if current_rating < 12:
                # Good ratings: 70% stay, 20% downgrade, 10% upgrade
                change = rng.choice([0, 1, -1], p=[0.7, 0.2, 0.1])
            elif current_rating > 12:
                # Bad ratings: 60% stay, 20% downgrade, 20% upgrade
                change = rng.choice([0, 1, -1], p=[0.6, 0.2, 0.2])
            else:
                # Middle rating: 70% stay, 15% each direction
                change = rng.choice([0, 1, -1], p=[0.7, 0.15, 0.15])

            current_rating = int(np.clip(current_rating + change, 1, 21))

        # Check for default
        base_default_prob = _rating_to_monthly_pd(current_rating)
        monthly_default_prob = base_default_prob * default_mult

        if not is_defaulted and rng.random() < monthly_default_prob:
            is_defaulted = True
            current_rating = 24  # Default

        # Convert to external rating
        from src.data.synthetic_generator import internal_rating_to_sp_rating
        external_rating = internal_rating_to_sp_rating(current_rating)

        history.append({
            'month': month,
            'borrower_id': borrower_id,
            'internal_rating': current_rating,
            'external_rating': external_rating,
            'is_defaulted': is_defaulted,
            'sector': sector
        })

        # Exit if defaulted
        if is_defaulted:
            break

    return pd.DataFrame(history)


def _rating_to_monthly_pd(rating: int) -> float:
    """Convert internal rating to monthly probability of default."""
    # Annual PD by rating (from RATING_SYSTEM_UPDATE.md)
    annual_pds = {
        1: 0.0001, 2: 0.0002, 3: 0.0003, 4: 0.0005, 5: 0.0008,
        6: 0.0010, 7: 0.0015, 8: 0.0030, 9: 0.0050, 10: 0.0075,
        11: 0.0120, 12: 0.0200, 13: 0.0350, 14: 0.0550, 15: 0.0850,
        16: 0.1200, 17: 0.1800, 18: 0.2500, 19: 0.3500, 20: 0.5000,
        21: 0.7000, 22: 1.0, 23: 1.0, 24: 1.0
    }

    annual_pd = annual_pds.get(rating, 0.25)
    # Convert to monthly: 1 - (1 - annual_pd)^(1/12)
    monthly_pd = 1 - (1 - annual_pd) ** (1/12)
    return monthly_pd


def generate_historical_portfolio(
    n_borrowers_final: int = 5000,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate 15 years of historical portfolio data with realistic dynamics.

    This creates:
    - Borrowers entering/exiting over time (varying history lengths)
    - Monthly rating snapshots for each borrower
    - Economic events affecting sectors
    - Rating migrations and defaults

    Parameters
    ----------
    n_borrowers_final : int
        Target number of borrowers in final snapshot (Dec 2024)
    start_date : str
        Portfolio start date (YYYY-MM-DD)
    end_date : str
        Portfolio end date (YYYY-MM-DD)
    seed : int
        Random seed

    Returns
    -------
    Dict[str, pd.DataFrame]
        {
            'rating_history': Monthly ratings for all borrowers,
            'borrower_metadata': Static borrower info (entry/exit, sector),
            'monthly_snapshots': Aggregated monthly statistics,
            'events': Economic events timeline
        }
    """
    rng = np.random.default_rng(seed)

    # Calculate number of months
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    n_months = ((end.year - start.year) * 12 + (end.month - start.month) + 1)

    print(f"Generating {n_months} months of history ({start_date} to {end_date})...")
    print(f"Target: {n_borrowers_final} borrowers in final snapshot")

    # Industries
    sectors = ["IND001", "IND002", "IND003", "IND004", "IND005"]
    sector_names = {
        "IND001": "Oil & Gas",
        "IND002": "Financial Services",
        "IND003": "Manufacturing",
        "IND004": "Real Estate",
        "IND005": "Services"
    }

    # Generate borrower population
    # Total borrowers needed = final + those who exited
    # Assume 20% churn over 15 years → need 1.25x final count
    n_total_borrowers = int(n_borrowers_final * 1.25)

    borrowers_metadata = []
    all_rating_history = []

    print(f"Simulating {n_total_borrowers} total borrowers (including exits)...")

    for i in range(n_total_borrowers):
        borrower_id = f"B{i:05d}"

        # Entry month (spread over first 10 years, weighted toward earlier)
        # More borrowers join early, fewer join late
        entry_weights = np.exp(-np.arange(120) / 40)  # Exponential decay
        entry_weights = entry_weights / entry_weights.sum()
        entry_month = rng.choice(120, p=entry_weights)

        # Exit month (20% exit before end)
        will_exit = rng.random() < 0.20
        if will_exit:
            # Exit at least 12 months after entry
            min_exit = entry_month + 12
            max_exit = n_months
            if min_exit < max_exit:
                exit_month = rng.integers(min_exit, max_exit)
            else:
                exit_month = None  # Stay until end
        else:
            exit_month = None

        # Sector (evenly distributed)
        sector = rng.choice(sectors)

        # Initial rating (normal distribution around BB = 12)
        # Mean 12, std 4 → mostly BB to BBB range
        initial_rating = int(np.clip(rng.normal(12, 4), 1, 21))

        # Generate rating history
        borrower_history = generate_borrower_lifecycle(
            borrower_id=borrower_id,
            entry_month=entry_month,
            exit_month=exit_month,
            initial_rating=initial_rating,
            sector=sector,
            rng=rng
        )

        all_rating_history.append(borrower_history)

        # Metadata
        entry_date = start + relativedelta(months=int(entry_month))
        exit_date = (start + relativedelta(months=int(exit_month))) if exit_month else None

        borrowers_metadata.append({
            'borrower_id': borrower_id,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_month': entry_month,
            'exit_month': exit_month,
            'sector': sector,
            'sector_name': sector_names[sector],
            'initial_rating': initial_rating,
            'months_on_book': len(borrower_history)
        })

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{n_total_borrowers} borrowers...")

    # Combine all rating histories
    rating_history = pd.concat(all_rating_history, ignore_index=True)
    rating_history['calendar_date'] = rating_history['month'].apply(
        lambda m: start + relativedelta(months=int(m))
    )

    borrowers_metadata = pd.DataFrame(borrowers_metadata)

    # Generate monthly snapshots (aggregated stats)
    monthly_snapshots = []
    for month in range(n_months):
        month_data = rating_history[rating_history['month'] == month]

        # Count active borrowers by sector
        sector_counts = month_data['sector'].value_counts()

        # Average rating
        avg_rating = month_data['internal_rating'].mean()

        # Default rate
        default_rate = month_data['is_defaulted'].mean()

        # Check active events
        active_events = [e.name for e in ECONOMIC_EVENTS if e.is_active(month)]

        snapshot_date = start + relativedelta(months=int(month))

        monthly_snapshots.append({
            'month': month,
            'calendar_date': snapshot_date,
            'n_borrowers': len(month_data),
            'avg_internal_rating': avg_rating,
            'default_rate': default_rate,
            'n_defaults': month_data['is_defaulted'].sum(),
            'active_events': ', '.join(active_events) if active_events else None,
            **{f'n_{sector}': sector_counts.get(sector, 0) for sector in sectors}
        })

    monthly_snapshots = pd.DataFrame(monthly_snapshots)

    # Create events timeline
    events_timeline = []
    for event in ECONOMIC_EVENTS:
        start_date_event = start + relativedelta(months=int(event.start_month))
        end_date_event = start + relativedelta(months=int(event.end_month))

        events_timeline.append({
            'event_name': event.name,
            'start_date': start_date_event,
            'end_date': end_date_event,
            'start_month': event.start_month,
            'end_month': event.end_month,
            'duration_months': event.end_month - event.start_month,
            'affected_sectors': ', '.join(event.affected_sectors),
            'rating_impact': event.rating_impact,
            'default_multiplier': event.default_multiplier
        })

    events_timeline = pd.DataFrame(events_timeline)

    print("\n" + "="*80)
    print("HISTORICAL SIMULATION COMPLETE")
    print("="*80)
    print(f"Total borrowers simulated: {len(borrowers_metadata):,}")
    print(f"Active in final month: {len(rating_history[rating_history['month'] == n_months - 1]):,}")
    print(f"Exited before end: {borrowers_metadata['exit_date'].notna().sum():,}")
    print(f"Total defaults: {rating_history['is_defaulted'].sum():,}")
    print(f"Economic events: {len(events_timeline)}")

    return {
        'rating_history': rating_history,
        'borrower_metadata': borrowers_metadata,
        'monthly_snapshots': monthly_snapshots,
        'events': events_timeline
    }


def get_snapshot_at_date(
    historical_data: Dict[str, pd.DataFrame],
    snapshot_date: str
) -> pd.DataFrame:
    """
    Extract borrower snapshot at a specific date.

    Parameters
    ----------
    historical_data : Dict[str, pd.DataFrame]
        Output from generate_historical_portfolio()
    snapshot_date : str
        Date for snapshot (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame
        Borrower ratings at that date with their complete history
    """
    """
    Extract borrower snapshot at specific date with historical features.

    Vectorized implementation ~40x faster than iterative version.
    Uses pandas groupby and merge operations instead of per-borrower loops.

    Performance: 300-400s per snapshot reduced to 8-10s per snapshot.
    """
    rating_history = historical_data['rating_history']
    target_date = pd.to_datetime(snapshot_date)

    # Calculate time windows once
    three_months_ago = target_date - relativedelta(months=3)
    six_months_ago = target_date - relativedelta(months=6)
    twelve_months_ago = target_date - relativedelta(months=12)

    # Extract snapshot at target date
    snapshot = rating_history[rating_history['calendar_date'] == target_date].copy()

    if len(snapshot) == 0:
        raise ValueError(f"No data found for {snapshot_date}")

    # OPTIMIZATION: Pre-filter to only relevant borrowers and dates
    # Reduces dataset from 373K rows to ~60-150K rows (10x smaller)
    relevant_borrowers = snapshot['borrower_id'].unique()
    history_subset = rating_history[
        (rating_history['borrower_id'].isin(relevant_borrowers)) &
        (rating_history['calendar_date'] <= target_date)
    ].copy()

    # VECTORIZED: Count months on book (replaces per-borrower loop)
    months_on_book = history_subset.groupby('borrower_id').size().reset_index(name='months_on_book')
    snapshot = snapshot.merge(months_on_book, on='borrower_id', how='left')

    # VECTORIZED: Get ratings at 3/6/12 months ago
    history_3m = history_subset[history_subset['calendar_date'] <= three_months_ago]
    history_6m = history_subset[history_subset['calendar_date'] <= six_months_ago]
    history_12m_filter = history_subset[history_subset['calendar_date'] <= twelve_months_ago]

    # Get last rating before each window (groupby.last is vectorized)
    if len(history_3m) > 0:
        rating_3m = history_3m.groupby('borrower_id')['internal_rating'].last().reset_index()
        rating_3m.columns = ['borrower_id', 'rating_3m_ago']
        snapshot = snapshot.merge(rating_3m, on='borrower_id', how='left')
    else:
        snapshot['rating_3m_ago'] = None

    if len(history_6m) > 0:
        rating_6m = history_6m.groupby('borrower_id')['internal_rating'].last().reset_index()
        rating_6m.columns = ['borrower_id', 'rating_6m_ago']
        snapshot = snapshot.merge(rating_6m, on='borrower_id', how='left')
    else:
        snapshot['rating_6m_ago'] = None

    if len(history_12m_filter) > 0:
        rating_12m = history_12m_filter.groupby('borrower_id')['internal_rating'].last().reset_index()
        rating_12m.columns = ['borrower_id', 'rating_12m_ago']
        snapshot = snapshot.merge(rating_12m, on='borrower_id', how='left')
    else:
        snapshot['rating_12m_ago'] = None

    # Handle missing values (borrowers who didn't exist at past dates)
    # Get first rating for each borrower as fallback
    first_rating = history_subset.groupby('borrower_id')['internal_rating'].first().reset_index()
    first_rating.columns = ['borrower_id', 'first_rating']
    snapshot = snapshot.merge(first_rating, on='borrower_id', how='left')

    # Fill NaN with first rating (for borrowers newer than 3/6/12 months)
    snapshot['rating_3m_ago'] = snapshot['rating_3m_ago'].fillna(snapshot['first_rating']).astype(float)
    snapshot['rating_6m_ago'] = snapshot['rating_6m_ago'].fillna(snapshot['first_rating']).astype(float)
    snapshot['rating_12m_ago'] = snapshot['rating_12m_ago'].fillna(snapshot['first_rating']).astype(float)

    snapshot = snapshot.drop(columns=['first_rating'])

    # VECTORIZED: Calculate rating changes (no loops!)
    snapshot['rating_change_3m'] = snapshot['internal_rating'] - snapshot['rating_3m_ago']
    snapshot['rating_change_6m'] = snapshot['internal_rating'] - snapshot['rating_6m_ago']
    snapshot['rating_change_12m'] = snapshot['internal_rating'] - snapshot['rating_12m_ago']

    # VECTORIZED: Count upgrades/downgrades in last 12 months
    history_last_12m = history_subset[history_subset['calendar_date'] > twelve_months_ago]

    if len(history_last_12m) > 0:
        # Sort to ensure correct diff calculation
        history_last_12m = history_last_12m.sort_values(['borrower_id', 'calendar_date'])

        # Calculate rating changes within each borrower group
        history_last_12m['rating_diff'] = history_last_12m.groupby('borrower_id')['internal_rating'].diff()

        # Count downgrades (positive diff = rating number increased = worse rating)
        downgrades = history_last_12m[history_last_12m['rating_diff'] > 0].groupby('borrower_id').size()
        downgrades = downgrades.reset_index(name='downgrades_12m')

        # Count upgrades (negative diff = rating number decreased = better rating)
        upgrades = history_last_12m[history_last_12m['rating_diff'] < 0].groupby('borrower_id').size()
        upgrades = upgrades.reset_index(name='upgrades_12m')

        # Merge back to snapshot
        snapshot = snapshot.merge(downgrades, on='borrower_id', how='left')
        snapshot = snapshot.merge(upgrades, on='borrower_id', how='left')

        # Fill NaN with 0 (borrowers with no upgrades/downgrades)
        snapshot['downgrades_12m'] = snapshot['downgrades_12m'].fillna(0).astype(int)
        snapshot['upgrades_12m'] = snapshot['upgrades_12m'].fillna(0).astype(int)
    else:
        # No history in last 12 months (should rarely happen)
        snapshot['downgrades_12m'] = 0
        snapshot['upgrades_12m'] = 0

    # Rename columns for consistency with notebook expectations
    snapshot = snapshot.rename(columns={
        'internal_rating': 'current_rating',
        'external_rating': 'current_external_rating'
    })

    return snapshot


def save_historical_data(
    historical_data: Dict[str, pd.DataFrame],
    output_dir: str = "data/historical"
):
    """Save historical simulation to CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    historical_data['rating_history'].to_csv(
        f"{output_dir}/rating_history.csv", index=False
    )
    historical_data['borrower_metadata'].to_csv(
        f"{output_dir}/borrower_metadata.csv", index=False
    )
    historical_data['monthly_snapshots'].to_csv(
        f"{output_dir}/monthly_snapshots.csv", index=False
    )
    historical_data['events'].to_csv(
        f"{output_dir}/events_timeline.csv", index=False
    )

    print(f"\nHistorical data saved to {output_dir}/")
    print("Files:")
    print("  - rating_history.csv (all monthly ratings)")
    print("  - borrower_metadata.csv (borrower lifecycle)")
    print("  - monthly_snapshots.csv (portfolio stats)")
    print("  - events_timeline.csv (economic events)")


def generate_network_relationships_for_snapshot(
    snapshot: pd.DataFrame,
    network_density: str = 'medium',
    cluster_connectivity: float = 0.82,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate network relationships (facilities, guarantees, collateral) for a snapshot.

    This adds the missing network structure to historical snapshots so they can be
    used with build_hetero_graph() for GNN training.

    Parameters
    ----------
    snapshot : pd.DataFrame
        Snapshot from get_snapshot_at_date() with borrower features
    network_density : str
        Network density: 'low', 'medium', or 'high'
    cluster_connectivity : float
        Probability that relationships stay within the same cluster (0.8-0.85)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Dict[str, pd.DataFrame]
        Data dictionary compatible with build_hetero_graph():
        - 'borrowers': borrower features
        - 'facilities': facility relationships
        - 'guarantees': guarantee relationships
        - 'collateral': collateral relationships
        - 'co_borrows': co-borrowing relationships
        - 'industries': industry nodes
        - 'locations': location nodes
    """
    rng = np.random.RandomState(seed)
    n_borrowers = len(snapshot)

    # Prepare borrower DataFrame with cluster_id
    borrowers = snapshot.copy()
    borrowers = borrowers.rename(columns={
        'current_rating': 'internal_rating',
        'current_external_rating': 'external_rating',
        'is_defaulted': 'default_label'
    })

    # Add rating_numeric if not present
    if 'rating_numeric' not in borrowers.columns:
        from src.data.synthetic_generator import sp_rating_to_numeric
        borrowers['rating_numeric'] = borrowers['external_rating'].apply(sp_rating_to_numeric)

    # Generate synthetic financial features if not present
    # These are approximated based on rating to match expected input features for build_hetero_graph
    if 'total_assets' not in borrowers.columns:
        # Financial features correlated with credit quality (worse rating = smaller assets)
        # Base assets inversely related to rating (lower number = better rating = more assets)
        base_assets = np.maximum(1_000_000, 50_000_000 - borrowers['internal_rating'] * 2_000_000)
        borrowers['total_assets'] = base_assets * rng.lognormal(0, 0.5, n_borrowers)

        # Liabilities as % of assets (worse rating = higher leverage)
        leverage_ratio = 0.4 + (borrowers['internal_rating'] / 24) * 0.4  # 40-80% leverage
        borrowers['total_liabilities'] = borrowers['total_assets'] * leverage_ratio * rng.uniform(0.8, 1.2, n_borrowers)

        # Revenue roughly proportional to assets
        borrowers['revenue'] = borrowers['total_assets'] * rng.uniform(0.2, 0.5, n_borrowers)

        # EBITDA margin decreases with worse rating (better companies = higher margins)
        ebitda_margin = 0.25 - (borrowers['internal_rating'] / 24) * 0.15  # 25% down to 10%
        borrowers['ebitda'] = borrowers['revenue'] * ebitda_margin * rng.uniform(0.7, 1.3, n_borrowers)

    # Add cluster_id if not present (use sector as cluster)
    if 'cluster_id' not in borrowers.columns:
        # Map sector to cluster_id (0-4 for 5 clusters)
        sector_to_cluster = {
            'IND001': 0,  # Financial
            'IND002': 1,  # Manufacturing
            'IND003': 2,  # Services
            'IND004': 3,  # Retail
            'IND005': 4,  # O&G
        }
        borrowers['cluster_id'] = borrowers['sector'].map(sector_to_cluster).fillna(0).astype(int)

    # Map sector to industry_code
    borrowers['industry_code'] = borrowers['sector']

    # Assign location codes based on sector (for geographic clustering)
    location_map = {
        'IND001': 'LOC001',  # Financial
        'IND002': 'LOC002',  # Manufacturing
        'IND003': 'LOC003',  # Services
        'IND004': 'LOC001',  # Retail
        'IND005': 'LOC002',  # O&G
    }
    borrowers['location_code'] = borrowers['industry_code'].map(location_map).fillna('LOC004')

    # Network density parameters
    density_params = {
        'low': {'facilities_per_borrower': 1.5, 'guarantee_rate': 0.05, 'coborrow_rate': 0.10, 'fac_range': (1, 2)},
        'medium': {'facilities_per_borrower': 2.0, 'guarantee_rate': 0.10, 'coborrow_rate': 0.20, 'fac_range': (1, 3)},
        'high': {'facilities_per_borrower': 2.5, 'guarantee_rate': 0.15, 'coborrow_rate': 0.30, 'fac_range': (2, 4)}
    }
    params = density_params[network_density]

    # =========================================================================
    # 1. GENERATE FACILITIES
    # =========================================================================
    facilities = []
    fac_id = 0

    for idx, borrower_row in borrowers.iterrows():
        borrower_id = borrower_row["borrower_id"]
        # RandomState.randint doesn't have endpoint parameter, upper bound is exclusive
        n_fac = rng.randint(params['fac_range'][0], params['fac_range'][1] + 1)

        for _ in range(n_fac):
            limit = rng.uniform(100_000, 10_000_000)
            utilization_rate = np.clip(rng.normal(0.6, 0.25), 0.0, 1.0)
            drawn = np.clip(limit * utilization_rate, 0, limit)

            facilities.append({
                "facility_id": f"F{fac_id:05d}",
                "borrower_id": borrower_id,
                "limit": limit,
                "drawn": drawn,
                "interest_rate": rng.uniform(0.02, 0.15),
                "maturity_years": rng.choice([1, 3, 5, 7, 10]),
                "dpd": rng.choice([0, 7, 30], p=[0.80, 0.15, 0.05]),
            })
            fac_id += 1

    facilities = pd.DataFrame(facilities) if facilities else pd.DataFrame(columns=[
        'facility_id', 'borrower_id', 'limit', 'drawn', 'interest_rate', 'maturity_years', 'dpd'
    ])

    # =========================================================================
    # 2. GENERATE COLLATERAL (0-2 per facility)
    # =========================================================================
    collateral = []
    col_id = 0

    for idx, fac_row in facilities.iterrows():
        facility_id = fac_row["facility_id"]
        n_col = rng.randint(0, 3)  # 0-2 collateral per facility (upper bound is exclusive)

        for _ in range(n_col):
            value = rng.uniform(50_000, 5_000_000)

            collateral.append({
                "collateral_id": f"C{col_id:05d}",
                "facility_id": facility_id,
                "collateral_type": rng.choice([
                    "Real Estate", "Equipment", "Inventory", "Receivables"
                ]),
                "value": value,
                "lien_position": rng.choice([1, 2]),
            })
            col_id += 1

    collateral = pd.DataFrame(collateral) if collateral else pd.DataFrame(columns=[
        'collateral_id', 'facility_id', 'collateral_type', 'value', 'lien_position'
    ])

    # =========================================================================
    # 3. GENERATE GUARANTEES (cluster-aware)
    # =========================================================================
    guarantors_list = []
    guarantees_list = []
    guar_id = 0
    n_guarantors = int(n_borrowers * params['guarantee_rate'])

    for _ in range(n_guarantors):
        # 80% within-cluster guarantees, 20% between-cluster
        if rng.random() < cluster_connectivity:
            # Within-cluster: select from same cluster
            cluster_id = rng.choice(borrowers['cluster_id'].unique())
            cluster_borrowers = borrowers[borrowers['cluster_id'] == cluster_id]['borrower_id'].values
            if len(cluster_borrowers) >= 2:
                guarantor_borrower_id, guaranteed_borrower_id = rng.choice(
                    cluster_borrowers, size=2, replace=False
                )
            else:
                # Fallback if cluster too small
                guarantor_borrower_id = rng.choice(borrowers["borrower_id"].values)
                guaranteed_borrower_id = rng.choice(borrowers["borrower_id"].values)
        else:
            # Between-cluster: randomly select
            guarantor_borrower_id = rng.choice(borrowers["borrower_id"].values)
            guaranteed_borrower_id = rng.choice(borrowers["borrower_id"].values)

        # Avoid self-guarantees
        if guarantor_borrower_id != guaranteed_borrower_id:
            strength = rng.uniform(0.5, 1.0)

            guarantors_list.append({
                "guarantor_id": f"G{guar_id:05d}",
                "guarantor_type": rng.choice(["Personal", "Corporate"]),
                "strength": strength,
            })

            guarantees_list.append({
                "guarantor_id": f"G{guar_id:05d}",
                "borrower_id": guaranteed_borrower_id,
                "guarantor_borrower_id": guarantor_borrower_id,
                "coverage_amount": rng.uniform(100_000, 5_000_000),
                "relationship_strength": rng.uniform(0.6, 1.0),
            })
            guar_id += 1

    guarantors_df = pd.DataFrame(guarantors_list) if guarantors_list else pd.DataFrame(columns=[
        'guarantor_id', 'guarantor_type', 'strength'
    ])
    guarantees = pd.DataFrame(guarantees_list) if guarantees_list else pd.DataFrame(columns=[
        'guarantor_id', 'borrower_id', 'guarantor_borrower_id', 'coverage_amount', 'relationship_strength'
    ])

    # 4. Generate co-borrowing relationships (cluster-aware)
    n_coborrows = int(n_borrowers * params['coborrow_rate'])
    co_borrows = []

    # Within-cluster co-borrows (80-85%)
    n_within_cluster = int(n_coborrows * cluster_connectivity)
    for _ in range(n_within_cluster):
        cluster = rng.choice(borrowers['cluster_id'].unique())
        cluster_borrowers = borrowers[borrowers['cluster_id'] == cluster]['borrower_id'].values
        if len(cluster_borrowers) >= 2:
            b1, b2 = rng.choice(cluster_borrowers, size=2, replace=False)
            amount = rng.lognormal(14, 1.0)  # ~$1-5M
            co_borrows.append({
                'borrower_id_1': b1,
                'borrower_id_2': b2,
                'coborrow_amount': amount,
                'relationship_strength': rng.uniform(0.6, 1.0)
            })

    # Cross-cluster co-borrows (15-20%)
    n_cross_cluster = n_coborrows - n_within_cluster
    for _ in range(n_cross_cluster):
        b1, b2 = rng.choice(borrowers['borrower_id'].values, size=2, replace=False)
        amount = rng.lognormal(14, 1.0)
        co_borrows.append({
            'borrower_id_1': b1,
            'borrower_id_2': b2,
            'coborrow_amount': amount,
            'relationship_strength': rng.uniform(0.3, 0.7)
        })

    co_borrows_df = pd.DataFrame(co_borrows) if co_borrows else pd.DataFrame(columns=[
        'borrower_id_1', 'borrower_id_2', 'coborrow_amount', 'relationship_strength'
    ])

    # 5. Create industry and location nodes (renamed to match graph_builder expectations)
    unique_industries = borrowers['industry_code'].unique()
    industry_stress = pd.DataFrame({
        'industry_code': unique_industries,
        'industry_name': ['Financial Services', 'Manufacturing', 'Services', 'Retail', 'Oil & Gas'][:len(unique_industries)],
        'stress_index': [1.0] * len(unique_industries),  # Required by graph_builder
        'default_rate': [0.1] * len(unique_industries)    # Required by graph_builder
    })

    unique_locations = borrowers['location_code'].unique()
    location_stress = pd.DataFrame({
        'location_code': unique_locations,
        'location_name': ['Region A', 'Region B', 'Region C', 'Region D'][:len(unique_locations)],
        'unemployment_rate': [0.05] * len(unique_locations),  # Required by graph_builder
        'gdp_growth': [0.02] * len(unique_locations)          # Required by graph_builder
    })

    # 6. Create edge relationships dictionary (required by build_hetero_graph)
    # Extract edge relationships with proper src/dst naming convention

    # has_facility: borrower -> facility
    E_has_fac = facilities[['borrower_id', 'facility_id']].copy()
    E_has_fac.columns = ['src_borrower_id', 'dst_facility_id']
    # Add edge attributes
    E_has_fac['utilization'] = (facilities['drawn'] / facilities['limit']).fillna(0)
    E_has_fac['relationship_years'] = 3.0  # Default relationship duration

    # secures: collateral -> facility
    E_secures = collateral[['collateral_id', 'facility_id']].copy()
    E_secures.columns = ['src_collateral_id', 'dst_facility_id']
    # Add edge attributes (LTV ratio)
    collateral_fac = collateral.merge(facilities[['facility_id', 'limit']], on='facility_id')
    E_secures['ltv_ratio'] = (collateral_fac['value'] / collateral_fac['limit']).fillna(0.5)

    # guarantees: guarantor -> borrower
    E_guarantees = guarantees[['guarantor_id', 'borrower_id']].copy()
    E_guarantees.columns = ['src_guarantor_id', 'dst_borrower_id']
    # Add edge attributes
    E_guarantees['coverage_amount'] = guarantees['coverage_amount']
    E_guarantees['relationship_strength'] = guarantees['relationship_strength']

    # in_industry: borrower -> industry
    E_in_industry = borrowers[['borrower_id', 'industry_code']].copy()
    E_in_industry.columns = ['src_borrower_id', 'dst_industry_code']
    # Add edge attribute
    E_in_industry['exposure_weight'] = 1.0  # Equal weight per borrower

    # in_location: borrower -> location
    E_in_location = borrowers[['borrower_id', 'location_code']].copy()
    E_in_location.columns = ['src_borrower_id', 'dst_location_code']
    # Add edge attribute
    E_in_location['regional_exposure'] = 1.0  # Equal weight per borrower

    # co_borrows: borrower <-> borrower (symmetric)
    if len(co_borrows_df) > 0:
        E_co_borrows = co_borrows_df[['borrower_id_1', 'borrower_id_2']].copy()
        E_co_borrows.columns = ['src_borrower_id', 'dst_borrower_id']
        # Add edge attributes
        E_co_borrows['transaction_volume'] = co_borrows_df['coborrow_amount']
        E_co_borrows['interaction_frequency'] = co_borrows_df['relationship_strength']
    else:
        E_co_borrows = pd.DataFrame(columns=['src_borrower_id', 'dst_borrower_id', 'transaction_volume', 'interaction_frequency'])

    edges = {
        'has_facility': E_has_fac,
        'secures': E_secures,
        'guarantees': E_guarantees,
        'in_industry': E_in_industry,
        'in_location': E_in_location,
        'co_borrows': E_co_borrows
    }

    return {
        'borrowers': borrowers,
        'facilities': facilities,
        'guarantees': guarantees,
        'guarantors': guarantors_df,
        'collateral': collateral,
        'co_borrows': co_borrows_df,
        'industry_stress': industry_stress,  # Renamed from 'industries'
        'location_stress': location_stress,  # Renamed from 'locations'
        'edges': edges  # Added edge relationships
    }


def generate_temporal_training_data(
    rating_history: pd.DataFrame,
    borrower_metadata: pd.DataFrame,
    train_start: str = "2010-01-01",
    train_end: str = "2019-12-31",
    val_start: str = "2020-01-01",
    val_end: str = "2020-06-30",
    test_start: str = "2020-07-01",
    test_end: str = "2021-12-31",
    network_density: str = 'medium',
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate temporal training data for GNN from historical simulation.

    Creates monthly snapshots with network relationships for:
    - Training period (e.g., 2010-2019): Pre-crisis normal conditions
    - Validation period (e.g., 2020 H1): Early crisis period
    - Test period (e.g., 2020 H2-2021): Peak crisis period

    This enables the GNN to learn from historical patterns and test on crisis periods.

    Parameters
    ----------
    rating_history : pd.DataFrame
        Rating history from generate_historical_portfolio()
    borrower_metadata : pd.DataFrame
        Borrower metadata from generate_historical_portfolio()
    train_start, train_end : str
        Training period dates
    val_start, val_end : str
        Validation period dates
    test_start, test_end : str
        Test period dates
    network_density : str
        Network density for relationship generation
    seed : int
        Random seed

    Returns
    -------
    Dict[str, Any]
        {
            'train_snapshots': List[Dict] - Monthly snapshots for training
            'val_snapshots': List[Dict] - Monthly snapshots for validation
            'test_snapshots': List[Dict] - Monthly snapshots for testing
            'train_period': (start, end) - Training period
            'val_period': (start, end) - Validation period
            'test_period': (start, end) - Test period
        }
    """
    from dateutil.relativedelta import relativedelta

    def get_monthly_dates(start: str, end: str) -> List[str]:
        """Generate list of month-start dates between start and end."""
        dates = []
        current = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        while current <= end_date:
            dates.append(current.strftime('%Y-%m-%d'))
            current += relativedelta(months=1)
        return dates

    # Get monthly dates for each period
    train_dates = get_monthly_dates(train_start, train_end)
    val_dates = get_monthly_dates(val_start, val_end)
    test_dates = get_monthly_dates(test_start, test_end)

    print(f"\nGenerating temporal training data:")
    print(f"  Training: {train_start} to {train_end} ({len(train_dates)} months)")
    print(f"  Validation: {val_start} to {val_end} ({len(val_dates)} months)")
    print(f"  Test: {test_start} to {test_end} ({len(test_dates)} months)")

    def generate_snapshots_for_period(dates: List[str], period_name: str) -> List[Dict]:
        """Generate snapshots with network relationships for a period."""
        snapshots = []
        rng = np.random.RandomState(seed)

        # Prepare historical_data dictionary for get_snapshot_at_date
        historical_data_dict = {
            'rating_history': rating_history,
            'borrower_metadata': borrower_metadata
        }

        for i, date in enumerate(dates):
            # Get borrower snapshot at this date
            snapshot = get_snapshot_at_date(historical_data_dict, date)

            if len(snapshot) == 0:
                continue

            # Generate network relationships for this snapshot
            # Use date-specific seed for reproducibility but variation across months
            month_seed = seed + hash(date) % 10000
            data_with_network = generate_network_relationships_for_snapshot(
                snapshot,
                network_density=network_density,
                seed=month_seed
            )

            # Add metadata
            data_with_network['snapshot_date'] = date
            data_with_network['snapshot_index'] = i

            snapshots.append(data_with_network)

            if (i + 1) % 12 == 0:  # Print progress every year
                print(f"    {period_name}: Generated {i + 1}/{len(dates)} snapshots...")

        print(f"  [SUCCESS] {period_name}: {len(snapshots)} snapshots generated")
        return snapshots

    # Generate snapshots for each period
    train_snapshots = generate_snapshots_for_period(train_dates, "Training")
    val_snapshots = generate_snapshots_for_period(val_dates, "Validation")
    test_snapshots = generate_snapshots_for_period(test_dates, "Test")

    return {
        'train_snapshots': train_snapshots,
        'val_snapshots': val_snapshots,
        'test_snapshots': test_snapshots,
        'train_period': (train_start, train_end),
        'val_period': (val_start, val_end),
        'test_period': (test_start, test_end),
        'n_train_months': len(train_snapshots),
        'n_val_months': len(val_snapshots),
        'n_test_months': len(test_snapshots)
    }


if __name__ == "__main__":
    # Generate 15 years of historical data
    historical_data = generate_historical_portfolio(
        n_borrowers_final=5000,
        start_date="2010-01-01",
        end_date="2024-12-31",
        seed=42
    )

    # Save to CSV
    save_historical_data(historical_data, output_dir="data/historical")

    # Show snapshot at specific date
    snapshot = get_snapshot_at_date(historical_data, "2024-12-01")
    print(f"\nSnapshot at 2024-12-01:")
    print(f"  Active borrowers: {len(snapshot):,}")
    print(f"  Avg rating: {snapshot['internal_rating'].mean():.1f}")
    print(f"  Default rate: {snapshot['is_defaulted'].mean():.1%}")
