"""
================================================================================
MWDT CALCULATION PIPELINE
Minutes Weighted Dyadic Tenure — NBA Team Familiarity Metric
================================================================================

Author:  Luca Stuebel
Advisor: Dr. Tan Kim
School:  Texas A&M University San Antonio
Paper:   "Familiarity as a Dyadic Phenomenon: Introducing MWDT"
Sloan:   MIT Sloan Sports Analytics Conference 2027

--------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES:
    1. Loads raw 5-man lineup data (shared court minutes per lineup)
    2. Extracts all unique player pairs (dyads) from each lineup
    3. Aggregates shared minutes across all lineups for each dyad
    4. Computes MWDT = Σ minutes(i,j) / N_pairs per team per season
    5. Outputs: all_dyads.csv + mwdt_team_ranking.csv

THEORY:
    Traditional team tenure measures average individual experience (additive)
    or time together as a unit (collective). Both fail to capture the relational
    structure that Wegner's (1987) Transactive Memory System predicts should
    drive coordination. MWDT operationalizes TMS at the dyadic level — the level
    where the mechanism actually lives.

    MWDT = Σ(i<j) minutes(i,j) / N_pairs

    where N_pairs = number of unique player pairs on the roster.
    This captures network density, not just volume.

DATA SOURCE:
    NBA.com 5-man lineup data via pbpstats.com
    Each row = one unique 5-player lineup + total minutes played together
    Season coverage: 2022-23, 2023-24, 2024-25 (N = 90 team-seasons)
================================================================================
"""

import pandas as pd
import numpy as np
from itertools import combinations
import os
import warnings
warnings.filterwarnings('ignore')


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

SEASONS = {
    '2022-23': 'data/raw/lineups_2022-23/',
    '2023-24': 'data/raw/lineups_2023-24/',
    '2024-25': 'data/raw/lineups_2024-25/',
}

OUTPUT_DYADS   = 'data/processed/all_dyads.csv'
OUTPUT_MWDT    = 'data/processed/mwdt_team_ranking.csv'
MIN_MINUTES    = 0   # Minimum minutes for a lineup to be included


# ── CORE FUNCTIONS ─────────────────────────────────────────────────────────────

def extract_dyads_from_lineup(lineup_str: str, minutes: float,
                               team: str, season: str) -> list:
    """
    Extract all unique player pairs (dyads) from a 5-man lineup.

    A 5-man lineup contains C(5,2) = 10 unique pairs. For each pair,
    we record the minutes they shared the court together.

    Args:
        lineup_str: Comma-separated string of 5 player names
        minutes:    Total minutes this lineup played together
        team:       NBA team abbreviation (e.g. 'BOS')
        season:     Season string (e.g. '2022-23')

    Returns:
        List of dicts: [{team, season, player_a, player_b, minutes_together}]

    Example:
        lineup = "LeBron James, Anthony Davis, Austin Reaves, ..."
        → 10 dyad records, each with minutes shared
    """
    players = [p.strip() for p in str(lineup_str).split(',')]

    if len(players) != 5:
        return []

    dyads = []
    for player_a, player_b in combinations(sorted(players), 2):
        dyads.append({
            'Team':             team,
            'Season':           season,
            'Player_A':         player_a,
            'Player_B':         player_b,
            'Minutes_Together': minutes
        })

    return dyads


def process_team_file(filepath: str, team: str, season: str) -> pd.DataFrame:
    """
    Process a single team's lineup CSV file and extract all dyads.

    Args:
        filepath: Path to the team's lineup CSV
        team:     NBA team abbreviation
        season:   Season identifier

    Returns:
        DataFrame with all dyads for this team-season
    """
    df = pd.read_csv(filepath)

    # Identify lineup and minutes columns (flexible naming)
    lineup_col  = df.columns[0]
    minutes_col = 'Minutes' if 'Minutes' in df.columns else df.columns[1]

    # Filter minimum minutes
    df = df[pd.to_numeric(df[minutes_col], errors='coerce') >= MIN_MINUTES]
    df = df.dropna(subset=[lineup_col, minutes_col])

    all_dyads = []
    for _, row in df.iterrows():
        dyads = extract_dyads_from_lineup(
            lineup_str=row[lineup_col],
            minutes=float(row[minutes_col]),
            team=team,
            season=season
        )
        all_dyads.extend(dyads)

    return pd.DataFrame(all_dyads)


def compute_mwdt(dyads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MWDT (Minutes Weighted Dyadic Tenure) for each team-season.

    FORMULA:
        MWDT = Σ(i<j) minutes(i,j) / N_pairs

    where N_pairs = number of unique player pairs with any shared minutes.

    THEORETICAL BASIS:
        - Numerator captures total dyadic familiarity accumulated
        - Denominator normalizes by roster complexity (network size)
        - Result: average depth of familiarity per pair
        - High MWDT = concentrated, deep ties (tight rotation)
        - Low MWDT  = dispersed, shallow ties (many players used)

    Args:
        dyads_df: DataFrame with columns [Team, Season, Player_A, Player_B,
                  Minutes_Together]

    Returns:
        DataFrame with MWDT per team-season, sorted descending
    """
    # Aggregate: sum all minutes for each unique dyad
    dyad_totals = dyads_df.groupby(
        ['Team', 'Season', 'Player_A', 'Player_B'],
        as_index=False
    )['Minutes_Together'].sum()

    dyad_totals = dyad_totals.rename(
        columns={'Minutes_Together': 'Total_Minutes_Together'}
    )

    # Compute MWDT per team-season
    mwdt_records = []

    for (team, season), group in dyad_totals.groupby(['Team', 'Season']):
        n_pairs     = len(group)
        total_min   = group['Total_Minutes_Together'].sum()
        mwdt        = total_min / n_pairs if n_pairs > 0 else 0
        mwdt_sq     = mwdt ** 2  # For H2 quadratic term

        # Additional descriptives
        top_pair    = group.nlargest(1, 'Total_Minutes_Together').iloc[0]

        mwdt_records.append({
            'Team':              team,
            'Season':            season,
            'MWDT':              round(mwdt, 4),
            'MWDT_Squared':      round(mwdt_sq, 4),
            'N_Pairs':           n_pairs,
            'Total_Dyadic_Min':  round(total_min, 2),
            'Top_Pair':          f"{top_pair['Player_A']} / {top_pair['Player_B']}",
            'Top_Pair_Minutes':  round(top_pair['Total_Minutes_Together'], 2),
        })

    mwdt_df = pd.DataFrame(mwdt_records)
    mwdt_df = mwdt_df.sort_values('MWDT', ascending=False).reset_index(drop=True)
    mwdt_df['MWDT_Rank'] = mwdt_df.index + 1

    return mwdt_df, dyad_totals


def run_pipeline():
    """
    Main pipeline: process all team-season lineup files → compute MWDT.
    """
    print("=" * 70)
    print("MWDT CALCULATION PIPELINE")
    print("Minutes Weighted Dyadic Tenure — Stuebel & Kim (2026)")
    print("=" * 70)

    all_dyads = []

    for season, folder in SEASONS.items():
        print(f"\nProcessing Season: {season}")
        print("-" * 40)

        if not os.path.exists(folder):
            print(f"  ⚠️  Folder not found: {folder}")
            print(f"      Place lineup CSVs in {folder}")
            continue

        files_found = [f for f in os.listdir(folder) if f.endswith('.csv')]
        print(f"  Found {len(files_found)} team files")

        for filename in sorted(files_found):
            team = filename.replace('.csv', '').split('_')[0].upper()
            filepath = os.path.join(folder, filename)

            try:
                team_dyads = process_team_file(filepath, team, season)
                all_dyads.append(team_dyads)
                print(f"  ✓ {team}: {len(team_dyads)} dyad records")
            except Exception as e:
                print(f"  ✗ {team}: {e}")

    if not all_dyads:
        print("\n⚠️  No data processed. See data/raw/ for required file structure.")
        return

    # Combine all dyads
    dyads_df = pd.concat(all_dyads, ignore_index=True)
    print(f"\n✅ Total dyad records: {len(dyads_df):,}")

    # Compute MWDT
    mwdt_df, dyad_totals = compute_mwdt(dyads_df)

    # Save outputs
    os.makedirs('data/processed', exist_ok=True)
    dyad_totals.to_csv(OUTPUT_DYADS, index=False)
    mwdt_df.to_csv(OUTPUT_MWDT, index=False)

    print(f"\n📊 MWDT RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Rank':<6} {'Team':<25} {'Season':<12} {'MWDT':>8}")
    print("-" * 55)
    for _, row in mwdt_df.head(10).iterrows():
        print(f"{int(row['MWDT_Rank']):<6} {row['Team']:<25} {row['Season']:<12} {row['MWDT']:>8.1f}")
    print("  ...")
    print(f"\n  Range: {mwdt_df['MWDT'].min():.1f} — {mwdt_df['MWDT'].max():.1f} minutes")
    print(f"  Mean:  {mwdt_df['MWDT'].mean():.1f} minutes")
    print(f"  SD:    {mwdt_df['MWDT'].std():.1f} minutes")

    print(f"\n✅ Saved: {OUTPUT_DYADS}")
    print(f"✅ Saved: {OUTPUT_MWDT}")

    return mwdt_df, dyad_totals


if __name__ == '__main__':
    run_pipeline()
