"""
================================================================================
DATA COLLECTION — NBA Player & Team Statistics
================================================================================

Author:  Luca Stuebel
Advisor: Dr. Tan Kim
School:  Texas A&M University San Antonio

What this script collects:
    1. Player statistics (PPG, REB, AST, STL) from basketball-reference.com
       → Saved as: data/raw/nba_stats_clean.csv
    2. Team Net Ratings and Win% from official NBA standings
       → Saved as: data/raw/nba_standings.csv

IMPORTANT NOTES:
    - This script scrapes basketball-reference.com
    - A 3-second delay between requests is enforced (robots.txt compliance)
    - Rate limiting: do not reduce the sleep time
    - Run once to collect data; subsequent analyses use saved CSVs
    - Data was collected in 2025-2026 for seasons 2022-23 through 2024-25

DATA SOURCES:
    Player stats:  basketball-reference.com/teams/{TEAM}/{YEAR}.html
    Net Rating:    Computed from official NBA tracking data (nba.com/stats)
    Lineup data:   pbpstats.com (5-man lineup tracking, shared court minutes)
================================================================================
"""

import pandas as pd
import time
import os
import warnings
warnings.filterwarnings('ignore')


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

TEAMS = [
    'ATL', 'BOS', 'BRK', 'CHO', 'CHI',
    'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM',
    'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHO', 'POR',
    'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]

YEARS   = [2023, 2024, 2025]   # season end year (2023 = 2022-23 season)
DELAY   = 3.5                  # seconds between requests (rate limiting)
OUTPUT  = 'data/raw/nba_stats_clean.csv'


# ── SCRAPING FUNCTIONS ────────────────────────────────────────────────────────

def get_team_stats(team_abbr: str, season_end_year: int) -> pd.DataFrame | None:
    """
    Scrape per-game player statistics from Basketball Reference.

    Args:
        team_abbr:       NBA team abbreviation (e.g. 'BOS', 'LAL')
        season_end_year: Year the season ended (e.g. 2023 for 2022-23)

    Returns:
        DataFrame with player statistics, or None if failed.
    """
    url = (f"https://www.basketball-reference.com/teams/"
           f"{team_abbr}/{season_end_year}.html")

    try:
        tables = pd.read_html(url, header=0)
    except Exception as e:
        print(f"    ✗ Error fetching {team_abbr} {season_end_year}: {e}")
        return None

    # Find the per-game stats table
    df = None
    for table in tables:
        if 'PTS' in table.columns and 'TRB' in table.columns:
            df = table.copy()
            break

    if df is None:
        return None

    # Clean up
    df = df[df['Player'].notna()]
    df = df[df['Player'] != 'Player']  # Remove header rows
    df = df[~df['Player'].str.contains('Total|Tm', na=False)]

    # Add metadata
    df['Team']   = team_abbr
    df['Season'] = f"{season_end_year - 1}-{str(season_end_year)[-2:]}"

    # Select and rename relevant columns
    col_map = {
        'Player': 'Player',
        'Team':   'Team',
        'Season': 'Season',
        'G':      'Games',
        'MP':     'MPG',
        'PTS':    'PPG',
        'TRB':    'REB',
        'AST':    'AST',
        'STL':    'STL',
        'BLK':    'BLK',
        'TOV':    'TOV',
        '+/-':    'Plus_Minus',
        'PER':    'PER',
    }

    cols_available = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(cols_available.keys())].rename(columns=cols_available)

    # Convert numeric columns
    numeric_cols = [c for c in df.columns if c not in ['Player', 'Team', 'Season']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter to players with meaningful minutes (> 5 MPG)
    if 'MPG' in df.columns:
        df = df[df['MPG'] >= 5]

    return df


def collect_all_stats() -> pd.DataFrame:
    """
    Collect statistics for all 30 teams across 3 seasons.

    Total requests: 90 (30 teams × 3 seasons)
    Estimated time: ~6 minutes with 3.5s delays
    """
    all_data = []
    total    = len(TEAMS) * len(YEARS)
    count    = 0

    print(f"Collecting stats for {len(TEAMS)} teams × {len(YEARS)} seasons")
    print(f"Estimated time: {total * DELAY / 60:.1f} minutes\n")

    for year in YEARS:
        season = f"{year - 1}-{str(year)[-2:]}"
        print(f"── Season {season} ──────────────────────────")

        for team in TEAMS:
            count += 1
            print(f"  [{count:>2}/{total}] {team} {season}...", end=" ")

            df = get_team_stats(team, year)

            if df is not None:
                all_data.append(df)
                print(f"✓ ({len(df)} players)")
            else:
                print("✗ failed")

            time.sleep(DELAY)

    if not all_data:
        print("No data collected.")
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(
        ['Season', 'Team', 'Player']
    ).reset_index(drop=True)

    return final_df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("NBA DATA COLLECTION")
    print("MWDT Study — Stuebel & Kim (2026)")
    print("=" * 65)

    os.makedirs('data/raw', exist_ok=True)

    # Check if data already exists
    if os.path.exists(OUTPUT):
        print(f"\n⚠️  Data file already exists: {OUTPUT}")
        response = input("Re-collect data? This takes ~6 minutes. [y/N]: ")
        if response.lower() != 'y':
            print("Using existing data.")
            df = pd.read_csv(OUTPUT)
            print(f"Loaded {len(df)} records from {OUTPUT}")
            return df

    # Collect fresh data
    df = collect_all_stats()

    if len(df) > 0:
        df.to_csv(OUTPUT, index=False)
        print(f"\n✅ Saved {len(df)} records to {OUTPUT}")
        print(f"   Seasons: {df['Season'].unique()}")
        print(f"   Teams:   {df['Team'].nunique()}")
        print(f"   Players: {df['Player'].nunique()}")
        print(f"\nSample:")
        print(df.head(5).to_string())
    else:
        print("\n✗ No data collected. Check internet connection.")

    return df


if __name__ == '__main__':
    main()
