"""
================================================================================
01 — MWDT CALCULATION FROM RAW LINEUP DATA
Minutes Weighted Dyadic Tenure — Stubel (2026)
================================================================================

INPUT
    data/raw/lineups/<SEASON>/<TEAM>.csv — pbpstats.com 5-man lineup exports,
    regular season, one file per team-season (90 files). Columns used:
    ShortName (5 player surnames), TeamAbbreviation, Minutes (integer).

WHAT THIS SCRIPT DOES
    1. Splits every lineup into its C(5,2) = 10 player pairs and sums shared
       minutes per pair  -> data/processed/all_dyads_<SEASON>.csv
    2. Computes per team-season:
         MWDT          = sum of pair-minutes / N_Pairs
         N_Pairs       = number of distinct pairs that shared the floor
         Top10_Share   = share of pair-minutes held by the 10 most-used pairs
         Lineup_Minutes, Truncated (export hit the 500-row limit)
       in two versions:
         *_all  — every exported lineup
         *_h    — harmonized: only lineups with >= 3 minutes (see below)
       -> data/processed/mwdt_team_ranking.csv

WHY A HARMONIZED VERSION
    pbpstats exports at most 500 lineups, sorted by minutes. 59 of 90 team-
    seasons hit that limit; their cut-off lies at 0–2 minutes, so they are
    missing many very short lineups (up to ~14% of a season's minutes).
    Truncation is not random: it hits teams that used many lineups, i.e.
    disrupted rosters. Every lineup with >= 3 minutes is present in all 90
    files, so restricting all teams to those lineups makes the measures
    comparable. The harmonized version is the primary specification; the
    all-lineups version is reported as a robustness check.

NOTE ON THE DENOMINATOR
    Every game minute puts exactly 10 pairs on the floor, so total pair-
    minutes are nearly constant across teams. MWDT is therefore almost exactly
    (constant / N_Pairs): it measures rotation size. Script 02 tests this.

UNITS: game-clock minutes throughout. No rescaling.
================================================================================
"""

import os
import glob
from itertools import combinations

import numpy as np
import pandas as pd

RAW = 'data/raw/lineups'
SEASONS = ['2022-23', '2023-24', '2024-25']
OUT_DYADS = 'data/processed/all_dyads_{season}.csv'
OUT_TEAM = 'data/processed/mwdt_team_ranking.csv'
EXPORT_LIMIT = 500
H_THRESHOLD = 3  # minutes


def pair_minutes(lineups: pd.DataFrame) -> pd.Series:
    pairs = {}
    for lineup, minutes in zip(lineups['ShortName'], lineups['Minutes']):
        players = sorted(p.strip() for p in str(lineup).split(','))
        if len(players) != 5:
            continue
        for a, b in combinations(players, 2):
            pairs[(a, b)] = pairs.get((a, b), 0) + minutes
    s = pd.Series(pairs, dtype=float)
    s.index = s.index.set_names(['Player1', 'Player2'])
    return s.sort_values(ascending=False)


def summarize(pm: pd.Series, suffix: str) -> dict:
    total = pm.sum()
    return {
        f'MWDT{suffix}': total / len(pm),
        f'N_Pairs{suffix}': len(pm),
        f'Top10_Share{suffix}': pm.head(10).sum() / total,
        f'Pair_Minutes{suffix}': total,
    }


def main():
    print('=' * 70)
    print('01 — MWDT CALCULATION FROM RAW LINEUPS')
    print('=' * 70)
    team_rows = []
    for season in SEASONS:
        files = sorted(glob.glob(os.path.join(RAW, season, '*.csv')))
        season_dyads = []
        for f in files:
            d = pd.read_csv(f)
            team = d['TeamAbbreviation'].iloc[0]
            d['Minutes'] = pd.to_numeric(d['Minutes'], errors='coerce').fillna(0)
            pm_all = pair_minutes(d)
            pm_h = pair_minutes(d[d['Minutes'] >= H_THRESHOLD])
            dy = pm_all.reset_index(name='SharedMinutes')
            dy['Team'], dy['Season'] = team, season
            season_dyads.append(dy)
            row = {'Team': team, 'Season': season,
                   'Lineups': len(d), 'Lineup_Minutes': int(d['Minutes'].sum()),
                   'Truncated': int(len(d) >= EXPORT_LIMIT)}
            row.update(summarize(pm_all, '_all'))
            row.update(summarize(pm_h, '_h'))
            p1, p2 = pm_all.index[0]
            row['Top_Pair'] = f'{p1} / {p2}'
            team_rows.append(row)
        pd.concat(season_dyads, ignore_index=True)[
            ['Player1', 'Player2', 'SharedMinutes', 'Team', 'Season']
        ].to_csv(OUT_DYADS.format(season=season), index=False)
        print(f'  {season}: {len(files)} teams -> {OUT_DYADS.format(season=season)}')

    t = pd.DataFrame(team_rows)
    # backward-compatible column names = all-lineups version
    t['MWDT'], t['N_Pairs'], t['Top10_Share'] = t['MWDT_all'], t['N_Pairs_all'], t['Top10_Share_all']
    t = t.sort_values('MWDT_h', ascending=False).reset_index(drop=True)
    t.to_csv(OUT_TEAM, index=False)

    print(f"\nTeam-seasons: {len(t)} | truncated at {EXPORT_LIMIT} rows: {t['Truncated'].sum()}")
    print(f"Lineup minutes: truncated mean {t.loc[t.Truncated==1,'Lineup_Minutes'].mean():.0f}, "
          f"complete mean {t.loc[t.Truncated==0,'Lineup_Minutes'].mean():.0f}")
    for v in ['_all', '_h']:
        pmin = t[f'Pair_Minutes{v}']
        print(f"MWDT{v}: mean {t[f'MWDT{v}'].mean():.2f}, SD {t[f'MWDT{v}'].std():.2f}, "
              f"range {t[f'MWDT{v}'].min():.1f}–{t[f'MWDT{v}'].max():.1f} | pair-minutes CV {pmin.std()/pmin.mean():.3f} | "
              f"r(MWDT, 1/N) = {np.corrcoef(t[f'MWDT{v}'], 1/t[f'N_Pairs{v}'])[0,1]:.3f}")
    print(f'\nSaved: {OUT_TEAM}')


if __name__ == '__main__':
    main()
