# Data Dictionary

All minutes are regular-season game-clock minutes. `Season` strings use the form `2022-23`; in `team_data_corrected.csv`, `Season` is the ending year (2023 = 2022-23).

## Raw data — `data/raw/lineups/<season>/<TEAM>.csv`

pbpstats.com five-man lineup exports, one file per team-season (90 files). Exports are capped at 500 lineups, sorted by minutes; 59 files hit the cap (see README, "Harmonized measures"). Columns used by the analysis:

| Column | Meaning |
|---|---|
| `ShortName` | The five players in the lineup, surnames only, comma-separated |
| `TeamAbbreviation` | Team code |
| `Minutes` | Minutes the lineup played together (integer, rounded by pbpstats) |

The remaining columns (possessions, shooting splits, plus-minus) are exported by pbpstats and not used.

## Processed data — `data/processed/`

| File | Columns | Built by |
|---|---|---|
| `all_dyads_<season>.csv` | `Player1`, `Player2` (surnames, alphabetical), `SharedMinutes`, `Team`, `Season` | script 01 |
| `mwdt_team_ranking.csv` | `Lineups` (rows exported), `Lineup_Minutes`, `Truncated` (1 = hit 500-row cap); for all lineups (`_all`) and lineups ≥ 3 min (`_h`): `MWDT` (pair-minutes ÷ pairs), `N_Pairs`, `Top10_Share` (share of pair-minutes in the 10 most-used pairs), `Pair_Minutes`; `Top_Pair`. Unsuffixed `MWDT`, `N_Pairs`, `Top10_Share` = `_all` | script 01 |
| `team_data_corrected.csv` | `Team`, `Season`, `MWDT` (all lineups, for cross-checking), `AvgAge`, `NetRating`, `Payroll` / `Payroll_raw` (USD), `Payroll_Source`, `Continuity` (0–1), `WinPct_REAL` | assembled manually; see README, Data Sources |
| `payroll_spotrac.csv` | `TeamAbbr`, `SeasonStr`, `Payroll_Spotrac` (total cap allocations, USD) | transcribed from Spotrac; robustness R11 |
| `roster_continuity_bref.csv` | `TeamAbbr`, `SeasonStr`, `Cont_BRef` (0–1) | transcribed from Basketball-Reference; verification only |
| `nba_stats_clean.csv` | `Player` (full name), `Team`, `Season`, `PPG`, `REB`, `AST`, `STL` | script 00 |
| `H3_final_results.csv` | `LastName`, `PersonalMWDT` (season-1 shared minutes with all teammates), season-2 `PPG`, `REB`, `AST`, `STL`, `LagPair`, `MWDT_squared` | earlier pipeline; input to script 03 |
| `H3_player_panel.csv` | `Teams_S1`, `Teams_S2`, `Mover`, `Minutes_S1` (= PersonalMWDT ÷ 4), `N_Partners`, `Partner_Top4_Share`, `PPG_S1`, `PPG_S2`, `Surname_Collision` | script 03 |

## Data licensing

Lineup data are from pbpstats.com; payroll from HoopsHype and Spotrac; roster continuity, win percentages, Net Rating, age and player statistics from Basketball-Reference and official NBA standings. These third-party data are redistributed here in small, derived form solely for non-commercial research reproducibility; all rights remain with the original providers, and users should consult their terms before reuse. The MIT license in this repository applies to the code and documentation only.
