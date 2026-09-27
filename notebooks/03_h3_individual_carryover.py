"""
================================================================================
03 — PLAYER-LEVEL ANALYSIS: H3 (CARRYOVER) + MOVERS TEST
Minutes Weighted Dyadic Tenure — Stubel (2026)
================================================================================

WHAT THIS SCRIPT DOES
    1. Rebuilds the player panel from files in this repo:
         H3_final_results.csv      PersonalMWDT (season 1), PPG (season 2)
         all_dyads_*.csv           teams per player-season, partner structure
         nba_stats_clean.csv       prior-season PPG
       -> data/processed/H3_player_panel.csv
    2. H3: PPG_S2 ~ PersonalMWDT_S1 (+ PPG_S1)
    3. Mechanism check: PersonalMWDT vs. minutes played
    4. Movers test: players whose season-2 team(s) share no team with season 1
    5. Name-collision sensitivity check

DEFINITIONS
    PersonalMWDT = sum of a player's shared minutes with all teammates in
    season 1. Because every minute on court is shared with exactly four
    teammates, PersonalMWDT = 4 x minutes played (exactly, up to lineup data
    coverage). It is therefore a measure of playing time, not of how
    concentrated a player's partnerships are. We add a partner-concentration
    measure that is independent of total minutes:
        Partner_Top4_Share = share of the player's shared minutes that went to
                             his four most frequent partners.

STANDARD ERRORS
    Clustered by player (a player can appear in both season transitions).

LIMITATION
    Dyad files identify players by last name only (pbpstats "ShortName").
    Same-surname teammates (e.g. Antetokounmpo, Green, Jackson) are merged.
    Section 5 re-runs H3 without any surname that is shared by more than one
    player in the league in that season.
================================================================================
"""

import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

H3_FILE = 'data/processed/H3_final_results.csv'
STATS_FILE = 'data/processed/nba_stats_clean.csv'
PANEL_OUT = 'data/processed/H3_player_panel.csv'
OUT_TXT = 'results/player_level_results.txt'
TEAM_FIX = {'BRK': 'BKN', 'PHO': 'PHX', 'CHO': 'CHA'}
SUFFIX = {'Jr', 'Sr', 'II', 'III', 'IV'}


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for st in self.s:
            st.write(x)

    def flush(self):
        for st in self.s:
            st.flush()


def surname(full):
    """'Kelly Oubre Jr.' -> 'Oubre Jr.' to match pbpstats ShortName."""
    toks = full.split()
    if len(toks) >= 3 and toks[-1].replace('.', '') in SUFFIX:
        return f'{toks[-2]} {toks[-1]}'
    return toks[-1]


def strip_accents(s):
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def build_panel():
    dy = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('data/processed/all_dyads_*.csv'))])
    long = pd.concat([
        dy.rename(columns={'Player1': 'P', 'Player2': 'Partner'}),
        dy.rename(columns={'Player2': 'P', 'Player1': 'Partner'}),
    ])[['P', 'Partner', 'Team', 'Season', 'SharedMinutes']]

    # per player-season: total shared minutes, teams, partner concentration
    by_team = long.groupby(['P', 'Season', 'Team'])['SharedMinutes'].sum().reset_index()
    def conc(g):
        tot = g['SharedMinutes'].sum()
        return pd.Series({
            'MWDT_check': tot,
            'Partner_Top4_Share': g.nlargest(4, 'SharedMinutes')['SharedMinutes'].sum() / tot,
            'N_Partners': g['Partner'].nunique(),
        })
    ps = long.groupby(['P', 'Season']).apply(conc).reset_index()
    teams = by_team.groupby(['P', 'Season'])['Team'].apply(lambda x: '/'.join(sorted(x))).rename('Teams')
    ps = ps.merge(teams.reset_index(), on=['P', 'Season'])

    h = pd.read_csv(H3_FILE)
    h['S1'] = h['LagPair'].str[:7]
    h['S2'] = h['LagPair'].str[-7:]
    p = h.merge(ps.rename(columns={'Teams': 'Teams_S1'}), left_on=['LastName', 'S1'],
                right_on=['P', 'Season'], how='left').drop(columns=['P', 'Season'])
    assert (p['PersonalMWDT'] == p['MWDT_check']).all(), 'PersonalMWDT does not match dyad files'
    p = p.merge(ps[['P', 'Season', 'Teams']].rename(columns={'Teams': 'Teams_S2'}),
                left_on=['LastName', 'S2'], right_on=['P', 'Season'], how='left').drop(columns=['P', 'Season'])

    # prior-season PPG from player stats, weighted by dyadic minutes when traded mid-season
    st = pd.read_csv(STATS_FILE)
    st['Team'] = st['Team'].replace(TEAM_FIX)
    st['Surname'] = st['Player'].map(surname)
    st['SurnameKey'] = st['Surname'].map(strip_accents).str.replace('.', '', regex=False)
    by_team['Key'] = by_team['P'].map(strip_accents).str.replace('.', '', regex=False)
    j = by_team.merge(st, left_on=['Key', 'Season', 'Team'], right_on=['SurnameKey', 'Season', 'Team'])
    n_players = j.groupby(['P', 'Season', 'Team'])['Player'].transform('nunique')
    j = j[n_players == 1]  # drop same-team surname collisions
    ppg1 = (j.groupby(['P', 'Season'])
              .apply(lambda g: np.average(g['PPG'], weights=g['SharedMinutes']))
              .rename('PPG_S1').reset_index())
    p = p.merge(ppg1, left_on=['LastName', 'S1'], right_on=['P', 'Season'], how='left').drop(columns=['P', 'Season'])

    # league-wide surname collisions in S1 or S2
    coll = st.groupby(['SurnameKey', 'Season'])['Player'].nunique()
    coll = set(coll[coll > 1].index)
    key = p['LastName'].map(strip_accents).str.replace('.', '', regex=False)
    p['Surname_Collision'] = [(k, s1) in coll or (k, s2) in coll for k, s1, s2 in zip(key, p['S1'], p['S2'])]

    p['PPG_S2'] = p['PPG']
    p['Minutes_S1'] = p['PersonalMWDT'] / 4
    p['Mover'] = [int(isinstance(b, str) and not set(a.split('/')) & set(b.split('/')))
                  for a, b in zip(p['Teams_S1'], p['Teams_S2'])]
    p['PlayerID'] = p['LastName']
    p['MWDT_k'] = p['PersonalMWDT'] / 1000
    cols = ['LastName', 'LagPair', 'Teams_S1', 'Teams_S2', 'Mover', 'PersonalMWDT', 'Minutes_S1',
            'N_Partners', 'Partner_Top4_Share', 'PPG_S1', 'PPG_S2', 'REB', 'AST', 'STL',
            'Surname_Collision']
    p[cols].to_csv(PANEL_OUT, index=False)
    return p


def fit(f, d):
    return smf.ols(f, d).fit(cov_type='cluster', cov_kwds={'groups': d['PlayerID']})


def show(m, label, var='MWDT_k'):
    print(f"  {label:58} b = {m.params[var]:+.3f} per 1,000 min  SE {m.bse[var]:.3f}  "
          f"p = {m.pvalues[var]:.4f}  N = {int(m.nobs)}  R² = {m.rsquared:.3f}")


def main():
    os.makedirs('results', exist_ok=True)
    log = open(OUT_TXT, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log)
    print('=' * 78)
    print('03 — PLAYER-LEVEL ANALYSIS (H3 + MOVERS)   Stubel (2026)')
    print('=' * 78)

    p = build_panel()
    d = p.dropna(subset=['PPG_S1', 'PPG_S2'])
    print(f'Panel: {len(p)} player-transitions; {len(d)} with prior-season PPG matched '
          f'({len(p) - len(d)} unmatched, mostly name-format differences).')
    print(f"Transitions: {p['LagPair'].value_counts().to_dict()}")
    print(f"Movers: {int(d['Mover'].sum())}  Stayers: {int((1 - d['Mover']).sum())}")
    print(f'Saved panel: {PANEL_OUT}')

    print('\n1. MECHANISM CHECK')
    print('-' * 70)
    print('  PersonalMWDT = 4 × minutes played in season 1 (each court minute is shared with 4 teammates).')
    print(f"  r(PersonalMWDT, Minutes_S1) = {np.corrcoef(p['PersonalMWDT'], p['Minutes_S1'])[0,1]:.3f} "
          '(identity) — H3 therefore tests playing time, not partnership depth.')
    print(f"  r(PersonalMWDT, PPG_S1) = {d[['PersonalMWDT','PPG_S1']].corr().iloc[0,1]:.3f};  "
          f"r(PersonalMWDT, PPG_S2) = {d[['PersonalMWDT','PPG_S2']].corr().iloc[0,1]:.3f}")

    print('\n2. H3 — PPG in season 2 (SEs clustered by player)')
    print('-' * 70)
    show(fit('PPG_S2 ~ MWDT_k', p), 'Bivariate (all rows)')
    show(fit('PPG_S2 ~ MWDT_k', d), 'Bivariate (rows with prior PPG)')
    m2 = fit('PPG_S2 ~ MWDT_k + PPG_S1', d)
    show(m2, '+ prior PPG')
    m2b = fit('PPG_S2 ~ MWDT_k + PPG_S1 + C(LagPair)', d)
    show(m2b, '+ prior PPG + transition FE')

    print('\n3. PARTNER CONCENTRATION (independent of total minutes)')
    print('-' * 70)
    m3 = fit('PPG_S2 ~ Partner_Top4_Share + MWDT_k + PPG_S1', d)
    print(f"  Partner_Top4_Share: b = {m3.params['Partner_Top4_Share']:+.3f}  SE {m3.bse['Partner_Top4_Share']:.3f}  "
          f"p = {m3.pvalues['Partner_Top4_Share']:.4f}  (controls: minutes, prior PPG; N = {int(m3.nobs)})")

    print('\n4. MOVERS TEST (PPG_S2 ~ PersonalMWDT + PPG_S1)')
    print('-' * 70)
    mv, sy = d[d['Mover'] == 1], d[d['Mover'] == 0]
    show(fit('PPG_S2 ~ MWDT_k + PPG_S1', mv), 'Movers')
    show(fit('PPG_S2 ~ MWDT_k + PPG_S1', sy), 'Stayers')
    mi = fit('PPG_S2 ~ MWDT_k * Mover + PPG_S1', d)
    print(f"  Interaction MWDT × Mover: b = {mi.params['MWDT_k:Mover']:+.3f}, p = {mi.pvalues['MWDT_k:Mover']:.4f}")
    print('  Reading: because PersonalMWDT is playing time, a carryover among movers shows that')
    print('  players who got minutes last year score more next year — it does not isolate')
    print('  transportable relational knowledge.')

    print('\n5. SENSITIVITY — exclude league-wide surname collisions')
    print('-' * 70)
    dc = d[~d['Surname_Collision']]
    show(fit('PPG_S2 ~ MWDT_k + PPG_S1', dc), 'H3 + prior PPG, no collisions')
    show(fit('PPG_S2 ~ MWDT_k + PPG_S1', dc[dc['Mover'] == 1]), 'Movers, no collisions')

    sys.stdout = sys.__stdout__
    log.close()
    print(f'Results written to {OUT_TXT}')


if __name__ == '__main__':
    main()
