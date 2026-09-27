"""
Reproducibility tests. Run from the repository root after scripts 01–04:
    pytest -q
They check that the pipeline rebuilds the dyads exactly, that the key
mathematical property holds, and that every number quoted in the SSAC 2027
abstract matches the model output.
"""
import glob
import importlib.util
import re
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('reg', 'notebooks/02_regression_analysis.py')
reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg)


@pytest.fixture(scope='module')
def df():
    return reg.load()


def test_ninety_raw_files():
    assert len(glob.glob('data/raw/lineups/*/*.csv')) == 90


def test_dyads_rebuild_exactly_from_raw():
    raw = pd.read_csv('data/raw/lineups/2022-23/WAS.csv')
    pairs = {}
    for lineup, m in zip(raw['ShortName'], raw['Minutes']):
        for a, b in combinations(sorted(x.strip() for x in lineup.split(',')), 2):
            pairs[(a, b)] = pairs.get((a, b), 0) + m
    dy = pd.read_csv('data/processed/all_dyads_2022-23.csv')
    dy = dy[dy['Team'] == 'WAS']
    shipped = {(a, b): v for a, b, v in zip(dy['Player1'], dy['Player2'], dy['SharedMinutes'])}
    assert pairs == shipped


def test_each_lineup_minute_contains_ten_pairs():
    raw = pd.read_csv('data/raw/lineups/2023-24/BOS.csv')
    dy = pd.read_csv('data/processed/all_dyads_2023-24.csv')
    assert dy.loc[dy['Team'] == 'BOS', 'SharedMinutes'].sum() == 10 * raw['Minutes'].sum()


def test_team_file_complete(df):
    assert len(df) == 90
    assert df['Payroll_M'].notna().all()
    assert abs(df['WinPct'].mean() - 0.5) < 0.005


def test_continuity_matches_basketball_reference(df):
    b = pd.read_csv('data/processed/roster_continuity_bref.csv')
    m = df.merge(b, on=['TeamAbbr', 'SeasonStr'])
    assert len(m) == 90 and (m['Continuity'] - m['Cont_BRef']).abs().max() < 1e-9


def test_rotation_size_identity(df):
    r = np.corrcoef(df['MWDT'], 1 / df['N_Pairs'])[0, 1]
    assert r > 0.99
    cv = df['Pair_Minutes_h'].std() / df['Pair_Minutes_h'].mean()
    assert cv < 0.04


def test_abstract_numbers(df):
    C = reg.CONTROLS
    m1 = reg.fit(f'NetRating ~ MWDT_c + {C}', df)
    m2 = reg.fit(f'NetRating ~ MWDT_c + N_Pairs + {C}', df)
    m3 = reg.fit(f'NetRating ~ Top10_Share + N_Pairs + {C}', df)
    text = open('sloan/abstract.md', encoding='utf-8').read()
    assert 'r = .52' in text and round(np.corrcoef(df['MWDT'], df['NetRating'])[0, 1], 2) == 0.52
    assert 'associated with +2.0 points' in text and round(m1.params['MWDT_c'] * df['MWDT'].std(), 1) == 2.0
    assert 'p = .30' in text and round(m2.pvalues['MWDT_c'], 2) == 0.30
    assert 'p = .02' in text and round(m3.pvalues['Top10_Share'], 2) == 0.02
    assert '+1.2 Net Rating' in text and round(m3.params['Top10_Share'] * df['Top10_Share'].std(), 1) == 1.2
    assert 'r = .994' in text and round(np.corrcoef(df['MWDT'], 1 / df['N_Pairs'])[0, 1], 3) == 0.994


def test_abstract_under_500_words():
    text = open('sloan/abstract.md', encoding='utf-8').read()
    assert len(re.sub(r'[#*]', '', text).split()) < 500


def test_wild_bootstrap_reported_value(df):
    C = reg.CONTROLS
    _, p = reg.wild_cluster_bootstrap(f'NetRating ~ Top10_Share + N_Pairs + {C}', df, 'Top10_Share')
    assert round(p, 2) == 0.05
