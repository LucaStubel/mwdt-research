"""
================================================================================
02 — TEAM-LEVEL ANALYSIS: H1, H2, H4 + ROBUSTNESS
Minutes Weighted Dyadic Tenure — Stubel & Kim (2026)
================================================================================

INPUTS
    data/processed/team_data_corrected.csv   team covariates (N = 90)
    data/processed/mwdt_team_ranking.csv     MWDT, N_Pairs, Top10_Share
                                             (run 01_mwdt_calculation.py first)

PRIMARY SPECIFICATION
    Harmonized measures (lineups >= 3 minutes, identical coverage for all 90
    team-seasons; see script 01). The all-lineups version is robustness R7.
    Payroll: HoopsHype for 87 team-seasons; Philadelphia's three values (which
    duplicated Orlando's in the original file) are taken from Spotrac. R11
    re-runs the key models with Spotrac payroll for all 90 team-seasons.

MODELS (OLS, standard errors clustered by franchise, 30 clusters)
    H1   NetRating ~ MWDT_c + Payroll_M + AvgAge + Continuity + East
    H2   NetRating ~ MWDT_c + MWDT_c^2 + controls
         turning point = -b1 / (2 b2), delta-method CI,
         Lind & Mehlum (2010) test: slope must be significantly positive at
         the sample minimum AND significantly negative at the sample maximum.
    H4   Within-sample comparison of MWDT with roster continuity (Williams
         test for dependent correlations) + both in the same model.
         Meta-analytic benchmarks (Gonzalez-Mule et al., 2020) are shown for
         context only; they are artifact-corrected population estimates and
         are not directly comparable to a single-sample r.

ROBUSTNESS (rotation size)
    MWDT = (near-constant total dyadic minutes) / N_Pairs, so it is almost a
    pure function of how many player pairs a team used. R1-R5 test whether
    MWDT carries information beyond that.

UNITS
    MWDT is in game-clock minutes throughout, mean-centered at the sample
    mean (248.2). No rescaling constant is applied. (Earlier drafts reported
    b = .1235; that value is the coefficient on *uncentered* MWDT in the
    quadratic model, i.e. the slope extrapolated to MWDT = 0, not a unit
    conversion.)

OUTPUTS
    results/team_level_results.txt
    figures/fig1_h1_scatter.png ... fig4_rotation_size.png
================================================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

TEAM_FILE = 'data/processed/team_data_corrected.csv'
MWDT_FILE = 'data/processed/mwdt_team_ranking.csv'
OUT_TXT = 'results/team_level_results.txt'

ABBR = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Knicks': 'NYK', 'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'Oklahoma': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
}
EAST = {'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DET', 'IND', 'MIA', 'MIL',
        'NYK', 'ORL', 'PHI', 'TOR', 'WAS'}
SEASON = {2023: '2022-23', 2024: '2023-24', 2025: '2024-25'}
CONTROLS = 'Payroll_M + AvgAge + Continuity + East'


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def fit(formula, df):
    """OLS with franchise-clustered SEs; drops rows with missing payroll when payroll is used."""
    d = df.dropna(subset=['Payroll_M']) if 'Payroll_M' in formula else df
    return smf.ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': d['TeamAbbr']})


def wild_cluster_bootstrap(formula, df, var, B=9999, seed=2027):
    """Wild cluster restricted (WCR) bootstrap p-value, Rademacher weights,
    clusters = franchises (Cameron, Gelbach & Miller 2008; Roodman et al. 2019).
    Recommended when the number of clusters is small (here 30)."""
    import patsy
    d = df.dropna(subset=['Payroll_M']) if 'Payroll_M' in formula else df
    y, X = patsy.dmatrices(formula, d, return_type='dataframe')
    y, cols = y.values.ravel(), list(X.columns)
    X = X.values
    k = cols.index(var)
    groups = pd.factorize(d['TeamAbbr'])[0]
    G, n, p = groups.max() + 1, len(y), X.shape[1]
    XtX_inv = np.linalg.inv(X.T @ X)
    c = G / (G - 1) * (n - 1) / (n - p)

    def t_stats(Y):  # Y: n x R
        beta = XtX_inv @ X.T @ Y
        U = Y - X @ beta
        S = np.zeros((G, p, Y.shape[1]))
        np.add.at(S, groups, X[:, :, None] * U[:, None, :])
        a = XtX_inv[k]                      # row k of bread
        proj = np.einsum('p,gpr->gr', a, S)  # G x R
        se = np.sqrt(c * (proj ** 2).sum(axis=0))
        return beta[k] / se

    t_obs = t_stats(y[:, None])[0]
    Xr = np.delete(X, k, axis=1)
    fit_r = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    u_r = y - fit_r
    rng = np.random.default_rng(seed)
    t_star = []
    for chunk in range(0, B, 1000):
        R = min(1000, B - chunk)
        w = rng.choice([-1.0, 1.0], size=(G, R))
        Ystar = fit_r[:, None] + u_r[:, None] * w[groups]
        t_star.append(t_stats(Ystar))
    t_star = np.concatenate(t_star)
    hits = int((np.abs(t_star) >= abs(t_obs)).sum())
    wild_cluster_bootstrap.last = {'hits': hits, 'B': B, 'seed': seed}
    return float(t_obs), hits / B


# ── DATA ──────────────────────────────────────────────────────────────────────

def load():
    t = pd.read_csv(TEAM_FILE)
    t['TeamAbbr'] = t['Team'].map(ABBR)
    t['SeasonStr'] = t['Season'].map(SEASON)
    assert t['TeamAbbr'].notna().all(), 'unmapped team name'
    t = t.rename(columns={'MWDT': 'MWDT_teamfile'})
    m = pd.read_csv(MWDT_FILE).rename(columns={'Team': 'TeamAbbr', 'Season': 'SeasonStr'})
    df = t.merge(m, on=['TeamAbbr', 'SeasonStr'], how='left')
    sp = pd.read_csv('data/processed/payroll_spotrac.csv')
    df = df.merge(sp, on=['TeamAbbr', 'SeasonStr'], how='left')
    df['PayrollSp_M'] = df['Payroll_Spotrac'] / 1e6
    assert len(df) == 90 and df['MWDT_h'].notna().all(), 'merge failed'
    assert (df['MWDT_teamfile'] - df['MWDT_all']).abs().max() < 1e-6, 'MWDT mismatch between files'

    # primary = harmonized measures
    df['MWDT'] = df['MWDT_h']
    df['N_Pairs'] = df['N_Pairs_h']
    df['Top10_Share'] = df['Top10_Share_h']

    df['Payroll_M'] = df['Payroll_raw'] / 1e6
    df['WinPct'] = df['WinPct_REAL']
    df['East'] = df['TeamAbbr'].isin(EAST).astype(int)
    df['MWDT_c'] = df['MWDT'] - df['MWDT'].mean()
    df['InvPairs'] = 1000 / df['N_Pairs']
    df['LogPairs'] = np.log(df['N_Pairs'])
    df['MWDT_resid'] = smf.ols('MWDT ~ InvPairs', df).fit().resid
    df['MWDTall_c'] = df['MWDT_all'] - df['MWDT_all'].mean()
    return df


# ── TABLES ────────────────────────────────────────────────────────────────────

def descriptives(df):
    cols = [('MWDT', 'MWDT (min)'), ('NetRating', 'Net Rating'), ('WinPct', 'Win %'),
            ('Payroll_M', 'Payroll ($M)'), ('AvgAge', 'Avg Age'),
            ('Continuity', 'Roster Continuity'), ('N_Pairs', 'N pairs used')]
    print('\nTABLE 1 — DESCRIPTIVES AND CORRELATIONS (N = 90)')
    print('-' * 100)
    head = f"{'':22}{'Mean':>9}{'SD':>9}" + ''.join(f'{i+1:>8}' for i in range(len(cols) - 1))
    print(head)
    for i, (c, lab) in enumerate(cols):
        line = f'{i+1}. {lab:19}{df[c].mean():9.2f}{df[c].std():9.2f}'
        for j in range(i):
            pair = df[[c, cols[j][0]]].dropna()
            r, p = stats.pearsonr(pair[c], pair[cols[j][0]])
            line += f'{r:6.2f}{stars(p):<2}'
        print(line)
    print('* p<.05  ** p<.01  *** p<.001')


def regression_table(models, names, rows, title):
    print(f'\n{title}')
    print('-' * (26 + 16 * len(models)))
    print(f"{'':26}" + ''.join(f'{n:>16}' for n in names))
    for var, lab in rows:
        if not any(var in m.params for m in models):
            continue
        b = ''.join(f"{m.params[var]:>13.4f}{stars(m.pvalues[var]):<3}" if var in m.params else f"{'—':>16}"
                    for m in models)
        se = ''.join(f"{'(' + format(m.bse[var], '.4f') + ')':>16}" if var in m.params else f"{'':16}"
                     for m in models)
        print(f'{lab:26}{b}\n{"":26}{se}')
    print(f"{'R²':26}" + ''.join(f'{m.rsquared:>16.3f}' for m in models))
    print(f"{'N':26}" + ''.join(f'{int(m.nobs):>16}' for m in models))
    print('Clustered SEs (30 franchises). * p<.05  ** p<.01  *** p<.001')


def h2_tests(df, m):
    q = 'I(MWDT_c ** 2)'
    b1, b2 = m.params['MWDT_c'], m.params[q]
    V = m.cov_params().loc[['MWDT_c', q], ['MWDT_c', q]].values
    tp_c = -b1 / (2 * b2)
    g = np.array([-1 / (2 * b2), b1 / (2 * b2 ** 2)])
    tp_se = float(np.sqrt(g @ V @ g))
    mean = df['MWDT'].mean()
    tp = tp_c + mean
    pct = (df['MWDT'] < tp).mean() * 100
    n_beyond = int((df['MWDT'] > tp).sum())

    print('\nH2 — CURVILINEARITY')
    print('-' * 70)
    print(f'b(MWDT_c)   = {b1:.5f}  p = {m.pvalues["MWDT_c"]:.4f}')
    print(f'b(MWDT_c²)  = {b2:.6f}  p = {m.pvalues[q]:.4f}')
    print(f'Turning point = {tp:.1f} min (delta-method 95% CI {tp - 1.96*tp_se:.1f} – {tp + 1.96*tp_se:.1f})')
    print(f'Share of team-seasons below turning point: {pct:.0f}%  ({n_beyond} of 90 beyond it)')
    print('\nLind–Mehlum slope test (slope = b1 + 2·b2·(x − mean)):')
    res = {}
    for lab, x in [('sample minimum', df['MWDT'].min()), ('sample maximum', df['MWDT'].max())]:
        w = np.array([1, 2 * (x - mean)])
        s = float(w @ np.array([b1, b2]))
        se = float(np.sqrt(w @ V @ w))
        p_one = stats.norm.sf(s / se) if lab == 'sample minimum' else stats.norm.cdf(s / se)
        res[lab] = (s, p_one)
        print(f'  at {lab} ({x:.1f} min): slope = {s:+.4f}, SE = {se:.4f}, one-sided p = {p_one:.3f}')
    inverted_u = res['sample minimum'][1] < .05 and res['sample maximum'][1] < .05
    verdict = ('both slopes significant with opposite signs' if inverted_u else
               'slope at the upper end is not significantly negative; the data show '
               'diminishing returns, not a downturn')
    print(f'  -> Inverted U {"supported" if inverted_u else "NOT supported"}: {verdict}.')
    return tp


def williams_test(r12, r13, r23, n):
    """Williams (1959) t for H0: rho12 = rho13 (dependent correlations)."""
    R = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
    rbar = (r12 + r13) / 2
    t = (r12 - r13) * np.sqrt(((n - 1) * (1 + r23)) /
                              (2 * ((n - 1) / (n - 3)) * R + rbar**2 * (1 - r23)**3))
    return t, 2 * stats.t.sf(abs(t), n - 3)


def h4(df, m1):
    y = df['NetRating']
    r_m = stats.pearsonr(df['MWDT'], y)[0]
    r_c = stats.pearsonr(df['Continuity'], y)[0]
    r_mc = stats.pearsonr(df['MWDT'], df['Continuity'])[0]
    t, p = williams_test(r_m, r_c, r_mc, len(df))
    print('\nH4 — MWDT vs. TRADITIONAL STABILITY MEASURE (within this sample)')
    print('-' * 70)
    print(f'r(MWDT, NetRating)              = {r_m:.3f}')
    print(f'r(Roster Continuity, NetRating) = {r_c:.3f}')
    print(f'r(MWDT, Continuity)             = {r_mc:.3f}')
    print(f'Williams test, H0 equal correlations: t({len(df)-3}) = {t:.2f}, p = {p:.3f}')
    print(f"Joint model (Table 2, H1): MWDT p = {m1.pvalues['MWDT_c']:.4f}, "
          f"Continuity p = {m1.pvalues['Continuity']:.4f} — both contribute.")
    print('\nContext only (not a like-for-like comparison): Gonzalez-Mulé et al. (2020)')
    print('report artifact-corrected meta-analytic rho = .200 (additive), .110 (collective),')
    print('.080 (dispersion) across organizational teams.')


# ── FIGURES ───────────────────────────────────────────────────────────────────

def figures(df, m_h2, tp):
    os.makedirs('figures', exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    xs = np.linspace(df['MWDT'].min(), df['MWDT'].max(), 100)

    # Fig 1 — H1 scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df['MWDT'], df['NetRating'], s=28, alpha=.7, color='#1f4e79')
    b = np.polyfit(df['MWDT'], df['NetRating'], 1)
    ax.plot(xs, np.polyval(b, xs), color='#c0392b')
    r = stats.pearsonr(df['MWDT'], df['NetRating'])[0]
    ax.set(xlabel='MWDT (clock-minutes per pair)', ylabel='Net Rating',
           title=f'Fig. 1 — MWDT and team Net Rating (r = {r:.2f}, N = 90)')
    fig.savefig('figures/fig1_h1_scatter.png', dpi=200, bbox_inches='tight'); plt.close(fig)

    # Fig 2 — H2 curve (controls at means)
    fig, ax = plt.subplots(figsize=(7, 5))
    mean = df['MWDT'].mean()
    grid = pd.DataFrame({'MWDT_c': xs - mean})
    for c in ['Payroll_M', 'AvgAge', 'Continuity', 'East']:
        grid[c] = df[c].mean()
    pred = m_h2.get_prediction(grid).summary_frame()
    ax.scatter(df['MWDT'], df['NetRating'], s=22, alpha=.45, color='#7f8c8d')
    ax.plot(xs, pred['mean'], color='#1f4e79')
    ax.fill_between(xs, pred['mean_ci_lower'], pred['mean_ci_upper'], color='#1f4e79', alpha=.15)
    ax.axvline(tp, ls='--', color='#c0392b', lw=1)
    ax.text(tp, ax.get_ylim()[1], f'turning point {tp:.0f} min ', ha='right', va='top',
            color='#c0392b', fontsize=9)
    ax.set(xlabel='MWDT (clock-minutes per pair)', ylabel='Net Rating (controls at mean)',
           title='Fig. 2 — Quadratic fit: diminishing returns at high MWDT')
    fig.savefig('figures/fig2_h2_curve.png', dpi=200, bbox_inches='tight'); plt.close(fig)

    # Fig 3 — within-sample correlations with 95% CIs
    fig, ax = plt.subplots(figsize=(7.5, 4))
    labs, rs, lo, hi = [], [], [], []
    for c, lab in [('MWDT', 'MWDT'), ('InvPairs', '1 / N pairs\n(rotation tightness)'),
                   ('Continuity', 'Roster\ncontinuity'), ('MWDT_resid', 'MWDT net of\nrotation size')]:
        r = stats.pearsonr(df[c], df['NetRating'])[0]
        z, se = np.arctanh(r), 1 / np.sqrt(len(df) - 3)
        labs.append(lab); rs.append(r)
        lo.append(r - np.tanh(z - 1.96 * se)); hi.append(np.tanh(z + 1.96 * se) - r)
    ax.bar(labs, rs, yerr=[lo, hi], capsize=4, color=['#1f4e79', '#2e86c1', '#7f8c8d', '#bdc3c7'])
    ax.axhline(0, color='black', lw=.8)
    ax.set(ylabel='r with Net Rating (95% CI)', title='Fig. 3 — Correlates of Net Rating in this sample (N = 90)')
    fig.savefig('figures/fig3_h4_comparison.png', dpi=200, bbox_inches='tight'); plt.close(fig)

    # Fig 4 — MWDT vs rotation size
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df['N_Pairs'], df['MWDT'], s=28, alpha=.7, color='#1f4e79')
    ns = np.linspace(df['N_Pairs'].min(), df['N_Pairs'].max(), 100)
    k = (df['MWDT'] * df['N_Pairs']).mean()
    ax.plot(ns, k / ns, color='#c0392b', label=f'MWDT = {k:,.0f} / N pairs')
    r = np.corrcoef(df['MWDT'], 1 / df['N_Pairs'])[0, 1]
    ax.set(xlabel='Number of player pairs that shared the floor', ylabel='MWDT (clock-minutes per pair)',
           title=f'Fig. 4 — MWDT tracks rotation size almost exactly (r with 1/N = {r:.3f})')
    ax.legend(frameon=False)
    fig.savefig('figures/fig4_rotation_size.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    print('\nFigures saved to figures/ (fig1–fig4).')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs('results', exist_ok=True)
    log = open(OUT_TXT, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log)

    print('=' * 78)
    print('02 — TEAM-LEVEL ANALYSIS (H1, H2, H4, ROBUSTNESS)   Stubel & Kim (2026)')
    print('=' * 78)
    df = load()
    print(f"N = {len(df)} team-seasons | MWDT mean {df['MWDT'].mean():.2f} min, SD {df['MWDT'].std():.2f}")
    print(f"Win % mean = {df['WinPct'].mean():.3f}")
    descriptives(df)

    m0 = fit(f'NetRating ~ {CONTROLS}', df)
    m1 = fit(f'NetRating ~ MWDT_c + {CONTROLS}', df)
    m2 = fit(f'NetRating ~ MWDT_c + I(MWDT_c**2) + {CONTROLS}', df)
    m3 = fit(f'WinPct ~ MWDT_c + {CONTROLS}', df)
    rows = [('MWDT_c', 'MWDT (centered, min)'), ('I(MWDT_c ** 2)', 'MWDT² (centered)'),
            ('Payroll_M', 'Payroll ($M)'), ('AvgAge', 'Average age'),
            ('Continuity', 'Roster continuity'), ('East', 'East conference'), ('Intercept', 'Constant')]
    regression_table([m0, m1, m2, m3], ['Controls', 'H1 linear', 'H2 quad.', 'Win % (H1)'], rows,
                     'TABLE 2 — OLS, DV = NET RATING (Models 1–3) / WIN % (Model 4)')
    b = m1.params['MWDT_c']
    print(f'\nH1: +10 min of MWDT ≈ {10*b:+.2f} Net Rating; +1 SD ({df["MWDT"].std():.1f} min) ≈ '
          f'{df["MWDT"].std()*b:+.2f} Net Rating, controls held constant.')

    tp = h2_tests(df, m2)
    h4(df, m1)

    r1 = fit(f'NetRating ~ MWDT_c + N_Pairs + {CONTROLS}', df)
    r2 = fit(f'NetRating ~ MWDT_c + LogPairs + {CONTROLS}', df)
    r3 = fit(f'NetRating ~ InvPairs + {CONTROLS}', df)
    r4 = fit(f'NetRating ~ MWDT_resid + InvPairs + {CONTROLS}', df)
    r5 = fit(f'NetRating ~ Top10_Share + N_Pairs + {CONTROLS}', df)
    rows_r = [('MWDT_c', 'MWDT (centered)'), ('MWDT_resid', 'MWDT net of 1/N'),
              ('N_Pairs', 'N pairs'), ('LogPairs', 'log N pairs'), ('InvPairs', '1000 / N pairs'),
              ('Top10_Share', 'Top-10 pair share'), ('Continuity', 'Roster continuity')]
    regression_table([r1, r2, r3, r4, r5], ['R1', 'R2', 'R3', 'R4', 'R5'], rows_r,
                     'TABLE 3 — ROBUSTNESS: DOES MWDT ADD INFORMATION BEYOND ROTATION SIZE?\n'
                     '(DV = Net Rating; all models also include payroll, age, East)')
    print(f"\nr(MWDT, 1/N_Pairs) = {np.corrcoef(df['MWDT'], df['InvPairs'])[0,1]:.3f};  "
          f"r(1/N_Pairs, NetRating) = {stats.pearsonr(df['InvPairs'], df['NetRating'])[0]:.3f};  "
          f"r(MWDT net of 1/N, NetRating) = {stats.pearsonr(df['MWDT_resid'], df['NetRating'])[0]:.3f}")

    r7a = fit(f'NetRating ~ MWDTall_c + {CONTROLS}', df)
    r7b = fit(f'NetRating ~ MWDTall_c + N_Pairs_all + {CONTROLS}', df)
    r7c = fit(f'NetRating ~ Top10_Share_all + N_Pairs_all + {CONTROLS}', df)
    r8 = fit('NetRating ~ Top10_Share + N_Pairs + Payroll_M + Continuity + East', df)
    r9 = fit(f'NetRating ~ Top10_Share + N_Pairs + Truncated + {CONTROLS}', df)
    r10 = fit('NetRating ~ Top10_Share + N_Pairs + AvgAge + Continuity + East', df)
    print('\nFURTHER ROBUSTNESS (DV = Net Rating)')
    print('-' * 70)
    for lab, mod, v in [('R7a all lineups (no harmonization): MWDT', r7a, 'MWDTall_c'),
                        ('R7b all lineups: MWDT + N pairs', r7b, 'MWDTall_c'),
                        ('R7c all lineups: Top-10 share + N pairs', r7c, 'Top10_Share_all'),
                        ('R8 Top-10 share, without age control', r8, 'Top10_Share'),
                        ('R9 Top-10 share + export-truncation dummy', r9, 'Top10_Share'),
                        ('R10 Top-10 share, without payroll (N = 90)', r10, 'Top10_Share')]:
        print(f"  {lab:48} b = {mod.params[v]:+.4f}  p = {mod.pvalues[v]:.4f}  N = {int(mod.nobs)}")
    sd = df['Top10_Share'].std()
    print(f"  Top-10 share SD = {sd:.3f}; +1 SD ≈ {r5.params['Top10_Share']*sd:+.2f} Net Rating (R5, p = {r5.pvalues['Top10_Share']:.4f})")
    print(f"  r(Top-10 share, N pairs) = {np.corrcoef(df['Top10_Share'], df['N_Pairs'])[0,1]:.3f}; "
          f"r(Truncated, N pairs) = {np.corrcoef(df['Truncated'], df['N_Pairs'])[0,1]:.3f}")

    csp = CONTROLS.replace('Payroll_M', 'PayrollSp_M')
    print('\nR11 SPOTRAC PAYROLL FOR ALL 90 TEAM-SEASONS')
    for lab, f, v in [('H1 MWDT', f'NetRating ~ MWDT_c + {csp}', 'MWDT_c'),
                      ('R1 MWDT + N pairs', f'NetRating ~ MWDT_c + N_Pairs + {csp}', 'MWDT_c'),
                      ('R5 Top-10 share + N pairs', f'NetRating ~ Top10_Share + N_Pairs + {csp}', 'Top10_Share'),
                      ('R9 R5 + truncation dummy', f'NetRating ~ Top10_Share + N_Pairs + Truncated + {csp}', 'Top10_Share')]:
        mod = fit(f, df)
        print(f"  {lab:48} b = {mod.params[v]:+.4f}  p = {mod.pvalues[v]:.4f}  N = {int(mod.nobs)}")

    print('\nR12 SMALL-CLUSTER INFERENCE: WILD CLUSTER RESTRICTED BOOTSTRAP')
    print('  Method: restricted (null imposed), Rademacher weights at franchise level, 30 clusters,')
    print('  B = 9,999 replications, seed = 2027, two-sided p = share of |t*| >= |t|, CRV1 small-sample factor.')
    for lab, f, v in [('H1 MWDT', f'NetRating ~ MWDT_c + {CONTROLS}', 'MWDT_c'),
                      ('R1 MWDT + N pairs', f'NetRating ~ MWDT_c + N_Pairs + {CONTROLS}', 'MWDT_c'),
                      ('R5 Top-10 share + N pairs', f'NetRating ~ Top10_Share + N_Pairs + {CONTROLS}', 'Top10_Share'),
                      ('R9 R5 + truncation dummy', f'NetRating ~ Top10_Share + N_Pairs + Truncated + {CONTROLS}', 'Top10_Share')]:
        t, pb = wild_cluster_bootstrap(f, df, v)
        pc = fit(f, df).pvalues[v]
        h = wild_cluster_bootstrap.last
        print(f"  {lab:32} t = {t:+.3f}  clustered p = {pc:.6f}  wild-bootstrap p = {pb:.6f} ({h['hits']}/{h['B']})")

    r6 = fit(f'NetRating ~ MWDT_c + {CONTROLS} + C(Season)', df)
    print(f"R6 season fixed effects (no rotation control): b(MWDT_c) = {r6.params['MWDT_c']:.4f}, "
          f"p = {r6.pvalues['MWDT_c']:.4f}")

    print('\nCASES CITED IN THE PAPER')
    for t, s in [('NYK', '2022-23'), ('NYK', '2023-24'), ('GSW', '2022-23'),
                 ('MEM', '2022-23'), ('MEM', '2023-24'), ('BOS', '2023-24')]:
        row = df[(df['TeamAbbr'] == t) & (df['SeasonStr'] == s)].iloc[0]
        print(f"  {t} {s}: MWDT {row['MWDT']:.1f}, N pairs {int(row['N_Pairs'])}, "
              f"Net Rating {row['NetRating']:+.2f}")

    figures(df, m2, tp)
    sys.stdout = sys.__stdout__
    log.close()
    print(f'Results written to {OUT_TXT}')


if __name__ == '__main__':
    main()
