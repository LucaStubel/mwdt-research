"""
================================================================================
04 — EXHIBITS FOR THE SSAC 2027 ABSTRACT
Minutes Weighted Dyadic Tenure — Stubel (2026)
================================================================================

Builds the one figure and one table used in the Sloan abstract directly from
the same data and models as script 02, so every number in the abstract can be
regenerated here.

    figures/sloan_figure1.png   (a) MWDT vs number of pairs used
                                (b) Net Rating vs Top-10 pair share, both net
                                    of rotation size and controls
    results/sloan_table1.txt    three models: MWDT, MWDT + rotation size,
                                concentration + rotation size

Run after 01_mwdt_calculation.py.
================================================================================
"""

import importlib.util
import os

import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location('reg', 'notebooks/02_regression_analysis.py')
reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg)

df = reg.load()
C = reg.CONTROLS

m1 = reg.fit(f'NetRating ~ MWDT_c + {C}', df)
m2 = reg.fit(f'NetRating ~ MWDT_c + N_Pairs + {C}', df)
m3 = reg.fit(f'NetRating ~ Top10_Share + N_Pairs + {C}', df)


def cell(m, var, scale):
    if var not in m.params:
        return '—'
    return f'{m.params[var] * scale:+.2f}{reg.stars(m.pvalues[var])} (p = {m.pvalues[var]:.3f})'


rows = [('MWDT, per 10 min', 'MWDT_c', 10),
        ('Pairs used, per 10 pairs', 'N_Pairs', 10),
        ('Top-10 pair share, per 10 pp', 'Top10_Share', 0.10)]
os.makedirs('results', exist_ok=True)
with open('results/sloan_table1.txt', 'w', encoding='utf-8') as f:
    head = f"{'DV: Net Rating':30}{'(1) MWDT':>26}{'(2) + rotation size':>26}{'(3) Concentration':>26}"
    f.write('TABLE 1 — What does pair-averaged familiarity measure?\n' + head + '\n')
    for lab, v, s in rows:
        f.write(f'{lab:30}' + ''.join(f'{cell(m, v, s):>26}' for m in (m1, m2, m3)) + '\n')
    f.write(f"{'R²':30}" + ''.join(f'{m.rsquared:>26.2f}' for m in (m1, m2, m3)) + '\n')
    f.write('N = 90 team-seasons. Controls: payroll, average age, roster continuity, conference. '
            'Stars and p-values: franchise-clustered SEs. * p<.05 ** p<.01 *** p<.001\n')
print(open('results/sloan_table1.txt', encoding='utf-8').read())

# ── Figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={'wspace': 0.28})
navy, red, grey = '#16325c', '#c0392b', '#8a94a6'

# (a) identity
k = (df['MWDT'] * df['N_Pairs']).mean()
ns = np.linspace(df['N_Pairs'].min() - 3, df['N_Pairs'].max() + 3, 200)
a.plot(ns, k / ns, color=red, lw=1.6, zorder=1, label=f'MWDT = {k:,.0f} ÷ pairs used')
a.scatter(df['N_Pairs'], df['MWDT'], s=20, color=navy, alpha=.85, zorder=2, edgecolor='white', lw=.4)
r = np.corrcoef(df['MWDT'], 1 / df['N_Pairs'])[0, 1]
a.set(xlabel='Player pairs that shared the floor', ylabel='MWDT (shared minutes per pair)')
a.set_title(f'(a) MWDT vs. pairs used  (r = {r:.3f})', loc='left', fontsize=10, fontweight='bold')
a.legend(frameon=False, loc='lower left', fontsize=8.5)
for t, s, dx, dy in [('NYK', '2022-23', 25, 5), ('MEM', '2023-24', -45, 45)]:
    row = df[(df['TeamAbbr'] == t) & (df['SeasonStr'] == s)].iloc[0]
    a.annotate(f'{t} {s}', (row['N_Pairs'], row['MWDT']), xytext=(row['N_Pairs'] + dx, row['MWDT'] + dy),
               fontsize=8, color=navy, arrowprops=dict(arrowstyle='-', color=grey, lw=.7))

# (b) added-variable plot
base = f'N_Pairs + {C}'
d = df.dropna(subset=['Payroll_M'])
ry = smf.ols(f'NetRating ~ {base}', d).fit().resid
rx = smf.ols(f'Top10_Share ~ {base}', d).fit().resid
b.axhline(0, color='#d0d5dd', lw=.8)
b.scatter(rx * 100, ry, s=20, color=navy, alpha=.85, edgecolor='white', lw=.4)
xs = np.linspace(rx.min(), rx.max(), 50)
b.plot(xs * 100, m3.params['Top10_Share'] * xs, color=red, lw=1.6)
b.set(xlabel='Top-10 pair share (pp, net of rotation size & controls)',
      ylabel='Net Rating (net of rotation size & controls)')
b.set_title("(b) Concentration, net of rotation size",
            loc='left', fontsize=10, fontweight='bold')

os.makedirs('figures', exist_ok=True)
fig.savefig('figures/sloan_figure1.png', dpi=300, bbox_inches='tight')
print('Saved figures/sloan_figure1.png')
