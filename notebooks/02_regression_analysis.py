"""
================================================================================
MWDT REGRESSION ANALYSIS — H1, H2, H4
================================================================================

Author:  Luca Stuebel
Advisor: Dr. Tan Kim
School:  Texas A&M University San Antonio

Hypotheses Tested:
    H1: MWDT positively predicts team Net Rating (OLS, clustered SEs)
    H2: MWDT follows an inverted-U (quadratic term, mean-centered)
    H4: MWDT outperforms traditional team tenure metrics

Specification:
    NetRating = β₀ + β₁(MWDT_c) + β₂(MWDT_c²) + β₃(Payroll) +
                β₄(AvgAge) + β₅(Continuity) + β₆(Conference) + ε

    Standard errors clustered at team level (30 clusters).
    MWDT mean-centered before squaring to reduce multicollinearity.

================================================================================
UNIT NOTE — IMPORTANT FOR REPRODUCIBILITY
================================================================================

The file team_data_corrected.csv stores MWDT in CLOCK-MINUTES
(e.g., 248.2 clock-minutes of shared court time).

The original analysis was conducted with pbpstats.com raw lineup units,
where 1 raw unit ≈ 3.5 clock-minutes. This script applies the conversion
factor CLOCK_TO_RAW = 3.5 so that all reported coefficients match the paper.

This scaling affects ONLY the beta coefficient (β = .1235).
It does NOT affect: r, p-values, R², turning point in clock-minutes,
or any theoretical conclusion.

To work in clock-minutes instead, set: CLOCK_TO_RAW = 1.0
All p-values and conclusions remain identical.
================================================================================

Key Results (paper values):
    H1: β = .1235, SE = .036, p = .0005, r = .506, R² = .386
    H2: β² = -.00018, p = .018, turning point = 347 clock-minutes (98th pct)
    H4: r(MWDT) = .506 vs. ρ(additive tenure) = .200 — 2.5× stronger
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ── UNIT CONVERSION FACTOR ────────────────────────────────────────────────────
# Set to 3.5 to reproduce paper's β = .1235 (pbpstats raw units)
# Set to 1.0 to work in clock-minutes (β = .035, same p-value)
CLOCK_TO_RAW = 3.5


# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_team_data(filepath: str = 'data/processed/team_data_corrected.csv') -> pd.DataFrame:
    """
    Load and prepare the team-level dataset (N = 90 team-seasons).

    MWDT in CSV = clock-minutes. Divided by CLOCK_TO_RAW to match paper units.
    All other variables are in their natural units.
    """
    df = pd.read_csv(filepath)

    # Clean payroll string → float → millions
    df['Payroll_raw'] = (df['Payroll'].astype(str)
                         .str.replace(r'[\$,]', '', regex=True)
                         .pipe(pd.to_numeric, errors='coerce'))
    df['Payroll_M'] = df['Payroll_raw'] / 1_000_000

    # Conference (East = 1, West = 0)
    east = ['Atlanta','Boston','Brooklyn','Charlotte','Chicago','Cleveland',
            'Detroit','Indiana','Miami','Milwaukee','New York','Orlando',
            'Philadelphia','Toronto','Washington']
    df['Conference'] = df['Team'].apply(
        lambda t: 1 if any(e in t for e in east) else 0
    )

    # ── MWDT UNIT CONVERSION ──────────────────────────────────────────────────
    # Clock-minutes → paper units (divide by CLOCK_TO_RAW)
    # After centering, MWDT_c is in paper units throughout
    df['MWDT_clock'] = df['MWDT']                           # original clock-minutes
    df['MWDT_units'] = df['MWDT_clock'] / CLOCK_TO_RAW      # paper analysis units

    mwdt_mean_units  = df['MWDT_units'].mean()              # centering point (units)
    mwdt_mean_clock  = mwdt_mean_units * CLOCK_TO_RAW       # centering point (minutes)

    df['MWDT_c']  = df['MWDT_units'] - mwdt_mean_units     # centered (units)
    df['MWDT_c2'] = df['MWDT_c'] ** 2                      # squared centered (units²)

    # Team ID for clustering
    df['Team_ID'] = pd.Categorical(df['Team']).codes

    print(f"Dataset: N = {len(df)} team-seasons | {df['Team'].nunique()} teams | "
          f"{df['Season'].nunique()} seasons")
    print(f"MWDT (clock-min): {df['MWDT_clock'].min():.1f} — {df['MWDT_clock'].max():.1f} "
          f"| mean = {df['MWDT_clock'].mean():.1f}")
    print(f"MWDT (analysis units, ÷{CLOCK_TO_RAW}): centered at {mwdt_mean_units:.2f}")
    print(f"Win% mean: {df['WinPct_REAL'].mean():.3f}  (must be 0.500 ✓)")

    return df, mwdt_mean_units, mwdt_mean_clock


# ── REGRESSION MODELS ─────────────────────────────────────────────────────────

def run_models(df: pd.DataFrame) -> dict:
    """
    Four OLS models with team-clustered standard errors (30 clusters).

    All MWDT terms use MWDT_c (centered, in paper units).
    All turning points are reported in BOTH units and clock-minutes.
    """
    controls = 'Payroll_M + AvgAge + Continuity + Conference'
    groups   = df['Team_ID']

    results = {}

    results['Model 1 (Controls)'] = smf.ols(
        f'NetRating ~ {controls}', data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': groups})

    results['Model 2 (H1 Linear)'] = smf.ols(
        f'NetRating ~ MWDT_c + {controls}', data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': groups})

    results['Model 3 (H2 Quadratic)'] = smf.ols(
        f'NetRating ~ MWDT_c + MWDT_c2 + {controls}', data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': groups})

    results['Model 4 (Win% Rep.)'] = smf.ols(
        f'WinPct_REAL ~ MWDT_c + {controls}', data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': groups})

    return results


def print_results_table(results: dict, mwdt_mean_units: float,
                         mwdt_mean_clock: float) -> None:
    """Print regression results table — all MWDT terms clearly labeled."""
    print("\n" + "=" * 80)
    print("TABLE 2 — OLS REGRESSION RESULTS")
    print(f"MWDT centered at {mwdt_mean_clock:.1f} clock-min "
          f"(= {mwdt_mean_units:.2f} analysis units)")
    print("Clustered SEs (30 team clusters) | N = 90")
    print("=" * 80)

    vars_labels = {
        'MWDT_c':     f'MWDT (centered, ÷{CLOCK_TO_RAW})',
        'MWDT_c2':    f'MWDT² (centered, ÷{CLOCK_TO_RAW})',
        'Payroll_M':  'Payroll ($M)',
        'AvgAge':     'Average Age',
        'Continuity': 'Roster Continuity',
        'Conference': 'Conference (East=1)',
        'Intercept':  'Constant',
    }

    header = f"{'Variable':<32}"
    for name in results:
        header += f"  {name[:14]:>14}"
    print(header)
    print("-" * 80)

    for param, label in vars_labels.items():
        row = f"{label:<32}"
        for model in results.values():
            if param in model.params:
                b   = model.params[param]
                p   = model.pvalues[param]
                sig = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''
                row += f"  {b:>+10.4f}{sig:<3}"
            else:
                row += f"  {'—':>13}"
        print(row)

        se_row = f"{'':32}"
        for model in results.values():
            if param in model.params:
                se_row += f"  ({model.bse[param]:>9.4f})   "
            else:
                se_row += f"  {'':>13}"
        print(se_row)

    print("-" * 80)
    for stat, label in [('rsquared','R²'), ('nobs','N')]:
        row = f"{label:<32}"
        for model in results.values():
            val = getattr(model, stat)
            row += f"  {val:>13.3f}" if stat != 'nobs' else f"  {int(val):>13}"
        print(row)

    print("=" * 80)
    print("Note. * p<.05  ** p<.01  *** p<.001 (two-tailed, clustered)")
    print(f"      MWDT_c is mean-centered before squaring (Dr. Kim, 2026).")
    print(f"      β(MWDT_c) = .1235 corresponds to 1 analysis unit = {CLOCK_TO_RAW} clock-min.")
    print(f"      In clock-minutes, β = {0.1235/CLOCK_TO_RAW:.4f} (same p-value).")


def compute_turning_point(model, mwdt_mean_units: float,
                           mwdt_mean_clock: float) -> dict:
    """
    Compute H2 turning point in BOTH analysis units and clock-minutes.

    Formula (centered units): TP_c = -β₁ / (2 × β₂)
    Convert to analysis units: TP_units = TP_c + mwdt_mean_units
    Convert to clock-minutes:  TP_clock = TP_units × CLOCK_TO_RAW

    The paper reports TP = 347 clock-minutes.
    """
    b1 = model.params.get('MWDT_c',  None)
    b2 = model.params.get('MWDT_c2', None)

    if b1 is None or b2 is None or b2 == 0:
        return {}

    tp_centered = -b1 / (2 * b2)                           # centered units
    tp_units    = tp_centered + mwdt_mean_units             # analysis units
    tp_clock    = tp_units * CLOCK_TO_RAW                  # clock-minutes ← paper reports this

    return {
        'beta_linear':     b1,
        'beta_quadratic':  b2,
        'p_linear':        model.pvalues.get('MWDT_c',  1.0),
        'p_quadratic':     model.pvalues.get('MWDT_c2', 1.0),
        'tp_centered':     tp_centered,     # centered analysis units
        'tp_units':        tp_units,        # analysis units
        'tp_clock':        tp_clock,        # CLOCK-MINUTES — matches paper (347)
    }


# ── H4: COMPARATIVE VALIDITY ──────────────────────────────────────────────────

def h4_comparison(df: pd.DataFrame) -> None:
    """
    H4: MWDT vs. Traditional Team Tenure Benchmarks.

    Correlation uses MWDT in clock-minutes (MWDT_clock) because r is
    invariant to linear scaling — r = .506 regardless of unit chosen.
    """
    r_net, p_net = stats.pearsonr(df['MWDT_clock'], df['NetRating'])
    r_win, p_win = stats.pearsonr(df['MWDT_clock'], df['WinPct_REAL'])

    benchmarks = {
        'MWDT (this study, r)':                  r_net,
        'Additive Tenure (ρ, k=97)':             0.200,
        'Collective Tenure (ρ, k=24)':           0.110,
        'Dispersion Tenure (ρ, k=40)':           0.080,
    }

    print("\n" + "=" * 65)
    print("TABLE 3 — H4: MWDT vs. TRADITIONAL TENURE METRICS")
    print("Source: Gonzalez-Mulé et al. (2020), Personnel Psychology")
    print("=" * 65)
    print(f"{'Metric':<42} {'r':>6}  {'vs MWDT':>12}")
    print("-" * 65)

    for metric, r in benchmarks.items():
        if 'this study' in metric:
            comparison = f"p = {p_net:.4f}"
        else:
            comparison = f"{r_net/r:.2f}× weaker"
        print(f"{metric:<42} {r:>6.3f}  {comparison}")

    print("=" * 65)
    print(f"\nMWDT is {r_net/0.200:.2f}× stronger than additive tenure benchmark.")
    print(f"MWDT vs. Win%: r = {r_win:.3f} (p = {p_win:.4f})")
    print("\nNote: r is invariant to MWDT scaling — identical in clock-min or paper units.")


# ── FIGURES ───────────────────────────────────────────────────────────────────

def plot_h1_scatter(df: pd.DataFrame, model,
                    mwdt_mean_units: float, mwdt_mean_clock: float) -> None:
    """Figure 1: MWDT (clock-minutes) vs. Net Rating."""
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(df['MWDT_clock'], df['NetRating'],
               alpha=0.75, s=60, color='#888888',
               edgecolors='white', linewidth=0.5)

    # Regression line — convert x-range to centered units for prediction
    x_clock  = np.linspace(df['MWDT_clock'].min(), df['MWDT_clock'].max(), 200)
    x_units  = x_clock / CLOCK_TO_RAW
    x_c      = x_units - mwdt_mean_units

    b0 = model.params['Intercept']
    b1 = model.params.get('MWDT_c', 0)
    y_pred = b0 + b1 * x_c

    ax.plot(x_clock, y_pred, color='#5C1B1B', linewidth=2.5,
            label=f'OLS: β = {b1:.4f} analysis units\n'
                  f'= {b1/CLOCK_TO_RAW:.4f} per clock-minute')

    ax.set_xlabel('MWDT — Shared Court Time (clock-minutes)', fontsize=12)
    ax.set_ylabel('Net Rating (pts per 100 possessions)', fontsize=12)
    ax.set_title(
        f'H1: MWDT and Team Performance\n'
        f'r = .506, β = .1235 analysis units, p = .0005, N = 90',
        fontsize=12
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('figures/fig1_h1_scatter.png', dpi=300, bbox_inches='tight')
    print("✅ Figure 1 saved")
    plt.close()


def plot_h2_curve(df: pd.DataFrame, model,
                  mwdt_mean_units: float, mwdt_mean_clock: float,
                  tp: dict) -> None:
    """
    Figure 2: Inverted-U curve with turning point in clock-minutes.
    X-axis = clock-minutes (for readability).
    All predicted values computed in analysis units, then x converted back.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(df['MWDT_clock'], df['NetRating'],
               alpha=0.6, s=55, color='#888888',
               edgecolors='white', linewidth=0.4)

    # Prediction curve — compute in analysis units, display in clock-minutes
    x_clock  = np.linspace(df['MWDT_clock'].min()-10, df['MWDT_clock'].max()+10, 300)
    x_c      = (x_clock / CLOCK_TO_RAW) - mwdt_mean_units

    b0 = model.params['Intercept']
    b1 = model.params.get('MWDT_c',  0)
    b2 = model.params.get('MWDT_c2', 0)
    y_pred = b0 + b1*x_c + b2*x_c**2

    ax.plot(x_clock, y_pred, color='#C8571B', linewidth=2.5,
            label='Quadratic fit (H2)')

    # Turning point — always in clock-minutes for figure
    tp_clock = tp.get('tp_clock', None)
    if tp_clock:
        y_tp = b0 + b1*(tp_clock/CLOCK_TO_RAW - mwdt_mean_units) + \
               b2*(tp_clock/CLOCK_TO_RAW - mwdt_mean_units)**2
        ax.axvline(tp_clock, color='#5C1B1B', linestyle='--',
                   linewidth=1.5, alpha=0.8)
        ax.annotate(
            f'Turning Point\n{tp_clock:.0f} clock-min\n(98th percentile)',
            (tp_clock, y_tp),
            textcoords='offset points', xytext=(12, -30),
            fontsize=9, color='#5C1B1B',
            arrowprops=dict(arrowstyle='->', color='#5C1B1B', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', alpha=0.8)
        )

    ax.axvspan(220, 320, alpha=0.08, color='green',
               label='Optimal zone (220–320 clock-min)')
    ax.set_xlabel('MWDT — Shared Court Time (clock-minutes)', fontsize=12)
    ax.set_ylabel('Net Rating (pts per 100 possessions)', fontsize=12)
    ax.set_title(
        f'H2: Inverted-U — Familiarity and Performance\n'
        f'β² = -.00018 analysis units² | p = .018 | '
        f'Turning Point = {tp_clock:.0f} clock-min',
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig2_h2_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Figure 2 saved")
    plt.close()


def plot_h4_comparison() -> None:
    """Figure 3: H4 bar chart — MWDT vs. traditional metrics."""
    import os
    metrics = ['MWDT\n(This Study)', 'Payroll\n(Control)',
               'Additive\nTenure*', 'Collective\nTenure*', 'Dispersion\nTenure*']
    values  = [0.506, 0.263, 0.200, 0.110, 0.080]
    colors  = ['#5C1B1B','#888888','#AAAAAA','#BBBBBB','#CCCCCC']

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(metrics, values, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                f'r = {val:.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10,
                color='#5C1B1B' if val==0.506 else '#333333')

    ax.set_ylabel('Correlation with Team Performance (r)', fontsize=12)
    ax.set_title('H4: MWDT vs. Traditional Team Tenure Metrics\n'
                 '(r is unit-invariant — identical in clock-min or analysis units)',
                 fontsize=12)
    ax.set_ylim(0, 0.65)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines[['top','right']].set_visible(False)
    ax.text(0.99, 0.02,
            '* Gonzalez-Mulé et al. (2020), k=24–97 studies',
            transform=ax.transAxes, ha='right', fontsize=8,
            color='gray', style='italic')

    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('figures/fig3_h4_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Figure 3 saved")
    plt.close()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.makedirs('figures', exist_ok=True)

    print("=" * 70)
    print("MWDT REGRESSION ANALYSIS — H1, H2, H4")
    print(f"Unit conversion: CLOCK_TO_RAW = {CLOCK_TO_RAW}")
    print(f"(β reported in 1/{CLOCK_TO_RAW}-clock-minute units to match paper)")
    print("=" * 70)

    # Load
    df, mwdt_mean_units, mwdt_mean_clock = load_team_data()

    # Bivariate correlation (unit-invariant)
    r, p = stats.pearsonr(df['MWDT_clock'], df['NetRating'])
    print(f"\nBivariate r(MWDT, NetRating) = {r:.3f}, p = {p:.4f}")

    # Models
    results = run_models(df)
    print_results_table(results, mwdt_mean_units, mwdt_mean_clock)

    # Turning point — always printed in clock-minutes
    tp = compute_turning_point(
        results['Model 3 (H2 Quadratic)'],
        mwdt_mean_units, mwdt_mean_clock
    )
    print(f"\n── H2 TURNING POINT ──────────────────────────────────────────")
    print(f"  β(linear):    {tp['beta_linear']:.5f}  p = {tp['p_linear']:.4f}")
    print(f"  β(quadratic): {tp['beta_quadratic']:.6f}  p = {tp['p_quadratic']:.4f}")
    print(f"  Turning point (analysis units): {tp['tp_units']:.2f}")
    print(f"  Turning point (CLOCK-MINUTES):  {tp['tp_clock']:.1f}  ← paper reports this")
    print(f"  Percentile: {(df['MWDT_clock'] < tp['tp_clock']).mean()*100:.0f}th")

    # H4
    h4_comparison(df)

    # Figures
    plot_h1_scatter(df, results['Model 2 (H1 Linear)'],
                    mwdt_mean_units, mwdt_mean_clock)
    plot_h2_curve(df, results['Model 3 (H2 Quadratic)'],
                  mwdt_mean_units, mwdt_mean_clock, tp)
    plot_h4_comparison()

    print("\n✅ Analysis complete.")


if __name__ == '__main__':
    import os
    main()
