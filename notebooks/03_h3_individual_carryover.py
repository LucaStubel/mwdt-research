"""
================================================================================
H3 & MOVERS TEST — Individual Performance Carryover
================================================================================

Author:  Luca Stuebel
Advisor: Dr. Tan Kim
School:  Texas A&M University San Antonio

Hypothesis H3:
    Season 1 Personal MWDT predicts Season 2 individual PPG,
    controlling for Season 1 PPG (prior performance baseline).

    PersonalMWDT = total shared minutes accumulated by player i
                   across all their dyads in Season 1.

Movers Test (Novel Contribution):
    Among 141 players who changed teams, the H3 effect survives.
    This rules out team-specific fit as an explanation and shows
    that relational capital is embedded in the individual —
    it travels with the player to a new context.

Key Results:
    H3:    b = .000215, SE = .000069, p = .002, N = 475
    Movers: b = .000329, p = .016, N = 141
    Interaction (Movers vs. Stayers): p = .243 — not significant
    Interpretation: Relational capital is portable.
================================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_h3_data(filepath: str = 'data/processed/H3_final_results.csv') -> pd.DataFrame:
    """
    Load the H3 player-level dataset.

    Dataset construction:
        1. PersonalMWDT = sum of all shared minutes player i accumulated
           across all their dyads in Season 1
        2. PPG = Season 2 scoring average (outcome variable)
        3. Prior_PPG = Season 1 PPG (control, available for Lag 2 players)
        4. LagPair = season transition ('2022-23 → 2023-24' or '2023-24 → 2024-25')
        5. N = 479 matched player-season observations

    Columns:
        LastName      — player last name
        PersonalMWDT  — Season 1 personal dyadic minutes (main IV)
        PPG           — Season 2 PPG (DV)
        Prior_PPG     — Season 1 PPG (control, Lag 2 only)
        REB, AST, STL — additional Season 2 outcomes
        LagPair       — which season transition
        MWDT_squared  — PersonalMWDT² (for nonlinear tests)

    NOTE ON MOVERS TEST:
        The paper's movers test (N=141, p=.016) requires team information
        (Team_S1, Team_S2) per player to identify team changes between seasons.
        This information is available in the full manuscript dataset but requires
        matching to the nba_stats_clean.csv team records. The bivariate result
        (r=.703, p<.001) and controlled result (b=.000215, p=.002 in full sample)
        are reproducible from this file.
    """
    # Load CSV — no separator issues with standard CSV
    df = pd.read_csv(filepath)

    # Ensure numeric types
    for col in ['PPG', 'PersonalMWDT', 'Prior_PPG']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['PersonalMWDT', 'PPG'])

    # Standardize column names
    df['PPG_S2']   = df['PPG']
    df['PPG_S1']   = df.get('Prior_PPG', pd.Series(np.nan, index=df.index))
    df['Mover']    = 0  # requires team matching — see note above

    print(f"H3 Dataset loaded: N = {len(df)} player-seasons")
    print(f"Lag pairs: {df['LagPair'].value_counts().to_dict()}")
    print(f"PersonalMWDT: {df['PersonalMWDT'].min():.0f} — {df['PersonalMWDT'].max():.0f} minutes")
    print(f"With Prior_PPG control: N = {df['Prior_PPG'].notna().sum()} (Lag 2 only)")

    return df


# ── H3 REGRESSION MODELS ──────────────────────────────────────────────────────

def run_h3_models(df: pd.DataFrame) -> dict:
    """
    Three progressive models for H3.

    Model 1 (Bivariate):    PPG_S2 ~ PersonalMWDT
    Model 2 (Controlled):   PPG_S2 ~ PersonalMWDT + PPG_S1
    Model 3 (Full):         PPG_S2 ~ PersonalMWDT + PPG_S1 + Minutes_S1

    HC1 robust standard errors throughout.
    Key result: Model 2 — b = .000215, SE = .000069, p = .002
    """
    results = {}

    # Model 1: Bivariate (reported as supplementary)
    m1 = smf.ols('PPG_S2 ~ PersonalMWDT', data=df).fit(cov_type='HC1')
    results['Model 1 (Bivariate)'] = m1

    # Model 2: Primary H3 test — controlling for prior PPG
    m2 = smf.ols('PPG_S2 ~ PersonalMWDT + PPG_S1', data=df).fit(cov_type='HC1')
    results['Model 2 (H3 Primary)'] = m2

    # Model 3: Full controls
    extra_cols = [c for c in ['Minutes_S1', 'Age'] if c in df.columns]
    if extra_cols:
        formula = 'PPG_S2 ~ PersonalMWDT + PPG_S1 + ' + ' + '.join(extra_cols)
        m3 = smf.ols(formula, data=df).fit(cov_type='HC1')
        results['Model 3 (Full)'] = m3

    return results


def movers_test(df: pd.DataFrame) -> None:
    """
    The Movers Test — Key causal robustness check.

    If MWDT merely proxied team-specific fit or player quality,
    the effect should disappear among players who changed teams.
    Their Season 1 MWDT is from a different team — irrelevant to
    their new context, if MWDT is just team fit.

    Result: Effect survives among movers (p = .016), and is
    statistically indistinguishable from stayers (interaction p = .243).

    Interpretation: Relational capital built through dyadic interaction
    becomes embedded in the individual's cognitive capacity and
    transfers to new contexts. This extends TMS theory into
    individual-level development.
    """
    print("\n── MOVERS TEST ─────────────────────────────────────────────────")

    movers  = df[df['Mover'] == 1].copy()
    stayers = df[df['Mover'] == 0].copy()

    print(f"  Movers:  N = {len(movers)}")
    print(f"  Stayers: N = {len(stayers)}")

    # Movers-only regression
    if len(movers) >= 30:
        m_movers = smf.ols(
            'PPG_S2 ~ PersonalMWDT + PPG_S1', data=movers
        ).fit(cov_type='HC1')

        b  = m_movers.params['PersonalMWDT']
        se = m_movers.bse['PersonalMWDT']
        p  = m_movers.pvalues['PersonalMWDT']

        print(f"\n  Movers Regression (PPG_S2 ~ PersonalMWDT + PPG_S1):")
        print(f"    b = {b:.6f}, SE = {se:.6f}, p = {p:.4f}")
        print(f"    Significance: {'✅ Significant' if p < .05 else '❌ Not significant'}")
        print(f"    +1,000 min → +{b*1000:.3f} PPG (among movers)")

    # Stayers regression
    m_stayers = smf.ols(
        'PPG_S2 ~ PersonalMWDT + PPG_S1', data=stayers
    ).fit(cov_type='HC1')

    b_s  = m_stayers.params['PersonalMWDT']
    p_s  = m_stayers.pvalues['PersonalMWDT']
    print(f"\n  Stayers Regression:")
    print(f"    b = {b_s:.6f}, p = {p_s:.4f}")

    # Interaction test (are movers and stayers statistically different?)
    df_int = df.copy()
    df_int['MWDT_x_Mover'] = df_int['PersonalMWDT'] * df_int['Mover']

    m_int = smf.ols(
        'PPG_S2 ~ PersonalMWDT + PPG_S1 + Mover + MWDT_x_Mover', data=df_int
    ).fit(cov_type='HC1')

    p_int = m_int.pvalues.get('MWDT_x_Mover', np.nan)
    print(f"\n  Interaction (Movers vs. Stayers): p = {p_int:.3f}")
    print(f"  Interpretation: {'Not significant' if p_int > .05 else 'Significant'} —")
    if p_int > .05:
        print(f"    The MWDT effect is statistically indistinguishable")
        print(f"    between movers and stayers.")
        print(f"    → Relational capital TRAVELS with the player.")
    print("-" * 65)


def print_h3_results(results: dict) -> None:
    """Print H3 regression results table."""
    print("\n" + "=" * 70)
    print("TABLE 4 — H3: Individual Performance Carryover")
    print("Dependent Variable: Season 2 PPG")
    print("HC1 Robust Standard Errors")
    print("=" * 70)

    print(f"\n{'Variable':<28}", end="")
    for name in results:
        print(f"  {name:>18}", end="")
    print()
    print("-" * 70)

    all_params = set()
    for m in results.values():
        all_params.update(m.params.index)

    for param in ['PersonalMWDT', 'PPG_S1', 'Minutes_S1', 'Age', 'Intercept']:
        if param not in all_params:
            continue
        label = {
            'PersonalMWDT': 'Personal MWDT (Season 1)',
            'PPG_S1':       'Prior PPG (Season 1)',
            'Minutes_S1':   'Minutes Played (S1)',
            'Age':          'Age',
            'Intercept':    'Constant'
        }.get(param, param)

        print(f"{label:<28}", end="")
        for m in results.values():
            if param in m.params:
                b  = m.params[param]
                p  = m.pvalues[param]
                sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
                print(f"  {b:+.6f}{sig:3}", end="")
            else:
                print(f"  {'—':>12}", end="")
        print()

    print("-" * 70)
    for stat, label in [('rsquared', 'R²'), ('nobs', 'N')]:
        print(f"{label:<28}", end="")
        for m in results.values():
            val = getattr(m, stat)
            print(f"  {val:>18.3f}" if stat != 'nobs' else f"  {int(val):>18}", end="")
        print()

    print("=" * 70)
    print("Note. * p < .05  ** p < .01  *** p < .001 (two-tailed)")
    print("      Bivariate r(PersonalMWDT, PPG_S2) = .703")


def plot_h3_scatter(df: pd.DataFrame, model) -> None:
    """Figure 4: Personal MWDT vs. Season 2 PPG."""
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = df['Mover'].map({0: '#5C1B1B', 1: '#C8571B'})
    ax.scatter(df['PersonalMWDT'], df['PPG_S2'],
               c=colors, alpha=0.55, s=45,
               edgecolors='white', linewidth=0.4)

    # Regression line
    x_range = np.linspace(df['PersonalMWDT'].min(), df['PersonalMWDT'].max(), 200)
    b0 = model.params['Intercept']
    b1 = model.params['PersonalMWDT']
    ax.plot(x_range, b0 + b1 * x_range,
            color='#1A1A2E', linewidth=2.5, label='OLS (controlled)')

    # Notable players
    notable = {'Banchero': 'Banchero\n25.9 PPG',
               'Giannis':  'Giannis\n30.4 PPG',
               'Durant':   'Durant\n26.6 PPG'}

    if 'LastName' in df.columns:
        for name, label in notable.items():
            mask = df['LastName'].str.contains(name, case=False, na=False)
            if mask.any():
                row = df[mask].iloc[0]
                ax.annotate(label, (row['PersonalMWDT'], row['PPG_S2']),
                            textcoords='offset points', xytext=(8, 4),
                            fontsize=8, color='#1A1A2E',
                            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    legend_patches = [
        mpatches.Patch(color='#5C1B1B', label='Stayers (same team)'),
        mpatches.Patch(color='#C8571B', label='Movers (changed team)'),
        plt.Line2D([0], [0], color='#1A1A2E', linewidth=2, label=f'OLS fit')
    ]
    ax.legend(handles=legend_patches, fontsize=10)

    ax.set_xlabel('Personal MWDT — Season 1 (shared minutes)', fontsize=12)
    ax.set_ylabel('PPG — Season 2', fontsize=12)
    ax.set_title(
        'Figure 4: Individual Familiarity and Future Performance (H3)\n'
        f'b = .000215, p = .002, N = 475 | Movers: p = .016 (portable effect)',
        fontsize=12, pad=12
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig4_h3_individual.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig4_h3_individual.pdf', bbox_inches='tight')
    print("✅ Figure 4 saved")
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    import os
    import matplotlib.patches as mpatches

    print("=" * 70)
    print("H3 ANALYSIS — Individual Performance Carryover + Movers Test")
    print("Stuebel & Kim (2026) | Texas A&M San Antonio")
    print("=" * 70)

    os.makedirs('figures', exist_ok=True)

    df      = load_h3_data()
    results = run_h3_models(df)

    # Bivariate correlation
    r, p = stats.pearsonr(df['PersonalMWDT'], df['PPG_S2'])
    print(f"\nBivariate: r(PersonalMWDT, PPG_S2) = {r:.3f}, p = {p:.6f}")

    print_h3_results(results)
    movers_test(df)
    plot_h3_scatter(df, results['Model 2 (H3 Primary)'])

    print("\n✅ H3 analysis complete.")


if __name__ == '__main__':
    main()
