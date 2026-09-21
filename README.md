# MWDT: Minutes Weighted Dyadic Tenure
### *Why 169 Studies Got Team Familiarity Wrong*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SSAC27](https://img.shields.io/badge/Submitted-MIT%20Sloan%202027-red.svg)](https://sloansportsconference.com)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)]()

**Author:** Luca Stuebel — Texas A&M University San Antonio  
**Advisor:** Dr. Tan Kim — Department of Management  
**Contact:** lucastuebel@gmail.com

---

## The Core Argument

For decades, organizational research has tried to measure team familiarity by asking: *how long has the average player been on this roster?* Across **169 studies and over 11,000 teams**, the answer has produced correlations of .08 to .20 with team performance. Inconsistent. Weak. Contested.

Nobody questioned the ruler.

Every study measured familiarity at the wrong level of analysis. **Familiarity is not a property of a roster. It is a property of a relationship** — between two specific people, built through hours of working together under pressure. Daniel Wegner called this *transactive memory* in 1987. For 38 years, researchers cited him and measured something else.

This repository introduces **Minutes Weighted Dyadic Tenure (MWDT)**: the first team familiarity metric operationalized at the dyadic level — where the mechanism actually lives.

---

## The Metric

$$\text{MWDT} = \frac{\sum_{i<j} \text{minutes}(i,j)}{N_{\text{pairs}}}$$

Where:
- $\text{minutes}(i,j)$ = total shared court minutes between player pair $(i, j)$
- $N_{\text{pairs}}$ = number of unique player pairs on the roster

**Why pairwise minutes?** In basketball, the total volume of dyadic court time is near-constant across all teams (5 players × every minute = 10 active dyads per possession). What varies is *distribution* — whether minutes concentrate in a few deeply familiar pairs or disperse across many shallow ones.

**Why divide by $N_\text{pairs}$?** This captures network density, not volume. A tight 8-man rotation concentrates its minutes into few deep ties (high MWDT). An injury-disrupted team cycling 15 players dissipates the same minutes across many shallow connections (low MWDT).

The measure does not approximate Wegner's theory of transactive memory. **It is the theory, rendered in a form computable from lineup data.**

---

## Results

| Hypothesis | Prediction | Result | Key Statistic |
|---|---|---|---|
| **H1** | MWDT → Team Net Rating | ✅ **Supported** | r = .506, β = .124, p = .0005, R² = .386 |
| **H2** | Inverted-U (ossification) | ✅ **Supported** | β² = -.00018, p = .018, turning point = 347 min (98th pct) |
| **H3** | Season 1 MWDT → Season 2 PPG | ✅ **Supported** | b = .000215, p = .002, N = 475 |
| **H4** | MWDT > traditional metrics | ✅ **Supported** | 2.5× stronger than best meta-analytic benchmark |
| **Movers Test** | Effect survives team change | ✅ **Supported** | b = .000329, p = .016, N = 141 |

### H4: MWDT vs. Traditional Tenure Metrics

| Metric | r with Performance | Source |
|---|---|---|
| **MWDT (this study)** | **r = .506** | N = 90 team-seasons |
| Additive Tenure | ρ = .200 | Gonzalez-Mulé et al. (2020), k = 97 studies |
| Collective Tenure | ρ = .110 | Gonzalez-Mulé et al. (2020), k = 24 studies |
| Dispersion Tenure | ρ = .080 | Gonzalez-Mulé et al. (2020), k = 40 studies |

**MWDT is 2.53× stronger than the best traditional metric — and the traditional measure is not significant when entered alongside MWDT in the same regression (p = .218).**

### The Movers Test — Key Theoretical Contribution

If MWDT merely proxied team-specific fit, traded players should show no effect — their old MWDT is irrelevant to a new team. Instead, the effect **survives** among 141 players who changed teams (p = .016) and is **statistically indistinguishable** from stayers (interaction p = .243).

*Relational capital, once built, is embedded in the individual's cognitive capacity. It travels.*

---

## Data

### Sample
- **Team-level (H1, H2, H4):** N = 90 team-seasons — all 30 NBA teams × 3 seasons (2022–23, 2023–24, 2024–25). Full population, no sampling.
- **Player-level (H3):** N = 475 matched player observations across two consecutive season-pairs

### Data Sources
| Variable | Source |
|---|---|
| 5-man lineup data | pbpstats.com (minute-level tracking) |
| Net Rating, Win % | Official NBA standings (nba.com/stats) |
| Player statistics | Basketball-Reference.com |
| Payroll | Spotrac.com |
| Roster Continuity, Age | Basketball-Reference.com |

### Repository Structure

```
mwdt-research/
├── README.md
├── LICENSE
├── requirements.txt
│
├── notebooks/
│   ├── 00_data_collection.py        # NBA stats scraper (Basketball-Reference)
│   ├── 01_mwdt_calculation.py       # Core MWDT metric from lineup CSVs
│   ├── 02_regression_analysis.py    # H1, H2, H4 — OLS with clustered SEs
│   └── 03_h3_individual_carryover.py # H3 + Movers Test
│
├── data/
│   ├── raw/
│   │   └── lineups_YYYY-YY/         # 5-man lineup CSVs (30 teams per season)
│   └── processed/
│       ├── team_data_corrected.csv  # Team-level dataset (N = 90)
│       ├── H3_final_results.csv     # Player-level dataset (N = 475)
│       ├── all_dyads.csv            # Full dyadic dataset
│       └── mwdt_team_ranking.csv    # MWDT scores per team-season
│
├── results/
│   └── regression_tables.xlsx       # Full regression output
│
└── figures/
    ├── fig1_mwdt_vs_netrating.pdf   # H1 scatter
    ├── fig2_inverted_u.pdf          # H2 curve
    ├── fig3_h4_comparison.pdf       # H4 bar chart
    └── fig4_h3_individual.pdf       # H3 scatter
```

---

## Reproducing the Analysis

### Requirements
```bash
pip install pandas numpy scipy statsmodels matplotlib openpyxl
```

### Quick Start

```bash
# 1. Collect NBA player statistics (takes ~6 min due to rate limiting)
python notebooks/00_data_collection.py

# 2. Compute MWDT from lineup CSVs
#    (place lineup files in data/raw/lineups_YYYY-YY/)
python notebooks/01_mwdt_calculation.py

# 3. Run H1, H2, H4 regression analysis
python notebooks/02_regression_analysis.py

# 4. Run H3 individual carryover + movers test
python notebooks/03_h3_individual_carryover.py
```

### Lineup Data Format

Each team-season lineup CSV should have this structure:

```
Lineup,Minutes
LeBron James, Anthony Davis, Austin Reaves, D'Angelo Russell, Jarred Vanderbilt,12.3
LeBron James, Anthony Davis, Austin Reaves, D'Angelo Russell, Rui Hachimura,8.7
...
```

Source: pbpstats.com → Team Lineups → 5-Man → Export CSV

---

## Key Theoretical Background

### Why Transactive Memory is Irreducibly Dyadic

Wegner's (1987) original insight was about *couples*. When two people work together over time, each uses the other as an external memory — storing not information, but *the location of information*. Three properties make this mechanism irreducibly dyadic:

1. **Cognitive Addressing** — The mind can build a detailed "who knows what" index only for specific individuals, not for an abstraction called a roster
2. **Relationship-Specific Knowledge** — Reagans et al. (2005) showed in cardiac surgery teams that coordination improvements came from working with *this specific anesthesiologist*, not from general surgical experience
3. **Retrieval Under Pressure** — A point guard in a fast break passes to *this specific center* based on thousands of shared minutes. No player passes to a roster average.

### The Category Error

Kozlowski & Bell (2003) distinguish **compositional constructs** (legitimately averaged: team age = mean individual ages) from **compilational constructs** (destroyed by averaging: they live in the *pattern* of relationships). Transactive memory is compilational. Measuring it as a roster average is not a minor imprecision — it is a category error, and it mechanically attenuates the estimated relationship. This explains the meta-analytic record: ρ = .08 to .20 for a mechanism theory predicts should be strong.

---

## Limitations

1. **Cross-sectional design:** H1 is best interpreted as a robust cross-sectional association. The cross-lagged test (reverse causality: past performance → future MWDT) was underpowered with only two transition periods (p = .381) and cannot definitively establish causal direction.

2. **NBA-specific institutional constraints:** The salary cap and draft system keep 98% of teams below the ossification turning point (347 min). Organizations without these constraints may show earlier and more frequent ossification.

3. **Endogeneity in H3:** Star players both accumulate more MWDT and perform better. We address this by controlling for prior PPG and through the movers test, but full causal identification would require an instrumental variable approach.

4. **Roster complexity:** MWDT denominator grows with roster size; the Roster Continuity control addresses this but cannot fully separate familiarity concentration from roster stability.

---

## Citation

If you use this code or metric in your research, please cite:

```bibtex
@article{stuebel2026mwdt,
  title   = {Familiarity as a Dyadic Phenomenon: Introducing Minutes Weighted
             Dyadic Tenure and Its Relationship to NBA Team and Individual Performance},
  author  = {Stuebel, Luca and Kim, Tan},
  journal = {Under Review},
  year    = {2026},
  school  = {Texas A&M University San Antonio}
}
```

---

## References

- Gonzalez-Mulé, E., Cockburn, B. S., McCormick, B. W., & Zhao, P. (2020). Team tenure and team performance: A meta-analysis and process model. *Personnel Psychology, 73*, 151–198.
- Kozlowski, S. W. J., & Bell, B. S. (2003). Work groups and teams in organizations. *Handbook of Psychology*, 12, 333–375.
- Reagans, R., Argote, L., & Brooks, D. (2005). Individual experience and experience working together. *Management Science, 51*, 869–881.
- Wegner, D. M. (1987). Transactive memory: A contemporary analysis of the group mind. In *Theories of Group Behavior* (pp. 185–208). Springer.

---

*"The question was never whether familiarity matters. It was whether anyone was measuring it correctly."*
