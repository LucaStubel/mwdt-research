# Chemistry or Rotation Size? Auditing Pair-Level Familiarity Metrics in the NBA

## Introduction

Lineup data make it easy to turn shared minutes into "chemistry" metrics, and front offices value continuity. The evidence is weaker than the intuition: across 169 studies, team tenure predicts performance only modestly (ρ = .08–.20; Gonzalez-Mulé et al., 2020). One explanation is that familiarity lives between specific people (transactive memory; Wegner, 1987), so it should be measured between pairs (e.g., Huckman et al., 2009). This study asks what such a metric measures when the game clock fixes how many minutes teammates can share.

## Methods

Using pbpstats regular-season five-man lineup exports for all 30 teams from 2022-23 to 2024-25 (90 team-seasons, 14,796 pair-seasons), I compute a pair-averaged familiarity metric, Minutes Weighted Dyadic Tenure (MWDT): total shared minutes across all player pairs, divided by the number of pairs that shared the floor. Because exports cap at 500 lineups, all measures use lineups of three or more minutes, complete for every team. Net Rating is regressed on MWDT with controls for payroll, age, roster continuity and conference, clustering standard errors by franchise. Robustness checks vary payroll source, lineup threshold and controls. All data and code: github.com/LucaStubel/mwdt-research.

## Results

Every minute puts exactly ten pairs on the floor, so total pair-minutes barely vary across teams (CV = 3.4%), and MWDT collapses to a constant divided by the number of pairs a team used (r = .994; Figure 1a). MWDT is therefore almost mechanically determined by rotation breadth.

This mechanical relationship explains the initially large association with Net Rating. MWDT correlates with Net Rating at r = .52, and one standard deviation (58 minutes) is associated with +2.0 points per 100 possessions after controls (p < .001; Table 1). Once rotation size is controlled, MWDT adds nothing (p = .30). Returns flatten at the top but never turn down (Lind–Mehlum p = .24).

Pair-minute concentration remains associated with Net Rating after adjustment, although the evidence is sensitive to specification. Holding rotation size fixed, teams that ran more of their minutes through their ten most-used pairs performed better (+1 SD ≈ +1.2 Net Rating, p = .02; wild-cluster bootstrap p = .05; Figure 1b), and the estimate weakens further when controlling for whether a team's export hit the 500-lineup cap.

## Conclusion

Pair-averaged familiarity, a natural chemistry metric for lineup data, mostly counts how many players a team used. Because familiarity and performance are measured in the same season, none of these associations is causal: healthy, strong teams can afford tight rotations. Any chemistry or continuity metric built from shared minutes should be benchmarked against rotation size before it informs roster decisions. Future studies should test concentration measures constructed exclusively from prior shared experience.

**Table 1.** OLS, DV = Net Rating, N = 90; controls: payroll, age, continuity, conference; stars use franchise-clustered p-values.

**Figure 1.** (a) MWDT vs. pairs used, with constant ÷ pairs curve. (b) Net Rating vs. top-10 pair share, net of rotation size and controls.
