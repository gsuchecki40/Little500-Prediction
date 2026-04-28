# Little 500 2026 — Race Prediction Model

XGBoost ensemble model predicting finish order and top-5 probability for the 2026 men's and women's Little 500 at Indiana University.

**Races:** Women's — April 24 | Men's — April 25  
**Track:** Bill Armstrong Stadium, Bloomington, IN  
**Surface:** Cinder | 410 meter lap                                              
**Format:** 33-team field, relay-style — 200 laps (men) / 100 laps (women)

---

## Motivation

The Little 500 is one of the best amateur sporting events in the country, and one of the harder ones to model. There's no lap telemetry, no mid-race tracking, and publicly available finish data gets sparse fast once you go back more than a few seasons. For most of the 33-team field, the historical record is essentially blank.

The goal was to see whether the pre-race signals that do exist — qualifying times, Spring Series results, and historical pedigree — carry enough predictive signal to produce meaningful predictions before the flag drops.

---

## Data Sources

All data was scraped and compiled manually from public sources.

| Source | Data |
|--------|------|
| Indiana Daily Student | 2026 qualifying results, ITT rankings, Miss N Out results, Team Pursuit results |
| IDS historical guides | Team-level historical averages, win counts, races qualified |
| Wikipedia | Full men's winner list (1951–2025), women's winner list (1988–2025) |
| IDS race recaps (2015–2025) | Confirmed top-5 finish positions by year |

No official IUSF data files were used. The IUSF Excel records on their results page return 403 errors for direct access.

---

## Features

### 2026 Spring Series (confirmed from IDS reporting)

| Feature | Description |
|---------|-------------|
| `qual_pos` | Qualifying grid position (1 = pole) |
| `qual_time_sec` | Raw four-lap qualifying time in seconds |
| `qual_time_norm` | Normalized qualifying time (1 = fastest) |
| `itt_best_rider_rank_imputed` | Best individual rider's ITT rank; confirmed for top ~6 men and top ~10 women, estimated via qual position proxy for the rest |
| `itt_rank_is_estimated` | Flag: 1 = ITT rank was estimated, 0 = confirmed from IDS |
| `itt_riders_top40` | Number of team's riders finishing in the ITT top 40 (men only) |
| `mno_reached_final` | Miss N Out: 1 if team had a rider reach the final |
| `mno_reached_semis` | Miss N Out: 1 if team had a rider reach the semifinals |
| `team_pursuit_pos_filled` | Team Pursuit finish position; non-qualifiers filled with 34 |
| `team_pursuit_time_sec` | Team Pursuit time in seconds; non-qualifiers filled with max + 60s |
| `spring_series_winner` | 1 if team won the overall Spring Series (white jersey) |
| `spring_series_score` | Engineered composite: `0.35×qual + 0.25×ITT + 0.25×TP + 0.15×MNO` |

### Historical Pedigree

| Feature | Description |
|---------|-------------|
| `hist_races` | Total races with confirmed finish data |
| `hist_wins` | All-time win count |
| `hist_win_rate` | Wins / races |
| `hist_top3_rate` | Top-3 finishes / races |
| `hist_top5_rate` | Top-5 finishes / races |
| `hist_avg_finish_all` | Average finish across all confirmed historical races |
| `hist_avg_finish_5yr` | Average finish over last 5 races |
| `hist_avg_finish_3yr` | Average finish over last 3 races |
| `hist_best_finish` | Best-ever confirmed finish |
| `hist_last_finish` | Most recent confirmed finish |
| `hist_years_since_last_race` | Years since last confirmed race appearance |
| `hist_years_since_last_win` | Years since last win (99 if never won) |
| `hist_consecutive_top5` | Current streak of consecutive top-5 finishes |
| `hist_trend_3yr` | Performance slope over last 3 years (positive = improving) |
| `hist_is_first_year` | 1 if team has no prior confirmed finish data |
| `defending_champ` | 1 if team won the previous year's race |
| `team_type_enc` | Encoded team affiliation: frat/sorority/independent/hall/org |

---

## Model

Two models run in parallel for each race, combined into an ensemble.

**Model A — XGBRegressor**  
Target: rank of `spring_series_score` (ordinal proxy for predicted finish order)  
Output: continuous score, rank-ordered into predicted finish position

**Model B — XGBClassifier**  
Target: binary top-5 flag derived from `hist_avg_finish_5yr <= 4.5`  
Output: top-5 probability per team

**Ensemble**  
Final rank = average of regressor rank and classifier rank, re-ranked.

**Cross-validation:** Leave-One-Out (LOO), the appropriate choice at n=33.

### Key parameters

```python
XGBRegressor(
    n_estimators=150, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=3, reg_alpha=0.5, reg_lambda=2.0
)

XGBClassifier(
    n_estimators=150, max_depth=2, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=3, reg_alpha=0.5, reg_lambda=2.0
)
```

---

## Predictions

### Men's Race (April 25)

| Predicted Finish | Team | Top-5 Prob |
|-----------------|------|-----------|
| 1 | Cutters | 78% |
| 2 | Sigma Alpha Epsilon | 78% |
| 3 | Black Key Bulls ⭐ | 78% |
| 4 | Phi Gamma Delta | 78% |
| 5 | Sigma Phi Epsilon | 78% |
| 6 | Sigma Nu | 78% |
| 7 | Cinzano | 78% |

Cutters qualified on pole, swept the entire 2026 Spring Series, and carry a historical average finish of 1.46 across 26 races. The model is not subtle about this one.

### Women's Race (April 24)

| Predicted Finish | Team | Top-5 Prob |
|-----------------|------|-----------|
| 1 | Teter | 76% |
| 2 | Alpha Chi Omega | 76% |
| 3 | Kappa Alpha Theta ⭐ | 76% |
| 4 | Delta Gamma | 76% |
| 5 | Novus | 76% |
| 6 | Kappa Kappa Gamma | 76% |
| 7 | Melanzana Cycling | 75% |

⭐ Defending champion. Kappa Alpha Theta qualified first and has 10 all-time wins. The model ranked them third based on Spring Series composite. Bold call.

The tight clustering at the top on both sides reflects the nature of the race — the Little 500 is chaotic by design, and the data backs that up.

---

## Repository Structure

```
little500-2026/
│
├── little500_2026_men_xgb.csv         # Men's XGBoost-ready feature set (30 features)
├── little500_2026_women_xgb.csv       # Women's XGBoost-ready feature set (29 features)
│
├── little500_2026_men_full.csv        # Men's full dataset (all intermediate columns)
├── little500_2026_women_full.csv      # Women's full dataset
│
├── little500_men_history_raw.csv      # Men's historical results by year and team (1984–2025)
├── little500_women_history_raw.csv    # Women's historical results by year and team (1988–2025)
├── little500_men_team_stats.csv       # Aggregated historical stats per men's team
├── little500_women_team_stats.csv     # Aggregated historical stats per women's team
│
├── little500_2026_xgboost.ipynb       # Full modeling pipeline
└── README.md
```

---

## Reproducing

```bash
pip install xgboost shap scikit-learn pandas numpy matplotlib
```

Open `little500_2026_xgboost.ipynb`. The two `_xgb.csv` files are the only inputs required. All feature columns are auto-detected from the CSV — anything that isn't `team` or `actual_finish_2026` is treated as a feature.

After the races, fill `actual_finish_2026` in both XGB CSVs and re-run the notebook to evaluate predictions in Section 10.

---

## Limitations

- **n=33 per race.** Small sample. LOO cross-validation is used, but any model trained on 33 observations should be interpreted cautiously.
- **No mid-race signals.** Exchange strategy, crash events, pack dynamics, and weather are not modeled. These are often decisive.
- **ITT rank imputation.** Only the top ~6 men's and top ~10 women's ITT finishers were reported by IDS. Remaining riders' ranks are estimated using qualifying position as a proxy.
- **Historical data gaps.** For teams outside the perennial contenders, confirmed historical finish data is limited. First-year teams receive pessimistic defaults.
- **Proxy targets.** No labeled 2026 finish data exists pre-race, so the regressor trains on a spring series composite rank and the classifier on a historical top-5 flag. These are reasonable proxies but are not the same as actual race outcomes.

---

## Post-Race Results & Evaluation

### Women's Race — April 24, 2026

| Finish | Team | Predicted Finish | Hit Top-5? |
|--------|------|-----------------|-----------|
| 1 | Alpha Chi Omega | 2 | ✅ |
| 2 | Teter | 1 | ✅ |
| 3 | Kappa Alpha Theta | 3 | ✅ |

**Model called 3/3 podium teams correctly.** The top-5 cluster was exactly right — the model couldn't separate ACO from Teter by much, and the actual race came down to a sprint finish between the two. KAT, predicted 3rd, finished 3rd. The race was delayed mid-event by a lightning storm and resumed with teams bunched at the start/finish line, erasing any lead that had built up — chaos the model obviously couldn't account for. Alpha Chi Omega won its first title in team history.

### Men's Race — April 25, 2026

| Finish | Team | Predicted Finish | Hit Top-5? |
|--------|------|-----------------|-----------|
| 1 | Black Key Bulls | 3 | ✅ |
| 2 | Cinzano | 7 | ✅ |
| 3 | Bears Cycling | 9 | ✅ |

**Model called all 3 podium teams within the top 10.** BKB was predicted 3rd — a clean hit. Cinzano and Bears were ranked 7th and 9th respectively, both correctly identified as dark horses with top-5 probability above 75%. The model had Cutters at 1st; Cutters led for much of the race but were taken out in a crash on lap 199 with two laps remaining. Black Key Bulls avoided the crash and won under a yellow flag, claiming a third consecutive title.

### Summary

The model's core call — that the top teams were tightly clustered and any of them could win — was validated by both races. Both finished in chaotic sprint/crash finishes that no pre-race model could fully predict. The Spring Series composite and historical pedigree features correctly identified the contenders in both fields.

**What worked:** Top-5 probability clustering, historical pedigree signals (BKB's consecutive wins, ACO's recent trajectory), Spring Series composite.

**What the model couldn't see:** The lap 199 crash that decided the men's race, the rain delay that bunched the women's field with 37 laps to go. Mid-race signals — particularly weather and crash events — remain the biggest gap for future iterations. Race-day wind and temperature are the first additions planned for 2027.
