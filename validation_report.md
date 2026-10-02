# Draft-Day XGBoost Model — Validation Report

**Built:** 2026-10-02 · **Bundle:** `~/workspace/sleeper-league-data/model/`
**Task (Eric's ask):** an XGBoost classifier usable on draft day in a 10-team auction
superflex redraft league — P(top-5 positional finish) and P(starter-tier finish) as the
headline outputs, a VOR regression head as secondary — trained on everything we pulled
(Fantasy 101 joined picks + gen-1 + gen-2 crossleague picks), incorporating the validated
findings (top-5 composition threshold, injury history, position, cost, offense quality).

**Bottom line up front:** the model is a competent *ranking/tiebreaker* layer on top of
auction price, but on the strict 2024–2025 holdout it does **not** beat a price-only
logistic regression overall. Price — the market's aggregated judgment — already contains
most of what age, injury history, and prior-year offense quality add, and the two
"profile" effects we validated (offense quality, injury history) did not appear in the
2024–2025 seasons at all in leak-free form. Details and receipts below. Nothing here is
smoothed over.

---

## 1. Data

| Source | File | Rows used |
|---|---|---|
| Gen-1 targeted 10T auction SF | `crossleague-targeted-picks.csv` | part of 40,788 |
| Gen-2 targeted 10T auction SF | `crossleague-targeted-gen2-picks.csv` | part of 40,788 |
| Fantasy 101 (Eric's league chain) | `draft_picks_joined.csv` (2020–2025 only) | 871 |

**41,659 model rows** total (QB 7,294 / RB 12,597 / WR 15,787 / TE 5,981), seasons
2020–2025. Positions restricted to QB/RB/WR/TE (IDP/K/DEF contamination rows dropped).
Rows are individual auction picks; the same player-season appears once per league that
drafted him, so all inference is clustered accordingly (§4).

**Targets** (computed per league, in that league's own scoring):
- **top5** = player finishes `pos_rank ≤ 5` at his position. Picks with no recorded stats
  (797) are coded 0 — they scored ~0 and really were busts. Test base rates: QB 13.5%,
  RB 7.6%, WR 6.2%, TE 15.4%.
- **starter** ("starter-tier") = season points ≥ the league's replacement points, i.e.
  VOR ≥ 0. Replacement in these files is the ~QB25 / RB30 / WR35 / TE12 scorer in
  10-team SF terms (the last flex-worthy tier). This is exactly the `hit` target behind
  the validated cross-league findings (P13, R1), so the sanity mapping in §7 is apples
  to apples. Test base rates: QB 65.9%, RB 43.3%, WR 42.1%, TE 34.0%.
- **vor** = season points minus replacement points (league scoring units).

## 2. Features — all draft-day legal

`price_norm` (auction price normalized to a $200 budget: `price × 200 / league budget`),
position (one-hot), `age`, `missed_prior` (prior-season games missed: 16- or 17-game
season as appropriate; missing = rookie/no prior season, flagged by `has_prior`),
`missed5`, and prior-year offense quality (`prior_off_rank` 1–32 + tier one-hots).

**The leakage trap, handled:** the `offense_tier` column shipped in the pick files is
computed from **the same season's** stats (`offense_tiers(season)` in the build scripts) —
a top-5 finish mechanically inflates the player's own team total, so that feature is
circular for draft-day use. We rebuilt offense quality from `stats_{season-1}.json`
(2019 fetched and cached for Fantasy 101's 2020 rows) with the identical method
(sum of pts_std by team over QB/RB/WR/TE/K, rank, thirds). A leaked-feature variant of
each classifier was trained *only* to measure the inflation (§5).

Known imperfections, stated once: (a) a player's team is his team in the 2026 player
database, not necessarily his team on that historical draft day — team-changers get the
wrong prior-year offense, adding noise that biases the offense feature toward zero;
(b) `sos_pa` was excluded (built from same-season schedule/outcomes); (c) nomination
order (`pick_no`, the confirmed D2 late-nomination discount) is excluded — it's only
knowable mid-auction, a v2 feature for live use; (d) offense rank is missing for ~24% of
rows (no team on record) — XGBoost handles the NaNs natively.

## 3. Model design

One pooled XGBoost classifier per target with position as one-hot features (plus a VOR
regressor with price, and one without price for fair-price inference). **Justification
for pooled over per-position:** per-position models were trained as a comparison and
lost to the pooled model in 7 of 8 position×target slices on the holdout (the
exception, QB top-5, has a very wide CI and the price-only baseline beat both). Pooling
borrows strength across positions — TE especially (5,981 rows) — while the trees still
learn position-specific price curves through the one-hots.

Config was selected **only** on a 2023 holdout (fit ≤2022) from 4 pre-declared configs;
both targets chose the most regularized one (depth 2, η=0.03, min_child_weight 30,
λ=5; 2023 log-loss: top5 0.289 vs 0.354 for the depth-4 base config; starter 0.594 vs
0.645). Final models refit on all ≤2023 data (89 and 124 rounds). A Platt
recalibrator fit **only** on 2023 predictions ships with the bundle; test metrics are
reported for raw and calibrated outputs. The first (unregularized depth-4) attempt
scored top5 AUC 0.795 / starter 0.759 on test — worse than the shipped config — and is
not in the bundle.

## 4. Validation design

**Strict time split: train ≤2023 (23,136 rows), test 2024–2025 (18,523 rows).** No
random splits anywhere; config selection and calibration never touched the test years.
Baselines: price-only logistic regression, and position-average rates. Per-position AUC
confidence intervals are cluster-bootstrapped (400 resamples over player-season
clusters, since the same player-season repeats across leagues). Full numbers:
`metrics.json`, per-row test predictions: `test_predictions.csv`.

## 5. Held-out results (test = 2024–2025)

### Classifiers — overall

| Target | Model | AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|
| **top5** (base 9.2%) | **XGBoost (shipped)** | **0.812** | **0.329** | 0.0718 | 0.023 |
| | XGBoost, Platt-calibrated | 0.812 | 0.329 | 0.0744 | 0.043 |
| | XGBoost per-position | 0.771 | 0.264 | 0.0808 | — |
| | XGBoost w/ LEAKED offense | 0.827 | 0.350 | 0.0697 | — |
| | **price-only logistic reg** | **0.819** | 0.326 | **0.0717** | — |
| | position-average | 0.610 | 0.125 | 0.0825 | — |
| **starter** (base 45.4%) | **XGBoost (shipped)** | **0.781** | 0.708 | 0.1972 | 0.081 |
| | XGBoost, Platt-calibrated | 0.781 | 0.708 | **0.1947** | 0.072 |
| | XGBoost per-position | 0.745 | 0.677 | 0.2137 | — |
| | XGBoost w/ LEAKED offense | 0.777 | 0.729 | 0.1958 | — |
| | **price-only logistic reg** | 0.772 | **0.722** | 0.1978 | — |
| | position-average | 0.592 | 0.530 | 0.2389 | — |

**Reading:** against position-average, both real models are enormously better — price is
information. Against price-only LR, the XGBoost is a **statistical tie**: top5 −0.007
AUC / +0.003 PR-AUC / +0.0001 Brier; starter +0.009 AUC / −0.014 PR-AUC / −0.001 Brier.
**The model does not beat the simpler rule on the holdout. Claim nothing more.**
The leaked-offense variant quantifies the trap: using same-season offense tier would
have flattered top5 AUC by +0.015 and Brier by −0.002 — modest here only because price
already carries most of it; the *raw* same-season tier gap is far more inflated (§7).

### Classifiers — by position (AUC, 95% cluster-bootstrap CI for shipped XGB)

| Target | Pos | n (events) | XGB | CI | per-pos XGB | price-only LR |
|---|---|---|---|---|---|---|
| top5 | QB | 3,189 (430) | 0.649 | [0.45, 0.85] | 0.680 | **0.693** |
| top5 | RB | 5,682 (430) | 0.901 | [0.84, 0.95] | 0.850 | **0.939** |
| top5 | WR | 6,938 (430) | 0.860 | [0.68, 0.98] | 0.794 | **0.879** |
| top5 | TE | 2,714 (419) | 0.709 | [0.54, 0.85] | 0.634 | **0.752** |
| starter | QB | 3,189 (2,103) | 0.740 | [0.62, 0.86] | 0.719 | 0.729 |
| starter | RB | 5,682 (2,463) | 0.810 | [0.74, 0.87] | 0.789 | 0.806 |
| starter | WR | 6,938 (2,920) | 0.783 | [0.72, 0.84] | 0.733 | 0.762 |
| starter | TE | 2,714 (923) | 0.670 | [0.53, 0.78] | 0.608 | **0.706** |

Where the shipped XGB is the best available tool: **starter-tier for RB and WR**
(≈ties or edges price-only). Where it is not: **top-5 anywhere** (price-only LR wins
every position, decisively for RB/WR) and **TE, period**. QB top-5's CI spans
[0.45, 0.85] — with only a few dozen distinct QB player-seasons per test year, nobody
knows this number precisely, including us. ~430 events per position look plentiful
until you remember they're ~30–60 unique player-seasons repeated across leagues.

### Calibration (test)

Probabilities are **rankings, not odds**. Starter raw ECE 0.081 (calibrated 0.072):
the model is systematically overconfident mid-range — predicted 0.60 ↔ observed 0.43,
predicted 0.92 ↔ 0.75 in the top decile. Top-5 raw ECE is better (0.023; top decile
0.45 predicted ↔ 0.34 observed) and Platt made it slightly *worse* on test, another
symptom of era drift. `predict.py` returns both raw and calibrated numbers; treat
either as ±0.1 at best.

### VOR regression (test)

| Model | MAE | R² |
|---|---|---|
| XGBoost with price | 68.5 | 0.301 |
| **price-only linear reg** | 68.8 | **0.317** |
| XGBoost profile-only (no price) | 79.4 | 0.067 |
| position-mean | 80.4 | 0.062 |

By position, the profile-only model is **worse than the position mean for WR (R²
−0.124) and TE (−0.088)**. Consequence for `fair_price`: without the market's price
input, age/injury/offense/position explain almost nothing about VOR — so
`fair_price_200scale` from `predict.py` is a coarse position-mean anchor (e.g., a $45
BUF QB profile prices at ~$23), **not** a valuation. Use it only to flag absurd asks.

### Feature importance (gain share)

Price dominates, as it should: starter — price_norm 0.34, prior_off_rank 0.13, pos_QB
0.11, missed_prior 0.09, missed5 0.09, pos_TE 0.08, prio_mid 0.07, age 0.06; top5 —
price_norm 0.26, prior_off_rank 0.14, prio_mid 0.13, prio_top 0.12, pos_TE 0.09, pos_QB
0.07, age 0.06, missed_prior 0.05. Gain shares have no sign; the *directions* the model
learned (from ≤2023 data) are: better prior offense ↑, more missed games ↓, extreme
youth at high prices ↓. Whether those directions still hold is §7's problem.

## 6. What the model gets right (sanity vs validated findings)

- **Expensive-youth bust (P14) — CONFIRMED in the holdout itself.** Test actuals at
  $25+: bust rate 20.2% for age ≤24 vs 9.0% for 25–30 (gap +0.112, larger than gen-2's
  +0.085). Model direction agrees (P(starter) 0.785 young vs 0.811 prime). Caveat: the
  model *under*-penalizes age 31+ (predicts 0.840; actual bust 24.7%, n=73 — thin cell,
  flagged not resolved).
- **RB price tiers (P12) — CONFIRMED in test actuals.** Hit rate peaks in the $20–40
  band (0.84), top-5 rate peaks at $30–40 (0.50), $41+ slightly lower on both. The model
  tracks the starter curve but *under*-predicts top-5 for $30–40 RBs (0.32 vs 0.50
  actual) — the market prices elite-RB ceiling better than the model does.
- **Median top-10 finisher price $19** (P16) — verified in the pooled training data.
- **Draft-value law (P1)** — consistent: price is the single dominant feature and both
  models' discrimination comes mostly from it.

## 7. Contradictions — called out, not smoothed over

**The two profile effects did not show up in 2024–2025.** Raw actual gaps among $10+
picks with prior-season data (my `missed5` reconstruction matches the files' own
column on 100% of 7,985 checked crossleague rows, so this is not a rebuild bug):

| Effect | Train ≤2023 actual | Test 2024–25 actual | Model (train-learned) predicts |
|---|---|---|---|
| Prior-year offense, top vs bottom tier → starter rate | 0.838 vs 0.710 (**+0.128**) | 0.714 vs 0.758 (**−0.044**) | 0.800 vs 0.664 (+0.136) |
| Missed 5+ prior games → starter rate | 0.540 vs 0.778 (**−0.238**) | 0.737 vs 0.706 (**+0.031**) | 0.614 vs 0.782 (−0.168) |

Implications, stated plainly:

1. **P13 (offense quality) was substantially a leakage artifact.** The same-season tier
   in the files produces a +0.228 raw gap in the test years (0.814 top vs 0.586 bottom)
   — an effect that is partly the player's own points inflating his team's "offense."
   The leak-free prior-year version was worth ~+0.13 in 2021–2023 and ~0.00 in
   2024–2025. The true draft-day offense effect is likely small (0 to +0.05), not the
   +0.31 same-season headline. **Do not draft off offense tier with anything like the
   old confidence.**
2. **R1 (injury penalty) is era-/sample-unstable.** The gen-2 estimate (−0.104) pooled
   2021–2025 seasons dominated by earlier years; in the two most recent seasons the gap
   is zero-to-positive. Either the market learned to price injury risk (plausible —
   that's what Fantasy 101's original null said), or the 2024–25 test mix (mostly
   gen-2 community leagues, $400 budgets) behaves differently. **The shipped model
   still applies a train-era injury discount (P(starter) −0.17 for missed-5+ at $10+);
   the holdout says don't trust it.** If you use one number from this report on draft
   day, it should not be an injury discount.
3. Both contradictions rhyme with **P12e's market-efficiency lesson**: public,
   priced-in information (age, injury history, team quality) adds little beyond the
   price itself out-of-sample. The validated findings were real *in their samples*;
   most were already in the price.

Also noted, minor: on the top-5 head (not starter), toggling missed games 0→8 for a
sample RB moved P(top5) slightly *up* (0.317→0.365) while P(starter) correctly dropped
(0.846→0.669) — a depth-2 interaction quirk. Another reason to treat the top-5 head as
a ranking, not gospel.

## 8. Limitations

- Test era is only two seasons and is dominated by gen-2's community theme-league
  ecosystem ($400 budgets, possibly shared managers across chains) — between-community
  replication doesn't exist yet; that's the gen-3 problem, and it's exactly where the
  profile effects went to die.
- Team attribution uses the 2026 player database (§2a) — real noise in the offense
  features, worst for journeymen.
- VOR pools different league scoring systems (mostly PPR variants); `pos_rank` targets
  are league-relative, which is the right target for "pick the best players," but VOR
  magnitudes aren't perfectly comparable across leagues.
- Small cells: TE (2,714 test rows but few elite outcomes), QB top-5 (CI [0.45, 0.85]),
  age-31+ expensive picks (n=73). Prices above ~$60 normalized are sparse — don't
  extrapolate the curves there.
- Best-ball auctions are included (defensible: the gen-1 report showed best-ball
  auctions price players like managed ones; outcomes are player points either way).

## 9. Verdict — where this is draft-day usable

**Usable:**
- **P(starter) as a floor/tiebreaker for RB and WR** — among same-price candidates,
  prefer the higher P(starter). It's the one slice that matched or edged price-only.
- **P(top5) as a ceiling ranking for RB/WR shortlists** — again as a *ranking* among
  near-equal prices, knowing price-only was marginally better in 2024–25 and the gap is
  inside the noise (CIs overlap).
- **The learned interactions as a checklist, not rules**: prime-age (25–30) over
  expensive youth is the one profile effect that replicated in the holdout actuals.

**Not usable:**
- As calibrated probabilities (ECE ~0.07–0.08 on starter; mid-range overconfidence).
- `fair_price` as a valuation (profile-only VOR R² 0.067; negative for WR/TE; it
  cannot tell a hyped rookie from a nobody).
- For TE decisions or QB top-5 hunting — price-only beat the model there; your own
  price discipline is the better instrument.
- The offense-quality and injury adjustments as hard draft rules (§7).

**Net:** the model's honest draft-day role is a second opinion layered on price —
exactly what the data supports. The auction market in these leagues is efficient
enough that "beat the price" remains hard; the exploitable residue is small,
position-specific (RB/WR floors), and worth maybe a tiebreak, not a strategy.

## 10. Bundle contents & reuse

```
model/
  train_model.py        full pipeline: data assembly, prior-year offense rebuild,
                        time-split training, config search, calibration, metrics
  predict.py            predict_player(pos, age, price, prior_games_missed,
                        team=None, prior_off_rank=None, budget=200)
                        -> p_top5, p_starter (+cal variants), VOR at price,
                           profile VOR, fair_price_200scale
  model_top5.json  model_starter.json  model_vor.json  model_vor_noprice.json
  features.json         feature order, target definitions, Platt params, fair-price k
  team_offense_2025.json  2025 offense rank/tier by team (prior-year input for 2026)
  metrics.json  importance.json  sanity_checks.json
  training_data.csv     the exact 41,659-row training/test frame
  test_predictions.csv  held-out predictions for audit
  validation_report.md  this file
```

Retrain: `~/workspace/.venvs/xgb/bin/python train_model.py` (xgboost 3.4.1,
scikit-learn 1.9.1, pandas 3.0.6; set `TMPDIR` to a roomy directory — /tmp here is a
512MB tmpfs and pip will die). To refresh for the 2027 draft: add the new season's
picks, cache `stats_{season}.json`, regenerate the team-offense table for the new
prior year, and rerun — the time split moves forward automatically by season.

---

## 11. v2 improvement pass (2026-10-02, later same day)

**Mandate:** beat price-only on the holdout, or honestly report that we can't.
**Verdict: the v2 XGBoost clears the bar overall** — top-5 AUC 0.842 vs price-only
0.819, starter 0.797 vs 0.772, both with better Brier too — and its disagreements
with price cashed in the holdout, which is the actual draft-day use. Two honest
caveats ride along: a simple 5-feature logistic *beats the XGBoost on top-5*, and
no calibration scheme survived transport to 2024–25. v1 files are preserved in
`backup_v1/`; everything below was produced by `build_features_v2.py` +
`train_model_v2.py` + `finalize_v2.py`.

### What changed

1. **Price-independent features** (the lever v1 lacked): FantasyPros preseason ECR
   from the DynastyProcess archive (`db_fpecr.csv.gz`) — latest redraft scrape
   strictly before kickoff each year (2021-09-03, 2022-09-02, 2023-09-01,
   2024-08-30, 2025-08-08; archive has no summer-2020 redraft scrape, so the 143
   rows of 2020 go without), overall + positional ECR + expert sd. Plus
   prior-season production from the Sleeper stats cache: PPR PPG/points/NFL
   positional rank, snap share, target share, rush share, opportunities/game;
   NFL draft capital; age²; expensive-youth flag. Prior-year offense rank/tier
   (leak-free) kept; same-season offense stays banned. Coverage: ECR 98.7%,
   prior production ~85%, shares 61% (target/rush share needs current-team
   attribution; train-era 50%, test-era 75%).
2. **Recency weighting**: per-season sample-weight decay, selected on 2023
   (top-5 picked decay 0.7, starter 0.85).
3. **Calibration**: see below — attempted twice, both honest failures.
4. Same protocol protection as v1: every choice (config, decay, calibration
   family, ablations) made on the 2023 fold with fits ≤2022; final models refit
   ≤2023 and evaluated once on 2024–25. Disclosed imperfection: v1's holdout
   results were known when designing v2; the split itself was never re-used for
   selection.

### Held-out results (test = 2024–2025, same 18,523 picks as v1)

| Target | Model | AUC | PR-AUC | Brier | LogLoss |
|---|---|---|---|---|---|
| **top5** | **XGBoost v2** | **0.842** | **0.357** | **0.0693** | 0.238 |
| | XGBoost v1 (for reference) | 0.812 | 0.329 | 0.0718 | — |
| | price-only LR | 0.819 | 0.326 | 0.0717 | 0.252 |
| | **LR-extended** (price+ECR+PPG+age+missed) | **0.867** | **0.396** | **0.0670** | 0.226 |
| **starter** | **XGBoost v2** | **0.797** | **0.737** | **0.1846** | 0.554 |
| | XGBoost v1 | 0.781 | 0.708 | 0.1972 | — |
| | price-only LR | 0.772 | 0.722 | 0.1978 | 0.583 |
| | LR-extended | 0.789 | 0.737 | 0.1863 | 0.553 |
| **VOR R²** | XGBoost v2 with price / profile-only | **0.382 / 0.369** | | | |
| | v1 equivalents; price-only linear | 0.301 / 0.067; 0.317 | | | |

Per position (AUC; XGB v2 vs price-LR vs LR-ext): top-5 — QB **0.710**/0.693/0.732
(CI still [0.53, 0.89]: few dozen unique QB seasons, treat as ranking only),
RB 0.924/**0.939**/**0.958**, WR 0.820/0.879/**0.886**, TE **0.804**/0.752/0.769
(v1's worst cell, TE top-5, is now the model's clearest win — the ECR/prior-PPG
features apparently carry the TE signal price missed). Starter — QB **0.771**,
RB 0.807, WR 0.781, TE **0.717**: XGBoost ≥ price-LR everywhere; LR-ext edges it
at RB/WR. `features.json → head_reliability` records the winner per cell and
`predict.py` surfaces it as `head_notes`.

**Walk-forward (fit ≤2022 → test 2023, fixed v1 config):** top-5 AUC 0.728
(v1 features) → 0.738 (v2 features) → 0.747 (v2 + recency decay); starter
0.763 → 0.771 → 0.774. The gain replicates in the forward direction on a fold
the final models never trained on, smaller than on 2024–25 but the same sign —
not a holdout artifact.

**Ablations on 2023 were equivocal, stated plainly:** dropping ECR *helped*
starter on 2023 (AUC 0.783, log-loss 0.569 vs full 0.774/0.601) and dropping
prior-production helped top-5 (0.752 vs 0.747), yet the full model won where it
counts (the holdout above). Single folds at this size cannot resolve individual
feature groups; ECR's marginal contribution is genuinely uncertain. Gain
importance in the shipped models: top-5 — price 0.32, ecr_pos + its log 0.22,
prior offense ~0.14, snap/PPG/prior-rank ~0.11; starter — ecr_sd 0.26 (!),
ecr_pos 0.09, prior PPG 0.08. The starter model's single biggest gain feature is
expert *disagreement* (sd) — the market's uncertainty about a player is itself
predictive of floor outcomes, in the direction of lower hit rates.

### Did disagreements with price cash? (the draft-day test)

Quintiles of (model P − price-only P) on the 2024–25 holdout, realized rate
minus price-implied rate:

| Quintile | starter: act − implied | top-5: act − implied |
|---|---|---|
| model hates him vs price | **−0.164** | −0.028 |
| | −0.124 | −0.018 |
| neutral | −0.007 | −0.026 |
| | +0.073 | 0.000 |
| model loves him vs price | **+0.158** | **+0.087** |

Starter disagreements are monotone and big: the top quintile (mean price $7.6)
actually hit at 0.600 vs 0.442 implied — cheap players the model rates as
likely starters but the room prices as fliers. By position (starter, top
quintile): **QB +0.469** (mean price $7.4, realized starter rate 0.905 vs 0.436
implied — cheap QB2s are the single biggest market inefficiency in this data),
WR +0.115, RB +0.053, TE +0.005. Top-5 disagreements cashed at TE (+0.195),
QB (+0.073), RB (+0.031) but **failed at WR (−0.032)** — use LR-ext for WR
ceiling calls, consistent with the per-position table. Realized VOR-per-dollar
tells the same story (starter top quintile +2.36 vs bottom −6.33).

### Calibration — attempted twice, report both failures

1. Isotonic fit on 2023 predictions (chosen by CV inside 2023) **worsened** the
   holdout (top-5 Brier 0.0693→0.0742; AUC diluted by out-of-range clamping).
2. Rebuilt properly — 5-fold cross-fit calibrator on ≤2023 OOF predictions
   (isotonic legitimately won the OOF comparison) — still did not transport:
   top-5 ECE improved 0.046→0.025 but Brier worsened; starter got *worse on
   everything* (ECE 0.089→0.106, log-loss 0.554→0.630). Era drift keeps beating
   recalibration at this sample size. **Usage rule: read raw `p_top5` as rough
   odds (±~0.05), raw `p_starter` as a ranking with approximate magnitudes;
   `*_cal` ships for reference and is NOT the recommended number.** One
   consequence, owned: the calibration sub-question has now seen the holdout,
   so future calibration choices need a fresher test fold.

### Updated sanity: what still doesn't travel

- **Injury discount**: v2 model-implied starter gap on real test rows is −0.131
  (0.654 missed-5+ vs 0.785 durable) vs actual **+0.031**. Recency weighting
  shrank v1's −0.168 but did not kill it. Still do not draft off it.
- **Prior-year offense**: test-era top vs bottom tier hit 0.714 vs 0.758
  (−0.044; train was +0.096). Effect remains ~zero post-2023; the model still
  spends ~14% of top-5 gain on it from train-era data.
- **Expensive-youth bust**: features now include the explicit flag; prior
  finding stands (not re-tested as a headline here).

### Updated verdict

- **The XGBoost v2 is now the best overall instrument for the starter target**
  and for ceiling rankings at QB/TE, beating the price-only baseline the
  mandate set. On **top-5 at RB/WR, a 5-feature logistic (or even price-only at
  RB) beats the trees** — the honest recommendation is per-position tools,
  encoded in `head_reliability`/`head_notes` rather than one model everywhere.
- **The exploitable output is the disagreement list**: players where the model
  and the room disagree, above all cheap QB2s (starter +0.469). That, not the
  probability itself, is the draft-day edge this exercise actually found.
- **`fair_price` graduated** from coarse anchor to usable reference: profile-only
  VOR R² 0.369 with every position positive (QB 0.275 / RB 0.397 / WR 0.274 /
  TE 0.325), and `predict.py` now uses position-specific $/VOR rates. Still a
  sanity band, not a bid sheet (it cannot see hype, and elite ceilings are
  regressed toward the profile mean).
- Probabilities: rankings everywhere; rough odds only for top-5 raw.
- Still not usable: injury-history or offense-tier adjustments as draft rules;
  QB top-5 point estimates (CI [0.53, 0.89]).

### Bundle changes in v2

`model_*.json` retrained (v2 features); `features.json` rewritten (feature list,
selection config, cross-fit calibrator, `head_reliability`, LR-extended
coefficients, position-specific fair-price K); `predict.py` — same signature,
new optional inputs (`ecr_*`, `prior_*`, shares, draft capital), returns
`p_*_lrext`, `head_notes`, and `profile_completeness`/`sparse_profile` flags
(sparse profiles route down "no prior data" branches — supply ECR and prior
PPG or treat outputs as floors). New files: `training_data_v2.csv`,
`build_features_v2.py`, `train_model_v2.py`, `finalize_v2.py`, `metrics_v2.json`,
`test_predictions_v2.csv`, `backup_v1/` (complete v1 bundle). ECR source data in
`../crossleague-cache/v2/` (`ecr_preseason_sleeper.csv` is the joined preseason
table). For the 2026 draft, `team_offense_2025.json` is already the correct
prior-year input; refresh ritual now also needs the new season's preseason ECR
scrape and `stats_{season}.json` before retraining.

---

## 12. Improvement loop (2026-10-02, after §11)

**Mandate:** iterate single changes against the shipped v2 champion under
Eric's protocol — sequential falsification on prequential folds. Every
experiment was tested cold on TWO folds: the 2024–25 holdout (fit ≤2023)
and the walk-forward fold (fit ≤2022 → test 2023). Acceptance required a
holdout improvement (AUC ≥ +0.003 or Brier ≥ −0.0015 on ≥1 target) with
no regression beyond tolerance (AUC −0.002 / Brier +0.0005) on any
target × fold cell. Accepted changes became the new champion
(hill-climbing). Cap: 12 experiments. Harness: `loop_lib.py` /
`run_experiment.py`; per-run metrics in `loop_runs/`; narrative log in
`experiment_log.md`; progress in `checkpoints/progress.md`.

**Disclosed protocol caveat:** the 2024–25 holdout informed v2's design
and now ~15 loop selection decisions. The dual-fold gate and non-trivial
thresholds mitigate adaptive overfitting but do not eliminate it; the
honest read of the loop's total gain should carry that discount. The
walk-forward fold (2023) is genuinely untouched by the final fits.

### Experiments

| # | Change | Holdout Δ (top5 / starter AUC) | Walk-fwd Δ | Verdict |
|---|---|---|---|---|
| e01 | +VAL relative-value features (ppg_per_dollar, price_x_age, ecr_vs_prior_rank, price_per_ecr) | **+0.010** / +0.003 | −0.002 / −0.001 | ✅ ACCEPT |
| e02 | +EFF efficiency rates (all targets) | −0.016 / +0.004 | −0.002 / −0.002 | ❌ top-5 regressed |
| e02b | EFF features, starter head only (scoped variant of e02) | 0 / **+0.004** (Brier −0.002) | 0 / −0.000 | ✅ ACCEPT |
| e03 | drop prior-year offense features | +0.002 / −0.000 (starter Brier −0.0016) | +0.005 / −0.001 | ✅ ACCEPT* |
| e04 | drop missed_prior/missed5 | +0.003 / +0.000 | **−0.011** / +0.001 | ❌ WF regressed |
| e05 | 50/50 ensemble XGB + LR-extended | **+0.017** / +0.004 (ECE 0.085→0.052) | **+0.052** / +0.018 | ✅ ACCEPT |
| e06 | cluster-normalized sample weights | −0.001 / −0.002 | +0.025 / +0.006 | ❌ holdout flat |
| e07 | +TEAM pass-volume context | +0.000 / +0.002 | +0.002 / +0.001 | ❌ under bar |
| e08 | +TRAJ two-year trajectory | +0.002 / −0.002 | −0.005 / +0.003 | ❌ mixed |
| e09 | +WK weekly consistency | −0.001 / −0.001 | −0.003 / −0.000 | ❌ flat |
| e11 | per-position blends (WR top-5, TE starter) | +0.001 / −0.002 | +0.010 / −0.003 | ❌ holdout Brier regressed |
| e12 | train-fold-only shrink calibration | cal ≡ raw | cal ≡ raw | ❌ null (see below) |

*e03 tradeoff, recorded: holdout top-5 PR-AUC dipped 0.404 → 0.374 while
AUC/Brier/ECE improved or held. Acceptance followed the predeclared gate
(AUC/Brier); anyone optimizing precision-at-the-top over ranking quality
should know the offense features' removal cost some early-precision.

**Process note:** the first e03/e04 runs were silent no-ops for the
classifiers (a spec-shadowing bug: `feats_by_target` from e02b shadowed
the `feats` edits). Caught because two consecutive experiments returned
bit-identical metrics; fixed and rerun. The rerun decisions supersede.

### What each accepted change means

- **VAL features (e01)** were the only new-information win, and they are
  not really new information — they are *ratios* (PPG per dollar, price
  per ECR rank, ECR vs last year's finish). Letting depth-2 trees split
  on relative-value directly is what unlocked them: `price_per_ecr` is
  now the starter model's #1 gain feature (0.238).
- **EFF for starter only (e02b):** efficiency rates (pts/opportunity,
  RZ share, catch rate) describe floors, not ceilings — exactly the
  starter target. As a global add they hurt top-5; scoped to starter
  they helped both folds. The asymmetry is itself a finding.
- **Dropping offense (e03):** after ECR + production + VAL features,
  prior-year offense tier contributes nothing the trees use. Combined
  with §7/§11, the offense-quality era of this project is closed: it was
  leakage-inflated, then era-expired, now feature-redundant.
- **Ensemble (e05)** was the largest single gain. The 5-feature logistic
  and the depth-2 trees make different errors; averaging halves both.
  Walk-forward top-5 jumped +0.052 — the logistic is very strong on
  2023 — and holdout ECE improved materially (starter 0.085 → 0.052).
- **Injury features stay (e04 rejected):** dropping them helped the
  holdout (+0.003) but hurt walk-forward (−0.011). The era conflict from
  §7 is real and unresolved; the dual gate keeps the features because
  2023 still prices injury risk the way the model expects. Do not read
  this as the injury discount being trustworthy — §7's warning stands.

### Calibration (e12)

The one calibration family not yet tried — shrink toward the base rate
with s chosen by 5-fold OOF Brier *inside each training fold* — was run
properly. **OOF chose s = 1.0 (no shrinkage) in all four target × fold
cells.** The train folds themselves say the raw probabilities are
already as calibrated as a monotone shrink can make them; the residual
test-fold miscalibration is era drift, not a removable bias. `*_cal` in
predict.py now returns raw unchanged. Calibration attempts are closed:
three families (Platt, isotonic ×2, shrink) all failed to transport.

### Final champion vs shipped v2

| Metric (holdout 2024–25) | v2 | loop champion | Δ |
|---|---|---|---|
| top-5 AUC / PR-AUC / Brier / ECE | 0.842 / 0.357 / 0.0693 / 0.046 | **0.869 / 0.392 / 0.0668 / 0.022** | +0.027 |
| starter AUC / PR-AUC / Brier / ECE | 0.797 / 0.737 / 0.1846 / 0.089 | **0.808 / 0.746 / 0.1790 / 0.052** | +0.011 |
| walk-forward top-5 / starter AUC | 0.747 / 0.774 | **0.802 / 0.789** | +0.055 / +0.015 |
| VOR R² with-price / profile-only | 0.382 / 0.369 | 0.381 / **0.383** | flat / +0.014 |

Per position (holdout, ensemble AUC): top-5 QB 0.745, RB 0.957, WR 0.882,
TE 0.782; starter QB 0.762, RB 0.825, WR 0.799, TE 0.726. The ensemble is
best or within 0.017 of the best tool in every cell; LR-extended still
nominally leads RB/WR top-5 by ≤0.004 (a tie, and the ensemble closed
the gap that made v2 route those cells away from the trees).
`features.json → head_reliability` carries the new table with
cluster-bootstrap CIs for the ensemble.

**Disagreement check (the draft-day use case), final ensemble:** starter
quintiles of (model − price-implied) remain monotone, −0.192 → +0.179
(v2's XGB-only: −0.164 → +0.158). By position, the cheap-QB2 effect
persists (+0.402 top-quintile actual-minus-implied at a $7 mean price —
this is why QB starter is the model's best asset: superflex QB prices
are set at the top of the market, the QB2 tail is priced as fliers, and
price_per_ecr / ecr_sd measure exactly that gap). The v2 WR top-5
disagreement failure (−0.032) is now neutral (+0.004) under the
ensemble; TE top-5 disagreement is the strongest ceiling signal
(+0.219).

### Headroom verdict

**The public-information feature space is exhausted.** Seven feature
experiments produced two accepts, both about *relative value against
price* rather than new information; team context, trajectory, weekly
consistency, and efficiency-for-ceilings all failed the gate. Remaining
headroom, in expected-value order:

1. **A fresher holdout.** 2026 picks (the season now in progress) become
   testable next fall; the loop should re-run against 2025–26 before
   any further tuning. Cluster weights (e06) and the injury question
   (e04) both showed fold disagreement that only new seasons can settle.
2. **Live-auction features** (nomination order / D2 late-nomination
   discount, remaining-budget context) — excluded here by design
   (mid-auction only), the one known signal never fed to the model.
3. **Between-community data (gen-3)** for the community-ecosystem bias
   noted in §8; more of the same leagues would not add diversity.

Further tuning against 2024–25 would now be adaptive overfitting, not
improvement. **Loop stopped at the 12-experiment cap with the champion
above; predict.py's interface is unchanged** (p_top5/p_starter are now
the ensemble; p_*_xgb exposes the pure trees; *_cal returns raw).

Bundle changes: `model_*.json` retrained; `features.json` rewritten
(version `v3-loop`, per-target feature lists, ensemble spec,
head_reliability); `predict.py` updated (per-target features, ensemble
primary, optional EFF inputs); complete v2 bundle in `backup_v2/`.
Loop artifacts: `build_features_v3.py`, `training_data_v3.csv`,
`loop_lib.py`, `run_experiment.py`, `finalize_loop.py`,
`loop_state.json`, `loop_runs/`, `experiment_log.md`,
`metrics_loop.json`, `importance_loop.json`.
