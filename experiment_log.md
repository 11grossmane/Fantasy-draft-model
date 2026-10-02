# Improvement loop — experiment log

Started 2026-10-02. Champion = shipped v2 recipe. Protocol: sequential
falsification on prequential folds — every change tested cold on the 2024-25
holdout AND the walk-forward fold (fit <=2022 -> 2023); accepted only if it
improves the holdout (AUC +0.003 or Brier -0.0015 on >=1 target) with no
regression beyond tolerance (AUC -0.002 / Brier +0.0005) on any
target x fold cell. Hill-climbing: accepted changes become the new champion.
Cap: 12 experiments. Known caveat: the 2024-25 holdout informed v2's design,
so loop acceptances carry adaptive-overfitting risk; the dual-fold gate and
non-trivial thresholds are the mitigation.

## e01_val — add VAL relative-value/interaction features (ppg_per_dollar, price_x_age, ecr_vs_prior_rank, price_per_ecr)

- decision: **ACCEPT** ()
- deltas vs champion: top5/wf AUC -0.0018 Brier +0.0004; top5/holdout AUC +0.0095 Brier -0.0015; starter/wf AUC -0.0015 Brier -0.0013; starter/holdout AUC +0.0029 Brier -0.0007
- new metrics: `{"top5_wf": {"auc": 0.7455, "brier": 0.082}, "top5_holdout": {"auc": 0.8513, "brier": 0.0678}, "starter_wf": {"auc": 0.7726, "brier": 0.2028}, "starter_holdout": {"auc": 0.7998, "brier": 0.1839}, "vor_wf": {"r2_price": 0.281, "r2_profile": 0.292}, "vor_holdout": {"r2_price": 0.384, "r2_profile": 0.367}}`

## e02_eff — add EFF prior-season efficiency-rate features (pts/opp, pts/snap, catch rate, aDOT, RZ share, QB pass/rush rates)

- decision: **REJECT** (top5/wf Brier regress +0.0010; top5/holdout AUC regress -0.0157)
- deltas vs champion: top5/wf AUC +0.0002 Brier +0.0010; top5/holdout AUC -0.0157 Brier +0.0003; starter/wf AUC -0.0002 Brier -0.0014; starter/holdout AUC +0.0043 Brier -0.0020
- new metrics: `{"top5_wf": {"auc": 0.7457, "brier": 0.0829}, "top5_holdout": {"auc": 0.8356, "brier": 0.0681}, "starter_wf": {"auc": 0.7724, "brier": 0.2013}, "starter_holdout": {"auc": 0.804, "brier": 0.1819}, "vor_wf": {"r2_price": 0.294, "r2_profile": 0.29}, "vor_holdout": {"r2_price": 0.381, "r2_profile": 0.371}}`

## e02b_eff_starter — EFF efficiency features applied to the STARTER head only (post-hoc scoped variant of rejected e02: EFF helped starter +0.004 AUC / hurt top-5 -0.016 as a global add; gated identically)

- decision: **ACCEPT** ()
- deltas vs champion: top5/wf AUC +0.0000 Brier +0.0000; top5/holdout AUC +0.0000 Brier +0.0000; starter/wf AUC -0.0002 Brier -0.0014; starter/holdout AUC +0.0043 Brier -0.0020
- new metrics: `{"top5_wf": {"auc": 0.7455, "brier": 0.082}, "top5_holdout": {"auc": 0.8513, "brier": 0.0678}, "starter_wf": {"auc": 0.7724, "brier": 0.2013}, "starter_holdout": {"auc": 0.804, "brier": 0.1819}, "vor_wf": {"r2_price": 0.281, "r2_profile": 0.292}, "vor_holdout": {"r2_price": 0.384, "r2_profile": 0.367}}`

## e03_dropoff — drop prior-year offense features (prior_off_rank, prio_*) - effect measured ~0 post-2023 in v2 report s.11

- decision: **REJECT** (no target cleared the holdout improvement bar (AUC +0.003 or Brier -0.0015))
- deltas vs champion: top5/wf AUC +0.0000 Brier +0.0000; top5/holdout AUC +0.0000 Brier +0.0000; starter/wf AUC +0.0000 Brier +0.0000; starter/holdout AUC +0.0000 Brier +0.0000
- new metrics: `{"top5_wf": {"auc": 0.7455, "brier": 0.082}, "top5_holdout": {"auc": 0.8513, "brier": 0.0678}, "starter_wf": {"auc": 0.7724, "brier": 0.2013}, "starter_holdout": {"auc": 0.804, "brier": 0.1819}, "vor_wf": {"r2_price": 0.282, "r2_profile": 0.294}, "vor_holdout": {"r2_price": 0.381, "r2_profile": 0.383}}`

## e04_dropmissed — drop missed_prior/missed5 (injury effect flipped sign in 2024-25 test actuals; model still applies stale discount)

- decision: **REJECT** (no target cleared the holdout improvement bar (AUC +0.003 or Brier -0.0015))
- deltas vs champion: top5/wf AUC +0.0000 Brier +0.0000; top5/holdout AUC +0.0000 Brier +0.0000; starter/wf AUC +0.0000 Brier +0.0000; starter/holdout AUC +0.0000 Brier +0.0000
- new metrics: `{"top5_wf": {"auc": 0.7455, "brier": 0.082}, "top5_holdout": {"auc": 0.8513, "brier": 0.0678}, "starter_wf": {"auc": 0.7724, "brier": 0.2013}, "starter_holdout": {"auc": 0.804, "brier": 0.1819}, "vor_wf": {"r2_price": 0.28, "r2_profile": 0.294}, "vor_holdout": {"r2_price": 0.367, "r2_profile": 0.373}}`

## e03_dropoff — drop prior-year offense features (prior_off_rank, prio_*) - effect measured ~0 post-2023 in v2 report s.11

- decision: **ACCEPT** ()
- deltas vs champion: top5/wf AUC +0.0046 Brier -0.0015; top5/holdout AUC +0.0015 Brier +0.0000; starter/wf AUC -0.0009 Brier -0.0003; starter/holdout AUC -0.0005 Brier -0.0015
- new metrics: `{"top5_wf": {"auc": 0.7502, "brier": 0.0805}, "top5_holdout": {"auc": 0.8528, "brier": 0.0678}, "starter_wf": {"auc": 0.7714, "brier": 0.201}, "starter_holdout": {"auc": 0.8036, "brier": 0.1803}, "vor_wf": {"r2_price": 0.282, "r2_profile": 0.294}, "vor_holdout": {"r2_price": 0.381, "r2_profile": 0.383}}`

## e04_dropmissed — drop missed_prior/missed5 (injury effect flipped sign in 2024-25 test actuals; model still applies stale discount)

- decision: **REJECT** (top5/wf AUC regress -0.0110; top5/wf Brier regress +0.0009)
- deltas vs champion: top5/wf AUC -0.0110 Brier +0.0009; top5/holdout AUC +0.0031 Brier -0.0001; starter/wf AUC +0.0018 Brier -0.0010; starter/holdout AUC +0.0004 Brier +0.0000
- new metrics: `{"top5_wf": {"auc": 0.7391, "brier": 0.0814}, "top5_holdout": {"auc": 0.8559, "brier": 0.0677}, "starter_wf": {"auc": 0.7733, "brier": 0.2001}, "starter_holdout": {"auc": 0.804, "brier": 0.1803}, "vor_wf": {"r2_price": 0.273, "r2_profile": 0.287}, "vor_holdout": {"r2_price": 0.368, "r2_profile": 0.383}}`

> **Corrigendum (e03/e04):** the first e03/e04 runs were no-ops for the
> classifiers — since e02b the champion spec carries `feats_by_target`,
> which shadowed the `feats` edits those variants made (only VOR moved).
> The runner was fixed (`map_feats` applies edits to both) and e03/e04
> were rerun; the rerun decisions supersede the first ones. e03 note:
> acceptance came via starter Brier and walk-forward gains while holdout
> top-5 PR-AUC dipped 0.404 -> 0.374 — a real tradeoff, recorded.

## e05_ensemble — fixed 50/50 probability average of XGBoost with the LR-extended head (LR-ext beat XGB on holdout top-5)

- decision: **ACCEPT** ()
- deltas vs champion: top5/wf AUC +0.0515 Brier -0.0062; top5/holdout AUC +0.0166 Brier -0.0010; starter/wf AUC +0.0180 Brier -0.0153; starter/holdout AUC +0.0041 Brier -0.0014
- new metrics: `{"top5_wf": {"auc": 0.8017, "brier": 0.0743}, "top5_holdout": {"auc": 0.8694, "brier": 0.0668}, "starter_wf": {"auc": 0.7894, "brier": 0.1857}, "starter_holdout": {"auc": 0.8077, "brier": 0.179}, "vor_wf": {"r2_price": 0.282, "r2_profile": 0.294}, "vor_holdout": {"r2_price": 0.381, "r2_profile": 0.383}}`

## e06_cluster — cluster-normalized sample weights: recency decay / times player-season appears across leagues, renormalized

- decision: **REJECT** (starter/holdout AUC regress -0.0021; starter/holdout Brier regress +0.0006; no target cleared the holdout improvement bar (AUC +0.003 or Brier -0.0015))
- deltas vs champion: top5/wf AUC +0.0249 Brier -0.0019; top5/holdout AUC -0.0011 Brier -0.0002; starter/wf AUC +0.0055 Brier -0.0031; starter/holdout AUC -0.0021 Brier +0.0006
- new metrics: `{"top5_wf": {"auc": 0.8265, "brier": 0.0724}, "top5_holdout": {"auc": 0.8683, "brier": 0.0666}, "starter_wf": {"auc": 0.7949, "brier": 0.1826}, "starter_holdout": {"auc": 0.8056, "brier": 0.1796}, "vor_wf": {"r2_price": 0.282, "r2_profile": 0.294}, "vor_holdout": {"r2_price": 0.381, "r2_profile": 0.383}}`

## e07_team — add TEAM prior-year team pass-volume context (pass att/game, pass share, pass rank, plays/game)

- decision: **REJECT** (no target cleared the holdout improvement bar (AUC +0.003 or Brier -0.0015))
- deltas vs champion: top5/wf AUC +0.0022 Brier +0.0005; top5/holdout AUC +0.0002 Brier +0.0000; starter/wf AUC +0.0008 Brier +0.0003; starter/holdout AUC +0.0017 Brier -0.0004
- new metrics: `{"top5_wf": {"auc": 0.8038, "brier": 0.0748}, "top5_holdout": {"auc": 0.8695, "brier": 0.0668}, "starter_wf": {"auc": 0.7902, "brier": 0.186}, "starter_holdout": {"auc": 0.8094, "brier": 0.1785}, "vor_wf": {"r2_price": 0.28, "r2_profile": 0.271}, "vor_holdout": {"r2_price": 0.381, "r2_profile": 0.388}}`

## e08_traj — add TRAJ two-year trajectory (prior2 PPG/rank, ppg_delta, rank_improve)

- decision: **REJECT** (top5/wf AUC regress -0.0047; top5/wf Brier regress +0.0018; starter/holdout AUC regress -0.0022; starter/holdout Brier regress +0.0008; no target cleared the holdout improvement bar (AUC +0.003 or Brier -0.0015))
- deltas vs champion: top5/wf AUC -0.0047 Brier +0.0018; top5/holdout AUC +0.0020 Brier -0.0002; starter/wf AUC +0.0028 Brier -0.0009; starter/holdout AUC -0.0022 Brier +0.0008
- new metrics: `{"top5_wf": {"auc": 0.7969, "brier": 0.0761}, "top5_holdout": {"auc": 0.8713, "brier": 0.0666}, "starter_wf": {"auc": 0.7922, "brier": 0.1848}, "starter_holdout": {"auc": 0.8055, "brier": 0.1798}, "vor_wf": {"r2_price": 0.298, "r2_profile": 0.291}, "vor_holdout": {"r2_price": 0.373, "r2_profile": 0.37}}`

## e09_wk — add WK prior-season weekly consistency (points CV, floor-week share, 90th-pct week)

- decision: **REJECT** (top5/wf AUC regress -0.0032; no target cleared the holdout improvement bar (AUC +0.003 or Brier -0.0015))
- deltas vs champion: top5/wf AUC -0.0032 Brier +0.0001; top5/holdout AUC -0.0008 Brier +0.0002; starter/wf AUC -0.0001 Brier +0.0002; starter/holdout AUC -0.0010 Brier +0.0002
- new metrics: `{"top5_wf": {"auc": 0.7985, "brier": 0.0745}, "top5_holdout": {"auc": 0.8686, "brier": 0.067}, "starter_wf": {"auc": 0.7893, "brier": 0.1859}, "starter_holdout": {"auc": 0.8067, "brier": 0.1792}, "vor_wf": {"r2_price": 0.285, "r2_profile": 0.278}, "vor_holdout": {"r2_price": 0.385, "r2_profile": 0.388}}`

## e11_posblend — per-position blend for weak cells: 0.6 pooled + 0.4 per-position model for WR top-5 and TE starter (fixed weights)

- decision: **REJECT** (top5/holdout Brier regress +0.0009; starter/wf AUC regress -0.0033; starter/wf Brier regress +0.0017; starter/holdout Brier regress +0.0006; no target cleared the holdout improvement bar (AUC +0.003 or Brier -0.0015))
- deltas vs champion: top5/wf AUC +0.0102 Brier -0.0009; top5/holdout AUC +0.0009 Brier +0.0009; starter/wf AUC -0.0033 Brier +0.0017; starter/holdout AUC -0.0015 Brier +0.0006
- new metrics: `{"top5_wf": {"auc": 0.8118, "brier": 0.0734}, "top5_holdout": {"auc": 0.8703, "brier": 0.0677}, "starter_wf": {"auc": 0.7861, "brier": 0.1875}, "starter_holdout": {"auc": 0.8062, "brier": 0.1795}, "vor_wf": {"r2_price": 0.282, "r2_profile": 0.294}, "vor_holdout": {"r2_price": 0.381, "r2_profile": 0.383}}`

## e12_cal — train-fold-only shrink calibration: s in {1,.95,...,.7} toward train base rate chosen by 5-fold OOF Brier inside each training fold; transport to both test folds

- decision: **REJECT** (no target met the calibration improvement bar)
- deltas vs champion: top5/wf AUC +0.0000 Brier +0.0000; top5/holdout AUC +0.0000 Brier +0.0000; starter/wf AUC +0.0000 Brier +0.0000; starter/holdout AUC +0.0000 Brier +0.0000
- new metrics: `{"top5_wf": {"auc": 0.8017, "brier": 0.0743}, "top5_holdout": {"auc": 0.8694, "brier": 0.0668}, "starter_wf": {"auc": 0.7894, "brier": 0.1857}, "starter_holdout": {"auc": 0.8077, "brier": 0.179}, "vor_wf": {"r2_price": 0.282, "r2_profile": 0.294}, "vor_holdout": {"r2_price": 0.381, "r2_profile": 0.383}}`

## Loop closed (12-experiment cap reached)

Accepted: e01_val, e02b_eff_starter, e03_dropoff, e05_ensemble.
Rejected: e02_eff (global), e04_dropmissed, e06_cluster, e07_team,
e08_traj, e09_wk, e11_posblend, e12_cal (null: OOF chose s=1.0).
Cut before running: e10 depth-3 hyperparameters (v2's 2023 grid had
already rejected that direction, log-loss 0.291-0.301 vs depth-2 0.285).
Champion shipped as bundle version v3-loop; full write-up in
validation_report.md s.12. Headroom verdict: public-info feature space
exhausted; next honest improvement needs a fresher holdout (2026
season), live-auction features, or gen-3 between-community data.
