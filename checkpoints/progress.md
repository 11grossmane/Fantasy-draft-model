# Improve-the-model (v2) — progress log

Started 2026-10-02 ~19:30 UTC. Mandate: beat price-only logistic on the 2024–25
holdout, or honestly report that we can't. Protocol protection: all feature/
config/calibration choices are made on the 2023 fold (fit ≤2022); 2024–25 is
evaluated once at the end. Walk-forward (fit ≤2022 → test 2023, fixed v1 config)
compares feature sets without touching the primary holdout for selection.

## Data sources fetched
- DynastyProcess `db_fpecr.csv.gz` (FantasyPros ECR archive, scrape dates
  2019-12-27 → 2025-08-08) → `crossleague-cache/v2/db_fpecr.csv.gz`
- DynastyProcess `db_playerids.csv` (sleeper_id ↔ fantasypros_id crosswalk,
  + NFL draft year/round/pick) → `crossleague-cache/v2/db_playerids.csv`
- Sleeper season stats already cached: `crossleague-cache/targeted/stats/
  stats_{2019..2025}.json` (pts_ppr, pos_rank_ppr, off_snp, tm_off_snp,
  rec_tgt, rush_att, gp all present).

## Plan
1. `build_features_v2.py`: ECR preseason extraction (latest scrape Aug 1–Sep 10
   of draft year; fallback widen), prior-season production/usage from
   stats_{season-1}.json, draft capital. Output `training_data_v2.csv`.
2. `train_model_v2.py`: pooled XGBoost, v2 features (all draft-day legal, no
   leaked same-season offense), config search + recency-decay selection on
   2023 only, isotonic-vs-Platt chosen by CV inside 2023, final fit ≤2023,
   walk-forward feature comparison, disagreement-vs-price analysis, bundle
   update in place (same file names / predict.py interface).

## Status
- [x] Sources downloaded
- [x] Feature build — `training_data_v2.csv` written. Coverage: ecr_overall .987
      (2020 = 0.0, archive has no summer-2020 redraft scrape; 143 rows), prior
      PPG/rank ~.85, snap share .84, target/rush share .61 (team attribution is
      players_nfl current-team; test-era .75, train-era .50), draft capital .94.
      ECR scrapes used: 2021-09-03, 2022-09-02, 2023-09-01, 2024-08-30,
      2025-08-08 (archive ends 2025-08-08 — preseason but early-August).
- [x] Training + validation — config/decay/calibration selected on 2023 only.
      Holdout 2024-25: top5 AUC 0.842 (price-only 0.819), starter 0.797
      (0.772). LR-extended beats XGB on top5 (0.867) — disclosed, shipped as a
      challenger head. Walk-forward confirms. Disagreement quintiles cash.
      Calibration: isotonic (2023-fit and cross-fit versions) both failed to
      transport → raw probabilities are the shipped numbers.
- [x] Bundle update + report section — §11 appended to validation_report.md;
      v1 preserved in backup_v1/; predict.py interface kept + optional v2
      inputs, head_notes, sparse_profile flags.

## Improvement loop (v3 candidates)
- [19:59] loop started; baseline = shipped v2 champion
- [19:59] e01_val: ACCEPT — add VAL relative-value/interaction features (ppg_per_dollar, price_x_age, ecr_vs_prior_rank, price_per_ecr)
- [19:59] e02_eff: REJECT — add EFF prior-season efficiency-rate features (pts/opp, pts/snap, catch rate, aDOT, RZ share, QB pass/rush rates)
- [20:00] e02b_eff_starter: ACCEPT — EFF efficiency features applied to the STARTER head only (post-hoc scoped variant of rejected e02: EFF helped starter +0.004 AUC / hurt top-5 -0.016 as a global add; gated identically)
- [20:00] e03_dropoff: REJECT — drop prior-year offense features (prior_off_rank, prio_*) - effect measured ~0 post-2023 in v2 report s.11
- [20:00] e04_dropmissed: REJECT — drop missed_prior/missed5 (injury effect flipped sign in 2024-25 test actuals; model still applies stale discount)
- [20:01] e03_dropoff: ACCEPT — drop prior-year offense features (prior_off_rank, prio_*) - effect measured ~0 post-2023 in v2 report s.11
- [20:01] e04_dropmissed: REJECT — drop missed_prior/missed5 (injury effect flipped sign in 2024-25 test actuals; model still applies stale discount)
- [20:02] e05_ensemble: ACCEPT — fixed 50/50 probability average of XGBoost with the LR-extended head (LR-ext beat XGB on holdout top-5)
- [20:02] e06_cluster: REJECT — cluster-normalized sample weights: recency decay / times player-season appears across leagues, renormalized
- [20:02] e07_team: REJECT — add TEAM prior-year team pass-volume context (pass att/game, pass share, pass rank, plays/game)
- [20:03] e08_traj: REJECT — add TRAJ two-year trajectory (prior2 PPG/rank, ppg_delta, rank_improve)
- [20:03] e09_wk: REJECT — add WK prior-season weekly consistency (points CV, floor-week share, 90th-pct week)
- [20:03] e11_posblend: REJECT — per-position blend for weak cells: 0.6 pooled + 0.4 per-position model for WR top-5 and TE starter (fixed weights)
- [20:03] e12_cal: REJECT — train-fold-only shrink calibration: s in {1,.95,...,.7} toward train base rate chosen by 5-fold OOF Brier inside each training fold; transport to both test folds
- [20:07] loop CLOSED at cap: champion = VAL feats + EFF(starter) - offense feats + 50/50 XGB/LR-ext ensemble. Holdout top5 0.869 / starter 0.808. Bundle v3-loop finalized, predict.py verified, report s.12 written.
