# FAAB Bid Predictor — Fantasy 101 (Eric's league only)

Predicts what a waiver target will cost in Fantasy 101's Wednesday FAAB
auctions, from all 2,404 recorded bids (2020–2026, losers included).

## Files
- `faab_bids.csv` / `faab_bids_enriched.csv` — every bid: season, week,
  player, position, bidder, bid, won/lost, FAAB remaining before, plus
  demand features (prior-week points, season PPG, positional percentile).
- `FAAB_AUDIT.md` — descriptive fingerprints + model validation writeup.
- `manager_fingerprints.csv`, `manager_position_medians.csv` — audit tables.
- `faab_predict.py` — the tool. `model/` — trained quantile models,
  rival tables, validation metrics, test predictions.
- Pipeline: `collect.py` (Sleeper pull) → `build_dataset.py` →
  `fetch_stats.py` → `enrich.py` → `audit.py` / `model.py`. Raw JSON in
  `raw/`. Re-running fetch/collect is resumable (cached files are skipped).

## Using the tool
```python
from faab_predict import predict_player_faab, predict_position_faab
predict_player_faab('Jaylen Warren')   # current 2026 form auto-filled
predict_position_faab('RB', week=6, prev_week_pts=14.2, ppg_before=11.0,
                      prev_pts_pct=0.9)
```
Outputs: predicted clearing-price quantiles, Eric's needed bid at
~50/75/90% win probability (quantile + $1), and the top-3 habitual rival
bidders with their typical bid ranges and digit habits.

## Limitations (read before bidding real FAAB)
- Rival *identity* prediction validated at ~34% top-3 vs 33% chance —
  the names are context, not a call. Bid amounts are the reliable part.
- QB predictions are the weakest cell; small samples throughout (the
  league makes ~145 waiver auctions a season).
- FAAB-remaining is reconstructed, not official; check Sleeper for the
  live number before a big bid.
- Probabilities are calibrated on 2025–26 (n=173); the 50% band is
  conservative (71% actual coverage).
