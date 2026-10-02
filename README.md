# Fantasy Draft Model

XGBoost draft-day model for 10-team auction superflex redraft leagues, trained on 41,659 auction picks (Fantasy 101 league history + two waves of similar-league validation).

Outputs per player: P(top-5 positional finish), P(starter-tier finish), predicted VOR, and implied fair price (`predict.py`).

Headline validation (train ≤2023 → untouched 2024–25 holdout, plus walk-forward):
- Top-5 AUC 0.869 (vs 0.819 price-only), starter-tier AUC 0.808 (vs 0.772)
- Where the model disagrees with the room's price, it's right more often than not — biggest found inefficiency: cheap QB2s.

Key files:
- `predict.py` — scoring interface
- `features.json` — feature list + reliability flags per head
- `validation_report.md` — full validation write-up (honest limitations included)
- `loop_runs/` + `experiment_log.md` — the improvement loop's experiment trail
- `backup_v1/`, `backup_v2/` — earlier shipped bundles

Caveats from the report: treat probabilities as rankings, not odds; the offense-quality and injury "effects" partly leaked or flipped across seasons — sizing anything off them directly would be a mistake.

## Running it

```bash
pip install xgboost pandas numpy scikit-learn
python3 reassemble_models.py   # stitches model_vor.json + model_vor_noprice.json from their parts (one-time, after clone)
```

Then score any player:

```python
from predict import predict_player

predict_player(pos='RB', age=26, price=32, prior_games_missed=1,
               ecr_pos=8, prior_ppg=17.2, prior_rank=6)
```

**Required:** `pos` (QB/RB/WR/TE), `age`, `price` (in your league's budget — set `budget=` if not $200), `prior_games_missed` (None for rookies).
**Strongly recommended:** `ecr_pos`/`ecr_overall` (FantasyPros preseason consensus), `prior_ppg`, `prior_rank` — these are the features that let the model disagree with the room's price instead of echoing it.
**Optional:** snap/target/rush shares, opportunity per game, NFL draft capital, and the v3 efficiency stats (pts_per_opp, catch_rate, adot, etc.). Missing values are handled natively.

Returns `p_top5` / `p_starter` (the ensemble probabilities; also exposed pure as `p_*_xgb` and `p_*_lrext`), predicted VOR, implied fair price on a $200 scale, and `head_notes` flagging which output to trust for that position. Treat probabilities as rankings, not exact odds.

## Retraining

`python3 train_model_v2.py` then `python3 finalize_loop.py` rebuilds the bundle from the local training data (not in this repo — it regenerates from the Sleeper pull scripts). The improvement loop's full experiment trail is in `loop_runs/` + `experiment_log.md`; every accepted change had to win on both the 2024–25 holdout and a walk-forward fold.
