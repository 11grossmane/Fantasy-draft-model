# Fantasy-draft-model

Draft-day and FAAB models for a 10-team superflex auction league, trained
on Fantasy 101 (Eric's league) plus cross-league Sleeper data.

## Models

- **Draft model** (XGBoost ensemble + logistic heads): predicts
  P(top-5 finish), P(starter), and value over replacement from
  draft-day inputs. Cold-validated on untouched 2024-25 seasons
  (top-5 AUC 0.869, starter AUC 0.808). See `validation_report.md`.
- **FAAB predictor** (`faab/`): quantile price-to-beat model for
  Wednesday waivers in Fantasy 101, trained on 2,404 bids (winners and
  losers). Rival identification is at/below chance (~29% top-3 vs 33%);
  the calibrated bid ranges are the validated part. See `faab/README.md`.

## Web app

The tool's UI lives in `site/` - a fully static site (all model math
runs in the browser) deployed to GitHub Pages. Every push to `main`
that touches `site/` redeploys automatically
(`.github/workflows/pages.yml`). A weekly Action
(`.github/workflows/refresh.yml`) re-pulls Sleeper player data and
commits a fresh `site/assets/data/players.json`.

Live: https://11grossmane.github.io/Fantasy-draft-model/

## Retraining

- Draft model: retrain scripts live alongside the model bundle.
- FAAB model: `python faab/model.py`, then re-export quantiles.
- The trained pickle `faab/model/quantile_models.pkl` (~1 MB) is not
  in git; rebuild it with the faab pipeline.
