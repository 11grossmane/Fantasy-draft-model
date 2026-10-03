#!/usr/bin/env python3
"""FAAB clearing-price + rival model for Fantasy 101.
Frozen approach (written before testing):
 - Price-to-beat model: quantile HistGradientBoosting on winning bids of
   auctions NOT won by Eric (= max rival bid), quantiles .1/.25/.5/.75/.9.
 - Features (knowable pre-waiver): position, week, phase, prev_week_pts,
   ppg_before, games_before, prev_pts_pct, league FAAB landscape.
 - Rival module: per-manager historical bid propensity x position bid
   distribution (shrunk to league position distribution).
 - Split: train 2020-2024, cold-test 2025-2026. Baselines: position median,
   position x phase median. Report band coverage, MAE, pinball vs baselines,
   and rival top-3 hit rate. Failure reported plainly if baselines win.
"""
import json, os, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.expanduser('~/workspace/sleeper-league-data/faab')
ERIC = 'gizzle4'
df = pd.read_csv(os.path.join(ROOT, 'faab_bids_enriched.csv'), dtype={'player_id': str})
canon = (df.sort_values('season').groupby('bidder_user_id')['bidder_manager']
         .agg(lambda s: s.iloc[-1]))
df['manager'] = df['bidder_user_id'].map(canon)

# ---------- auction frame ----------
def phase(w):
    return 'early' if w <= 5 else ('mid' if w <= 10 else 'late')

au = (df.sort_values('bid', ascending=False).drop_duplicates('auction_id')
      [['auction_id', 'season', 'week', 'player', 'position', 'winning_bid',
        'num_bidders', 'prev_week_pts', 'ppg_before', 'games_before', 'prev_pts_pct']]
      .rename(columns={'winning_bid': 'win_bid'}))
# winner identity per auction
winners = df[df.won].set_index('auction_id')[['manager', 'bid', 'second_bid']]
au = au.join(winners, on='auction_id')
au['winner'] = au['manager']
au['phase'] = au.week.map(phase)
au['has_winner'] = au.win_bid.notna()
print(f'auctions total {len(au)}, with winner {int(au.has_winner.sum())}, '
      f'all-failed {int((~au.has_winner).sum())}')
au = au[au.has_winner].copy()
# league FAAB landscape per auction (from bidder rows; approx = bidders' landscape)
land = (df.groupby('auction_id')['faab_remaining_before']
        .agg(['max', 'median']).rename(columns={'max': 'faab_max', 'median': 'faab_med'}))
au = au.join(land, on='auction_id')

POS = ['QB', 'RB', 'WR', 'TE', 'DEF']
for p in POS:
    au[f'pos_{p}'] = (au.position == p).astype(int)
au['is_superflex_qb_need'] = 0  # placeholder kept for schema stability
FEATS = (['week', 'prev_week_pts', 'ppg_before', 'games_before', 'prev_pts_pct',
          'faab_max', 'faab_med'] + [f'pos_{p}' for p in POS] +
         ['phase_mid', 'phase_late'])
au['phase_mid'] = (au.phase == 'mid').astype(int)
au['phase_late'] = (au.phase == 'late').astype(int)

train = au[(au.season <= 2024) & (au.winner != ERIC)]
test = au[(au.season >= 2025) & (au.winner != ERIC)]
print(f'price-to-beat train auctions: {len(train)}  test: {len(test)}')

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
models = {}
for q in QUANTILES:
    m = HistGradientBoostingRegressor(loss='quantile', quantile=q,
                                      max_iter=300, learning_rate=0.06,
                                      max_depth=3, min_samples_leaf=20,
                                      random_state=7)
    m.fit(train[FEATS], train.win_bid)
    models[q] = m

preds = pd.DataFrame({f'q{int(q*100)}': models[q].predict(test[FEATS]) for q in QUANTILES})
preds.index = test.index
test = pd.concat([test, preds], axis=1)
for c in ['q10', 'q25', 'q50', 'q75', 'q90']:
    test[c] = test[c].clip(lower=0)

def pinball(y, yhat, q):
    d = y - yhat
    return np.mean(np.maximum(q * d, (q - 1) * d))

# baselines from train
base_pos = train.groupby('position').win_bid.median()
base_pp = train.groupby(['position', 'phase']).win_bid.median()
test['base_pos'] = test.position.map(base_pos)
test['base_pp'] = [base_pp.get((p, ph), base_pos.get(p, 4))
                   for p, ph in zip(test.position, test.phase)]

res = {}
res['band50_coverage'] = float(((test.win_bid >= test.q25) & (test.win_bid <= test.q75)).mean())
res['band80_coverage'] = float(((test.win_bid >= test.q10) & (test.win_bid <= test.q90)).mean())
res['mae_q50'] = float(np.abs(test.win_bid - test.q50).mean())
res['mae_base_pos'] = float(np.abs(test.win_bid - test.base_pos).mean())
res['mae_base_pp'] = float(np.abs(test.win_bid - test.base_pp).mean())
res['pinball_q50_model'] = float(pinball(test.win_bid, test.q50, 0.5))
res['pinball_q50_basepp'] = float(pinball(test.win_bid, test.base_pp, 0.5))
res['pinball_q75_model'] = float(pinball(test.win_bid, test.q75, 0.75))
res['pinball_q75_basepp'] = float(pinball(test.win_bid, test.base_pp, 0.75))
# bucket calibration: actual winning bid vs predicted quantile level
res['frac_actual_above_q75'] = float((test.win_bid > test.q75).mean())   # ideal .25
res['frac_actual_above_q90'] = float((test.win_bid > test.q90).mean())   # ideal .10
res['test_n'] = int(len(test))
print(json.dumps(res, indent=1))

# by-position coverage
cov = (test.assign(in50=(test.win_bid >= test.q25) & (test.win_bid <= test.q75))
       .groupby('position').agg(n=('win_bid', 'size'), cov50=('in50', 'mean'),
                                mae=('q50', lambda s: 0)))
print(cov[['n', 'cov50']].to_string())

# ---------- rival module ----------
riv_rows = []
train_bids = df[df.season <= 2024]
test_auctions = au[au.season >= 2025]
league_pos_q = (train_bids.groupby('position').bid
                .quantile([0.25, 0.5, 0.75]).unstack())
for (mgr, pos), g in train_bids.groupby(['manager', 'position']):
    n_pos_auctions = max(1, (train.position == pos).sum())
    qs = g.bid.quantile([0.25, 0.5, 0.75])
    wgt = len(g) / (len(g) + 12)  # shrinkage toward league position quantiles
    riv_rows.append(dict(manager=mgr, position=pos, n_bids=len(g),
                         p_bid=len(g) / n_pos_auctions,
                         q25=wgt * qs.get(0.25, 0) + (1 - wgt) * league_pos_q.loc[pos, 0.25],
                         med=wgt * qs.get(0.5, 0) + (1 - wgt) * league_pos_q.loc[pos, 0.5],
                         q75=wgt * qs.get(0.75, 0) + (1 - wgt) * league_pos_q.loc[pos, 0.75],
                         pct_49=(g.bid % 10).isin([4, 9]).mean(),
                         pct_05=(g.bid % 10).isin([0, 5]).mean()))
riv = pd.DataFrame(riv_rows)

MANAGERS = sorted(df.manager.unique())
def predict_rivals(position, top=3):
    cand = []
    for mgr in MANAGERS:
        if mgr == ERIC:
            continue
        r = riv[(riv.manager == mgr) & (riv.position == position)]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        cand.append(dict(manager=mgr, p_bid=float(r.p_bid), med=float(r.med),
                         q25=float(r.q25), q75=float(r.q75), score=float(r.p_bid) * float(r.med)))
    return sorted(cand, key=lambda c: -c['score'])[:top]

hits, inrange, n_eval = 0, 0, 0
for _, a in test_auctions.iterrows():
    top3 = predict_rivals(a.position)
    names = [c['manager'] for c in top3]
    if a.winner in names:
        hits += 1
        c = [c for c in top3 if c['manager'] == a.winner][0]
        if c['q25'] - 1 <= a.win_bid <= c['q75'] + 1:
            inrange += 1
    n_eval += 1
res['rival_top3_hit_rate'] = hits / max(1, n_eval)
res['rival_top3_chance'] = 3 / 9  # 9 rival managers
res['winner_bid_in_range_when_top3'] = inrange / max(1, hits)
res['rival_eval_n'] = n_eval
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()
       if k.startswith('rival')})

# ---------- persist ----------
os.makedirs(os.path.join(ROOT, 'model'), exist_ok=True)
with open(os.path.join(ROOT, 'model', 'quantile_models.pkl'), 'wb') as f:
    pickle.dump({'models': models, 'feats': FEATS, 'pos': POS}, f)
riv.to_csv(os.path.join(ROOT, 'model', 'rival_tables.csv'), index=False)
league_pos_q.to_csv(os.path.join(ROOT, 'model', 'league_position_quantiles.csv'))
with open(os.path.join(ROOT, 'model', 'validation.json'), 'w') as f:
    json.dump(res, f, indent=1)
test[['season', 'week', 'player', 'position', 'win_bid', 'winner', 'num_bidders',
      'q10', 'q25', 'q50', 'q75', 'q90', 'base_pos', 'base_pp']].to_csv(
    os.path.join(ROOT, 'model', 'test_predictions.csv'), index=False)
print('saved model artifacts')
