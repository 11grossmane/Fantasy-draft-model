#!/usr/bin/env python3
"""
Finalize the improvement-loop champion into the shipped bundle.

Champion recipe (loop_state.json): pooled XGBoost D_d2_reg, decays
top5 0.7 / starter 0.85, features = v2 - prior-offense + VAL (top5) and
+ EFF (starter), predictions = 0.5 * XGB + 0.5 * LR-extended ensemble.
Calibration: raw (loop e12: train-fold OOF chose shrink s=1.0 everywhere).

Backs the v2 bundle up to backup_v2/, retrains with the exact loop recipe,
recomputes head reliability + cluster-bootstrap CIs + disagreement table,
and rewrites model_*.json / features.json / metrics.json /
test_predictions.csv.
"""
import json, os, shutil, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import loop_lib as L

OUT = L.OUT
RNG = np.random.default_rng(23)

# ------------------------------------------------------------ backup v2
BK = os.path.join(OUT, 'backup_v2')
os.makedirs(BK, exist_ok=True)
for f in ['model_top5.json', 'model_starter.json', 'model_vor.json',
          'model_vor_noprice.json', 'features.json', 'metrics.json',
          'metrics_v2.json', 'test_predictions.csv', 'predict.py',
          'train_model_v2.py', 'finalize_v2.py', 'training_data_v2.csv']:
    src = os.path.join(OUT, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(BK, f))
print('v2 backed up to backup_v2/', flush=True)

state = json.load(open(L.STATE_F))
spec = state['champion_spec']
df = L.load_df()
tr23 = df[df['season'] <= 2023]
te = df[df['season'] >= 2024].copy()

# ------------------------------------------------- retrain via loop recipe
final = {}
pred_cols = {}
for target in ('top5', 'starter'):
    metrics, store = L.eval_target(df, spec, target)
    final[target] = store['ho_model']
    pred_cols[f'xgb_{target}'] = store['preds']['holdout']
    # LR pieces on the holdout with <=2023 fits (for ensemble + decomposition)
    feats_t = (spec.get('feats_by_target') or {}).get(target, spec['feats'])
    pred_cols[f'lre_{target}'] = L.fit_lr(tr23, te, L.LR_FEATS, target)
    pred_cols[f'lrp_{target}'] = L.fit_lr(tr23, te, ['price_norm'], target)
    pred_cols[f'ens_{target}'] = 0.5 * pred_cols[f'xgb_{target}'] \
        + 0.5 * pred_cols[f'lre_{target}']
    chk = metrics['holdout']
    print(f"{target}: holdout AUC {chk['auc']:.4f} Brier {chk['brier']:.4f} "
          f"(loop champion check)", flush=True)

# VOR with the champion's shared (top5) feature list
vor_res, vor_models = L.eval_vor(df, spec)
print('VOR holdout:', json.dumps(vor_res['holdout'], default=str), flush=True)

# ------------------------------------------------------- persist boosters
final['top5'].save_model(os.path.join(OUT, 'model_top5.json'))
final['starter'].save_model(os.path.join(OUT, 'model_starter.json'))
vor_models['with_price'].save_model(os.path.join(OUT, 'model_vor.json'))
vor_models['profile_only'].save_model(os.path.join(OUT, 'model_vor_noprice.json'))
print('boosters saved', flush=True)

# ------------------------------------------- per-position reliability + CIs
def cluster_boot_auc(d, ycol, pcol, n=200):
    key = d['player_id'].astype(str) + '_' + d['season'].astype(str)
    idx_by = d.groupby(key).indices
    keys = list(idx_by.keys())
    aucs = []
    yv, pv = d[ycol].values, d[pcol].values
    for _ in range(n):
        pick = RNG.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([idx_by[keys[i]] for i in pick])
        yy, pp = yv[rows], pv[rows]
        if yy.min() != yy.max():
            aucs.append(roc_auc_score(yy, pp))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]

for k, v in pred_cols.items():
    te[k] = v
flags = {}
for target in ('top5', 'starter'):
    flags[target] = {}
    for p in L.POS4:
        s = te[te['pos'] == p]
        aucs = {'xgb': roc_auc_score(s[target], s[f'xgb_{target}']),
                'ensemble': roc_auc_score(s[target], s[f'ens_{target}']),
                'lr_price': roc_auc_score(s[target], s[f'lrp_{target}']),
                'lr_extended': roc_auc_score(s[target], s[f'lre_{target}'])}
        best = max(aucs, key=aucs.get)
        ci = cluster_boot_auc(s, target, f'ens_{target}')
        note = ''
        if ci[1] - ci[0] > 0.3:
            note = 'very wide CI - ranking only, do not treat as odds'
        elif best != 'ensemble':
            note = f'p_* ships the ensemble; {best} was marginally better here'
        flags[target][p] = {'xgb_auc': float(aucs['xgb']),
                            'ensemble_auc': float(aucs['ensemble']),
                            'lr_price_auc': float(aucs['lr_price']),
                            'lr_ext_auc': float(aucs['lr_extended']),
                            'ensemble_auc_ci95': ci,
                            'best_tool': best, 'note': note}
        print(f'{target}/{p}: ens {aucs["ensemble"]:.3f} xgb {aucs["xgb"]:.3f} '
              f'lre {aucs["lr_extended"]:.3f} -> {best}', flush=True)

# ------------------------------------------------------- disagreement table
disagree = {}
for target in ('top5', 'starter'):
    d = te.copy()
    d['delta'] = d[f'ens_{target}'] - d[f'lrp_{target}']
    d['q'] = pd.qcut(d['delta'], 5, labels=False, duplicates='drop')
    rows = []
    for q, g in d.groupby('q'):
        rows.append({'quintile': int(q), 'n': int(len(g)),
                     'mean_price': float(g['price_norm'].mean()),
                     'actual_minus_lr': float(g[target].mean()
                                              - g[f'lrp_{target}'].mean())})
    disagree[target] = rows

# ------------------------------------------------------------- fair-price K
pos_vor = tr23[tr23['vor'] > 0]
K_FAIR = float((pos_vor['price_norm'] - 1).sum() / pos_vor['vor'].sum())
K_POS = {p: float((g['price_norm'] - 1).sum() / g['vor'].sum())
         for p, g in pos_vor.groupby('pos')}

# ------------------------------------------------------------- LR-ext meta
lre_meta = {}
for t in ('top5', 'starter'):
    lr = LogisticRegression(max_iter=1000, C=1.0)
    med = tr23[L.LR_FEATS].median()
    lr.fit(tr23[L.LR_FEATS].fillna(med), tr23[t])
    lre_meta[t] = {'features': L.LR_FEATS,
                   'coef': [float(c) for c in lr.coef_[0]],
                   'intercept': float(lr.intercept_[0]),
                   'medians': {f: float(med[f]) for f in L.LR_FEATS}}

feats_top5 = (spec.get('feats_by_target') or {}).get('top5', spec['feats'])
feats_starter = (spec.get('feats_by_target') or {}).get('starter',
                                                        spec['feats'])
features_json = {
    'version': 'v3-loop',
    'features': feats_top5,
    'features_by_target': {'top5': feats_top5, 'starter': feats_starter},
    'features_noprice': [f for f in feats_top5 if f != 'price_norm'],
    'positions': L.POS4,
    'targets': {'top5': "pos_rank <= 5 in the player's league that season",
                'starter': 'season points >= league replacement points '
                           '(VOR >= 0)',
                'vor': 'season points minus league replacement points '
                       '(league scoring)'},
    'train_seasons': '<=2023', 'test_seasons': '2024-2025',
    'selection': {
        'recipe': 'improvement loop champion (see validation_report.md s.12)',
        'params': spec['params'], 'decay': spec['decay'],
        'ensemble': {'xgb_weight': spec['ensemble_w'],
                     'other': 'lr_extended 5-feature logistic',
                     'note': 'p_top5/p_starter in predict.py are the '
                             '50/50 ensemble; p_*_xgb are the pure trees'},
        'accepted_experiments': ['e01_val', 'e02b_eff_starter',
                                 'e03_dropoff', 'e05_ensemble'],
    },
    'calibration_crossfit': {
        t: {'method': 'raw',
            'note': 'loop e12: train-fold OOF chose shrink s=1.0 in all '
                    'cells; raw probabilities ship. See report s.12.'}
        for t in ('top5', 'starter')},
    'head_reliability': flags,
    'lr_extended': lre_meta,
    'fair_price_k_dollars_per_vor': K_FAIR,
    'fair_price_k_by_position': K_POS,
    'xgb_version': xgb.__version__,
    'notes': ('v3 (improvement loop): VAL relative-value features in, '
              'prior-year offense features out, EFF efficiency features for '
              'the starter head, and p_* are a 50/50 XGBoost + LR-extended '
              'ensemble. Calibration: raw probabilities (no recalibrator '
              'transported; loop e12). See validation_report.md s.12.'),
}
json.dump(features_json, open(os.path.join(OUT, 'features.json'), 'w'), indent=1)

champ = json.load(open(os.path.join(L.RUNS_D, '_champion.json')))['results']
metrics_out = {'loop_champion_dual_fold': {k: v for k, v in champ.items()
                                           if k in ('top5', 'starter', 'vor')},
               'head_reliability': flags, 'disagreement_holdout': disagree,
               'fair_price_k': K_FAIR, 'fair_price_k_by_position': K_POS}
json.dump(metrics_out, open(os.path.join(OUT, 'metrics.json'), 'w'), indent=1,
          default=str)
json.dump(metrics_out, open(os.path.join(OUT, 'metrics_loop.json'), 'w'),
          indent=1, default=str)

cols = ['season', 'league_id', 'player_id', 'player', 'pos', 'price_norm',
        'top5', 'starter', 'vor', 'xgb_top5', 'xgb_starter', 'ens_top5',
        'ens_starter', 'lre_top5', 'lre_starter', 'lrp_top5', 'lrp_starter']
te[cols].to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)
imp = {}
for name, m in [('top5', final['top5']), ('starter', final['starter'])]:
    g = m.get_score(importance_type='gain')
    tot = sum(g.values()) or 1.0
    imp[name] = {k: round(float(v / tot), 4)
                 for k, v in sorted(g.items(), key=lambda x: -x[1])}
json.dump(imp, open(os.path.join(OUT, 'importance_loop.json'), 'w'), indent=1)
print('importance top5:', dict(list(imp['top5'].items())[:8]))
print('importance starter:', dict(list(imp['starter'].items())[:8]))
print('disagreement starter quintiles:',
      [round(r['actual_minus_lr'], 3) for r in disagree['starter']])
print('DONE')
