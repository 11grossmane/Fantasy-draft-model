#!/usr/bin/env python3
"""
Finalize the v2 bundle:
  - back up v1 bundle files to backup_v1/
  - cross-fit calibration on <=2023 (5 folds of the chosen config/decay;
    Platt vs isotonic chosen by out-of-fold Brier INSIDE <=2023 only),
    replacing the 2023-phase1 isotonic that failed to transport
  - head-reliability flags per position x target from the holdout
  - overwrite model_*.json / features.json / metrics.json / test_predictions.csv
  - fair-price constant recomputed on <=2023 (same definition as v1)
"""
import json, os, shutil, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss)

ROOT = os.path.expanduser('~/workspace/sleeper-league-data')
OUT = os.path.join(ROOT, 'model')
POS4 = ['QB', 'RB', 'WR', 'TE']
M2 = json.load(open(os.path.join(OUT, 'metrics_v2.json')))
STAGE = json.load(open(os.path.join(OUT, 'features_v2_staging.json')))
FEATS = STAGE['features_v2']
FEATS_NP = STAGE['features_v2_noprice']

# ------------------------------------------------------------ backup v1
BK = os.path.join(OUT, 'backup_v1')
os.makedirs(BK, exist_ok=True)
for f in ['model_top5.json', 'model_starter.json', 'model_vor.json',
          'model_vor_noprice.json', 'features.json', 'metrics.json',
          'test_predictions.csv', 'predict.py', 'importance.json',
          'sanity_checks.json', 'training_data.csv', 'train_model.py']:
    src = os.path.join(OUT, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(BK, f))
print('v1 backed up to backup_v1/', flush=True)

# ------------------------------------------------- cross-fit calibration
df = pd.read_csv(os.path.join(OUT, 'training_data_v2.csv'), low_memory=False)
tr = df[df['season'] <= 2023].copy()
te = pd.read_csv(os.path.join(OUT, 'test_predictions_v2.csv'), low_memory=False)

CLF_BASE = dict(max_depth=4, eta=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=20, reg_lambda=2.0, objective='binary:logistic',
                eval_metric='logloss', nthread=8, seed=7)
D2 = dict(CLF_BASE, max_depth=2, eta=0.03, min_child_weight=30, reg_lambda=5.0)
CH = {t: (D2, M2['chosen'][t]['key'].split('decay=')[1]) for t in ['top5', 'starter']}
CH = {t: (D2, float(M2['chosen'][t]['key'].split('decay=')[1])) for t in CH}
ROUNDS = {t: M2['chosen'][t]['rounds'] for t in CH}

def wts(d, decay, ref):
    return (decay ** (ref - d['season'])).values.astype(float)

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

crossfit = {}
for target in ['top5', 'starter']:
    params, decay = CH[target]
    kf = KFold(5, shuffle=True, random_state=5)
    oof = np.full(len(tr), np.nan)
    idx = tr.index.values
    for tri, tei in kf.split(idx):
        dfit, dho = tr.iloc[tri], tr.iloc[tei]
        m = xgb.train(params, xgb.DMatrix(dfit[FEATS], label=dfit[target],
                                          weight=wts(dfit, decay, 2023)),
                      num_boost_round=ROUNDS[target], verbose_eval=False)
        oof[tei] = m.predict(xgb.DMatrix(dho[FEATS]))
    y = tr[target].values
    out = {'oof_brier': {'raw': float(brier_score_loss(y, oof))}}
    pl = LogisticRegression(max_iter=1000, C=1e6).fit(logit(oof).reshape(-1, 1), y)
    p_pl = pl.predict_proba(logit(oof).reshape(-1, 1))[:, 1]
    out['oof_brier']['platt'] = float(brier_score_loss(y, p_pl))
    ir = IsotonicRegression(out_of_bounds='clip').fit(oof, y)
    out['oof_brier']['isotonic'] = float(brier_score_loss(y, ir.predict(oof)))
    pick = min(('platt', 'isotonic'), key=lambda k: out['oof_brier'][k])
    if out['oof_brier'][pick] >= out['oof_brier']['raw']:
        pick = 'raw'
    out['method'] = pick
    if pick == 'platt':
        out['platt'] = {'coef': float(pl.coef_[0][0]), 'intercept': float(pl.intercept_[0])}
    elif pick == 'isotonic':
        out['isotonic'] = {'x': [float(v) for v in ir.X_thresholds_],
                           'y': [float(v) for v in ir.y_thresholds_]}
    crossfit[target] = out
    print(f'{target}: crossfit calibration -> {pick} (oof brier {out["oof_brier"]})',
          flush=True)

def apply_cf(target, p):
    e = crossfit[target]
    if e['method'] == 'platt':
        c = e['platt']
        return 1.0 / (1.0 + np.exp(-(c['intercept'] + c['coef'] * logit(p))))
    if e['method'] == 'isotonic':
        return np.interp(p, e['isotonic']['x'], e['isotonic']['y'])
    return np.asarray(p, dtype=float)

# ------------------------------------------------- test-set calibration stats
def calib_tbl(y, p, bins=10):
    qs = np.quantile(p, np.linspace(0, 1, bins + 1)); qs[0], qs[-1] = -0.01, 1.01
    idxb = np.digitize(p, qs[1:-1])
    rows, ece = [], 0.0
    for b in range(bins):
        mk = idxb == b
        if mk.sum():
            rows.append(dict(bin=b, n=int(mk.sum()), pred=float(p[mk].mean()),
                             obs=float(y[mk].mean())))
            ece += abs(rows[-1]['pred'] - rows[-1]['obs']) * mk.sum()
    return rows, float(ece / len(y))

def clf_metrics(y, p):
    return dict(auc=float(roc_auc_score(y, p)), prauc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))))

final_metrics = {'v2_holdout_2024_25': {}, 'crossfit_calibration': crossfit}
for target in ['top5', 'starter']:
    raw = te[f'xgb_{target}'].values
    cal = apply_cf(target, raw)
    te[f'xgb_{target}_cfcal'] = cal
    y = te[target].values
    rows_r, ece_r = calib_tbl(y, raw)
    rows_c, ece_c = calib_tbl(y, cal)
    final_metrics['v2_holdout_2024_25'][target] = {
        'xgb_v2_raw': {**clf_metrics(y, raw), 'ece': ece_r},
        'xgb_v2_crossfit_cal': {**clf_metrics(y, cal), 'ece': ece_c},
        'calibration_bins_raw': rows_r, 'calibration_bins_cfcal': rows_c}
    print(f"{target}: raw ECE {ece_r:.3f} Brier {clf_metrics(y, raw)['brier']:.4f} | "
          f"cf-cal ECE {ece_c:.3f} Brier {clf_metrics(y, cal)['brier']:.4f}", flush=True)

# -------------------------------------- model-implied injury gap (real rows)
e = te[(te['price_norm'] >= 10) & (te['has_prior'] == 1)]
gi = e.groupby('missed5')
final_metrics['model_implied_injury_gap_test'] = {
    'mean_xgb_starter_missed5': float(gi['xgb_starter'].mean().loc[1.0]),
    'mean_xgb_starter_durable': float(gi['xgb_starter'].mean().loc[0.0]),
    'actual_hit_missed5': float(gi['starter'].mean().loc[1.0]),
    'actual_hit_durable': float(gi['starter'].mean().loc[0.0])}

# ------------------------------------------------- head reliability flags
flags = {}
for target in ['top5', 'starter']:
    flags[target] = {}
    for p in POS4:
        b = M2['by_position'][target][p]
        x, lp, le = b['xgb_v2']['auc'], b['lr_price_only']['auc'], b['lr_extended']['auc']
        best = max([('xgb', x), ('lr_price', lp), ('lr_extended', le)], key=lambda t: t[1])
        ci_w = b['xgb_v2_auc_ci95'][1] - b['xgb_v2_auc_ci95'][0]
        flags[target][p] = {
            'xgb_auc': x, 'lr_price_auc': lp, 'lr_ext_auc': le,
            'xgb_auc_ci95': b['xgb_v2_auc_ci95'], 'best_tool': best[0],
            'note': ('very wide CI - ranking only, do not treat as odds'
                     if ci_w > 0.3 else '')}
final_metrics['head_reliability'] = flags

# ------------------------------------------------------------- fair price K
pos_vor = tr[tr['vor'] > 0]
K_FAIR = float((pos_vor['price_norm'] - 1).sum() / pos_vor['vor'].sum())
final_metrics['fair_price_k'] = K_FAIR

# lr_extended coefficients + medians (train) for predict.py
lre_meta = {}
for t in ['top5', 'starter']:
    lr = LogisticRegression(max_iter=1000, C=1.0)
    feats = ['price_norm', 'ecr_pos', 'prior_ppg_ppr', 'age', 'missed_prior']
    med = tr[feats].median()
    lr.fit(tr[feats].fillna(med), tr[t])
    lre_meta[t] = {'features': feats, 'coef': [float(c) for c in lr.coef_[0]],
                   'intercept': float(lr.intercept_[0]),
                   'medians': {f: float(med[f]) for f in feats}}
final_metrics['lr_extended'] = lre_meta

# ------------------------------------------------------------- persist
for name in ['top5', 'starter', 'vor', 'vor_noprice']:
    shutil.move(os.path.join(OUT, f'model_{name}_v2.json'),
                os.path.join(OUT, f'model_{name}.json'))
os.remove(os.path.join(OUT, 'features_v2_staging.json'))

features_json = {
    'version': 'v2',
    'features': FEATS, 'features_noprice': FEATS_NP,
    'positions': POS4,
    'targets': {'top5': 'pos_rank <= 5 in the player\'s league that season',
                'starter': 'season points >= league replacement points (VOR >= 0)',
                'vor': 'season points minus league replacement points (league scoring)'},
    'train_seasons': '<=2023', 'test_seasons': '2024-2025',
    'selection': {t: M2['chosen'][t] for t in ['top5', 'starter']},
    'calibration_crossfit': crossfit,
    'platt_calibration_legacy_v1': None,
    'head_reliability': flags,
    'lr_extended': lre_meta,
    'fair_price_k_dollars_per_vor': K_FAIR,
    'xgb_version': xgb.__version__,
    'notes': ('v2 adds price-independent features (FantasyPros preseason ECR, '
              'prior-season production/usage, draft capital). p_*_cal outputs use '
              'the 5-fold cross-fit calibrator fit on <=2023 OOF predictions. '
              'head_reliability.best_tool says which tool won each position x '
              'target cell on the 2024-25 holdout. See validation_report.md s.11.'),
}
json.dump(features_json, open(os.path.join(OUT, 'features.json'), 'w'), indent=1)

merged = dict(M2)
merged.update(final_metrics)
json.dump(merged, open(os.path.join(OUT, 'metrics_v2.json'), 'w'), indent=1,
          default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
json.dump(merged, open(os.path.join(OUT, 'metrics.json'), 'w'), indent=1,
          default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
te.to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)

print('bundle overwritten: model_top5/starter/vor/vor_noprice.json, features.json, '
      'metrics.json, test_predictions.csv', flush=True)
print('DONE', flush=True)
