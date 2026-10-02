#!/usr/bin/env python3
"""
v2 improvement pass for the draft-day XGBoost bundle.

Changes vs v1 (see validation_report.md for v1 numbers):
  - New draft-day-legal, price-independent features: FantasyPros preseason ECR
    (overall + positional + expert sd), prior-season PPR points/PPG/NFL rank,
    snap share, target share, rush share, opportunities/game, NFL draft
    capital, age^2, expensive-youth flag. (Built by build_features_v2.py.
    Same-season offense features remain banned as leakage; prior-year offense
    rank/tier from v1 stays.)
  - Recency weighting: sample-weight decay per season, selected on 2023 only.
  - Calibration: none vs Platt vs isotonic, chosen by cross-validation INSIDE
    the 2023 fold (calibrator fit on <=2022 model predictions on 2023).
  - Walk-forward check: fixed v1 config, fit <=2022, v1 vs v2 features -> 2023.
  - Disagreement analysis on the 2024-25 holdout: do picks where the model
    rates a player far above his price actually cash?

Protocol: config search, decay, calibration choice, and ablations all happen
on 2023 (fit <=2022). Final models refit on <=2023 and are evaluated ONCE on
2024-25. The one imperfection, disclosed: v1's results on 2024-25 were known
when designing v2; the split discipline itself is unchanged.

Outputs: metrics_v2.json, test_predictions_v2.csv, new model boosters +
features_v2.json staging, disagreement tables, printed summary.
"""
import json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, mean_absolute_error, r2_score)

ROOT = os.path.expanduser('~/workspace/sleeper-league-data')
OUT = os.path.join(ROOT, 'model')
POS4 = ['QB', 'RB', 'WR', 'TE']
RNG = np.random.default_rng(11)

df = pd.read_csv(os.path.join(OUT, 'training_data_v2.csv'), low_memory=False)

FEATS_V1 = ['price_norm', 'age', 'missed_prior', 'missed5', 'has_prior',
            'prior_off_rank', 'prio_top', 'prio_mid', 'prio_bottom',
            'pos_QB', 'pos_RB', 'pos_WR', 'pos_TE']
NEW = ['ecr_overall', 'ecr_pos', 'ecr_sd', 'ecr_overall_log', 'ecr_pos_log',
       'prior_pts_ppr', 'prior_ppg_ppr', 'prior_rank_ppr', 'prior_snap_share',
       'prior_target_share', 'prior_rush_share', 'prior_opp_pg',
       'years_exp', 'draft_round', 'draft_pick', 'age_sq', 'exp_youth',
       'ppg_x_snap']
FEATS_V2 = FEATS_V1 + NEW
FEATS_V2_NOPRICE = [f for f in FEATS_V2 if f != 'price_norm']
print(f'rows {len(df)}; v2 features {len(FEATS_V2)}', flush=True)

tr = df[df['season'] <= 2023].copy()
te = df[df['season'] >= 2024].copy()
tr_fit, tr_es = tr[tr['season'] <= 2022].copy(), tr[tr['season'] == 2023].copy()
print(f'train<=2023 {len(tr)} (fit<=2022 {len(tr_fit)}, es2023 {len(tr_es)}), '
      f'test {len(te)}', flush=True)

CLF_BASE = dict(max_depth=4, eta=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=20, reg_lambda=2.0, objective='binary:logistic',
                eval_metric='logloss', nthread=8, seed=7)
MONO = tuple(1 if f == 'price_norm' else 0 for f in FEATS_V2)
CLF_CONFIGS = {
    'A_d4_base': dict(CLF_BASE),
    'B_d3_reg': dict(CLF_BASE, max_depth=3, eta=0.03, min_child_weight=50, reg_lambda=5.0),
    'C_d4_mono_price': dict(CLF_BASE, monotone_constraints=MONO),
    'D_d2_reg': dict(CLF_BASE, max_depth=2, eta=0.03, min_child_weight=30, reg_lambda=5.0),
}
DECAYS = [1.0, 0.85, 0.7]

def weights(d, decay, ref_season):
    return (decay ** (ref_season - d['season'])).astype(float).values if decay < 1.0 else None

def fit_eval(params, Xf, yf, wf, Xes, yes, feats):
    m = xgb.train(params, xgb.DMatrix(Xf[feats], label=yf, weight=wf),
                  num_boost_round=2000,
                  evals=[(xgb.DMatrix(Xes[feats], label=yes), 'es')],
                  early_stopping_rounds=100, verbose_eval=False)
    return m

def refit(params, X, y, w, feats, rounds):
    return xgb.train(params, xgb.DMatrix(X[feats], label=y, weight=w),
                     num_boost_round=rounds, verbose_eval=False)

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def clf_metrics(y, p):
    return dict(auc=float(roc_auc_score(y, p)),
                prauc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
                base_rate=float(np.mean(y)))

# baselines fit on <=2023 (v1 protocol) -------------------------------------
def lr_fit(feats, target):
    lr = LogisticRegression(max_iter=1000, C=1.0)
    X = tr[feats].fillna(tr[feats].median())
    lr.fit(X, tr[target])
    return lr

def lr_pred(lr, feats, d):
    return lr.predict_proba(d[feats].fillna(tr[feats].median()))[:, 1]

lr_price, lr_ext = {}, {}
for t in ['top5', 'starter']:
    lr_price[t] = lr_fit(['price_norm'], t)
    lr_ext[t] = lr_fit(['price_norm', 'ecr_pos', 'prior_ppg_ppr', 'age', 'missed_prior'], t)
    te[f'lrp_{t}'] = lr_pred(lr_price[t], ['price_norm'], te)
    te[f'lre_{t}'] = lr_pred(lr_ext[t], ['price_norm', 'ecr_pos', 'prior_ppg_ppr',
                                         'age', 'missed_prior'], te)

results = {}

# ---------------------------------------------------------------------------
# 1. Config + decay search on 2023 (fit <=2022), v2 features
# ---------------------------------------------------------------------------
chosen = {}
for target in ['top5', 'starter']:
    grid = {}
    fitted = {}
    for cname, params in CLF_CONFIGS.items():
        for decay in DECAYS:
            w = weights(tr_fit, decay, 2022)
            m = fit_eval(params, tr_fit, tr_fit[target], w, tr_es, tr_es[target], FEATS_V2)
            p = m.predict(xgb.DMatrix(tr_es[FEATS_V2]))
            grid[f'{cname}|decay={decay}'] = float(log_loss(tr_es[target], p))
            fitted[f'{cname}|decay={decay}'] = (m, params, decay)
    best = min(grid, key=grid.get)
    m0, params, decay = fitted[best]
    chosen[target] = dict(key=best, params=params, decay=decay,
                          rounds=int(max(m0.best_iteration + 50, 50)), grid=grid,
                          phase1=m0)
    print(f'{target}: chosen {best} rounds={chosen[target]["rounds"]} '
          f'(grid best 5: {sorted(grid.items(), key=lambda x: x[1])[:5]})', flush=True)

# ---------------------------------------------------------------------------
# 2. Ablations on 2023 (fixed chosen config/decay): what do new features add?
# ---------------------------------------------------------------------------
ablation = {}
ABL = {'full_v2': FEATS_V2, 'v1_feats': FEATS_V1,
       'no_ecr': [f for f in FEATS_V2 if not f.startswith('ecr')],
       'no_priorprod': [f for f in FEATS_V2 if f not in
                        ('prior_pts_ppr', 'prior_ppg_ppr', 'prior_rank_ppr',
                         'prior_snap_share', 'prior_target_share',
                         'prior_rush_share', 'prior_opp_pg', 'ppg_x_snap')]}
for target in ['top5', 'starter']:
    ch = chosen[target]
    ablation[target] = {}
    for name, feats in ABL.items():
        w = weights(tr_fit, ch['decay'], 2022)
        m = fit_eval(ch['params'], tr_fit, tr_fit[target], w, tr_es, tr_es[target], feats)
        p = m.predict(xgb.DMatrix(tr_es[feats]))
        ablation[target][name] = dict(
            auc=float(roc_auc_score(tr_es[target], p)),
            logloss=float(log_loss(tr_es[target], p)),
            brier=float(brier_score_loss(tr_es[target], p)))
    print(f'{target} ablations 2023: ' +
          ', '.join(f"{k} auc {v['auc']:.3f}" for k, v in ablation[target].items()),
          flush=True)

# ---------------------------------------------------------------------------
# 3. Calibration choice, cross-validated INSIDE 2023 predictions
# ---------------------------------------------------------------------------
def calib_cv_choose(y, p):
    kf = KFold(5, shuffle=True, random_state=3)
    scores = {'raw': [], 'platt': [], 'isotonic': []}
    for tri, tei in kf.split(p):
        scores['raw'].append(brier_score_loss(y[tei], p[tei]))
        pl = LogisticRegression(max_iter=1000, C=1e6)
        pl.fit(logit(p[tri]).reshape(-1, 1), y[tri])
        scores['platt'].append(brier_score_loss(
            y[tei], pl.predict_proba(logit(p[tei]).reshape(-1, 1))[:, 1]))
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(p[tri], y[tri])
        scores['isotonic'].append(brier_score_loss(y[tei], ir.predict(p[tei])))
    cv = {k: float(np.mean(v)) for k, v in scores.items()}
    # isotonic only wins if it actually helps; otherwise prefer platt (stable)
    pick = 'isotonic' if cv['isotonic'] < min(cv['platt'], cv['raw']) - 0.0005 else \
           ('platt' if cv['platt'] <= cv['raw'] else 'raw')
    return pick, cv

calibration = {}
for target in ['top5', 'starter']:
    ch = chosen[target]
    p_es = ch['phase1'].predict(xgb.DMatrix(tr_es[FEATS_V2]))
    y_es = tr_es[target].values
    pick, cv = calib_cv_choose(y_es, p_es)
    entry = {'method': pick, 'cv_brier_on_2023': cv}
    if pick == 'platt':
        pl = LogisticRegression(max_iter=1000, C=1e6)
        pl.fit(logit(p_es).reshape(-1, 1), y_es)
        entry['platt'] = {'coef': float(pl.coef_[0][0]), 'intercept': float(pl.intercept_[0])}
    elif pick == 'isotonic':
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(p_es, y_es)
        entry['isotonic'] = {'x': [float(v) for v in ir.X_thresholds_],
                             'y': [float(v) for v in ir.y_thresholds_]}
    calibration[target] = entry
    print(f'{target}: calibration -> {pick} (cv brier {cv})', flush=True)

def apply_cal(target, p):
    e = calibration[target]
    if e['method'] == 'platt':
        c = e['platt']
        return 1.0 / (1.0 + np.exp(-(c['intercept'] + c['coef'] * logit(p))))
    if e['method'] == 'isotonic':
        return np.interp(p, e['isotonic']['x'], e['isotonic']['y'])
    return p

# ---------------------------------------------------------------------------
# 4. Final fits on <=2023 + holdout evaluation (2024-25, evaluated now, once)
# ---------------------------------------------------------------------------
final = {}
for target in ['top5', 'starter']:
    ch = chosen[target]
    w = weights(tr, ch['decay'], 2023)
    m = refit(ch['params'], tr, tr[target], w, FEATS_V2, ch['rounds'])
    final[target] = m
    te[f'xgb_{target}'] = m.predict(xgb.DMatrix(te[FEATS_V2]))
    te[f'xgb_{target}_cal'] = apply_cal(target, te[f'xgb_{target}'].values)
    print(f'{target}: final model trained, rounds={ch["rounds"]}', flush=True)

# VOR regressors (with-price and profile-only) with v2 features
REG_PARAMS = dict(max_depth=4, eta=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=20, reg_lambda=2.0, objective='reg:squarederror',
                  eval_metric='rmse', nthread=8, seed=7)
def fit_reg(feats, decay=0.85):
    m0 = xgb.train(REG_PARAMS, xgb.DMatrix(tr_fit[feats], label=tr_fit['vor'],
                                           weight=weights(tr_fit, decay, 2022)),
                   num_boost_round=2000,
                   evals=[(xgb.DMatrix(tr_es[feats], label=tr_es['vor']), 'es')],
                   early_stopping_rounds=100, verbose_eval=False)
    rounds = int(max(m0.best_iteration + 50, 50))
    m = xgb.train(REG_PARAMS, xgb.DMatrix(tr[feats], label=tr['vor'],
                                          weight=weights(tr, decay, 2023)),
                  num_boost_round=rounds, verbose_eval=False)
    return m, rounds
vor_model, vor_rounds = fit_reg(FEATS_V2)
vornp_model, vornp_rounds = fit_reg(FEATS_V2_NOPRICE)
te['vor_hat_v2'] = vor_model.predict(xgb.DMatrix(te[FEATS_V2]))
te['vor_hat_np_v2'] = vornp_model.predict(xgb.DMatrix(te[FEATS_V2_NOPRICE]))
final['vor'], final['vor_noprice'] = vor_model, vornp_model

def cluster_boot_auc(d, ycol, pcol, n=300):
    key = d['player_id'].astype(str) + '_' + d['season'].astype(str)
    idx_by = d.groupby(key).indices
    keys = list(idx_by.keys())
    aucs = []
    for _ in range(n):
        pick = RNG.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([idx_by[keys[i]] for i in pick])
        yy, pp = d[ycol].values[rows], d[pcol].values[rows]
        if yy.min() != yy.max():
            aucs.append(roc_auc_score(yy, pp))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]

metrics = {'chosen': {t: {k: v for k, v in chosen[t].items()
                          if k not in ('params', 'phase1')} for t in chosen},
           'ablation_2023': ablation, 'calibration': calibration,
           'overall': {}, 'by_position': {}, 'vor': {}}
for target in ['top5', 'starter']:
    metrics['overall'][target] = {
        'xgb_v2_raw': clf_metrics(te[target], te[f'xgb_{target}']),
        'xgb_v2_cal': clf_metrics(te[target], te[f'xgb_{target}_cal']),
        'lr_price_only': clf_metrics(te[target], te[f'lrp_{target}']),
        'lr_extended': clf_metrics(te[target], te[f'lre_{target}'])}
    metrics['by_position'][target] = {}
    for p in POS4:
        s = te[te['pos'] == p]
        metrics['by_position'][target][p] = {
            'n': int(len(s)), 'events': int(s[target].sum()),
            'xgb_v2': clf_metrics(s[target], s[f'xgb_{target}']),
            'xgb_v2_auc_ci95': cluster_boot_auc(s, target, f'xgb_{target}'),
            'lr_price_only': clf_metrics(s[target], s[f'lrp_{target}']),
            'lr_extended': clf_metrics(s[target], s[f'lre_{target}'])}
metrics['vor'] = {
    'xgb_v2_with_price': {'mae': float(mean_absolute_error(te['vor'], te['vor_hat_v2'])),
                          'r2': float(r2_score(te['vor'], te['vor_hat_v2']))},
    'xgb_v2_no_price': {'mae': float(mean_absolute_error(te['vor'], te['vor_hat_np_v2'])),
                        'r2': float(r2_score(te['vor'], te['vor_hat_np_v2']))}}
metrics['vor_by_position_noprice'] = {}
for p in POS4:
    s = te[te['pos'] == p]
    metrics['vor_by_position_noprice'][p] = {
        'n': int(len(s)),
        'r2': float(r2_score(s['vor'], s['vor_hat_np_v2'])),
        'mae': float(mean_absolute_error(s['vor'], s['vor_hat_np_v2']))}

# ---------------------------------------------------------------------------
# 5. Walk-forward: fit <=2022 -> test 2023, v1 config vs feature sets
# ---------------------------------------------------------------------------
wf = {}
D_PARAMS = CLF_CONFIGS['D_d2_reg']
for target in ['top5', 'starter']:
    wf[target] = {}
    for name, feats, decay in [('v1_feats_d1.0', FEATS_V1, 1.0),
                               ('v2_feats_d1.0', FEATS_V2, 1.0),
                               ('v2_feats_decay', FEATS_V2, chosen[target]['decay'])]:
        m = fit_eval(D_PARAMS, tr_fit, tr_fit[target], weights(tr_fit, decay, 2022),
                     tr_es, tr_es[target], feats)
        p = m.predict(xgb.DMatrix(tr_es[feats]))
        wf[target][name] = dict(auc=float(roc_auc_score(tr_es[target], p)),
                                logloss=float(log_loss(tr_es[target], p)),
                                brier=float(brier_score_loss(tr_es[target], p)))
    print(f'walk-forward {target}: ' +
          ', '.join(f"{k} auc {v['auc']:.3f}" for k, v in wf[target].items()), flush=True)
metrics['walk_forward_2023'] = wf

# ---------------------------------------------------------------------------
# 6. Disagreement with price on the holdout (the draft-day use case)
# ---------------------------------------------------------------------------
disagree = {}
for target in ['top5', 'starter']:
    d = te.copy()
    d['delta'] = d[f'xgb_{target}'] - d[f'lrp_{target}']
    d['q'] = pd.qcut(d['delta'], 5, labels=False, duplicates='drop')
    rows = []
    for q, g in d.groupby('q'):
        rows.append({'delta_quintile': int(q), 'n': int(len(g)),
                     'mean_price': float(g['price_norm'].mean()),
                     'mean_model_p': float(g[f'xgb_{target}'].mean()),
                     'mean_lr_p': float(g[f'lrp_{target}'].mean()),
                     'actual_rate': float(g[target].mean()),
                     'actual_minus_lr': float(g[target].mean() - g[f'lrp_{target}'].mean()),
                     'mean_realized_vor': float(g['vor'].mean()),
                     'vor_per_dollar': float(g['vor'].sum() / g['price_norm'].sum())})
    by_pos = {}
    for p in POS4:
        s = d[d['pos'] == p].copy()
        if len(s) < 200:
            continue
        s['q'] = pd.qcut(s['delta'], 5, labels=False, duplicates='drop')
        top = s[s['q'] == s['q'].max()]
        by_pos[p] = {'n_top_quintile': int(len(top)),
                     'mean_price': float(top['price_norm'].mean()),
                     'actual_rate': float(top[target].mean()),
                     'position_base_rate': float(s[target].mean()),
                     'mean_lr_p': float(top[f'lrp_{target}'].mean()),
                     'actual_minus_lr': float(top[target].mean() - top[f'lrp_{target}'].mean())}
    disagree[target] = {'quintiles': rows, 'top_quintile_by_position': by_pos}
    print(f'disagreement {target}: top quintile actual-lr = '
          f'{rows[-1]["actual_minus_lr"]:+.3f}, bottom {rows[0]["actual_minus_lr"]:+.3f}',
          flush=True)
metrics['disagreement_holdout'] = disagree

# ---------------------------------------------------------------------------
# 7. Sanity: updated raw effects + model-implied injury effect, importances
# ---------------------------------------------------------------------------
imp = {}
for name, m in [('top5', final['top5']), ('starter', final['starter']),
                ('vor', final['vor'])]:
    g = m.get_score(importance_type='gain')
    tot = sum(g.values()) or 1.0
    imp[name] = {k: float(v / tot) for k, v in sorted(g.items(), key=lambda x: -x[1])}
metrics['importance'] = imp

sanity = {}
for era_name, d in [('train<=2023', tr), ('test2024-25', te)]:
    e = d[(d['price_norm'] >= 10) & (d['has_prior'] == 1)]
    gi = e.groupby('missed5')['starter'].agg(['mean', 'size'])
    sanity[f'injury_{era_name}'] = {
        'hit_missed5': float(gi.loc[1.0, 'mean']), 'n_missed5': int(gi.loc[1.0, 'size']),
        'hit_durable': float(gi.loc[0.0, 'mean']), 'n_durable': int(gi.loc[0.0, 'size'])}
    eo = d[(d['price_norm'] >= 10)].dropna(subset=['prior_off_tier'])
    go = eo.groupby('prior_off_tier')['starter'].mean()
    sanity[f'offense_{era_name}'] = {k: float(v) for k, v in go.items()}
metrics['sanity_effects'] = sanity

# model-implied injury effect: paired prediction delta teaser rows
def implied_delta(target):
    base = pd.DataFrame([dict(price_norm=30, age=27, missed_prior=0, missed5=0, has_prior=1,
                              prior_off_rank=12, prio_top=0, prio_mid=1, prio_bottom=0,
                              pos_QB=0, pos_RB=1, pos_WR=0, pos_TE=0,
                              **{f: np.nan for f in NEW})])
    hurt = base.copy(); hurt['missed_prior'] = 8; hurt['missed5'] = 1
    p0 = float(final[target].predict(xgb.DMatrix(base[FEATS_V2]))[0])
    p1 = float(final[target].predict(xgb.DMatrix(hurt[FEATS_V2]))[0])
    return {'p_missed0': p0, 'p_missed8': p1, 'delta': p1 - p0}
metrics['model_implied_injury_delta_RB_$30'] = {t: implied_delta(t) for t in
                                               ['top5', 'starter']}

# ---------------------------------------------------------------------------
# 8. Persist staging artifacts
# ---------------------------------------------------------------------------
final['top5'].save_model(os.path.join(OUT, 'model_top5_v2.json'))
final['starter'].save_model(os.path.join(OUT, 'model_starter_v2.json'))
final['vor'].save_model(os.path.join(OUT, 'model_vor_v2.json'))
final['vor_noprice'].save_model(os.path.join(OUT, 'model_vor_noprice_v2.json'))
te.to_csv(os.path.join(OUT, 'test_predictions_v2.csv'), index=False)
json.dump(metrics, open(os.path.join(OUT, 'metrics_v2.json'), 'w'), indent=1,
          default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
json.dump({'features_v2': FEATS_V2, 'features_v2_noprice': FEATS_V2_NOPRICE,
           'chosen': metrics['chosen'], 'calibration': calibration,
           'lr_price_coefs': {t: {'coef': float(lr_price[t].coef_[0][0]),
                                  'intercept': float(lr_price[t].intercept_[0])}
                              for t in lr_price},
           'lr_ext_coefs_note': 'lr_extended is a comparison baseline only'},
          open(os.path.join(OUT, 'features_v2_staging.json'), 'w'), indent=1, default=str)

print('\nHOLDOUT 2024-25:')
for t in ['top5', 'starter']:
    o = metrics['overall'][t]
    print(f"  {t}: XGBv2 AUC {o['xgb_v2_raw']['auc']:.3f} PR {o['xgb_v2_raw']['prauc']:.3f} "
          f"Brier {o['xgb_v2_raw']['brier']:.4f} | cal AUC {o['xgb_v2_cal']['auc']:.3f} "
          f"Brier {o['xgb_v2_cal']['brier']:.4f} | LRprice AUC {o['lr_price_only']['auc']:.3f} "
          f"PR {o['lr_price_only']['prauc']:.3f} Brier {o['lr_price_only']['brier']:.4f} | "
          f"LRext AUC {o['lr_extended']['auc']:.3f}")
print('VOR:', json.dumps(metrics['vor'], default=str))
print('DONE', flush=True)
