#!/usr/bin/env python3
"""
Improvement-loop harness for the draft-day model bundle.

Champion = the shipped v2 recipe (pooled XGBoost, D_d2_reg, per-target recency
decay, v2 feature set). Every experiment is a single change relative to the
current champion, evaluated on BOTH:
  - walk-forward fold: fit <=2022 (early stop on 2023) -> metrics on 2023
  - holdout fold:      fit <=2023 (rounds borrowed from the WF fit, +50)
                       -> metrics on 2024-25
Acceptance (see decide()): holdout AUC gain >= +0.003 (or Brier gain >= 0.0015)
on at least one target, with no regression beyond tolerance (AUC -0.002,
Brier +0.0005) on any target x fold cell. State lives in loop_state.json;
per-run metrics in loop_runs/.
"""
import copy, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss, mean_absolute_error,
                             r2_score)

OUT = os.path.dirname(os.path.abspath(__file__))
POS4 = ['QB', 'RB', 'WR', 'TE']
RNG = np.random.default_rng(11)

FEATS_V2 = ['price_norm', 'age', 'missed_prior', 'missed5', 'has_prior',
            'prior_off_rank', 'prio_top', 'prio_mid', 'prio_bottom',
            'pos_QB', 'pos_RB', 'pos_WR', 'pos_TE',
            'ecr_overall', 'ecr_pos', 'ecr_sd', 'ecr_overall_log', 'ecr_pos_log',
            'prior_pts_ppr', 'prior_ppg_ppr', 'prior_rank_ppr',
            'prior_snap_share', 'prior_target_share', 'prior_rush_share',
            'prior_opp_pg', 'years_exp', 'draft_round', 'draft_pick',
            'age_sq', 'exp_youth', 'ppg_x_snap']
GROUPS = {
    'VAL': ['ppg_per_dollar', 'price_x_age', 'ecr_vs_prior_rank', 'price_per_ecr'],
    'EFF': ['prior_pts_per_opp', 'prior_pts_per_snap', 'prior_catch_rate',
            'prior_adot', 'prior_rz_share', 'prior_pass_att_pg',
            'prior_pass_rtg', 'prior_pass_td_rate', 'prior_qb_rush_pg'],
    'TEAM': ['prior_team_pass_pg', 'prior_team_pass_share',
             'prior_team_pass_rank', 'prior_team_plays_pg'],
    'TRAJ': ['prior2_ppg_ppr', 'prior2_rank_ppr', 'ppg_delta', 'rank_improve'],
    'WK': ['prior_wk_cv', 'prior_wk_floor10', 'prior_wk_ceil90'],
}
CLF_BASE = dict(subsample=0.8, colsample_bytree=0.8,
                objective='binary:logistic', eval_metric='logloss',
                nthread=8, seed=7)
PARAMS = {
    'D_d2_reg': dict(CLF_BASE, max_depth=2, eta=0.03, min_child_weight=30,
                     reg_lambda=5.0),
    'B_d3_reg': dict(CLF_BASE, max_depth=3, eta=0.03, min_child_weight=50,
                     reg_lambda=5.0),
    'A_d4_base': dict(CLF_BASE, max_depth=4, eta=0.05, min_child_weight=20,
                      reg_lambda=2.0),
}
REG_PARAMS = dict(max_depth=4, eta=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=20, reg_lambda=2.0,
                  objective='reg:squarederror', eval_metric='rmse',
                  nthread=8, seed=7)
LR_FEATS = ['price_norm', 'ecr_pos', 'prior_ppg_ppr', 'age', 'missed_prior']

CHAMPION_SPEC = {
    'feats': FEATS_V2,
    'params': 'D_d2_reg',
    'decay': {'top5': 0.7, 'starter': 0.85},
    'weight_mode': 'decay',          # 'decay' | 'cluster'
    'ensemble_w': 0.0,               # weight on XGB in XGB/LR-ext prob average
    'pos_blend': {},                 # {"top5|WR": 0.4} -> w on per-position model
    'calibrator': None,              # None | 'shrink_oof'
    'vor_feats_extra': [],           # accepted feature groups also flow to VOR
}

_DF = None
def load_df():
    global _DF
    if _DF is None:
        _DF = pd.read_csv(os.path.join(OUT, 'training_data_v3.csv'),
                          low_memory=False)
    return _DF

def make_weights(d, spec, target, ref_season):
    decay = spec['decay'][target]
    w = (decay ** (ref_season - d['season'])).astype(float).values
    if spec.get('weight_mode') == 'cluster':
        cnt = d.groupby([d['player_id'], d['season']])['season'].transform('size')
        w = w / cnt.values
        w = w * (len(w) / w.sum())
    return w

def clf_metrics(y, p):
    p = np.asarray(p, dtype=float)
    return dict(auc=float(roc_auc_score(y, p)),
                prauc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))))

def ece(y, p, bins=10):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    qs = np.quantile(p, np.linspace(0, 1, bins + 1)); qs[0], qs[-1] = -0.01, 1.01
    idx = np.digitize(p, qs[1:-1])
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            out += abs(p[m].mean() - y[m].mean()) * m.sum()
    return float(out / len(y))

def fit_lr(d_fit, d_pred, feats, target):
    med = d_fit[feats].median()
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(d_fit[feats].fillna(med), d_fit[target])
    return lr.predict_proba(d_pred[feats].fillna(med))[:, 1]

def choose_shrink(d_fit, feats, target, spec, target_key):
    """5-fold OOF inside d_fit; pick shrink s toward train base rate by Brier."""
    y = d_fit[target].values.astype(float)
    base = float(y.mean())
    oof = np.full(len(d_fit), np.nan)
    kf = KFold(5, shuffle=True, random_state=5)
    idx = np.arange(len(d_fit))
    for tri, tei in kf.split(idx):
        fd, hd = d_fit.iloc[tri], d_fit.iloc[tei]
        m = xgb.train(PARAMS[spec['params']],
                      xgb.DMatrix(fd[feats], label=fd[target],
                                  weight=make_weights(fd, spec, target_key,
                                                      int(fd['season'].max()))),
                      num_boost_round=120, verbose_eval=False)
        oof[tei] = m.predict(xgb.DMatrix(hd[feats]))
    best_s, best_b = 1.0, brier_score_loss(y, oof)
    for s in [0.95, 0.9, 0.85, 0.8, 0.7]:
        b = brier_score_loss(y, base + s * (oof - base))
        if b < best_b:
            best_s, best_b = s, b
    return best_s

def eval_target(df, spec, target):
    """Dual-fold evaluation of one classifier target. Returns metrics + preds."""
    feats = (spec.get('feats_by_target') or {}).get(target, spec['feats'])
    params = PARAMS[spec['params']]
    tr22 = df[df['season'] <= 2022]
    s23 = df[df['season'] == 2023]
    tr23 = df[df['season'] <= 2023]
    te = df[df['season'] >= 2024]
    out = {}

    # ---- walk-forward: fit <=2022, early stop on 2023
    m_wf = xgb.train(params,
                     xgb.DMatrix(tr22[feats], label=tr22[target],
                                 weight=make_weights(tr22, spec, target, 2022)),
                     num_boost_round=2000,
                     evals=[(xgb.DMatrix(s23[feats], label=s23[target]), 'es')],
                     early_stopping_rounds=100, verbose_eval=False)
    rounds = int(max(m_wf.best_iteration + 50, 50))
    p_wf_xgb = m_wf.predict(xgb.DMatrix(s23[feats]))
    # ---- holdout: fit <=2023 with borrowed rounds
    m_ho = xgb.train(params,
                     xgb.DMatrix(tr23[feats], label=tr23[target],
                                 weight=make_weights(tr23, spec, target, 2023)),
                     num_boost_round=rounds, verbose_eval=False)
    p_ho_xgb = m_ho.predict(xgb.DMatrix(te[feats]))

    p = {'wf': p_wf_xgb, 'holdout': p_ho_xgb}
    fit_sets = {'wf': tr22, 'holdout': tr23}
    pred_sets = {'wf': s23, 'holdout': te}
    lr_preds = {}
    for fold in ('wf', 'holdout'):
        lr_preds[fold] = fit_lr(fit_sets[fold], pred_sets[fold], LR_FEATS, target)
        if spec.get('ensemble_w'):
            w = spec['ensemble_w']
            p[fold] = w * p[fold] + (1 - w) * lr_preds[fold]

    # ---- per-position blend (fixed weight on a per-position model)
    for key, wpos in (spec.get('pos_blend') or {}).items():
        tkey, pos = key.split('|')
        if tkey != target:
            continue
        for fold, dfit, dpred in (('wf', tr22, s23), ('holdout', tr23, te)):
            sub_fit = dfit[dfit['pos'] == pos]
            sub_es = s23[s23['pos'] == pos] if fold == 'wf' else None
            if fold == 'wf':
                mp = xgb.train(params,
                               xgb.DMatrix(sub_fit[feats], label=sub_fit[target],
                                           weight=make_weights(sub_fit, spec,
                                                               target, 2022)),
                               num_boost_round=2000,
                               evals=[(xgb.DMatrix(sub_es[feats],
                                                   label=sub_es[target]), 'es')],
                               early_stopping_rounds=100, verbose_eval=False)
                pr = mp.predict(xgb.DMatrix(dpred[feats]))
            else:
                mp = xgb.train(params,
                               xgb.DMatrix(sub_fit[feats], label=sub_fit[target],
                                           weight=make_weights(sub_fit, spec,
                                                               target, 2023)),
                               num_boost_round=rounds, verbose_eval=False)
                pr = mp.predict(xgb.DMatrix(dpred[feats]))
            mask = (dpred['pos'] == pos).values
            p[fold] = np.where(mask, (1 - wpos) * p[fold] + wpos * pr, p[fold])

    # ---- optional train-fold-only shrink calibration (reported alongside raw)
    cal = {}
    if spec.get('calibrator') == 'shrink_oof':
        for fold, dfit in fit_sets.items():
            s = choose_shrink(dfit, feats, target, spec, target)
            base = float(dfit[target].mean())
            cal[fold] = (s, base)

    for fold, dpred in pred_sets.items():
        y = dpred[target].values.astype(float)
        entry = clf_metrics(y, p[fold])
        entry['ece'] = ece(y, p[fold])
        entry['lr_price_auc'] = float(roc_auc_score(
            y, fit_lr(fit_sets[fold], dpred, ['price_norm'], target)))
        entry['lr_ext_auc'] = float(roc_auc_score(y, lr_preds[fold]))
        entry['n'] = int(len(y))
        if fold in cal:
            s, base = cal[fold]
            pc = base + s * (p[fold] - base)
            entry['cal'] = {**clf_metrics(y, pc), 'ece': ece(y, pc),
                            'shrink_s': s}
        entry['by_position'] = {}
        for pos in POS4:
            m = (dpred['pos'] == pos).values
            if y[m].min() != y[m].max():
                entry['by_position'][pos] = float(roc_auc_score(y[m], p[fold][m]))
        out[fold] = entry
    return out, {'wf_model': m_wf, 'ho_model': m_ho, 'rounds': rounds,
                 'preds': p}

def eval_vor(df, spec):
    feats = spec['feats'] + [f for g in spec.get('vor_feats_extra', [])
                             for f in GROUPS[g]]
    np_feats = [f for f in feats if f != 'price_norm']
    tr22 = df[df['season'] <= 2022]
    s23 = df[df['season'] == 2023]
    tr23 = df[df['season'] <= 2023]
    te = df[df['season'] >= 2024]
    res = {}
    models = {}
    for name, fl in (('with_price', feats), ('profile_only', np_feats)):
        def fit(dfit, w_season_ref):
            m0 = xgb.train(REG_PARAMS,
                           xgb.DMatrix(dfit[fl], label=dfit['vor'],
                                       weight=(0.85 ** (w_season_ref
                                                        - dfit['season'])).astype(float).values),
                           num_boost_round=2000,
                           evals=[(xgb.DMatrix(s23[fl], label=s23['vor']), 'es')],
                           early_stopping_rounds=100, verbose_eval=False)
            return m0
        mw = fit(tr22, 2022)
        rounds = int(max(mw.best_iteration + 50, 50))
        mh = xgb.train(REG_PARAMS,
                       xgb.DMatrix(tr23[fl], label=tr23['vor'],
                                   weight=(0.85 ** (2023 - tr23['season'])).astype(float).values),
                       num_boost_round=rounds, verbose_eval=False)
        for fold, dpred, mdl in (('wf', s23, mw), ('holdout', te, mh)):
            pr = mdl.predict(xgb.DMatrix(dpred[fl]))
            res.setdefault(fold, {})[name] = dict(
                mae=float(mean_absolute_error(dpred['vor'], pr)),
                r2=float(r2_score(dpred['vor'], pr)))
        models[name] = mh
    return res, models

def run_spec(spec, label='spec', vor=True):
    df = load_df()
    out = {'label': label, 'spec': {k: v for k, v in spec.items()}}
    store = {}
    for target in ('top5', 'starter'):
        m, st = eval_target(df, spec, target)
        out[target] = m
        store[target] = st
    if vor:
        v, vm = eval_vor(df, spec)
        out['vor'] = v
        store['vor'] = vm
    out['_store'] = store
    return out

def summarize(res):
    lines = []
    for target in ('top5', 'starter'):
        for fold in ('wf', 'holdout'):
            e = res[target][fold]
            lines.append(f"{target}/{fold}: AUC {e['auc']:.4f} PR {e['prauc']:.4f} "
                         f"Brier {e['brier']:.4f} LL {e['logloss']:.4f} "
                         f"ECE {e['ece']:.3f} (LRp {e['lr_price_auc']:.3f}, "
                         f"LRe {e['lr_ext_auc']:.3f})")
    if 'vor' in res:
        for fold in ('wf', 'holdout'):
            v = res['vor'][fold]
            lines.append(f"vor/{fold}: with-price R2 {v['with_price']['r2']:.3f} "
                         f"MAE {v['with_price']['mae']:.1f}; profile R2 "
                         f"{v['profile_only']['r2']:.3f} MAE {v['profile_only']['mae']:.1f}")
    return '\n'.join(lines)

def decide(base, new, kind='clf'):
    """Acceptance gate. Returns (accept, reasons list)."""
    reasons = []
    if kind == 'cal':
        improved = False
        for t in ('top5', 'starter'):
            for fold in ('wf', 'holdout'):
                bc = base[t][fold].get('cal') or base[t][fold]
                nc = new[t][fold].get('cal')
                if nc is None:
                    return False, [f'{t}/{fold}: no cal metrics']
                if nc['brier'] - bc['brier'] > 0.0005:
                    reasons.append(f"{t}/{fold} cal Brier regress "
                                   f"{nc['brier']-bc['brier']:+.4f}")
                if nc['ece'] - bc['ece'] > 0.01:
                    reasons.append(f"{t}/{fold} cal ECE regress "
                                   f"{nc['ece']-bc['ece']:+.3f}")
            if (base[t]['holdout'].get('cal') or base[t]['holdout'])['brier'] \
                    - new[t]['holdout']['cal']['brier'] >= 0.0015 and \
               (base[t]['holdout'].get('cal') or base[t]['holdout'])['ece'] \
                    - new[t]['holdout']['cal']['ece'] >= 0.01:
                improved = True
        return (improved and not reasons), reasons or (['calibration improved on '
                'holdout without fold regressions'] if improved else
                ['no target met the calibration improvement bar'])
    improved = False
    for t in ('top5', 'starter'):
        for fold in ('wf', 'holdout'):
            da = new[t][fold]['auc'] - base[t][fold]['auc']
            db = new[t][fold]['brier'] - base[t][fold]['brier']
            if da < -0.002:
                reasons.append(f"{t}/{fold} AUC regress {da:+.4f}")
            if db > 0.0005:
                reasons.append(f"{t}/{fold} Brier regress {db:+.4f}")
        da_h = new[t]['holdout']['auc'] - base[t]['holdout']['auc']
        db_h = base[t]['holdout']['brier'] - new[t]['holdout']['brier']
        if da_h >= 0.003 or db_h >= 0.0015:
            improved = True
    if not improved:
        reasons.append('no target cleared the holdout improvement bar '
                       '(AUC +0.003 or Brier -0.0015)')
    return (improved and not reasons), reasons

# ---------------------------------------------------------------- state I/O
STATE_F = os.path.join(OUT, 'loop_state.json')
RUNS_D = os.path.join(OUT, 'loop_runs')

def load_state():
    if os.path.exists(STATE_F):
        return json.load(open(STATE_F))
    return None

def save_run(exp_id, res, decision, extra=None):
    os.makedirs(RUNS_D, exist_ok=True)
    slim = {k: v for k, v in res.items() if k != '_store'}
    rec = {'exp_id': exp_id, 'results': slim, 'decision': decision}
    if extra:
        rec.update(extra)
    json.dump(rec, open(os.path.join(RUNS_D, f'{exp_id}.json'), 'w'), indent=1,
              default=str)
    return rec

def metric_table(res):
    """Compact per-target/fold AUC+Brier for the experiment log."""
    rows = {}
    for t in ('top5', 'starter'):
        for fold in ('wf', 'holdout'):
            e = res[t][fold]
            rows[f'{t}_{fold}'] = {'auc': round(e['auc'], 4),
                                   'brier': round(e['brier'], 4)}
    if 'vor' in res:
        for fold in ('wf', 'holdout'):
            rows[f'vor_{fold}'] = {
                'r2_price': round(res['vor'][fold]['with_price']['r2'], 3),
                'r2_profile': round(res['vor'][fold]['profile_only']['r2'], 3)}
    return rows
