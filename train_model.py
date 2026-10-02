#!/usr/bin/env python3
"""
Draft-day XGBoost model for 10-team auction SUPERFLEX redraft leagues.

Headline deliverables (classifiers):
  - model_top5.json     P(player finishes top-5 at his position)
  - model_starter.json  P(player finishes at/above replacement value, "starter-tier")
Secondary:
  - model_vor.json        predicted VOR given an asking price
  - model_vor_noprice.json predicted VOR from player profile only -> implied fair price

Features are 100% draft-day legal (knowable before the season starts):
  price_norm (auction price normalized to a $200 budget), position, age,
  prior-season games missed (injury history), prior-year NFL offense rank/tier.
The `offense_tier` column shipped in the pick files is built from SAME-season
stats (see targeted_picks.py `offense_tiers(season)`) and is therefore leakage;
we reconstruct prior-year offense quality from stats_{season-1}.json instead.
A leaked variant is trained only to quantify the leakage inflation.

Validation: strict time split. Train <=2023, test 2024-2025. No random splits.
"""
import json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             mean_absolute_error, r2_score)

ROOT = os.path.expanduser('~/workspace/sleeper-league-data')
CACHE = os.path.join(ROOT, 'crossleague-cache')
STATS = os.path.join(CACHE, 'targeted', 'stats')
OUT = os.path.join(ROOT, 'model')
os.makedirs(OUT, exist_ok=True)
POS4 = ['QB', 'RB', 'WR', 'TE']
TEAM_FIX = {'WSH': 'WAS'}
RNG = np.random.default_rng(7)

PLAYERS = json.load(open(os.path.join(CACHE, 'players_nfl.json')))
_NORM = {'DE':'DL','DT':'DL','NT':'DL','EDR':'DL','OLB':'LB','ILB':'LB','MLB':'LB',
         'CB':'DB','SS':'DB','FS':'DB','SAF':'DB'}
POSD = {pid: _NORM.get(v.get('position') or 'UNK', v.get('position') or 'UNK')
        for pid, v in PLAYERS.items()}

# ----------------------------------------------------------------------------
# Prior-year offense quality, reconstructed with the same method the pick files
# used for their (leaked) same-season tier: sum pts_std by NFL team over
# QB/RB/WR/TE/K, rank 1..32, thirds = top/mid/bottom.
# ----------------------------------------------------------------------------
def offense_table(season):
    fn = os.path.join(STATS, f'stats_{season}.json')
    if not os.path.exists(fn):
        return {}
    st = json.load(open(fn))
    tot = {}
    for pid, s in st.items():
        if not isinstance(s, dict):
            continue
        if POSD.get(pid) not in ('QB', 'RB', 'WR', 'TE', 'K'):
            continue
        team = (PLAYERS.get(pid) or {}).get('team')
        if not team:
            continue
        team = TEAM_FIX.get(team, team)
        tot[team] = tot.get(team, 0.0) + (s.get('pts_std') or 0.0)
    ranked = sorted(tot.items(), key=lambda x: -x[1])
    n = len(ranked)
    out = {}
    for i, (t, _) in enumerate(ranked):
        r = i + 1
        tier = 'top' if r <= n / 3 else ('bottom' if r > 2 * n / 3 else 'mid')
        out[t] = (r, tier)
    return out

OFF = {s: offense_table(s) for s in range(2019, 2026)}
for s, t in OFF.items():
    print(f'offense table {s}: {len(t)} teams', flush=True)

def prior_off(team, season):
    """(rank, tier) of the team's offense in season-1 (draft-day legal)."""
    if not isinstance(team, str):
        return (np.nan, None)
    team = TEAM_FIX.get(team, team)
    return OFF.get(season - 1, {}).get(team, (np.nan, None))

# ----------------------------------------------------------------------------
# Assemble the dataset
# ----------------------------------------------------------------------------
def base_frame(df, source):
    df = df[df['pos'].isin(POS4)].copy()
    df['source'] = source
    df['top5'] = (df['pos_rank'] <= 5).astype(float)
    df.loc[df['pos_rank'].isna(), 'top5'] = 0.0   # no recorded stats -> scored ~0
    df['starter'] = df['hit'].astype(float)
    return df

g1 = pd.read_csv(os.path.join(ROOT, 'crossleague-targeted-picks.csv'), low_memory=False)
g2 = pd.read_csv(os.path.join(ROOT, 'crossleague-targeted-gen2-picks.csv'), low_memory=False)
xl = base_frame(pd.concat([g1, g2], ignore_index=True), 'crossleague')
xl = xl[['season', 'league_id', 'chain_name', 'player_id', 'player', 'pos', 'price_norm',
         'season_points', 'pos_rank', 'vor', 'starter', 'top5', 'age', 'nfl_team',
         'gp_prior', 'offense_tier', 'source']]

j = pd.read_csv(os.path.join(ROOT, 'draft_picks_joined.csv'), low_memory=False)
j = j[(j['year'] >= 2020) & (j['position'].isin(POS4))].copy()
j['pos_rank'] = j.groupby(['year', 'position'])['season_points_league'].rank(
    ascending=False, method='min')
j = j.rename(columns={'year': 'season', 'position': 'pos', 'price': 'price_norm',
                      'season_points_league': 'season_points'})
j['league_id'] = 'fantasy101'
j['chain_name'] = 'Fantasy 101'
j['hit'] = (j['vor'] >= 0).astype(float)
j = base_frame(j, 'fantasy101')
j = j[['season', 'league_id', 'chain_name', 'player_id', 'player', 'pos', 'price_norm',
       'season_points', 'pos_rank', 'vor', 'starter', 'top5', 'age', 'nfl_team',
       'gp_prior', 'offense_tier', 'source']]

df = pd.concat([xl, j], ignore_index=True)
df = df[df['price_norm'].notna() & (df['price_norm'] > 0) & df['vor'].notna()].copy()
print(f'assembled rows: {len(df)}  (crossleague {(df.source=="crossleague").sum()}, '
      f'fantasy101 {(df.source=="fantasy101").sum()})', flush=True)

# draft-day legal derived features
def games_in(season_prior):
    return 17 if season_prior >= 2021 else 16

df['missed_prior'] = [ (games_in(s - 1) - g) if pd.notna(g) else np.nan
                       for s, g in zip(df['season'], df['gp_prior'])]
df['missed_prior'] = df['missed_prior'].clip(lower=0, upper=17)
df['has_prior'] = df['gp_prior'].notna().astype(float)
df['missed5'] = ((df['missed_prior'] >= 5) & (df['has_prior'] == 1)).astype(float)

po = [prior_off(t, s) for t, s in zip(df['nfl_team'], df['season'])]
df['prior_off_rank'] = [p[0] for p in po]
df['prior_off_tier'] = [p[1] for p in po]

# leaked variant uses the files' own same-season tier (joined uses good/mid/bad)
leak_map = {'top': 'top', 'mid': 'mid', 'bottom': 'bottom',
            'good': 'top', 'bad': 'bottom'}
df['leak_tier'] = df['offense_tier'].map(leak_map)

for c in POS4:
    df[f'pos_{c}'] = (df['pos'] == c).astype(float)
for c in ['top', 'mid', 'bottom']:
    df[f'prio_{c}'] = (df['prior_off_tier'] == c).astype(float)
    df[f'leak_{c}'] = (df['leak_tier'] == c).astype(float)

FEATS = ['price_norm', 'age', 'missed_prior', 'missed5', 'has_prior', 'prior_off_rank',
         'prio_top', 'prio_mid', 'prio_bottom',
         'pos_QB', 'pos_RB', 'pos_WR', 'pos_TE']
FEATS_LEAK = ['price_norm', 'age', 'missed_prior', 'missed5', 'has_prior',
              'leak_top', 'leak_mid', 'leak_bottom',
              'pos_QB', 'pos_RB', 'pos_WR', 'pos_TE']
FEATS_NOPRICE = [f for f in FEATS if f != 'price_norm']

df.to_csv(os.path.join(OUT, 'training_data.csv'), index=False)

# ----------------------------------------------------------------------------
# Time split
# ----------------------------------------------------------------------------
tr = df[df['season'] <= 2023].copy()
te = df[df['season'] >= 2024].copy()
tr_fit, tr_es = tr[tr['season'] <= 2022], tr[tr['season'] == 2023]
print(f'train<=2023: {len(tr)} (fit<=2022 {len(tr_fit)}, earlystop 2023 {len(tr_es)}), '
      f'test 2024-25: {len(te)}', flush=True)

CLF_PARAMS = dict(max_depth=4, eta=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=20, reg_lambda=2.0, objective='binary:logistic',
                  eval_metric='logloss', nthread=8, seed=7)
REG_PARAMS = dict(max_depth=4, eta=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=20, reg_lambda=2.0, objective='reg:squarederror',
                  eval_metric='rmse', nthread=8, seed=7)
MONO = tuple(1 if f == 'price_norm' else 0 for f in FEATS)
CLF_CONFIGS = {
    'A_d4_base': dict(CLF_PARAMS),
    'B_d3_reg': dict(CLF_PARAMS, max_depth=3, eta=0.03, min_child_weight=50,
                     reg_lambda=5.0),
    'C_d4_mono_price': dict(CLF_PARAMS, monotone_constraints=MONO),
    'D_d2_reg': dict(CLF_PARAMS, max_depth=2, eta=0.03, min_child_weight=30,
                     reg_lambda=5.0),
}

def fit_clf_params(params, Xf, yf, Xes, yes):
    dtr = xgb.DMatrix(Xf, label=yf)
    des = xgb.DMatrix(Xes, label=yes)
    return xgb.train(params, dtr, num_boost_round=2000,
                     evals=[(des, 'es')], early_stopping_rounds=100, verbose_eval=False)

def fit_clf(Xf, yf, Xes, yes):
    dtr = xgb.QuantileDMatrix(Xf, yf, ref=Xf) if False else xgb.DMatrix(Xf, label=yf)
    des = xgb.DMatrix(Xes, label=yes)
    m = xgb.train(CLF_PARAMS, dtr, num_boost_round=2000,
                  evals=[(des, 'es')], early_stopping_rounds=100, verbose_eval=False)
    return m

def fit_reg(Xf, yf, Xes, yes):
    dtr = xgb.DMatrix(Xf, label=yf)
    des = xgb.DMatrix(Xes, label=yes)
    return xgb.train(REG_PARAMS, dtr, num_boost_round=2000,
                     evals=[(des, 'es')], early_stopping_rounds=100, verbose_eval=False)

def refit(model_fn, X, y, rounds, feats, params=None):
    d = xgb.DMatrix(X, label=y)
    p = dict(params) if params else dict(CLF_PARAMS if model_fn == 'clf' else REG_PARAMS)
    return xgb.train(p, d, num_boost_round=rounds, verbose_eval=False)

from sklearn.metrics import log_loss

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

models = {}
fit_notes = {}
platt = {}
phase1_models = {}
for target in ['top5', 'starter']:
    # config selection on the 2023 holdout ONLY (fit <=2022); test untouched
    scores = {}
    fitted = {}
    for name, params in CLF_CONFIGS.items():
        mm = fit_clf_params(params, tr_fit[FEATS], tr_fit[target],
                            tr_es[FEATS], tr_es[target])
        pred_es = mm.predict(xgb.DMatrix(tr_es[FEATS]))
        scores[name] = float(log_loss(tr_es[target], pred_es))
        fitted[name] = (mm, params)
    best_name = min(scores, key=scores.get)
    m0, best_params = fitted[best_name]
    rounds = max(m0.best_iteration + 50, 50)
    m = refit('clf', tr[FEATS], tr[target], rounds, FEATS, params=best_params)
    models[f'model_{target}'] = m
    phase1_models[target] = m0
    # Platt recalibration fit ONLY on 2023 predictions of the <=2022 model
    p_es = m0.predict(xgb.DMatrix(tr_es[FEATS]))
    pl = LogisticRegression(max_iter=1000, C=1e6)
    pl.fit(logit(p_es).reshape(-1, 1), tr_es[target])
    platt[target] = {'coef': float(pl.coef_[0][0]), 'intercept': float(pl.intercept_[0])}
    fit_notes[target] = {'config_search_2023_logloss': scores,
                         'chosen_config': best_name,
                         'best_iter_phase1': int(m0.best_iteration),
                         'rounds_final': rounds}
    print(f'{target}: chosen {best_name} (2023 logloss {scores}) '
          f'phase1 best_iter={m0.best_iteration}', flush=True)

m0 = fit_reg(tr_fit[FEATS], tr_fit['vor'], tr_es[FEATS], tr_es['vor'])
model_vor = refit('reg', tr[FEATS], tr['vor'], max(m0.best_iteration + 50, 50), FEATS)
models['model_vor'] = model_vor
m0n = fit_reg(tr_fit[FEATS_NOPRICE], tr_fit['vor'], tr_es[FEATS_NOPRICE], tr_es['vor'])
model_vor_np = refit('reg', tr[FEATS_NOPRICE], tr['vor'], max(m0n.best_iteration + 50, 50),
                     FEATS_NOPRICE)
models['model_vor_noprice'] = model_vor_np

# leaked variants (comparison only)
leak_models = {}
for target in ['top5', 'starter']:
    m0 = fit_clf(tr_fit[FEATS_LEAK], tr_fit[target], tr_es[FEATS_LEAK], tr_es[target])
    leak_models[target] = refit('clf', tr[FEATS_LEAK], tr[target],
                                max(m0.best_iteration + 50, 50), FEATS_LEAK)

# per-position classifiers (comparison only)
perpos = {}
for p in POS4:
    a = tr_fit[tr_fit['pos'] == p]; b = tr_es[tr_es['pos'] == p]; allt = tr[tr['pos'] == p]
    feats_p = [f for f in FEATS if not f.startswith('pos_')]
    for target in ['top5', 'starter']:
        m0 = fit_clf(a[feats_p], a[target], b[feats_p], b[target])
        perpos[(p, target)] = (refit('clf', allt[feats_p], allt[target],
                                     max(m0.best_iteration + 50, 50), feats_p), feats_p)

# ----------------------------------------------------------------------------
# Predictions on the held-out 2024-2025 test set
# ----------------------------------------------------------------------------
dte = xgb.DMatrix(te[FEATS])
te = te.copy()
te['p_top5'] = models['model_top5'].predict(dte)
te['p_starter'] = models['model_starter'].predict(dte)
for target in ['top5', 'starter']:
    pc = platt[target]
    z = pc['intercept'] + pc['coef'] * logit(te[f'p_{target}'].values)
    te[f'p_{target}_cal'] = 1.0 / (1.0 + np.exp(-z))
te['p_top5_leak'] = leak_models['top5'].predict(xgb.DMatrix(te[FEATS_LEAK]))
te['p_starter_leak'] = leak_models['starter'].predict(xgb.DMatrix(te[FEATS_LEAK]))
te['vor_hat'] = models['model_vor'].predict(dte)
te['vor_hat_np'] = model_vor_np.predict(xgb.DMatrix(te[FEATS_NOPRICE]))
for p in POS4:
    mask = te['pos'] == p
    for target in ['top5', 'starter']:
        m, fp = perpos[(p, target)]
        te.loc[mask, f'p_{target}_perpos'] = m.predict(xgb.DMatrix(te.loc[mask, fp]))

# baselines
lr_price = {}
for target in ['top5', 'starter']:
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(tr[['price_norm']], tr[target])
    te[f'p_{target}_lrprice'] = lr.predict_proba(te[['price_norm']])[:, 1]
    lr_price[target] = lr
    rates = tr.groupby('pos')[target].mean()
    te[f'p_{target}_posavg'] = te['pos'].map(rates)

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def clf_metrics(y, p):
    return dict(auc=float(roc_auc_score(y, p)),
                prauc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                base_rate=float(np.mean(y)))

def cluster_boot_auc(d, ycol, pcol, n=400):
    clusters = (d['player_id'].astype(str) + '_' + d['season'].astype(str)).unique()
    idx_by_cluster = d.groupby(d['player_id'].astype(str) + '_' + d['season'].astype(str)).indices
    keys = list(idx_by_cluster.keys())
    aucs = []
    for _ in range(n):
        pick = RNG.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([idx_by_cluster[keys[i]] for i in pick])
        yy, pp = d[ycol].values[rows], d[pcol].values[rows]
        if yy.min() != yy.max():
            aucs.append(roc_auc_score(yy, pp))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

metrics = {'split': {'train_seasons': '<=2023', 'test_seasons': '2024-2025',
                     'n_train': int(len(tr)), 'n_test': int(len(te))},
           'fit_notes': fit_notes, 'overall': {}, 'by_position': {}, 'leakage': {}}

for target, col in [('top5', 'p_top5'), ('starter', 'p_starter')]:
    metrics['overall'][target] = {
        'xgb': clf_metrics(te[target], te[col]),
        'xgb_platt_calibrated': clf_metrics(te[target], te[f'{col}_cal']),
        'xgb_perpos': clf_metrics(te[target], te[f'{col}_perpos']),
        'xgb_leaked_offense': clf_metrics(te[target], te[f'{col}_leak']),
        'lr_price_only': clf_metrics(te[target], te[f'{col}_lrprice']),
        'pos_average': clf_metrics(te[target], te[f'{col}_posavg'])}
    metrics['by_position'][target] = {}
    for p in POS4:
        s = te[te['pos'] == p]
        lo, hi = cluster_boot_auc(s, target, col)
        metrics['by_position'][target][p] = {
            'n': int(len(s)), 'events': int(s[target].sum()),
            'xgb': clf_metrics(s[target], s[col]),
            'xgb_auc_ci95': [lo, hi],
            'xgb_perpos': clf_metrics(s[target], s[f'{col}_perpos']),
            'lr_price_only': clf_metrics(s[target], s[f'{col}_lrprice']),
            'pos_average_auc': float(roc_auc_score(s[target], s[f'{col}_posavg']))}

base_vor = tr.groupby('pos')['vor'].mean()
te['vor_posmean'] = te['pos'].map(base_vor)
lrv = LinearRegression().fit(tr[['price_norm']], tr['vor'])
te['vor_lrprice'] = lrv.predict(te[['price_norm']])
metrics['vor'] = {
    'xgb_with_price': {'mae': float(mean_absolute_error(te['vor'], te['vor_hat'])),
                       'r2': float(r2_score(te['vor'], te['vor_hat']))},
    'xgb_no_price': {'mae': float(mean_absolute_error(te['vor'], te['vor_hat_np'])),
                     'r2': float(r2_score(te['vor'], te['vor_hat_np']))},
    'lr_price_only': {'mae': float(mean_absolute_error(te['vor'], te['vor_lrprice'])),
                      'r2': float(r2_score(te['vor'], te['vor_lrprice']))},
    'pos_mean': {'mae': float(mean_absolute_error(te['vor'], te['vor_posmean'])),
                 'r2': float(r2_score(te['vor'], te['vor_posmean']))}}
metrics['vor_by_position'] = {}
for p in POS4:
    s = te[te['pos'] == p]
    metrics['vor_by_position'][p] = {
        'n': int(len(s)),
        'xgb_with_price': {'mae': float(mean_absolute_error(s['vor'], s['vor_hat'])),
                           'r2': float(r2_score(s['vor'], s['vor_hat']))},
        'xgb_no_price': {'mae': float(mean_absolute_error(s['vor'], s['vor_hat_np'])),
                         'r2': float(r2_score(s['vor'], s['vor_hat_np']))},
        'pos_mean_mae': float(mean_absolute_error(s['vor'], s['vor_posmean']))}

def calib(y, p, bins=10):
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -0.01, 1.01
    idx = np.digitize(p, qs[1:-1])
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append(dict(bin=b, n=int(m.sum()), pred=float(p[m].mean()),
                             obs=float(y[m].mean())))
    ece = float(np.sum([abs(r['pred'] - r['obs']) * r['n'] for r in rows]) / len(y))
    return rows, ece

cal, ece_t = calib(te['top5'].values, te['p_top5'].values)
cals, ece_s = calib(te['starter'].values, te['p_starter'].values)
calc, ece_tc = calib(te['top5'].values, te['p_top5_cal'].values)
calcs, ece_sc = calib(te['starter'].values, te['p_starter_cal'].values)
metrics['calibration'] = {'top5': {'ece': ece_t, 'bins': cal},
                          'starter': {'ece': ece_s, 'bins': cals},
                          'top5_platt': {'ece': ece_tc, 'bins': calc},
                          'starter_platt': {'ece': ece_sc, 'bins': calcs},
                          'platt_params_fit_on_2023': platt}

# ----------------------------------------------------------------------------
# Importance (gain) + sanity mappings against the validated findings
# ----------------------------------------------------------------------------
imp = {}
for name, m in [('top5', models['model_top5']), ('starter', models['model_starter']),
                ('vor', models['model_vor'])]:
    g = m.get_score(importance_type='gain')
    tot = sum(g.values()) or 1.0
    imp[name] = {k: float(v / tot) for k, v in sorted(g.items(), key=lambda x: -x[1])}

sanity = {}
exp = te[te['price_norm'] >= 10]
g_off = exp.dropna(subset=['prior_off_tier']).groupby('prior_off_tier')
sanity['offense_prior_year_10plus'] = {
    'n': {k: int(v) for k, v in g_off.size().items()},
    'actual_hit': {k: float(v) for k, v in g_off['starter'].mean().items()},
    'model_p_starter': {k: float(v) for k, v in g_off['p_starter'].mean().items()}}
lk = exp.dropna(subset=['leak_tier']).groupby('leak_tier')
sanity['offense_sameseason_leaked_10plus'] = {
    'n': {k: int(v) for k, v in lk.size().items()},
    'actual_hit': {k: float(v) for k, v in lk['starter'].mean().items()}}

inj = exp[exp['has_prior'] == 1]
gi = inj.groupby('missed5')
sanity['injury_10plus'] = {
    'n': {str(k): int(v) for k, v in gi.size().items()},
    'actual_hit': {str(k): float(v) for k, v in gi['starter'].mean().items()},
    'model_p_starter': {str(k): float(v) for k, v in gi['p_starter'].mean().items()}}

ey = te[(te['price_norm'] >= 25) & te['age'].notna()].copy()
ey['age_band'] = np.where(ey['age'] <= 24, '<=24', np.where(ey['age'] <= 30, '25-30', '31+'))
ge = ey.groupby('age_band')
sanity['expensive_by_age'] = {
    'n': {k: int(v) for k, v in ge.size().items()},
    'actual_bust': {k: float(1 - v) for k, v in ge['starter'].mean().items()},
    'model_p_starter': {k: float(v) for k, v in ge['p_starter'].mean().items()}}

rb = te[te['pos'] == 'RB'].copy()
bands = [0, 5, 10, 20, 30, 40, 1000]
rb['band'] = pd.cut(rb['price_norm'], bands)
grb = rb.groupby('band', observed=True)
sanity['rb_price_bands'] = {
    str(k): {'n': int(grb.size()[k]), 'actual_hit': float(grb['starter'].mean()[k]),
             'actual_top5': float(grb['top5'].mean()[k]),
             'model_p_starter': float(grb['p_starter'].mean()[k]),
             'model_p_top5': float(grb['p_top5'].mean()[k])}
    for k in grb.size().index}

top10 = df[(df['pos_rank'] <= 10) & (df['season'] >= 2021)]
sanity['median_price_top10_finishers'] = float(top10['price_norm'].median())
sanity['top5_rate_by_position_test'] = {
    p: float(te[te['pos'] == p]['top5'].mean()) for p in POS4}
sanity['starter_rate_by_position_test'] = {
    p: float(te[te['pos'] == p]['starter'].mean()) for p in POS4}

# fair-price calibration constant, from train: dollars paid above $1 per unit of
# positive realized VOR (the market's revealed $/VOR exchange rate).
pos_vor = tr[tr['vor'] > 0]
K_FAIR = float((pos_vor['price_norm'] - 1).sum() / pos_vor['vor'].sum())
metrics['fair_price_k'] = K_FAIR

# ----------------------------------------------------------------------------
# Persist the bundle
# ----------------------------------------------------------------------------
for name, m in models.items():
    m.save_model(os.path.join(OUT, f'{name}.json'))
models_meta = {
    'features': FEATS, 'features_noprice': FEATS_NOPRICE,
    'features_leaked_variant': FEATS_LEAK,
    'positions': POS4,
    'targets': {'top5': 'pos_rank <= 5 in the player\'s league that season',
                'starter': 'season points >= league replacement points (VOR >= 0); '
                           'replacement = QB~25th/RB~30th/WR~35th/TE~12th scorer in 10T SF',
                'vor': 'season points minus league replacement points (league scoring)'},
    'train_seasons': '<=2023', 'test_seasons': '2024-2025',
    'platt_calibration': platt,
    'prediction_note': 'predict.py returns raw booster probabilities AND Platt-calibrated '
                       'ones (calibrator fit on 2023 only). Held-out 2024-25 calibration is '
                       'mediocre either way (starter ECE ~0.07-0.08); treat probabilities as '
                       'rankings with rough magnitudes, not exact odds.',
    'fair_price_k_dollars_per_vor': K_FAIR,
    'xgb_version': xgb.__version__}
json.dump(models_meta, open(os.path.join(OUT, 'features.json'), 'w'), indent=1)
json.dump(metrics, open(os.path.join(OUT, 'metrics.json'), 'w'), indent=1,
          default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
json.dump(imp, open(os.path.join(OUT, 'importance.json'), 'w'), indent=1)
json.dump(sanity, open(os.path.join(OUT, 'sanity_checks.json'), 'w'), indent=1,
          default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
json.dump({t: {'rank': r, 'tier': tier} for t, (r, tier) in OFF[2025].items()},
          open(os.path.join(OUT, 'team_offense_2025.json'), 'w'), indent=1)
te.to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)

print(json.dumps({k: metrics[k] for k in ['overall', 'vor', 'fair_price_k']}, indent=1,
                 default=str), flush=True)
print('IMPORTANCE starter:', imp['starter'], flush=True)
print('IMPORTANCE top5:', imp['top5'], flush=True)
print('SANITY:', json.dumps(sanity, indent=1, default=str), flush=True)
print('DONE', flush=True)
