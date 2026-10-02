#!/usr/bin/env python3
"""
Draft-day inference for the 10T auction SUPERFLEX model bundle (v3, loop).

Usage:
    from predict import predict_player
    predict_player(pos='RB', age=26, price=32, prior_games_missed=1, team='DET',
                   ecr_pos=8, prior_ppg=17.2, prior_rank=6)

Core inputs (same interface as v1/v2):
    pos                 'QB' | 'RB' | 'WR' | 'TE'
    age                 age at draft
    price               auction price in YOUR league's budget
    prior_games_missed  games missed in the prior NFL season (0-17);
                        None = rookie / no prior NFL season on record
    team                NFL abbreviation. (Prior-year offense rank/tier is no
                        longer a model feature — loop e03 showed it adds
                        nothing once ECR/production are in. Kept for
                        backward compatibility; ignored by the models.)
    prior_off_rank      likewise ignored by the v3 models.
    budget              your league's auction budget (default 200)

Optional v2 inputs (all draft-day knowable; None -> model treats as missing):
    ecr_overall, ecr_pos  FantasyPros preseason consensus ranks (overall /
                          positional). The single most valuable add-on.
    ecr_sd                expert sd of the overall ECR (disagreement)
    prior_ppg             prior-season fantasy points/game (PPR)
    prior_pts             prior-season total PPR points
    prior_rank            prior-season NFL positional finish (PPR rank)
    snap_share            prior-season offensive snap share (0-1)
    target_share, rush_share   prior-season team opportunity shares (0-1)
    opp_pg                prior-season (targets + carries) per game
    years_exp, draft_round, draft_pick   NFL draft capital / experience

Optional v3 inputs (starter head only; None -> missing):
    pts_per_opp           prior-season PPR points per opportunity
                          (derived from prior_ppg/opp_pg when not supplied)
    pts_per_snap, catch_rate, adot, rz_share,
    pass_att_pg, pass_rtg, pass_td_rate, qb_rush_pg

Returns the v1/v2 keys PLUS p_top5_xgb / p_starter_xgb:
    p_top5, p_starter     the shipped probabilities: a 50/50 ensemble of the
        XGBoost model and the 5-feature LR-extended logistic (loop e05;
        holdout top-5 AUC 0.869, starter 0.808). When ecr_pos/prior_ppg are
        not supplied, the logistic leans on train medians and the ensemble
        leans more heavily on price - see sparse_profile.
    p_top5_xgb, p_starter_xgb   the pure XGBoost probabilities.
    p_top5_lrext, p_starter_lrext   the pure logistic probabilities.
    head_notes           per-target reliability guidance for this position.

Calibration honesty (validation_report.md s.12): no recalibrator survived
the loop - train-fold OOF chose shrink s=1.0 in every cell, so *_cal now
returns the raw probabilities unchanged. Read top-5 raw as rough odds
(holdout ECE 0.022), starter raw as a ranking with approximate magnitudes
(ECE 0.052).
"""
import json, math, os
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
META = json.load(open(os.path.join(HERE, 'features.json')))
FEATS_BY_TARGET = META.get('features_by_target') or {
    'top5': META['features'], 'starter': META['features']}
FEATS_NOPRICE = META['features_noprice']
CAL = META['calibration_crossfit']
K_FAIR = META['fair_price_k_dollars_per_vor']
K_POS = META.get('fair_price_k_by_position', {})
HEAD = META['head_reliability']
LRE = META['lr_extended']
TEAM_OFF = json.load(open(os.path.join(HERE, 'team_offense_2025.json')))
POS4 = META['positions']
ENS_W = (META.get('selection', {}).get('ensemble', {}) or {}).get(
    'xgb_weight', 0.5)

_B = {}
def _booster(name):
    if name not in _B:
        b = xgb.Booster()
        b.load_model(os.path.join(HERE, f'{name}.json'))
        _B[name] = b
    return _B[name]

def _cal(target, p):
    e = CAL[target]
    if e['method'] == 'platt':
        c = e['platt']
        p = min(max(float(p), 1e-6), 1 - 1e-6)
        z = c['intercept'] + c['coef'] * math.log(p / (1 - p))
        return 1.0 / (1.0 + math.exp(-z))
    if e['method'] == 'isotonic':
        return float(np.interp(p, e['isotonic']['x'], e['isotonic']['y']))
    return float(p)

def _lre(target, vals):
    m = LRE[target]
    z = m['intercept']
    for f, c in zip(m['features'], m['coef']):
        v = vals.get(f)
        z += c * (m['medians'][f] if v is None or (isinstance(v, float) and math.isnan(v)) else v)
    return 1.0 / (1.0 + math.exp(-z))

def predict_player(pos, age, price, prior_games_missed=None, team=None,
                   prior_off_rank=None, budget=200,
                   ecr_overall=None, ecr_pos=None, ecr_sd=None,
                   prior_ppg=None, prior_pts=None, prior_rank=None,
                   snap_share=None, target_share=None, rush_share=None,
                   opp_pg=None, years_exp=None, draft_round=None,
                   draft_pick=None,
                   pts_per_opp=None, pts_per_snap=None, catch_rate=None,
                   adot=None, rz_share=None, pass_att_pg=None, pass_rtg=None,
                   pass_td_rate=None, qb_rush_pg=None):
    pos = pos.upper()
    if pos not in POS4:
        raise ValueError(f'pos must be one of {POS4}')
    price_norm = float(price) * 200.0 / float(budget)
    age_f = float(age) if age is not None else np.nan
    if prior_games_missed is None:
        missed, has_prior, missed5 = np.nan, 0.0, 0.0
    else:
        missed = float(min(max(prior_games_missed, 0), 17))
        has_prior, missed5 = 1.0, 1.0 if missed >= 5 else 0.0
    rank, tier = np.nan, None
    if prior_off_rank is not None:
        rank = float(prior_off_rank)
        tier = 'top' if rank <= 10 else ('bottom' if rank >= 22 else 'mid')
    elif team:
        info = TEAM_OFF.get(team.upper())
        if info:
            rank, tier = float(info['rank']), info['tier']
    def num(v):
        return float(v) if v is not None else np.nan
    ppg, snap = num(prior_ppg), num(snap_share)
    if pts_per_opp is None and prior_ppg is not None and opp_pg:
        pts_per_opp = prior_ppg / opp_pg
    row = {
        'price_norm': price_norm, 'age': age_f,
        'missed_prior': missed, 'missed5': missed5, 'has_prior': has_prior,
        'prior_off_rank': rank,
        'prio_top': 1.0 if tier == 'top' else 0.0,
        'prio_mid': 1.0 if tier == 'mid' else 0.0,
        'prio_bottom': 1.0 if tier == 'bottom' else 0.0,
        'ecr_overall': num(ecr_overall), 'ecr_pos': num(ecr_pos),
        'ecr_sd': num(ecr_sd),
        'ecr_overall_log': np.log1p(ecr_overall) if ecr_overall is not None else np.nan,
        'ecr_pos_log': np.log1p(ecr_pos) if ecr_pos is not None else np.nan,
        'prior_pts_ppr': num(prior_pts), 'prior_ppg_ppr': ppg,
        'prior_rank_ppr': num(prior_rank), 'prior_snap_share': snap,
        'prior_target_share': num(target_share),
        'prior_rush_share': num(rush_share), 'prior_opp_pg': num(opp_pg),
        'years_exp': num(years_exp), 'draft_round': num(draft_round),
        'draft_pick': num(draft_pick),
        'age_sq': age_f ** 2 if age is not None else np.nan,
        'exp_youth': 1.0 if (price_norm >= 25 and age is not None and age <= 24) else 0.0,
        'ppg_x_snap': ppg * snap if (prior_ppg is not None and snap_share is not None) else np.nan,
        # v3 VAL features
        'ppg_per_dollar': ppg / max(price_norm, 1.0) if prior_ppg is not None else np.nan,
        'price_x_age': price_norm * age_f / 100.0 if age is not None else np.nan,
        'ecr_vs_prior_rank': (float(ecr_pos) - float(prior_rank))
                             if (ecr_pos is not None and prior_rank is not None) else np.nan,
        'price_per_ecr': price_norm / max(float(ecr_pos), 1.0) if ecr_pos is not None else np.nan,
        # v3 EFF features (starter head)
        'prior_pts_per_opp': num(pts_per_opp),
        'prior_pts_per_snap': num(pts_per_snap),
        'prior_catch_rate': num(catch_rate), 'prior_adot': num(adot),
        'prior_rz_share': num(rz_share),
        'prior_pass_att_pg': num(pass_att_pg), 'prior_pass_rtg': num(pass_rtg),
        'prior_pass_td_rate': num(pass_td_rate),
        'prior_qb_rush_pg': num(qb_rush_pg),
        **{f'pos_{p}': 1.0 if p == pos else 0.0 for p in POS4},
    }
    X5 = pd.DataFrame([row])[FEATS_BY_TARGET['top5']]
    XS = pd.DataFrame([row])[FEATS_BY_TARGET['starter']]
    Xn = pd.DataFrame([row])[FEATS_NOPRICE]
    x_top5 = float(_booster('model_top5').predict(xgb.DMatrix(X5))[0])
    x_starter = float(_booster('model_starter').predict(xgb.DMatrix(XS))[0])
    vor_price = float(_booster('model_vor').predict(xgb.DMatrix(X5))[0])
    vor_prof = float(_booster('model_vor_noprice').predict(xgb.DMatrix(Xn))[0])
    fair = max(1.0, 1.0 + K_POS.get(pos, K_FAIR) * max(vor_prof, 0.0))
    lre_vals = {'price_norm': price_norm, 'ecr_pos': row['ecr_pos'],
                'prior_ppg_ppr': ppg, 'age': age_f, 'missed_prior': missed}
    l_top5 = _lre('top5', lre_vals)
    l_starter = _lre('starter', lre_vals)
    p_top5 = ENS_W * x_top5 + (1 - ENS_W) * l_top5
    p_starter = ENS_W * x_starter + (1 - ENS_W) * l_starter
    notes = {t: {'best_tool': HEAD[t][pos]['best_tool'],
                 'ensemble_auc': HEAD[t][pos].get('ensemble_auc'),
                 'xgb_auc': HEAD[t][pos]['xgb_auc'],
                 'note': HEAD[t][pos]['note']} for t in ['top5', 'starter']}
    supplied = [ecr_pos is not None, prior_ppg is not None, prior_rank is not None,
                snap_share is not None, opp_pg is not None]
    return {
        'pos': pos, 'price_norm_200': round(price_norm, 1),
        'profile_completeness': round(float(np.mean(supplied)), 2),
        'sparse_profile': bool(np.mean(supplied) < 0.6),
        'p_top5': round(p_top5, 3), 'p_top5_cal': round(_cal('top5', p_top5), 3),
        'p_starter': round(p_starter, 3),
        'p_starter_cal': round(_cal('starter', p_starter), 3),
        'p_top5_xgb': round(x_top5, 3), 'p_starter_xgb': round(x_starter, 3),
        'p_top5_lrext': round(l_top5, 3), 'p_starter_lrext': round(l_starter, 3),
        'vor_expected_at_price': round(vor_price, 1),
        'vor_profile': round(vor_prof, 1),
        'fair_price_200scale': int(round(fair)),
        'head_notes': notes,
    }

if __name__ == '__main__':
    examples = [
        dict(pos='RB', age=26, price=32, prior_games_missed=1, team='DET',
             ecr_pos=8, ecr_overall=22, ecr_sd=3.1, prior_ppg=16.8, prior_pts=269,
             prior_rank=9, snap_share=0.62, target_share=0.17, rush_share=0.44,
             opp_pg=14.1, years_exp=4, draft_round=2),
        dict(pos='RB', age=23, price=45, prior_games_missed=6, team='NYG',
             ecr_pos=14, ecr_overall=41, prior_ppg=11.9, prior_pts=143,
             prior_rank=28, snap_share=0.55, opp_pg=13.0, years_exp=2),
        dict(pos='QB', age=28, price=45, prior_games_missed=0, team='BUF',
             ecr_pos=2, ecr_overall=30, ecr_sd=1.8, prior_ppg=24.5,
             prior_pts=392, prior_rank=1, snap_share=1.0, rush_share=0.09,
             opp_pg=36.0, years_exp=7, draft_round=1),
        dict(pos='TE', age=27, price=8, prior_games_missed=0, team='KC',
             ecr_pos=6, ecr_overall=75, prior_ppg=9.8, prior_pts=157,
             prior_rank=7, snap_share=0.85, target_share=0.21, opp_pg=6.2,
             years_exp=5, draft_round=3),
        dict(pos='WR', age=22, price=25, prior_games_missed=None, team='JAX',
             ecr_pos=24, ecr_overall=55, draft_round=1, years_exp=0),
    ]
    for e in examples:
        print(e, '\n  ->', predict_player(**e))
