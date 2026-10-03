#!/usr/bin/env python3
"""
FAAB bid predictor for Fantasy 101 (Eric's league only).

Usage:
    from faab_predict import predict_player_faab, predict_position_faab

    predict_player_faab('Jaylen Warren')          # looks up 2026 form
    predict_position_faab('RB', week=6, prev_week_pts=14.2, ppg_before=11.0,
                          prev_pts_pct=0.9)

Returns:
    clearing_quantiles  predicted distribution of the price to beat
                        (max rival bid): q10/q25/q50/q75/q90
    needed_bid          what Eric should bid for ~50/75/90% win probability
                        (quantile + $1, to clear ties)
    rivals              top-3 likely rival bidders with their likely bid
                        ranges and digit habits. HONEST CAVEAT: naming the
                        winner is barely better than chance in validation
                        (34% vs 33%); treat rivals as 'who usually pays up
                        for this position', not a pick'em. When the winner
                        was in the top-3, their bid landed in the predicted
                        range 81% of the time.

Validation (train 2020-24 -> cold test 2025-26, n=173 auctions):
    50% band coverage 71% (conservative), 80% band 89%;
    actual winning bid exceeded q75 28% of the time (ideal 25%) and
    q90 11% (ideal 10%) — the upper tail Eric bids against is calibrated.
    Median prediction MAE $4.22 vs $4.81 for the position x phase baseline.
"""
import json, os, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'model', 'quantile_models.pkl'), 'rb') as f:
    BUNDLE = pickle.load(f)
MODELS, FEATS, POS = BUNDLE['models'], BUNDLE['feats'], BUNDLE['pos']
RIVALS = pd.read_csv(os.path.join(HERE, 'model', 'rival_tables.csv'))
ERIC = 'gizzle4'
MANAGERS = sorted(RIVALS.manager.unique())

def _phase(w):
    return 'early' if w <= 5 else ('mid' if w <= 10 else 'late')

def _features(position, week, prev_week_pts, ppg_before, games_before,
              prev_pts_pct, faab_max, faab_med):
    x = dict(week=week, prev_week_pts=prev_week_pts, ppg_before=ppg_before,
             games_before=games_before, prev_pts_pct=prev_pts_pct,
             faab_max=faab_max, faab_med=faab_med,
             phase_mid=int(_phase(week) == 'mid'),
             phase_late=int(_phase(week) == 'late'))
    for p in POS:
        x[f'pos_{p}'] = int(position == p)
    return pd.DataFrame([x])[FEATS]

def predict_position_faab(position, week=6, prev_week_pts=None, ppg_before=None,
                          games_before=None, prev_pts_pct=None,
                          faab_max=60, faab_med=45):
    """faab_max/faab_med: richest / median rival FAAB remaining (league
    landscape). Defaults reflect a typical mid-season state; pass current
    standings for live use."""
    position = position.upper()
    if position == 'DST':
        position = 'DEF'
    X = _features(position, week, prev_week_pts, ppg_before, games_before,
                  prev_pts_pct, faab_max, faab_med)
    qs = {f'q{int(q*100)}': max(0.0, float(m.predict(X)[0]))
          for q, m in MODELS.items()}
    needed = {lvl: int(np.ceil(qs[q])) + 1
              for lvl, q in [('p50', 'q50'), ('p75', 'q75'), ('p90', 'q90')]}
    # rivals: validation-selected p_bid x q75 scoring
    cand = []
    sub = RIVALS[RIVALS.position == position]
    for _, r in sub.iterrows():
        if r.manager == ERIC:
            continue
        cand.append(dict(manager=r.manager,
                         bid_range=f"${r.q25:.0f}-${r.q75:.0f}",
                         typical_bid=f"${r.med:.0f}",
                         digit_note=('likes 4/9 endings (one under a multiple of 5)'
                                     if r.pct_49 >= 0.14 else
                                     ('bids round 0/5 numbers' if r.pct_05 >= 0.5 else '')),
                         score=float(r.p_bid) * float(r.q75)))
    cand.sort(key=lambda c: -c['score'])
    return dict(position=position, week=week,
                clearing_quantiles={k: round(v) for k, v in qs.items()},
                needed_bid=needed,
                rivals=[{k: v for k, v in c.items() if k != 'score'} for c in cand[:3]],
                note='Rival names are who-habitually-pays, not a reliable winner call '
                     '(validated ~34% top-3 vs 33% chance). Bid amounts are the '
                     'validated part.')

def _player_current_form(player_name, season=2026, through_week=4):
    """Look up a player's current-season form from cached Sleeper stats."""
    players = json.load(open(os.path.join(HERE, 'raw', 'players.json')))
    matches = [(pid, p) for pid, p in players.items()
               if (p.get('full_name') or '').lower() == player_name.lower()]
    if not matches:
        matches = [(pid, p) for pid, p in players.items()
                   if player_name.lower() in (p.get('full_name') or '').lower()]
    if not matches:
        raise ValueError(f'player not found: {player_name}')
    pid, p = matches[0]
    stats_dir = os.path.join(HERE, 'raw', 'stats')
    pts, last = [], None
    for wk in range(1, through_week + 1):
        f = os.path.join(stats_dir, f'{season}_w{wk:02d}.json')
        if os.path.exists(f):
            v = json.load(open(f)).get(pid)
            if v is not None:
                pts.append(v); last = v
    # positional percentile of last week's points
    pct = None
    if last is not None:
        pool = json.load(open(os.path.join(
            stats_dir, f'{season}_w{through_week:02d}.json')))
        peers = [v for p2, v in pool.items()
                 if (players.get(p2) or {}).get('position') == p.get('position')]
        if peers:
            pct = float(np.mean(np.array(peers) <= last))
    return dict(player=p.get('full_name'), position=p.get('position'),
                team=p.get('team'), prev_week_pts=last,
                ppg_before=float(np.mean(pts)) if pts else None,
                games_before=len(pts), prev_pts_pct=pct)

def predict_player_faab(player_name, week=6, **kw):
    form = _player_current_form(player_name)
    out = predict_position_faab(form['position'], week=week,
                                prev_week_pts=form['prev_week_pts'],
                                ppg_before=form['ppg_before'],
                                games_before=form['games_before'],
                                prev_pts_pct=form['prev_pts_pct'], **kw)
    out['player'] = form
    return out

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        print(json.dumps(predict_player_faab(' '.join(sys.argv[1:])), indent=1))
    else:
        print(json.dumps(predict_position_faab(
            'RB', week=6, prev_week_pts=14.2, ppg_before=11.0,
            games_before=4, prev_pts_pct=0.9), indent=1))
