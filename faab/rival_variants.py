#!/usr/bin/env python3
"""Rival-module scoring variants, selected on 2023-24 validation ONLY,
then evaluated once on 2025-26 test. No test peeking for selection."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.expanduser('~/workspace/sleeper-league-data/faab')
ERIC = 'gizzle4'
df = pd.read_csv(os.path.join(ROOT, 'faab_bids_enriched.csv'), dtype={'player_id': str})
canon = (df.sort_values('season').groupby('bidder_user_id')['bidder_manager']
         .agg(lambda s: s.iloc[-1]))
df['manager'] = df['bidder_user_id'].map(canon)

au = (df[df.won].drop_duplicates('auction_id')
      [['auction_id', 'season', 'week', 'position', 'winning_bid']])
win_mgr = df[df.won].set_index('auction_id')['manager']
au['winner'] = au.auction_id.map(win_mgr)
MANAGERS = sorted(m for m in df.manager.unique() if m != ERIC)

def build_tables(bids):
    league_q = bids.groupby('position').bid.quantile([0.25, 0.5, 0.75]).unstack()
    rows = {}
    for (mgr, pos), g in bids.groupby(['manager', 'position']):
        q = g.bid.quantile([0.25, 0.5, 0.75])
        wgt = len(g) / (len(g) + 12)
        rows[(mgr, pos)] = dict(
            n=len(g), p_bid=len(g) / max(1, len(bids)),
            med=wgt * q.get(0.5, 0) + (1 - wgt) * league_q.loc[pos, 0.5],
            q75=wgt * q.get(0.75, 0) + (1 - wgt) * league_q.loc[pos, 0.75],
            mx=float(g.bid.max()))
    return rows

def top3(tables, pos, mode):
    cand = []
    for mgr in MANAGERS:
        r = tables.get((mgr, pos))
        if r is None:
            continue
        score = {'p_x_med': r['p_bid'] * r['med'],
                 'p_x_q75': r['p_bid'] * r['q75'],
                 'q75_only': r['q75'],
                 'med_only': r['med'],
                 'max_only': r['mx'] * r['p_bid']}[mode]
        cand.append((mgr, score))
    return [m for m, _ in sorted(cand, key=lambda c: -c[1])[:3]]

def evaluate(train_seasons, eval_seasons, mode):
    tables = build_tables(df[df.season.isin(train_seasons)])
    ev = au[au.season.isin(eval_seasons)]
    hits = sum(a.winner in top3(tables, a.position, mode) for a in ev.itertuples())
    return hits / max(1, len(ev)), len(ev)

MODES = ['p_x_med', 'p_x_q75', 'q75_only', 'med_only', 'max_only']
print('validation (fit 2020-22 -> eval 2023-24):')
val = {}
for m in MODES:
    val[m], n = evaluate(range(2020, 2023), [2023, 2024], m)
    print(f'  {m:10s} top3 hit {val[m]:.3f} (n={n})')
best = max(val, key=val.get)
print(f'selected on validation: {best}')
print('test (fit 2020-24 -> eval 2025-26):')
for m in MODES:
    h, n = evaluate(range(2020, 2025), [2025, 2026], m)
    tag = ' <- selected' if m == best else ''
    print(f'  {m:10s} top3 hit {h:.3f} (n={n}){tag}')
print(f'chance = {3/9:.3f}')
