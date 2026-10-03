#!/usr/bin/env python3
"""Enrich bids with player demand features from weekly stats:
prev_week_pts, ppg_before (season-to-date before claim), games_before,
prev_pts percentile among same-position players that week."""
import json, os
import pandas as pd
import numpy as np

ROOT = os.path.expanduser('~/workspace/sleeper-league-data/faab')
STATS = os.path.join(ROOT, 'raw', 'stats')
df = pd.read_csv(os.path.join(ROOT, 'faab_bids.csv'), dtype={'player_id': str})
players = json.load(open(os.path.join(ROOT, 'raw', 'players.json')))

def load(season, wk):
    f = os.path.join(STATS, f'{season}_w{wk:02d}.json')
    return json.load(open(f)) if os.path.exists(f) else {}

# position lookup for percentile pools
pos_of = {pid: (p.get('position') or '?') for pid, p in players.items()}

cache = {}
def week_stats(season, wk):
    if (season, wk) not in cache:
        cache[(season, wk)] = load(season, wk)
    return cache[(season, wk)]

rows = []
auct = df.drop_duplicates('auction_id')[['auction_id', 'season', 'week', 'player_id', 'position']]
for _, r in auct.iterrows():
    s, w, pid = int(r.season), int(r.week), r.player_id
    prev = week_stats(s, w - 1).get(pid) if w > 1 else None
    tot, gms = 0.0, 0
    for wk in range(1, w):
        v = week_stats(s, wk).get(pid)
        if v is not None:
            tot += v; gms += 1
    # percentile of prev-week pts among same-position peers
    pct = np.nan
    if prev is not None and w > 1:
        pool = week_stats(s, w - 1)
        peers = [v for p2, v in pool.items() if pos_of.get(p2) == r.position]
        if peers:
            pct = float(np.mean(np.array(peers) <= prev))
    rows.append(dict(auction_id=r.auction_id, prev_week_pts=prev,
                     ppg_before=(tot / gms if gms else np.nan), games_before=gms,
                     prev_pts_pct=pct))
enr = pd.DataFrame(rows)
df = df.merge(enr, on='auction_id', how='left')
df.to_csv(os.path.join(ROOT, 'faab_bids_enriched.csv'), index=False)
print('enriched rows:', len(df))
print(df[['prev_week_pts', 'ppg_before', 'games_before', 'prev_pts_pct']].describe().round(2).to_string())
print('prev_week_pts coverage by season:')
print(df.groupby('season').prev_week_pts.apply(lambda s: s.notna().mean()).round(2).to_string())
