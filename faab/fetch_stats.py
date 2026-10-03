#!/usr/bin/env python3
"""Fetch weekly PPR points for all seasons 2020-2026 from Sleeper stats API.
Caches slim {player_id: pts_ppr} JSON per season-week under raw/stats/."""
import json, os, time, urllib.request

ROOT = os.path.expanduser('~/workspace/sleeper-league-data/faab')
OUT = os.path.join(ROOT, 'raw', 'stats')
os.makedirs(OUT, exist_ok=True)

def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'faab-stats/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))

for season in range(2020, 2027):
    maxw = 5 if season == 2026 else 18
    for wk in range(1, maxw + 1):
        f = os.path.join(OUT, f'{season}_w{wk:02d}.json')
        if os.path.exists(f):
            continue
        d = get(f'https://api.sleeper.app/v1/stats/nfl/regular/{season}/{wk}')
        slim = {pid: s.get('pts_ppr') for pid, s in d.items()
                if isinstance(s, dict) and s.get('pts_ppr') is not None}
        with open(f, 'w') as fh:
            json.dump(slim, fh)
        time.sleep(0.1)
    print(f'{season} done', flush=True)
print('ALL DONE')
