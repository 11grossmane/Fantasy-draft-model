#!/usr/bin/env python3
"""Collect all waiver transactions for the Fantasy 101 chain from Sleeper.
Saves raw JSON per league/season into faab/raw/. Durable, resumable."""
import json, os, time, urllib.request

ROOT = os.path.expanduser('~/workspace/sleeper-league-data/faab')
RAW = os.path.join(ROOT, 'raw')
API = 'https://api.sleeper.app/v1'

CHAIN = {  # season -> league_id
    2020: '596553726760632320',
    2021: '726144978962747392',
    2022: '862818439088189440',
    2023: '994041566085910528',
    2024: '1124822672346198016',
    2025: '1257436698095136768',
    2026: '1389719640493551616',
}

def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'faab-collect/1.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))

def save(name, obj):
    p = os.path.join(RAW, name)
    with open(p, 'w') as f:
        json.dump(obj, f)
    return p

def main():
    log = []
    for season, lid in CHAIN.items():
        meta_f = f'{season}_league.json'
        if not os.path.exists(os.path.join(RAW, meta_f)):
            save(meta_f, get(f'{API}/league/{lid}'))
            save(f'{season}_users.json', get(f'{API}/league/{lid}/users'))
            save(f'{season}_rosters.json', get(f'{API}/league/{lid}/rosters'))
        max_week = 6 if season == 2026 else 18
        n_tx = 0
        for wk in range(1, max_week + 1):
            f = f'{season}_tx_w{wk:02d}.json'
            fp = os.path.join(RAW, f)
            if os.path.exists(fp):
                n_tx += len(json.load(open(fp)))
                continue
            tx = get(f'{API}/league/{lid}/transactions/{wk}')
            save(f, tx)
            n_tx += len(tx)
            time.sleep(0.15)
        log.append(f'{season}: {n_tx} transactions cached')
        print(log[-1], flush=True)
    pf = os.path.join(RAW, 'players.json')
    if not os.path.exists(pf):
        print('fetching players DB (large)...', flush=True)
        save('players.json', get(f'{API}/players/nfl'))
        print('players DB saved', flush=True)
    with open(os.path.join(ROOT, 'checkpoints', 'collect_log.txt'), 'w') as f:
        f.write('\n'.join(log) + '\nDONE\n')

if __name__ == '__main__':
    main()
