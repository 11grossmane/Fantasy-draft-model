#!/usr/bin/env python3
"""Build the tidy FAAB bids dataset from raw Sleeper dumps.
One row per waiver BID (complete + failed). Auction = (season, week, player_id).
Also computes each roster's FAAB remaining before each bid, from completed
waiver spend (and trade FAAB transfers where Sleeper records them)."""
import json, os
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.expanduser('~/workspace/sleeper-league-data/faab')
RAW = os.path.join(ROOT, 'raw')
DENVER = ZoneInfo('America/Denver')
ERIC_ID = '597851820731203584'
BUDGET = 100

players = json.load(open(os.path.join(RAW, 'players.json')))

def pname(pid):
    p = players.get(str(pid))
    if not p:
        return str(pid), 'DEF'
    name = p.get('full_name') or p.get('last_name') or str(pid)
    return name, p.get('position') or '?'

rows = []
for season in range(2020, 2027):
    users = {u['user_id']: u for u in json.load(open(f'{RAW}/{season}_users.json'))}
    rosters = {r['roster_id']: r for r in json.load(open(f'{RAW}/{season}_rosters.json'))}
    rid2mgr = {}
    for rid, r in rosters.items():
        uid = r.get('owner_id')
        u = users.get(uid, {})
        rid2mgr[rid] = (uid, u.get('display_name') or u.get('username') or str(uid))
    max_week = 6 if season == 2026 else 18
    season_tx = []
    for wk in range(1, max_week + 1):
        season_tx += json.load(open(f'{RAW}/{season}_tx_w{wk:02d}.json'))
    # chronological spend ledger per roster (completed waivers + trade FAAB moves)
    events = []
    for t in season_tx:
        if t.get('type') == 'waiver' and t.get('status') == 'complete' and t.get('adds'):
            bid = (t.get('settings') or {}).get('waiver_bid') or 0
            events.append((t['status_updated'], 'spend', t['roster_ids'][0], bid))
        elif t.get('type') == 'trade':
            for mv in (t.get('waiver_budget') or []):
                events.append((t['status_updated'], 'trade', mv.get('roster_id'), mv.get('amount', 0),
                               mv.get('sender')))
    events.sort(key=lambda e: e[0])
    ledger = {rid: BUDGET for rid in rosters}
    spend_at = {}  # id(tx) not stable; use (status_updated, roster, player) key map built below
    # apply events in order, recording remaining-after for completed waivers by timestamp+roster
    remaining_before = {}
    for e in events:
        if e[1] == 'spend':
            _, _, rid, bid = e
            remaining_before[(e[0], rid)] = ledger.get(rid, BUDGET)
            ledger[rid] = ledger.get(rid, BUDGET) - bid
    # extract bids
    for t in season_tx:
        if t.get('type') != 'waiver' or not t.get('adds'):
            continue
        bid = (t.get('settings') or {}).get('waiver_bid')
        bid = 0 if bid is None else bid
        rid = t['roster_ids'][0]
        uid, mname = rid2mgr.get(rid, (None, str(rid)))
        proc = datetime.fromtimestamp(t['status_updated'] / 1000, tz=timezone.utc)
        created = datetime.fromtimestamp(t['created'] / 1000, tz=timezone.utc)
        for pid in t['adds'].keys():
            nm, pos = pname(pid)
            rows.append(dict(
                season=season, week=t.get('leg'), player_id=str(pid), player=nm, position=pos,
                bidder_roster=rid, bidder_user_id=uid, bidder_manager=mname,
                is_eric=(uid == ERIC_ID), bid=int(bid),
                won=(t['status'] == 'complete'),
                status=t['status'],
                processed_ts=t['status_updated'], created_ts=t['created'],
                processed_date=proc.astimezone(DENVER).strftime('%Y-%m-%d %a'),
                processed_weekday=proc.astimezone(DENVER).strftime('%a'),
                faab_remaining_before=remaining_before.get((t['status_updated'], rid)),
                drop_player_id=(list(t['drops'].keys())[0] if t.get('drops') else None),
            ))

df = pd.DataFrame(rows)
df['auction_id'] = (df.season.astype(str) + '_' + df.player_id + '_' +
                    df.processed_ts.astype(str))
# auction-level fields
grp = df.groupby('auction_id')
sizes = grp.size().rename('num_bidders')
df = df.merge(sizes, on='auction_id')
win = df[df.won].set_index('auction_id')['bid']
df['winning_bid'] = df.set_index('auction_id').index.map(win)
df['second_bid'] = df.groupby('auction_id')['bid'].transform(
    lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else 0)
df['won_by_eric'] = df.groupby('auction_id')['is_eric'].transform('max') & df.won
df = df.sort_values(['season', 'week', 'processed_ts', 'player']).reset_index(drop=True)
out = os.path.join(ROOT, 'faab_bids.csv')
df.to_csv(out, index=False)
print(f'rows (bids): {len(df)}  auctions: {df.auction_id.nunique()}')
print(df.groupby('season').agg(bids=('bid', 'size'), auctions=('player_id', 'nunique'),
                               med_bid=('bid', 'median'), med_win=('winning_bid', 'median')).to_string())
print('\nprocessed weekday share (Denver):')
print(df[df.won].processed_weekday.value_counts(normalize=True).head(4))
print('\nbids by status:'); print(df.status.value_counts().to_string())
print('\nmanagers:'); print(df.groupby('bidder_manager').agg(bids=('bid','size'), seasons=('season','nunique')).sort_values('bids', ascending=False).to_string())
df.to_pickle(os.path.join(ROOT, 'checkpoints', 'faab_bids.pkl'))
