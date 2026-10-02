#!/usr/bin/env python3
"""
Build v2 draft-day-legal features for the 10T auction SF model.

Adds on top of the v1 frame (training_data.csv, 41,659 rows):
  - FantasyPros preseason consensus (DynastyProcess db_fpecr archive):
    latest redraft scrape inside [Aug 1, Sep 10] of the draft year
    (if a season has no scrape in that window, widen to [Jul 1, Sep 15]).
    Features: ecr_overall (redraft-overall page), ecr_pos (redraft-<pos> page),
    ecr_sd (expert sd on the overall page), plus log ecr transforms.
  - Prior-season production/usage from stats_{season-1}.json (Sleeper cache):
    prior_ppg_ppr, prior_pts_ppr, prior_rank_ppr (NFL-wide positional finish),
    prior_snap_share (off_snp/tm_off_snp), prior_target_share,
    prior_rush_share (team totals via players_nfl team attribution, same
    method/noise as the v1 offense rebuild), prior_opp_pg (tgt+rush_att per gp).
  - NFL draft capital from db_playerids: draft_round, draft_pick,
    years_exp = season - draft_year.
  - Interactions: exp_youth = 1 if price>=25 and age<=24; age_sq.
ALL features are knowable before the season starts. No same-season data.
"""
import json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.expanduser('~/workspace/sleeper-league-data')
CACHE = os.path.join(ROOT, 'crossleague-cache')
STATS = os.path.join(CACHE, 'targeted', 'stats')
V2 = os.path.join(CACHE, 'v2')
OUT = os.path.join(ROOT, 'model')
POS4 = ['QB', 'RB', 'WR', 'TE']
TEAM_FIX = {'WSH': 'WAS'}

# ---------------------------------------------------------------- ECR
def load_ecr():
    cols = ['page_type', 'id', 'pos', 'ecr', 'sd', 'scrape_date', 'ecr_type']
    chunks = []
    for ch in pd.read_csv(os.path.join(V2, 'db_fpecr.csv.gz'), usecols=cols,
                          low_memory=False, chunksize=400_000):
        m = ch['page_type'].isin(['redraft-overall', 'redraft-qb', 'redraft-rb',
                                  'redraft-wr', 'redraft-te'])
        if m.any():
            chunks.append(ch[m])
    ecr = pd.concat(chunks, ignore_index=True)
    print('redraft ECR rows:', len(ecr), flush=True)
    print(ecr['page_type'].value_counts().to_dict(), flush=True)
    ecr['scrape_date'] = pd.to_datetime(ecr['scrape_date'])
    print('fp_page/ecr_type combos:', ecr.groupby(['page_type', 'ecr_type']).size().to_dict(),
          flush=True)
    return ecr

def preseason_ecr(ecr):
    """Latest scrape per (season, fp_id, page_type) in the preseason window."""
    out = []
    for season in range(2020, 2026):
        lo, hi = f'{season}-08-01', f'{season}-09-03'
        w = ecr[(ecr['scrape_date'] >= lo) & (ecr['scrape_date'] <= hi)]
        if w.empty:
            lo, hi = f'{season}-07-01', f'{season}-09-03'
            w = ecr[(ecr['scrape_date'] >= lo) & (ecr['scrape_date'] <= hi)]
        last = w['scrape_date'].max()
        w = w[w['scrape_date'] == last].copy()
        w['season'] = season
        print(f'ECR {season}: scrape {str(last)[:10]} rows {len(w)}', flush=True)
        out.append(w)
    pre = pd.concat(out, ignore_index=True)
    overall = pre[pre['page_type'] == 'redraft-overall'][
        ['season', 'id', 'ecr', 'sd']].rename(columns={'ecr': 'ecr_overall', 'sd': 'ecr_sd'})
    posp = pre[pre['page_type'] != 'redraft-overall'][
        ['season', 'id', 'ecr']].rename(columns={'ecr': 'ecr_pos'})
    m = overall.merge(posp, on=['season', 'id'], how='outer')
    return m

ecr = load_ecr()
pre = preseason_ecr(ecr)
pre.to_csv(os.path.join(V2, 'ecr_preseason.csv'), index=False)

# ------------------------------------------------- crosswalk (fp -> sleeper)
xw = pd.read_csv(os.path.join(V2, 'db_playerids.csv'), low_memory=False)
xw = xw.dropna(subset=['fantasypros_id', 'sleeper_id']).copy()
xw['fantasypros_id'] = xw['fantasypros_id'].astype(int)
xw['sleeper_id'] = xw['sleeper_id'].astype(int)
xw = xw.drop_duplicates('fantasypros_id')
fp2sl = dict(zip(xw['fantasypros_id'], xw['sleeper_id']))
pre['player_id'] = pre['id'].map(fp2sl)
n_match = pre['player_id'].notna().mean()
print(f'ECR crosswalk match rate: {n_match:.3f}', flush=True)
pre = pre.dropna(subset=['player_id']).copy()
pre['player_id'] = pre['player_id'].astype(int)
pre.to_csv(os.path.join(V2, 'ecr_preseason_sleeper.csv'), index=False)

# ------------------------------------------------ prior-season production
PLAYERS = json.load(open(os.path.join(CACHE, 'players_nfl.json')))
_NORM = {'DE':'DL','DT':'DL','NT':'DL','EDR':'DL','OLB':'LB','ILB':'LB','MLB':'LB',
         'CB':'DB','SS':'DB','FS':'DB','SAF':'DB'}
POSD = {pid: _NORM.get(v.get('position') or 'UNK', v.get('position') or 'UNK')
        for pid, v in PLAYERS.items()}

def team_of(pid):
    t = (PLAYERS.get(str(pid)) or {}).get('team')
    return TEAM_FIX.get(t, t) if t else None

def prior_stats(season):
    """Features from stats_{season-1}.json, keyed by sleeper player_id (int)."""
    fn = os.path.join(STATS, f'stats_{season-1}.json')
    st = json.load(open(fn))
    # team opportunity totals (players_nfl team attribution, same as v1)
    team_tgt, team_rush = {}, {}
    for pid, s in st.items():
        if not isinstance(s, dict) or not s:
            continue
        t = team_of(pid)
        if not t:
            continue
        team_tgt[t] = team_tgt.get(t, 0.0) + (s.get('rec_tgt') or 0.0)
        team_rush[t] = team_rush.get(t, 0.0) + (s.get('rush_att') or 0.0)
    rows = {}
    for pid, s in st.items():
        if not isinstance(s, dict) or not s or not str(pid).isdigit():
            continue
        gp = s.get('gp') or 0
        if gp <= 0:
            continue
        t = team_of(pid)
        pts = s.get('pts_ppr')
        snp, tsnp = s.get('off_snp'), s.get('tm_off_snp')
        rows[int(pid)] = {
            'prior_pts_ppr': pts,
            'prior_ppg_ppr': (pts / gp) if pts is not None else np.nan,
            'prior_rank_ppr': s.get('pos_rank_ppr'),
            'prior_gp_stats': gp,
            'prior_snap_share': (snp / tsnp) if (snp is not None and tsnp) else np.nan,
            'prior_target_share': ((s.get('rec_tgt') or 0.0) / team_tgt[t])
                                   if (t and team_tgt.get(t)) else np.nan,
            'prior_rush_share': ((s.get('rush_att') or 0.0) / team_rush[t])
                                 if (t and team_rush.get(t)) else np.nan,
            'prior_opp_pg': ((s.get('rec_tgt') or 0.0) + (s.get('rush_att') or 0.0)) / gp,
        }
    return rows

PRIOR = {}
for season in range(2020, 2026):
    PRIOR[season] = prior_stats(season)
    print(f'prior stats for {season}: {len(PRIOR[season])} players', flush=True)

# ------------------------------------------------------------- assemble
df = pd.read_csv(os.path.join(OUT, 'training_data.csv'), low_memory=False)
df['player_id'] = df['player_id'].astype(int)

df = df.merge(pre[['season', 'player_id', 'ecr_overall', 'ecr_pos', 'ecr_sd']],
              on=['season', 'player_id'], how='left')

pf = ['prior_pts_ppr', 'prior_ppg_ppr', 'prior_rank_ppr', 'prior_gp_stats',
      'prior_snap_share', 'prior_target_share', 'prior_rush_share', 'prior_opp_pg']
for c in pf:
    df[c] = [PRIOR[s].get(pid, {}).get(c, np.nan)
             for s, pid in zip(df['season'], df['player_id'])]

cap = xw.drop_duplicates('sleeper_id').set_index('sleeper_id')[
    ['draft_year', 'draft_round', 'draft_pick']]
dc = df['player_id'].map(cap['draft_year'])
dr = df['player_id'].map(cap['draft_round'])
dpk = df['player_id'].map(cap['draft_pick'])
df['years_exp'] = df['season'] - dc
df['draft_round'] = dr
df['draft_pick'] = dpk

df['ecr_overall_log'] = np.log1p(df['ecr_overall'])
df['ecr_pos_log'] = np.log1p(df['ecr_pos'])
df['age_sq'] = df['age'] ** 2
df['exp_youth'] = ((df['price_norm'] >= 25) & (df['age'] <= 24)).astype(float)
df['ppg_x_snap'] = df['prior_ppg_ppr'] * df['prior_snap_share']

df.to_csv(os.path.join(OUT, 'training_data_v2.csv'), index=False)

new = ['ecr_overall', 'ecr_pos', 'ecr_sd'] + pf + ['years_exp', 'draft_round',
      'ecr_overall_log', 'ecr_pos_log', 'age_sq', 'exp_youth', 'ppg_x_snap']
print('\ncoverage (share non-null) overall / by split:')
df['split'] = np.where(df['season'] <= 2023, 'train<=2023', 'test2024-25')
for c in new:
    print(f'  {c:20s} {df[c].notna().mean():.3f}  train {df[df.split=="train<=2023"][c].notna().mean():.3f}'
          f'  test {df[df.split=="test2024-25"][c].notna().mean():.3f}')
print('\nECR coverage by season:')
print(df.groupby('season')['ecr_overall'].apply(lambda s: s.notna().mean()).round(3).to_dict())
print('rows:', len(df))
print('DONE')
