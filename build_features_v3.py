#!/usr/bin/env python3
"""
Build v3 candidate feature groups for the improvement loop.

Extends training_data_v2.csv with draft-day-legal, prior-season-only features:

EFF (production rates from stats_{season-1}):
  prior_pts_per_opp   pts_ppr / (rec_tgt + rush_att)
  prior_pts_per_snap  pts_ppr / off_snp
  prior_catch_rate    rec / rec_tgt
  prior_adot          rec_air_yd / rec_tgt
  prior_rz_share      (rec_rz_tgt + rush_rz_att) / team (same sum)
  prior_pass_att_pg   pass_att / gp            (QB-relevant, NaN for others is fine)
  prior_pass_rtg      passer rating
  prior_pass_td_rate  pass_td / pass_att
  prior_qb_rush_pg    rush_att / gp for QBs (0-filled for non-QB? No: NaN unless pos==QB
                      is unknown here; we set for all rows from stats, XGB sorts it out.
                      Actually set only when pass_att>0 i.e. player threw passes.)

TEAM (prior-year team context; team = players_nfl attribution, same noise as v1/v2):
  prior_team_pass_pg     team pass_att / team games (16 pre-2021, else 17)
  prior_team_pass_share  team pass_att / (pass_att + rush_att)
  prior_team_pass_rank   1..32 rank of team pass attempts (1 = most)
  prior_team_plays_pg    team (pass_att + rush_att) / team games

TRAJ (two-year trajectory from stats_{season-2}):
  prior2_ppg_ppr, prior2_rank_ppr
  ppg_delta      = prior_ppg_ppr - prior2_ppg_ppr
  rank_improve   = prior2_rank_ppr - prior_rank_ppr  (positive = finish improved)

VAL (relative-value / interactions, computable from existing v2 columns):
  ppg_per_dollar    = prior_ppg_ppr / price_norm
  price_x_age       = price_norm * age / 100
  ecr_vs_prior_rank = ecr_pos - prior_rank_ppr  (positive = ECR worse than last finish)
  price_per_ecr     = price_norm / ecr_pos

WK (prior-season weekly consistency from stats_{season-1}_w*.json; 2021+ only):
  prior_wk_cv      std/mean of weekly pts_ppr (weeks with a recorded game)
  prior_wk_floor10 share of games with pts_ppr >= 10
  prior_wk_ceil90  90th percentile weekly pts_ppr

Output: training_data_v3.csv (v2 columns + new), coverage printed.
"""
import glob, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.expanduser('~/workspace/sleeper-league-data')
CACHE = os.path.join(ROOT, 'crossleague-cache')
STATS = os.path.join(CACHE, 'targeted', 'stats')
OUT = os.path.join(ROOT, 'model')
TEAM_FIX = {'WSH': 'WAS'}

PLAYERS = json.load(open(os.path.join(CACHE, 'players_nfl.json')))

def team_of(pid):
    t = (PLAYERS.get(str(pid)) or {}).get('team')
    return TEAM_FIX.get(t, t) if t else None

def load_season(season):
    fn = os.path.join(STATS, f'stats_{season}.json')
    return json.load(open(fn)) if os.path.exists(fn) else {}

# ------------------------------------------------ per-season raw ingredients
def season_ingredients(season):
    """Return (player_dict, team_dict) of prior-season features for `season`."""
    st = load_season(season)
    team_pass, team_rush, team_rz = {}, {}, {}
    for pid, s in st.items():
        if not isinstance(s, dict) or not s:
            continue
        t = team_of(pid)
        if not t:
            continue
        team_pass[t] = team_pass.get(t, 0.0) + (s.get('pass_att') or 0.0)
        team_rush[t] = team_rush.get(t, 0.0) + (s.get('rush_att') or 0.0)
        team_rz[t] = team_rz.get(t, 0.0) + ((s.get('rec_rz_tgt') or 0.0)
                                            + (s.get('rush_rz_att') or 0.0))
    games = 16 if season <= 2020 else 17
    tot_pass = sorted(team_pass.items(), key=lambda kv: -kv[1])
    pass_rank = {t: i + 1 for i, (t, _) in enumerate(tot_pass)}
    team = {}
    for t in set(team_pass) | set(team_rush):
        pa, ra = team_pass.get(t, 0.0), team_rush.get(t, 0.0)
        team[t] = {
            'prior_team_pass_pg': pa / games,
            'prior_team_pass_share': pa / (pa + ra) if (pa + ra) > 0 else np.nan,
            'prior_team_pass_rank': pass_rank.get(t, np.nan),
            'prior_team_plays_pg': (pa + ra) / games,
        }
    pl = {}
    for pid, s in st.items():
        if not isinstance(s, dict) or not s or not str(pid).isdigit():
            continue
        gp = s.get('gp') or 0
        if gp <= 0:
            continue
        pts = s.get('pts_ppr')
        tgt = s.get('rec_tgt') or 0.0
        rush = s.get('rush_att') or 0.0
        opp = tgt + rush
        snp = s.get('off_snp')
        d = {
            'prior_pts_per_opp': (pts / opp) if (pts is not None and opp > 0) else np.nan,
            'prior_pts_per_snap': (pts / snp) if (pts is not None and snp) else np.nan,
            'prior_catch_rate': ((s.get('rec') or 0.0) / tgt) if tgt > 0 else np.nan,
            'prior_adot': ((s.get('rec_air_yd') or 0.0) / tgt) if tgt > 0 else np.nan,
            'prior_rz_share': np.nan,
            'prior_pass_rtg': s.get('pass_rtg'),
        }
        t = team_of(pid)
        rz = (s.get('rec_rz_tgt') or 0.0) + (s.get('rush_rz_att') or 0.0)
        if t and team_rz.get(t):
            d['prior_rz_share'] = rz / team_rz[t]
        pa = s.get('pass_att') or 0.0
        if pa > 0:
            d['prior_pass_att_pg'] = pa / gp
            d['prior_pass_td_rate'] = (s.get('pass_td') or 0.0) / pa
            d['prior_qb_rush_pg'] = rush / gp
        else:
            d['prior_pass_att_pg'] = np.nan
            d['prior_pass_td_rate'] = np.nan
            d['prior_qb_rush_pg'] = np.nan
        pl[int(pid)] = d
    return pl, team

def weekly_ingredients(season):
    pl = {}
    files = sorted(glob.glob(os.path.join(STATS, f'stats_{season}_w*.json')))
    series = {}
    for fn in files:
        w = json.load(open(fn))
        for pid, s in w.items():
            if not isinstance(s, dict) or not s or not str(pid).isdigit():
                continue
            p = s.get('pts_ppr')
            if p is None:
                continue
            series.setdefault(int(pid), []).append(float(p))
    for pid, arr in series.items():
        if len(arr) < 4:
            continue
        a = np.array(arr)
        m = a.mean()
        pl[pid] = {
            'prior_wk_cv': float(a.std() / m) if m > 0 else np.nan,
            'prior_wk_floor10': float((a >= 10).mean()),
            'prior_wk_ceil90': float(np.percentile(a, 90)),
        }
    return pl

EFF = ['prior_pts_per_opp', 'prior_pts_per_snap', 'prior_catch_rate',
       'prior_adot', 'prior_rz_share', 'prior_pass_att_pg', 'prior_pass_rtg',
       'prior_pass_td_rate', 'prior_qb_rush_pg']
TEAM = ['prior_team_pass_pg', 'prior_team_pass_share', 'prior_team_pass_rank',
        'prior_team_plays_pg']
WK = ['prior_wk_cv', 'prior_wk_floor10', 'prior_wk_ceil90']

df = pd.read_csv(os.path.join(OUT, 'training_data_v2.csv'), low_memory=False)
df['player_id'] = df['player_id'].astype(int)

seasons = sorted(df['season'].unique())
PRIOR1, TEAMCTX, WKLY = {}, {}, {}
for s in seasons:
    PRIOR1[s], TEAMCTX[s] = season_ingredients(s - 1)
    WKLY[s] = weekly_ingredients(s - 1)
    print(f'prior-season ingredients for draft {s}: players {len(PRIOR1[s])}, '
          f'teams {len(TEAMCTX[s])}, weekly {len(WKLY[s])}', flush=True)
P2 = {}
for s in seasons:
    p2, _ = season_ingredients(s - 2)
    # need ppg/rank from stats_{s-2}; recompute light version below
    P2[s] = p2

# prior2 ppg/rank need pts/gp from stats_{s-2}: rebuild quickly
def ppg_rank(season):
    st = load_season(season)
    out = {}
    for pid, s in st.items():
        if not isinstance(s, dict) or not s or not str(pid).isdigit():
            continue
        gp = s.get('gp') or 0
        pts = s.get('pts_ppr')
        if gp > 0 and pts is not None:
            out[int(pid)] = {'prior2_ppg_ppr': pts / gp,
                             'prior2_rank_ppr': s.get('pos_rank_ppr')}
    return out

P2F = {s: ppg_rank(s - 2) for s in seasons}

for c in EFF:
    df[c] = [PRIOR1[s].get(pid, {}).get(c, np.nan)
             for s, pid in zip(df['season'], df['player_id'])]
for c in TEAM:
    df[c] = [TEAMCTX[s].get(team_of(pid), {}).get(c, np.nan)
             for s, pid in zip(df['season'], df['player_id'])]
for c in WK:
    df[c] = [WKLY[s].get(pid, {}).get(c, np.nan)
             for s, pid in zip(df['season'], df['player_id'])]
df['prior2_ppg_ppr'] = [P2F[s].get(pid, {}).get('prior2_ppg_ppr', np.nan)
                        for s, pid in zip(df['season'], df['player_id'])]
df['prior2_rank_ppr'] = [P2F[s].get(pid, {}).get('prior2_rank_ppr', np.nan)
                         for s, pid in zip(df['season'], df['player_id'])]
df['ppg_delta'] = df['prior_ppg_ppr'] - df['prior2_ppg_ppr']
df['rank_improve'] = df['prior2_rank_ppr'] - df['prior_rank_ppr']

df['ppg_per_dollar'] = df['prior_ppg_ppr'] / df['price_norm'].clip(lower=1)
df['price_x_age'] = df['price_norm'] * df['age'] / 100.0
df['ecr_vs_prior_rank'] = df['ecr_pos'] - df['prior_rank_ppr']
df['price_per_ecr'] = df['price_norm'] / df['ecr_pos'].clip(lower=1)

NEW_ALL = EFF + TEAM + WK + ['prior2_ppg_ppr', 'prior2_rank_ppr', 'ppg_delta',
                             'rank_improve', 'ppg_per_dollar', 'price_x_age',
                             'ecr_vs_prior_rank', 'price_per_ecr']
df.to_csv(os.path.join(OUT, 'training_data_v3.csv'), index=False)
print('\ncoverage by split:')
df['split'] = np.where(df['season'] <= 2023, 'train<=2023', 'test2024-25')
for c in NEW_ALL:
    print(f"  {c:22s} all {df[c].notna().mean():.3f}  train "
          f"{df[df.split=='train<=2023'][c].notna().mean():.3f}  test "
          f"{df[df.split=='test2024-25'][c].notna().mean():.3f}")
print('rows:', len(df))
print('DONE')
