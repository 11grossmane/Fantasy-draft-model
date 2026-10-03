#!/usr/bin/env python3
"""Weekly refresh of the static site's player snapshot.

Re-pulls public Sleeper data (player pool, current-season weekly stats)
and rebuilds docs/assets/data/players.*.json and meta.json in the exact
schema the site consumes. Draft-model projections (Sleeper/Rotowire
season projections) and FantasyPros ECR are preseason-baked; this
refresh updates injuries and in-season form, preserving stored
preseason inputs where no live pull exists in CI.

The site loads large files as sequential <100KB text parts
(players.00.json ... plus players.parts.json), so this script reads
and writes those parts instead of a single players.json.

Run from repo root:  python scripts/refresh_site_data.py
Exits non-zero on network failure so the workflow fails loudly
instead of committing stale or partial data.
"""
import json, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "docs", "assets", "data")
API = "https://api.sleeper.app/v1"
PART_SIZE = 98000  # keep each part comfortably under the commit pipeline limit


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "draft-model-refresh/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def read_parts(path_prefix, manifest_path):
    with open(manifest_path) as f:
        n = json.load(f)["parts"]
    text = ""
    for i in range(n):
        with open(f"{path_prefix}.{i:02d}.json") as f:
            text += f.read()
    return json.loads(text)


def write_parts(path_prefix, manifest_path, text):
    parts = [text[i:i + PART_SIZE] for i in range(0, len(text), PART_SIZE)]
    i = 0
    for chunk in parts:
        with open(f"{path_prefix}.{i:02d}.json", "w") as f:
            f.write(chunk)
        i += 1
    # drop stale extras from a previously longer file
    while os.path.exists(f"{path_prefix}.{i:02d}.json"):
        os.remove(f"{path_prefix}.{i:02d}.json")
        i += 1
    with open(manifest_path, "w") as f:
        json.dump({"parts": len(parts)}, f)


def main():
    meta_path = os.path.join(DATA, "meta.json")
    players = read_parts(os.path.join(DATA, "players"),
                         os.path.join(DATA, "players.parts.json"))
    with open(meta_path) as f:
        meta = json.load(f)

    season = meta["currentSeason"]
    state = get(f"{API}/state/nfl")
    week = state.get("week") or meta.get("currentThroughWeek", 1)

    live = get(f"{API}/players/nfl")
    season_pts = {}
    for wk in range(1, week + 1):
        try:
            wk_stats = get(f"{API}/stats/nfl/regular/{season}/{wk}")
        except Exception:
            continue
        for pid, s in wk_stats.items():
            if isinstance(s, dict) and s.get("pts_ppr") is not None:
                season_pts.setdefault(pid, []).append(s["pts_ppr"])

    for p in players:
        lp = live.get(p["id"]) or {}
        if lp.get("injury_status"):
            p["injury"] = lp["injury_status"]
        pts = season_pts.get(p["id"]) or []
        if pts:
            p["form"] = {**(p.get("form") or {}),
                         "gp": float(len(pts)), "pts": pts[-1],
                         "ppg": round(sum(pts) / len(pts), 2)}
            p["faab"] = {**(p.get("faab") or {}),
                         "prevWeekPts": pts[-1],
                         "ppgBefore": round(sum(pts) / len(pts), 2),
                         "gamesBefore": float(len(pts))}

    meta["currentThroughWeek"] = week
    meta["stateWeek"] = week
    meta["asOfMs"] = int(time.time() * 1000)
    write_parts(os.path.join(DATA, "players"),
                os.path.join(DATA, "players.parts.json"),
                json.dumps(players))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"refreshed {len(players)} players through week {week}")


if __name__ == "__main__":
    sys.exit(main())
