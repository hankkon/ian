"""Build a team / player / champion-pool dashboard (single HTML file) from a simulation run."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .data import DATA_DIR, ROLES, TAGS, ChampionPool, Team
from .tournament import Simulator, run_tournament

ROLE_ZH = {"TOP": "上路", "JNG": "打野", "MID": "中路", "BOT": "下路", "SUP": "輔助"}
TAG_ZH = {"engage": "開團", "frontline": "前排", "peel": "保護", "teamfight": "團戰", "pick": "抓單",
          "poke": "消耗", "burst": "爆發", "early": "前期", "scaling": "後期"}
# Data Dragon file names that differ from the display name.
DDRAGON_KEYS = {"Wukong": "MonkeyKing", "Renata Glasc": "Renata", "Kai'Sa": "Kaisa", "K'Sante": "KSante"}
DDRAGON_VERSION = "15.1.1"


def ddragon_key(name: str) -> str:
    return DDRAGON_KEYS.get(name) or re.sub(r"[^A-Za-z]", "", name)


def load_rosters() -> dict[str, list[str]]:
    p = DATA_DIR / "players.json"
    return json.loads(p.read_text(encoding="utf-8"))["rosters"] if p.exists() else {}


def build_data(pool: ChampionPool, teams: dict[str, Team], sim: Simulator, out: dict, sims: int) -> dict:
    st = sim.draft_stats()
    rosters = load_rosters()
    champs = {}
    for i, name in enumerate(pool.names):
        if pool.filler[i]:
            continue
        champs[name] = {
            "key": ddragon_key(name),
            "roles": [r for r, ok in zip(ROLES, pool.roles[i]) if ok],
            "power": round(float(pool.power[i]) * 100),
            "dmg": {1.0: "AP", 0.0: "AD"}.get(float(pool.ap[i]), "混合"),
            "tags": [TAG_ZH[t] for t, on in zip(TAGS, pool.tags[i]) if on],
            "counters": [pool.names[j] for j in np.flatnonzero(pool.counters[i])],
            "countered": [pool.names[j] for j in np.flatnonzero(pool.counters[:, i])],
            "presence": round(100 * (st.picks[name] + st.bans[name]) / max(st.games, 1), 1),
        }
    team_rows = []
    for code, t in teams.items():
        games = st.team_games[code] / sims
        wins = st.team_wins[code] / sims
        roster = rosters.get(code, [""] * 5)
        players = []
        for ri, role in enumerate(ROLES):
            picks = [(c, n) for (tm, r, c), n in st.team_picks.items() if tm == code and r == role]
            total = sum(n for _, n in picks) or 1
            pool_rows = []
            for c, n in sorted(picks, key=lambda x: -x[1]):
                share = n / total
                if share < 0.04 and len(pool_rows) >= 3:
                    continue
                w = st.team_pick_wins[(code, role, c)]
                i = pool.index[c]
                pool_rows.append({
                    "champ": c, "share": round(100 * share), "games": round(n / sims, 2),
                    "winrate": round(100 * w / n, 1), "score": round(float(t.values[i, ri]) * 100, 1),
                    "comfort": round(float(t.values[i, ri] - pool.power[i] + pool.off_role[i, ri]) * 100, 1),
                    "banned": round(100 * st.team_bans_against[(code, c)] / max(st.team_games[code], 1), 1),
                })
            pool_rows = pool_rows[:8]
            wsum = sum(p["share"] for p in pool_rows) or 1
            players.append({
                "role": role, "roleZh": ROLE_ZH[role], "name": roster[ri] if ri < len(roster) else "",
                "score": round(sum(p["score"] * p["share"] for p in pool_rows) / wsum, 1),
                "winrate": round(sum(p["winrate"] * p["share"] for p in pool_rows) / wsum, 1),
                "pool": pool_rows,
            })
        o = out[code]
        team_rows.append({
            "code": code, "name": t.name, "region": t.region, "seed": t.seed, "rating": int(t.rating),
            "games": round(games, 1), "wins": round(wins, 1), "losses": round(games - wins, 1),
            "advance": round(100 * o["advance"], 1), "semis": round(100 * o["semis"], 1),
            "final": round(100 * o["final"], 1), "champion": round(100 * o["champion"], 1),
            "record": {k: round(100 * o[k], 1) for k in ("3-0", "3-1", "3-2")},
            "topBans": [c for (tm, c), _ in sorted(((k, v) for k, v in st.team_bans_against.items() if k[0] == code),
                                                   key=lambda kv: -kv[1])[:5]],
            "players": players,
        })
    team_rows.sort(key=lambda r: (-r["champion"], -r["advance"]))
    return {"sims": sims, "ddragon": DDRAGON_VERSION, "teams": team_rows, "champs": champs}


STANDALONE_HEAD = ('<!doctype html>\n<html lang="zh-Hant">\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')


def render(data: dict, standalone: bool = True) -> str:
    """standalone=False leaves out the document skeleton (for hosts that add their own)."""
    template = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = template.replace("__DATA__", payload)
    return STANDALONE_HEAD + page if standalone else page


def write_dashboard(pool, teams, results, sims: int, seed, draft_samples: int, path: Path,
                    standalone: bool = True) -> Path:
    sim = Simulator(pool, teams, np.random.default_rng(seed), mode="fast", draft_samples=draft_samples)
    out = run_tournament(sim, results, sims)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(build_data(pool, teams, sim, out, sims), standalone), encoding="utf-8")
    return path
