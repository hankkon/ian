"""Build the prediction page (advancement odds, Swiss round matchups, pick/ban forecast) as one HTML file."""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

from .dashboard import STANDALONE_HEAD, ROLE_ZH
from .tournament import Simulator, role_of, run_tournament


def bo1_prob(sim: Simulator, a: str, b: str) -> float:
    """Chance `a` wins a single game with sides decided by coin flip."""
    return 0.5 * (sim.table[(a, b)] + 1 - sim.table[(b, a)])


def bo_prob(p: float, bo: int) -> float:
    need = bo // 2 + 1
    return sum(comb(need - 1 + k, k) * p ** need * (1 - p) ** k for k in range(need))


def build(sim: Simulator, out: dict, results: dict, sims: int, played: list[dict], base_rating: dict) -> dict:
    teams = sim.teams
    st = sim.draft_stats()
    g = max(st.games, 1)
    swiss = results.get("swiss", {})
    done = {(r["round"], frozenset((r["a"], r["b"]))): r for r in played}
    rec = {t: [0, 0] for t in teams}
    for r in played:
        loser = r["b"] if r["winner"] == r["a"] else r["a"]
        rec[r["winner"]][0] += 1
        rec[loser][1] += 1
    rounds = []
    for rnd in sorted(int(k) for k in swiss.get("pairings", {})):
        matches = []
        for a, b in swiss["pairings"][str(rnd)]:
            r = done.get((rnd, frozenset((a, b))))
            if r:
                pa = r["pa"] if r["a"] == a else round(100 - r["pa"], 1)
                matches.append({"a": a, "b": b, "pa": pa, "winner": r["winner"], "upset": r["upset"], "bo": None})
            else:
                bo = 3 if (rec[a][0] == 2 or rec[a][1] == 2) else 1
                p = bo_prob(bo1_prob(sim, a, b), bo)
                matches.append({"a": a, "b": b, "pa": round(100 * p, 1), "winner": None, "upset": False, "bo": bo,
                                "pool": f"{rec[a][0]}-{rec[a][1]}"})
        rounds.append({"round": rnd, "matches": matches, "done": all(m["winner"] for m in matches)})
    team_rows = [{
        "code": c, "name": t.name, "region": t.region, "seed": t.seed, "rating": int(t.rating),
        "form": round(t.rating - base_rating[c]), "record": f"{rec[c][0]}-{rec[c][1]}",
        **{k: round(100 * out[c][k], 1) for k in ("3-0", "3-1", "3-2", "advance", "semis", "final", "champion")},
    } for c, t in teams.items()]
    team_rows.sort(key=lambda r: (-r["champion"], -r["advance"]))
    champs = sorted(set(st.picks) | set(st.bans), key=lambda c: -(st.picks[c] + st.bans[c]))[:20]
    champ_rows = [{
        "champ": c, "role": ROLE_ZH[role_of(sim.pool, st, c)],
        "pick": round(100 * st.picks[c] / g, 1), "ban": round(100 * st.bans[c] / g, 1),
    } for c in champs]
    top8 = [r["code"] for r in sorted(team_rows, key=lambda r: -r["advance"])[:8]]
    grid = [[None if a == b else round(100 * bo_prob(bo1_prob(sim, a, b), 5)) for b in top8] for a in top8]
    return {"sims": sims, "teams": team_rows, "rounds": rounds, "champs": champ_rows, "top8": top8, "h2h": grid}


def write_predictions(pool, teams, results, sims: int, seed, draft_samples: int, path: Path,
                      standalone: bool = True, played: list[dict] | None = None,
                      base_rating: dict | None = None) -> Path:
    sim = Simulator(pool, teams, np.random.default_rng(seed), mode="fast", draft_samples=draft_samples)
    out = run_tournament(sim, results, sims)
    data = build(sim, out, results, sims, played or [], base_rating or {c: t.rating for c, t in teams.items()})
    template = (Path(__file__).parent / "predictions_template.html").read_text(encoding="utf-8")
    page = template.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STANDALONE_HEAD + page if standalone else page, encoding="utf-8")
    return path
