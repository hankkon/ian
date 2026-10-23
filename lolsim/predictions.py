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


def build(sim: Simulator, out: dict, results: dict, sims: int) -> dict:
    teams = sim.teams
    st = sim.draft_stats()
    g = max(st.games, 1)
    rnd1 = results.get("swiss", {}).get("pairings", {}).get("1", [])
    known = {frozenset(r["teams"]): r["winner"] for r in results.get("swiss", {}).get("results", [])
             if int(r.get("round", 1)) == 1}
    matches = []
    for a, b in rnd1:
        p = bo1_prob(sim, a, b)
        matches.append({"a": a, "b": b, "pa": round(100 * p, 1), "winner": known.get(frozenset((a, b)))})
    team_rows = [{
        "code": c, "name": t.name, "region": t.region, "seed": t.seed, "rating": int(t.rating),
        **{k: round(100 * out[c][k], 1) for k in ("3-0", "3-1", "3-2", "advance", "semis", "final", "champion")},
        "expWins": round(out[c]["exp_wins"], 2),
    } for c, t in teams.items()]
    team_rows.sort(key=lambda r: (-r["champion"], -r["advance"]))
    champs = sorted(set(st.picks) | set(st.bans), key=lambda c: -(st.picks[c] + st.bans[c]))[:20]
    champ_rows = [{
        "champ": c, "role": ROLE_ZH[role_of(sim.pool, st, c)],
        "pick": round(100 * st.picks[c] / g, 1), "ban": round(100 * st.bans[c] / g, 1),
        "wr": round(100 * st.wins[c] / st.picks[c], 1) if st.picks[c] else None,
    } for c in champs]
    # Head-to-head Bo3 grid for the top 8 by advancement odds.
    top8 = [r["code"] for r in sorted(team_rows, key=lambda r: -r["advance"])[:8]]
    grid = [[None if a == b else round(100 * bo_prob(bo1_prob(sim, a, b), 5), 0) for b in top8] for a in top8]
    return {"sims": sims, "teams": team_rows, "round1": matches, "champs": champ_rows,
            "top8": top8, "h2h": grid}


def write_predictions(pool, teams, results, sims: int, seed, draft_samples: int, path: Path,
                      standalone: bool = True) -> Path:
    sim = Simulator(pool, teams, np.random.default_rng(seed), mode="fast", draft_samples=draft_samples)
    out = run_tournament(sim, results, sims)
    data = build(sim, out, results, sims)
    template = (Path(__file__).parent / "predictions_template.html").read_text(encoding="utf-8")
    page = template.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STANDALONE_HEAD + page if standalone else page, encoding="utf-8")
    return path
