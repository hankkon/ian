"""Load champion / team / result data into numpy-friendly structures."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROLES = ["TOP", "JNG", "MID", "BOT", "SUP"]
TAGS = ["engage", "frontline", "peel", "teamfight", "pick", "poke", "burst", "early", "scaling"]
DATA_DIR = Path(__file__).parent / "data"

# Placeholder picks used only when fearless + bans empty a role's pool.
FILLER_POWER = 0.45
# A champion's first listed role is its main one; flex roles are slightly weaker.
OFF_ROLE_PENALTY = 0.05


@dataclass
class ChampionPool:
    names: list[str]
    roles: np.ndarray      # [N, 5] bool
    power: np.ndarray      # [N]
    tags: np.ndarray       # [N, T] float 0/1
    ap: np.ndarray         # [N] 1 = AP, 0 = AD, 0.5 = mixed
    counters: np.ndarray   # [N, N] bool, counters[i, j] = i wins lane vs j
    filler: np.ndarray     # [N] bool
    off_role: np.ndarray   # [N, 5] penalty for playing a champion outside its main role
    index: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.names)


@dataclass
class Team:
    code: str
    name: str
    region: str
    seed: str
    rating: float
    values: np.ndarray     # [N, 5] value of champion in role, -inf when not playable


def load_champions(path: Path | None = None) -> ChampionPool:
    raw = json.loads((path or DATA_DIR / "champions.json").read_text(encoding="utf-8"))["champions"]
    raw = list(raw) + [
        {"name": f"(other {r})", "roles": [r], "power": FILLER_POWER, "dmg": "MIX",
         "tags": [], "counters": [], "filler": True}
        for r in ROLES
    ]
    names = [c["name"] for c in raw]
    index = {n: i for i, n in enumerate(names)}
    n = len(raw)
    roles = np.zeros((n, len(ROLES)), dtype=bool)
    tags = np.zeros((n, len(TAGS)))
    counters = np.zeros((n, n), dtype=bool)
    for i, c in enumerate(raw):
        for r in c["roles"]:
            roles[i, ROLES.index(r)] = True
        for t in c["tags"]:
            tags[i, TAGS.index(t)] = 1.0
        for victim in c.get("counters", []):
            if victim not in index:
                raise ValueError(f"{c['name']} counters unknown champion {victim!r}")
            counters[i, index[victim]] = True
    power = np.array([c["power"] for c in raw], dtype=float)
    ap = np.array([{"AP": 1.0, "AD": 0.0, "MIX": 0.5}[c["dmg"]] for c in raw])
    filler = np.array([c.get("filler", False) for c in raw])
    off_role = OFF_ROLE_PENALTY * roles
    for i, c in enumerate(raw):
        off_role[i, ROLES.index(c["roles"][0])] = 0.0
    return ChampionPool(names, roles, power, tags, ap, counters, filler, off_role, index)


def load_teams(pool: ChampionPool, path: Path | None = None) -> dict[str, Team]:
    raw = json.loads((path or DATA_DIR / "teams.json").read_text(encoding="utf-8"))["teams"]
    teams = {}
    for t in raw:
        style = np.zeros(len(TAGS))
        for tag, w in t.get("style", {}).items():
            style[TAGS.index(tag)] = w
        values = np.where(pool.roles, (pool.power + pool.tags @ style)[:, None] - pool.off_role, -np.inf)
        for role, champs in t.get("pools", {}).items():
            r = ROLES.index(role)
            for champ, bonus in champs.items():
                i = pool.index[champ]
                if not pool.roles[i, r]:
                    raise ValueError(f"{t['code']}: {champ} cannot play {role}")
                values[i, r] += bonus
        teams[t["code"]] = Team(t["code"], t["name"], t["region"], t["seed"], float(t["rating"]), values)
    return teams


def load_results(path: Path | None = None) -> dict:
    p = path or DATA_DIR / "results.json"
    if not p.exists():
        return {"swiss": {"pairings": {}, "results": []}, "knockout": []}
    return json.loads(p.read_text(encoding="utf-8"))
