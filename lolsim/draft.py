"""Pick/ban simulator.

Each draft action scores every available champion with a heuristic that a
coach would recognise (meta power + team comfort + composition needs + lane
counters + denying the opponent) and samples from a softmax over the best
options, so repeated drafts give a probability distribution of picks and bans.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data import ROLES, TAGS, ChampionPool, Team

BLUE, RED = 0, 1

# Standard tournament draft: 3 bans each, 3 picks each (B R R B B R),
# 2 bans each (red first), 2 picks each (R B B R).
DRAFT_ORDER = (
    [(BLUE, "ban"), (RED, "ban")] * 3
    + [(BLUE, "pick"), (RED, "pick"), (RED, "pick"), (BLUE, "pick"), (BLUE, "pick"), (RED, "pick")]
    + [(RED, "ban"), (BLUE, "ban")] * 2
    + [(RED, "pick"), (BLUE, "pick"), (BLUE, "pick"), (RED, "pick")]
)

_T = {t: i for i, t in enumerate(TAGS)}
COUNTER_BONUS = 0.04
BLIND_RISK = 0.008
DENY_WEIGHT = 0.6
BAN_GAP_WEIGHT = 1.0
TEMPERATURE = 0.025
BAN_TEMPERATURE = 0.04
TOP_K = 8

# Game-level win model (logit units).
BLUE_SIDE_EDGE = 0.08
DRAFT_WEIGHT = 3.0


def comp_score(pool: ChampionPool, tag_sum: np.ndarray, ap: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Composition quality; vectorised over leading axis of tag_sum [K, T]."""
    t = tag_sum
    ad = count - ap
    ident = np.maximum.reduce([t[:, _T["teamfight"]], t[:, _T["pick"]], t[:, _T["poke"]]])
    return (
        0.06 * (t[:, _T["engage"]] >= 1) + 0.02 * (t[:, _T["engage"]] >= 2)
        + 0.06 * (t[:, _T["frontline"]] >= 1) + 0.02 * (t[:, _T["peel"]] >= 1)
        + 0.05 * (ap >= 1) + 0.05 * (ad >= 1)
        - 0.06 * np.maximum(0, ap - 3.5) - 0.06 * np.maximum(0, ad - 3.5)
        + 0.015 * np.clip(ident - 2, 0, 2)
    )


@dataclass
class DraftResult:
    picks: tuple[dict[str, int], dict[str, int]]   # side -> role -> champion index
    bans: tuple[list[int], list[int]]
    rating: tuple[float, float]                     # draft quality per side
    sequence: list[tuple[int, str, int, str | None]]  # (side, action, champion, role) in draft order

    def edge(self) -> float:
        return self.rating[BLUE] - self.rating[RED]


class Drafter:
    def __init__(self, pool: ChampionPool, rng: np.random.Generator, temperature: float = TEMPERATURE):
        self.pool = pool
        self.rng = rng
        self.temperature = temperature

    # ---- scoring helpers -------------------------------------------------
    def _side_rating(self, team: Team, picks: dict[str, int], opp: dict[str, int]) -> float:
        p = self.pool
        idx = list(picks.values())
        val = np.mean([team.values[c, ROLES.index(r)] for r, c in picks.items()])
        comp = comp_score(p, p.tags[idx].sum(0)[None], np.array([p.ap[idx].sum()]), np.array([len(idx)]))[0]
        lane = 0.0
        for r, c in picks.items():
            o = opp.get(r)
            if o is not None:
                lane += COUNTER_BONUS * (int(p.counters[c, o]) - int(p.counters[o, c]))
        return float(val + comp + lane)

    def _best_role_values(self, team: Team, open_roles: list[str]) -> np.ndarray:
        cols = [ROLES.index(r) for r in open_roles]
        return team.values[:, cols].max(1) if cols else np.full(self.pool.n, -np.inf)

    def _gap(self, team: Team, open_roles: list[str], avail: np.ndarray) -> np.ndarray:
        """How much worse `team` is off if each champion disappears (top1 - top2 per role)."""
        gap = np.zeros(self.pool.n)
        for r in open_roles:
            v = np.where(avail, team.values[:, ROLES.index(r)], -np.inf)
            order = np.argsort(-v)
            if np.isfinite(v[order[0]]):
                second = v[order[1]] if np.isfinite(v[order[1]]) else v[order[0]] - 0.2
                gap[order[0]] = max(gap[order[0]], v[order[0]] - second)
        return gap

    def _sample(self, scores: np.ndarray, temperature: float | None = None) -> int:
        temperature = self.temperature if temperature is None else temperature
        finite = np.flatnonzero(np.isfinite(scores))
        top = finite[np.argsort(-scores[finite])[:TOP_K]]
        s = scores[top]
        if temperature <= 0:
            return int(top[0])
        w = np.exp((s - s.max()) / temperature)
        return int(self.rng.choice(top, p=w / w.sum()))

    # ---- actions ---------------------------------------------------------
    def _pick(self, team: Team, opp: Team, mine: dict, theirs: dict, avail: np.ndarray, rotation: int) -> tuple[int, str]:
        p = self.pool
        my_open = [r for r in ROLES if r not in mine]
        opp_open = [r for r in ROLES if r not in theirs]
        idx = list(mine.values())
        base_tags = p.tags[idx].sum(0) if idx else np.zeros(len(TAGS))
        base_ap = p.ap[idx].sum() if idx else 0.0
        before = comp_score(p, base_tags[None], np.array([base_ap]), np.array([len(idx)]))[0]
        comp_delta = comp_score(p, base_tags[None] + p.tags, base_ap + p.ap, np.full(p.n, len(idx) + 1)) - before

        best_score = np.full(p.n, -np.inf)
        best_role = np.full(p.n, -1)
        for r in my_open:
            ri = ROLES.index(r)
            s = team.values[:, ri].copy()
            o = theirs.get(r)
            if o is not None:
                s += COUNTER_BONUS * (p.counters[:, o].astype(float) - p.counters[o, :])
            else:
                threats = (p.counters.T & (avail & p.roles[:, ri])[None, :]).sum(1)
                s -= BLIND_RISK * np.minimum(threats, 3)
            better = s > best_score
            best_score = np.where(better, s, best_score)
            best_role = np.where(better, ri, best_role)

        score = best_score + comp_delta
        if rotation == 0:
            flex = (p.roles[:, [ROLES.index(r) for r in my_open]].sum(1) > 1)
            score += 0.01 * flex
        score += DENY_WEIGHT * self._gap(opp, opp_open, avail)
        score = np.where(avail, score, -np.inf)
        c = self._sample(score)
        return c, ROLES[best_role[c]]

    def _ban(self, team: Team, opp: Team, mine: dict, theirs: dict, avail: np.ndarray, first_pick_next: bool) -> int:
        p = self.pool
        opp_open = [r for r in ROLES if r not in theirs]
        my_open = [r for r in ROLES if r not in mine]
        threat = self._best_role_values(opp, opp_open) + BAN_GAP_WEIGHT * self._gap(opp, opp_open, avail)
        if first_pick_next:
            # No need to ban what we can simply take with the next pick.
            threat -= 0.5 * np.maximum(self._best_role_values(team, my_open), 0)
        score = np.where(avail & ~p.filler, threat, -np.inf)
        return self._sample(score, self.temperature * BAN_TEMPERATURE / TEMPERATURE)

    # ---- full draft ------------------------------------------------------
    def run(self, blue: Team, red: Team, locked: set[int] = frozenset()) -> DraftResult:
        p = self.pool
        avail = np.ones(p.n, dtype=bool)
        for c in locked:
            avail[c] = False
        avail |= p.filler
        teams = (blue, red)
        picks: tuple[dict, dict] = ({}, {})
        bans: tuple[list, list] = ([], [])
        sequence = []
        for i, (side, action) in enumerate(DRAFT_ORDER):
            me, them = teams[side], teams[1 - side]
            if action == "ban":
                # Blue opens pick phase 1, red opens pick phase 2.
                first_next = (side == BLUE and i < 6) or (side == RED and i >= 12)
                c = self._ban(me, them, picks[side], picks[1 - side], avail, first_next)
                bans[side].append(c)
                sequence.append((side, "ban", c, None))
            else:
                rotation = 0 if i < 12 else 1
                c, role = self._pick(me, them, picks[side], picks[1 - side], avail, rotation)
                picks[side][role] = c
                sequence.append((side, "pick", c, role))
            if not p.filler[c]:
                avail[c] = False
        rating = (self._side_rating(blue, picks[BLUE], picks[RED]), self._side_rating(red, picks[RED], picks[BLUE]))
        return DraftResult(picks, bans, rating, sequence)


def game_win_prob(blue: Team, red: Team, draft_edge: float = 0.0) -> float:
    """Probability the blue side wins one game."""
    logit = math.log(10) * (blue.rating - red.rating) / 400 + BLUE_SIDE_EDGE + DRAFT_WEIGHT * draft_edge
    return 1 / (1 + math.exp(-logit))
