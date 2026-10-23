"""Series, Swiss stage and knockout bracket Monte Carlo."""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .data import ROLES, ChampionPool, Team
from .draft import BLUE, RED, Drafter, game_win_prob


@dataclass
class DraftStats:
    """Pick / ban / win counts per champion, aggregated over simulated games."""
    games: float = 0.0
    picks: Counter = field(default_factory=Counter)
    bans: Counter = field(default_factory=Counter)
    wins: Counter = field(default_factory=Counter)
    role_picks: dict = field(default_factory=lambda: defaultdict(Counter))
    # (team, role, champion) -> games / wins, and team -> games / wins
    team_picks: Counter = field(default_factory=Counter)
    team_pick_wins: Counter = field(default_factory=Counter)
    team_games: Counter = field(default_factory=Counter)
    team_wins: Counter = field(default_factory=Counter)
    team_bans_against: Counter = field(default_factory=Counter)   # (team, champion) banned by opponents

    def add(self, other: "DraftStats", weight: float = 1.0) -> None:
        self.games += other.games * weight
        for src, dst in ((other.picks, self.picks), (other.bans, self.bans), (other.wins, self.wins),
                         (other.team_picks, self.team_picks), (other.team_pick_wins, self.team_pick_wins),
                         (other.team_games, self.team_games), (other.team_wins, self.team_wins),
                         (other.team_bans_against, self.team_bans_against)):
            for k, v in src.items():
                dst[k] += v * weight
        for r, cnt in other.role_picks.items():
            for k, v in cnt.items():
                self.role_picks[r][k] += v * weight


class Simulator:
    """Plays games/series between teams.

    mode="fast": each game's blue-side win probability comes from a table
    averaged over `draft_samples` simulated drafts per (blue, red) pair.
    mode="full": every game is drafted, with fearless locks inside a series.
    """

    def __init__(self, pool: ChampionPool, teams: dict[str, Team], rng: np.random.Generator,
                 mode: str = "fast", fearless: bool = True, draft_samples: int = 60):
        self.pool, self.teams, self.rng = pool, teams, rng
        self.mode, self.fearless = mode, fearless
        self.drafter = Drafter(pool, rng)
        self.stats = DraftStats()                 # full mode: every game drafted
        self.pair_games: Counter = Counter()      # fast mode: games played per (blue, red)
        self.pair_stats: dict[tuple[str, str], DraftStats] = {}
        self.table: dict[tuple[str, str], float] = {}
        if mode == "fast":
            for b, r in itertools.permutations(teams, 2):
                st = DraftStats()
                probs = []
                for _ in range(draft_samples):
                    d = self.drafter.run(teams[b], teams[r])
                    p = game_win_prob(teams[b], teams[r], d.edge())
                    probs.append(p)
                    self._record(st, d, p, (b, r))
                self.table[(b, r)] = float(np.mean(probs))
                self.pair_stats[(b, r)] = st

    def _record(self, st: DraftStats, d, blue_win: float, codes: tuple[str, str]) -> None:
        names = self.pool.names
        st.games += 1
        for side, win in ((BLUE, blue_win), (RED, 1 - blue_win)):
            team, opp = codes[side], codes[1 - side]
            st.team_games[team] += 1
            st.team_wins[team] += win
            for role, c in d.picks[side].items():
                if self.pool.filler[c]:
                    continue
                st.picks[names[c]] += 1
                st.wins[names[c]] += win
                st.role_picks[role][names[c]] += 1
                st.team_picks[(team, role, names[c])] += 1
                st.team_pick_wins[(team, role, names[c])] += win
            for c in d.bans[side]:
                st.bans[names[c]] += 1
                st.team_bans_against[(opp, names[c])] += 1

    def play_game(self, blue: str, red: str, locked: set[int]) -> bool:
        if self.mode == "fast":
            self.pair_games[(blue, red)] += 1
            return self.rng.random() < self.table[(blue, red)]
        d = self.drafter.run(self.teams[blue], self.teams[red], locked)
        p = game_win_prob(self.teams[blue], self.teams[red], d.edge())
        won = self.rng.random() < p
        self._record(self.stats, d, float(won), (blue, red))
        if self.fearless:
            for side in (BLUE, RED):
                locked.update(c for c in d.picks[side].values() if not self.pool.filler[c])
        return won

    def play_series(self, a: str, b: str, bo: int) -> tuple[str, tuple[int, int]]:
        need = bo // 2 + 1
        wins = {a: 0, b: 0}
        locked: set[int] = set()
        blue, red = (a, b) if self.rng.random() < 0.5 else (b, a)
        while max(wins.values()) < need:
            blue_won = self.play_game(blue, red, locked)
            winner, loser = (blue, red) if blue_won else (red, blue)
            wins[winner] += 1
            blue, red = loser, winner          # loser of a game chooses blue side
        w = a if wins[a] == need else b
        return w, (wins[a], wins[b])

    def draft_stats(self) -> DraftStats:
        if self.mode == "full":
            return self.stats
        total = DraftStats()
        for pair, n in self.pair_games.items():
            st = self.pair_stats[pair]
            total.add(st, n / st.games)
        return total


# ---------------------------------------------------------------- Swiss ----

def _key(a: str, b: str) -> frozenset:
    return frozenset((a, b))


def _draw(group: list[str], played: set[frozenset], rng: np.random.Generator) -> list[tuple[str, str]] | None:
    """Random pairing inside one record group, avoiding rematches (backtracking)."""
    if not group:
        return []
    order = list(group)
    rng.shuffle(order)
    first, rest = order[0], order[1:]
    for opp in rest:
        if _key(first, opp) in played:
            continue
        sub = _draw([t for t in rest if t != opp], played, rng)
        if sub is not None:
            return [(first, opp)] + sub
    return None


def simulate_swiss(sim: Simulator, results: dict, rounds: int = 5) -> dict[str, tuple[int, int]]:
    teams = list(sim.teams)
    rec = {t: [0, 0] for t in teams}
    played: set[frozenset] = set()
    fixed_pairs = {int(k): v for k, v in results.get("pairings", {}).items()}
    known = {(int(r.get("round", 0)), _key(*r["teams"])): r["winner"] for r in results.get("results", [])}
    for rnd in range(1, rounds + 1):
        active = [t for t in teams if rec[t][0] < 3 and rec[t][1] < 3]
        if not active:
            break
        if rnd in fixed_pairs:
            pairs = [tuple(p) for p in fixed_pairs[rnd]]
        else:
            groups = defaultdict(list)
            for t in active:
                groups[tuple(rec[t])].append(t)
            pairs = []
            for g in groups.values():
                drawn = _draw(g, played, sim.rng) or _draw(g, set(), sim.rng)
                pairs += drawn
        for a, b in pairs:
            w_a, l_a = rec[a]
            bo = 3 if (w_a == 2 or l_a == 2) else 1
            winner = known.get((rnd, _key(a, b))) or known.get((0, _key(a, b)))
            if winner is None:
                winner, _ = sim.play_series(a, b, bo)
            loser = b if winner == a else a
            rec[winner][0] += 1
            rec[loser][1] += 1
            played.add(_key(a, b))
    return {t: tuple(r) for t, r in rec.items()}


# ------------------------------------------------------------- Knockout ----

def seed_quarterfinals(sim: Simulator, records: dict[str, tuple[int, int]], tries: int = 200) -> list[tuple[str, str]]:
    """3-0 teams meet 3-2 teams, the rest are drawn; same-region QFs avoided when possible.

    Returned in bracket order: QF1/QF2 feed SF1, QF3/QF4 feed SF2, and the two
    3-0 teams are placed in opposite halves.
    """
    adv = [t for t, (w, _) in records.items() if w == 3]
    by = defaultdict(list)
    for t in adv:
        by[records[t][1]].append(t)
    best = None
    for _ in range(tries):
        top = list(by[0]); low = list(by[2]); mid = list(by[1])
        sim.rng.shuffle(top); sim.rng.shuffle(low); sim.rng.shuffle(mid)
        pairs = []
        for t in top:
            pairs.append((t, low.pop()) if low else (t, mid.pop()))
        rest = mid + low
        sim.rng.shuffle(rest)
        pairs += [(rest[i], rest[i + 1]) for i in range(0, len(rest), 2)]
        clash = sum(sim.teams[a].region == sim.teams[b].region for a, b in pairs)
        if best is None or clash < best[0]:
            best = (clash, pairs)
        if clash == 0:
            break
    pairs = best[1]
    n_top = len(by[0])
    tops, others = pairs[:n_top], pairs[n_top:]
    if len(tops) == 2:
        return [tops[0], others[0], tops[1], others[1]]
    return pairs


def simulate_knockout(sim: Simulator, qfs: list[tuple[str, str]], known: list[dict]) -> dict[str, int]:
    """Returns furthest stage per team: 0 = QF loss, 1 = SF loss, 2 = final loss, 3 = champion."""
    res = {_key(*k["teams"]): k["winner"] for k in known}
    stage = {}
    current = [list(p) for p in qfs]
    level = 0
    while current:
        winners = []
        for a, b in current:
            w = res.get(_key(a, b)) or sim.play_series(a, b, 5)[0]
            stage[b if w == a else a] = level
            winners.append(w)
        level += 1
        if len(winners) == 1:
            stage[winners[0]] = level
            break
        current = [winners[i:i + 2] for i in range(0, len(winners), 2)]
    return stage


# ------------------------------------------------------------ Tournament ---

def run_tournament(sim: Simulator, results: dict, n: int) -> dict:
    teams = list(sim.teams)
    swiss_rec = defaultdict(Counter)
    stage_cnt = defaultdict(Counter)
    fixed_qf = results.get("knockout_pairings")
    for _ in range(n):
        rec = simulate_swiss(sim, results.get("swiss", {}))
        for t, r in rec.items():
            swiss_rec[t][r] += 1
        qfs = [tuple(p) for p in fixed_qf] if fixed_qf else seed_quarterfinals(sim, rec)
        for t, s in simulate_knockout(sim, qfs, results.get("knockout", [])).items():
            stage_cnt[t][s] += 1
    out = {}
    for t in teams:
        r = swiss_rec[t]
        s = stage_cnt[t]
        out[t] = {
            "3-0": r[(3, 0)] / n, "3-1": r[(3, 1)] / n, "3-2": r[(3, 2)] / n,
            "advance": sum(v for k, v in r.items() if k[0] == 3) / n,
            "semis": sum(v for k, v in s.items() if k >= 1) / n,
            "final": sum(v for k, v in s.items() if k >= 2) / n,
            "champion": s[3] / n,
            "exp_wins": sum(k[0] * v for k, v in r.items()) / n,
        }
    return out


def head_to_head(sim: Simulator, a: str, b: str, bo: int, n: int) -> dict:
    score = Counter()
    for _ in range(n):
        w, sc = sim.play_series(a, b, bo)
        score[sc] += 1
    win_a = sum(v for (x, y), v in score.items() if x > y) / n
    return {"win_a": win_a, "scores": {f"{x}-{y}": v / n for (x, y), v in sorted(score.items(), key=lambda kv: -kv[1])}}


def role_of(pool: ChampionPool, st: DraftStats, champ: str) -> str:
    roles = st.role_picks
    best = max(ROLES, key=lambda r: roles[r][champ]) if any(roles[r][champ] for r in ROLES) else None
    if best is None:
        i = pool.index[champ]
        best = ROLES[int(np.argmax(pool.roles[i]))]
    return best
