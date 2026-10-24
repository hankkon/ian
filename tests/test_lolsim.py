import unittest
from collections import Counter

import numpy as np

from lolsim.data import ROLES, load_champions, load_results, load_teams
from lolsim.draft import BLUE, RED, DRAFT_ORDER, Drafter, game_win_prob
from lolsim.tournament import Simulator, seed_quarterfinals, simulate_knockout, simulate_swiss


class LolsimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = load_champions()
        cls.teams = load_teams(cls.pool)

    def test_field_is_sixteen_teams(self):
        self.assertEqual(len(self.teams), 16)

    def test_draft_is_legal(self):
        d = Drafter(self.pool, np.random.default_rng(0))
        for _ in range(50):
            r = d.run(self.teams["GEN"], self.teams["IG"])
            for side in (BLUE, RED):
                self.assertEqual(sorted(r.picks[side]), sorted(ROLES))
                self.assertEqual(len(r.bans[side]), 5)
                for role, c in r.picks[side].items():
                    self.assertTrue(self.pool.roles[c, ROLES.index(role)])
            used = [c for c in r.bans[0] + r.bans[1] + list(r.picks[0].values()) + list(r.picks[1].values())
                    if not self.pool.filler[c]]
            self.assertEqual(len(used), len(set(used)), "champion used twice in one draft")
            self.assertEqual(len(r.sequence), len(DRAFT_ORDER))

    def test_fearless_locks_respected(self):
        d = Drafter(self.pool, np.random.default_rng(1))
        locked = {self.pool.index[c] for c in ("Rumble", "K'Sante", "Vi", "Orianna", "Kai'Sa")}
        for _ in range(20):
            r = d.run(self.teams["T1"], self.teams["BLG"], locked)
            chosen = set(r.picks[0].values()) | set(r.picks[1].values()) | set(r.bans[0]) | set(r.bans[1])
            self.assertFalse(chosen & locked)

    def test_stronger_team_favoured(self):
        self.assertGreater(game_win_prob(self.teams["GEN"], self.teams["LOS"]), 0.8)
        self.assertLess(game_win_prob(self.teams["LOS"], self.teams["GEN"]), 0.2)

    def test_swiss_produces_valid_records(self):
        sim = Simulator(self.pool, self.teams, np.random.default_rng(2), draft_samples=3)
        results = load_results()["swiss"]
        for _ in range(30):
            rec = simulate_swiss(sim, results)
            dist = Counter(rec.values())
            self.assertEqual(dist[(3, 0)], 2)
            self.assertEqual(dist[(3, 1)], 3)
            self.assertEqual(dist[(3, 2)], 3)
            self.assertEqual(sum(w for w, _ in rec.values()), sum(l for _, l in rec.values()))
            qfs = seed_quarterfinals(sim, rec)
            self.assertEqual(len({t for p in qfs for t in p}), 8)
            stage = simulate_knockout(sim, qfs, [])
            self.assertEqual(sorted(stage.values()), [0, 0, 0, 0, 1, 1, 2, 3])

    def test_known_results_are_applied(self):
        sim = Simulator(self.pool, self.teams, np.random.default_rng(3), draft_samples=3)
        results = load_results()["swiss"]
        results = {"pairings": {"1": results["pairings"]["1"]},
                   "results": [{"round": 1, "teams": ["LOS", "T1"], "winner": "LOS"}]}
        for _ in range(10):
            self.assertGreaterEqual(simulate_swiss(sim, results)["LOS"][0], 1)


if __name__ == "__main__":
    unittest.main()


class FormTest(unittest.TestCase):
    def test_upset_moves_ratings(self):
        from lolsim.tournament import apply_form
        pool = load_champions()
        teams = load_teams(pool)
        before = {c: t.rating for c, t in teams.items()}
        rows = apply_form(teams, {"swiss": {"results": [{"round": 1, "teams": ["GEN", "LOS"], "winner": "LOS"}]}})
        self.assertTrue(rows[0]["upset"])
        self.assertLess(teams["GEN"].rating, before["GEN"])
        self.assertAlmostEqual(teams["GEN"].rating - before["GEN"], before["LOS"] - teams["LOS"].rating)
