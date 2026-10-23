"""CLI: python -m lolsim {tournament,match,draft} ..."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .data import ROLES, load_champions, load_results, load_teams
from .draft import BLUE, RED, Drafter, game_win_prob
from .tournament import Simulator, head_to_head, role_of, run_tournament


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def presence_table(pool, st, top: int) -> str:
    g = max(st.games, 1)
    champs = set(st.picks) | set(st.bans)
    rows = sorted(champs, key=lambda c: -(st.picks[c] + st.bans[c]))[:top]
    return md_table(
        ["英雄", "位置", "出場率(P+B)", "選用率", "禁用率", "預測勝率"],
        [[c, role_of(pool, st, c), pct((st.picks[c] + st.bans[c]) / g), pct(st.picks[c] / g),
          pct(st.bans[c] / g), pct(st.wins[c] / st.picks[c]) if st.picks[c] else "  -  "] for c in rows],
    )


def cmd_tournament(args, pool, teams) -> str:
    results = load_results(Path(args.results) if args.results else None)
    rng = np.random.default_rng(args.seed)
    print(f"[1/2] 建立模擬器（模式: {args.mode}）...", file=sys.stderr)
    sim = Simulator(pool, teams, rng, mode=args.mode, fearless=not args.no_fearless, draft_samples=args.draft_samples)
    print(f"[2/2] 模擬整屆賽事 {args.sims} 次...", file=sys.stderr)
    out = run_tournament(sim, results, args.sims)
    order = sorted(out, key=lambda t: (-out[t]["champion"], -out[t]["advance"]))
    rows = [[t, teams[t].seed, pct(o["3-0"]), pct(o["3-1"]), pct(o["3-2"]), pct(o["advance"]),
             pct(o["semis"]), pct(o["final"]), pct(o["champion"])] for t, o in ((t, out[t]) for t in order)]
    parts = [
        f"## 晉級機率（{args.sims} 次模擬）\n",
        md_table(["隊伍", "種子", "3-0", "3-1", "3-2", "晉級八強", "四強", "決賽", "冠軍"], rows),
        f"\n## 全賽事 BP 預測（Top {args.top}）\n",
        presence_table(pool, sim.draft_stats(), args.top),
    ]
    return "\n".join(parts)


def cmd_match(args, pool, teams) -> str:
    a, b = args.team_a.upper(), args.team_b.upper()
    rng = np.random.default_rng(args.seed)
    sim = Simulator(pool, {a: teams[a], b: teams[b]}, rng, mode="full", fearless=not args.no_fearless)
    h = head_to_head(sim, a, b, args.bo, args.sims)
    st = sim.draft_stats()
    lines = [f"## {teams[a].name} vs {teams[b].name}（BO{args.bo}，{args.sims} 次模擬）\n",
             f"- {a} 勝率：**{pct(h['win_a']).strip()}**　{b} 勝率：**{pct(1 - h['win_a']).strip()}**",
             "- 比分分布：" + "、".join(f"{k} {pct(v).strip()}" for k, v in h["scores"].items()),
             f"\n### 此對戰 BP 預測（{int(st.games)} 場）\n", presence_table(pool, st, args.top)]
    return "\n".join(lines)


def cmd_draft(args, pool, teams) -> str:
    blue, red = teams[args.blue.upper()], teams[args.red.upper()]
    rng = np.random.default_rng(args.seed)
    drafter = Drafter(pool, rng)
    names = pool.names
    bans = (Counter(), Counter())
    picks = ({r: Counter() for r in ROLES}, {r: Counter() for r in ROLES})
    first = Counter()
    example = None
    probs = []
    for _ in range(args.sims):
        d = drafter.run(blue, red)
        probs.append(game_win_prob(blue, red, d.edge()))
        for s in (BLUE, RED):
            bans[s].update(names[c] for c in d.bans[s])
            for r, c in d.picks[s].items():
                picks[s][r][names[c]] += 1
        first[names[next(iter(d.picks[BLUE].values()))]] += 1  # dicts keep pick order
        example = example or d
    n = args.sims
    out = [f"## 單場 BP 模擬：藍方 {blue.name} vs 紅方 {red.name}（{n} 次）\n",
           f"- 藍方平均勝率（含 BP 影響）：**{pct(float(np.mean(probs))).strip()}**"
           f"（BP 造成的區間 P10–P90：{pct(np.percentile(probs, 10)).strip()} – {pct(np.percentile(probs, 90)).strip()}）",
           "- 藍方一選：" + "、".join(f"{c} {pct(v / n).strip()}" for c, v in first.most_common(5))]
    for s, team in ((BLUE, blue), (RED, red)):
        side = "藍方" if s == BLUE else "紅方"
        out.append(f"\n### {side} {team.code} 禁用\n")
        out.append("、".join(f"{c} {pct(v / n).strip()}" for c, v in bans[s].most_common(8)))
        out.append(f"\n### {side} {team.code} 各路選角\n")
        out.append(md_table(["位置", "最可能", "次可能", "第三"], [
            [r] + [f"{c} {pct(v / n).strip()}" for c, v in picks[s][r].most_common(3)] + ["-"] * (3 - len(picks[s][r].most_common(3)))
            for r in ROLES]))
    out.append("\n### 範例 BP 流程（其中一次模擬）\n")
    out.append(md_table(["#", "隊伍", "動作", "英雄", "位置"], [
        [str(i + 1), (blue, red)[s].code, "禁" if a == "ban" else "選", names[c], r or ""]
        for i, (s, a, c, r) in enumerate(example.sequence)]))
    return "\n".join(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="lolsim", description="LoL 世界賽 BP 與晉級機率模擬")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", help="同時把結果寫入 Markdown 檔")
    p.add_argument("--top", type=int, default=25, help="BP 表顯示的英雄數")
    p.add_argument("--no-fearless", action="store_true", help="關閉 Fearless（同系列賽不可重複使用英雄）")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tournament", help="模擬瑞士輪 + 淘汰賽，輸出晉級/奪冠機率")
    t.add_argument("--sims", type=int, default=5000)
    t.add_argument("--mode", choices=["fast", "full"], default="fast",
                   help="fast: 預先模擬 BP 建表；full: 每場比賽都跑完整 BP（慢、含 fearless）")
    t.add_argument("--draft-samples", type=int, default=60, help="fast 模式每組對戰的 BP 抽樣數")
    t.add_argument("--results", help="已賽結果 JSON（預設 lolsim/data/results.json）")

    m = sub.add_parser("match", help="兩隊系列賽勝率與 BP 預測")
    m.add_argument("team_a"); m.add_argument("team_b")
    m.add_argument("--bo", type=int, default=3, choices=[1, 3, 5])
    m.add_argument("--sims", type=int, default=1000)

    d = sub.add_parser("draft", help="單場 BP 細節：藍方 vs 紅方")
    d.add_argument("blue"); d.add_argument("red")
    d.add_argument("--sims", type=int, default=2000)

    db = sub.add_parser("dashboard", help="產生隊伍/選手/英雄池儀表板 HTML")
    db.add_argument("--sims", type=int, default=3000)
    db.add_argument("--draft-samples", type=int, default=60)
    db.add_argument("--results", help="已賽結果 JSON（預設 lolsim/data/results.json）")
    db.add_argument("--html", default="reports/dashboard.html", help="輸出路徑")
    db.add_argument("--fragment", action="store_true", help="不含 <!doctype>/<html> 外框（給會自動包外框的平台）")

    pr = sub.add_parser("predict", help="產生預測網頁（晉級機率、第一輪對戰、BP 預測）")
    pr.add_argument("--sims", type=int, default=5000)
    pr.add_argument("--draft-samples", type=int, default=60)
    pr.add_argument("--results", help="已賽結果 JSON（預設 lolsim/data/results.json）")
    pr.add_argument("--html", default="reports/predictions.html", help="輸出路徑")
    pr.add_argument("--fragment", action="store_true", help="不含 <!doctype>/<html> 外框")

    args = p.parse_args(argv)
    pool = load_champions()
    teams = load_teams(pool)
    for attr in ("team_a", "team_b", "blue", "red"):
        code = getattr(args, attr, None)
        if code and code.upper() not in teams:
            p.error(f"未知隊伍 {code}；可用：{', '.join(teams)}")
    if args.cmd == "predict":
        from .predictions import write_predictions
        results = load_results(Path(args.results) if args.results else None)
        path = write_predictions(pool, teams, results, args.sims, args.seed, args.draft_samples,
                                 Path(args.html), standalone=not args.fragment)
        print(f"已輸出 {path}")
        return
    if args.cmd == "dashboard":
        from .dashboard import write_dashboard
        results = load_results(Path(args.results) if args.results else None)
        path = write_dashboard(pool, teams, results, args.sims, args.seed, args.draft_samples,
                               Path(args.html), standalone=not args.fragment)
        print(f"已輸出 {path}")
        return
    report = {"tournament": cmd_tournament, "match": cmd_match, "draft": cmd_draft}[args.cmd](args, pool, teams)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
