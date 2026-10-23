# lolsim — LoL 世界賽 BP 與晉級機率模擬

這個模型用 Monte Carlo 模擬 **2026 英雄聯盟世界賽**（瑞士輪 → 八強淘汰賽），能做到：

1. **BP 模擬**：依照正式 BP 順序（3 禁 → 3 選 → 2 禁 → 2 選）一手一手模擬，預測每隊的禁用、各路選角、一選分布。
2. **晉級預測**：模擬整屆賽事，輸出每隊 3-0 / 3-1 / 3-2 晉級、八強、四強、決賽、冠軍機率。
3. **單一對戰**：任兩隊 BO1/BO3/BO5 勝率與比分分布（支援 Fearless：同一系列賽用過的英雄不能再選）。
4. **輸入實際賽果**：比賽打完後把結果填進 `lolsim/data/results.json`，剩下的比賽會以真實戰績繼續模擬。

只需要 Python 3.10+ 與 numpy。

## 快速開始

```bash
# 整屆賽事：晉級/奪冠機率 + 全賽事英雄出場率預測
python -m lolsim tournament --sims 5000

# 單場 BP 細節（藍方 GEN、紅方 IG）
python -m lolsim draft GEN IG

# 系列賽勝率（BO5，含 fearless BP）
python -m lolsim match T1 BLG --bo 5

# 寫入 Markdown 報告、固定亂數種子
python -m lolsim --seed 2026 --out reports/my_run.md tournament --sims 10000
```

常用參數：`--seed`（可重現）、`--top N`（BP 表顯示筆數）、`--no-fearless`、
`tournament --mode full`（每場比賽都完整跑 BP，較慢但最精確）。

已產生的範例報告在 [`reports/`](reports/)。

## 模型怎麼運作

### 1. BP 模型（`lolsim/draft.py`）
每一手對所有可用英雄打分，再用 softmax 抽樣（所以重複模擬會得到「機率分布」，而不是唯一答案）：

| 因子 | 說明 |
|---|---|
| 版本強度 `power` | 英雄在目前職業賽版本的強度（`champions.json`） |
| 隊伍熟練度 `pools` | 每隊各路的招牌英雄加成（`teams.json`） |
| 隊伍風格 `style` | 例如 LPL 偏前期、LCK 偏團戰/後期，依英雄標籤加分 |
| 陣容需求 | 開團、前排、保護、AP/AD 平衡、陣容主軸（團戰/抓單/消耗） |
| 對線克制 `counters` | 後選可以 counter；先選暴露在被 counter 的風險 |
| 搶奪/封鎖 | 對手最依賴、替代品又差的英雄會被優先搶或禁 |
| 彈性位置 | 第一輪選角偏好可打多路的英雄以隱藏分路 |

禁用則以「對手在尚未選的位置上能拿到多強、失去它有多痛」為主；藍方第一輪不太禁自己能一選拿到的英雄。

### 2. 單場勝率
```
logit(藍方勝) = ln10 × (藍方rating − 紅方rating) / 400 + 藍方優勢 0.08 + 3.0 × (BP 分數差)
```
`rating` 是 Elo 式隊伍實力（差 400 分 ≈ 單場 10 倍賠率），BP 分數差讓同一組對戰依 BP 結果有約 ±5–10% 的浮動。

### 3. 賽制（`lolsim/tournament.py`）
- **瑞士輪**：16 隊、最多 5 輪，3 勝晉級、3 敗淘汰；同戰績抽籤、不重複對手；
  晉級戰/淘汰戰（2 勝或 2 敗）為 BO3，其餘 BO1。第一輪使用官方抽籤結果。
- **八強**：3-0 隊伍對上 3-2 隊伍，其餘抽籤；盡量避免同賽區在八強相遇；兩支 3-0 隊伍分在不同半區；全部 BO5。
- 系列賽中，上一場的敗方選邊（選藍方）。
- `fast` 模式：先對每組 (藍, 紅) 跑數十次 BP 建出勝率表，再快速模擬上萬次賽事（全賽事 BP 統計依各組對戰實際出現次數加權）。
  `full` 模式：每場比賽都跑完整 BP，含 fearless 系列賽鎖英雄。

## 更新資料（越新越準）

| 檔案 | 內容 |
|---|---|
| `lolsim/data/teams.json` | 16 隊的 rating、賽區、種子、風格、各路招牌英雄 |
| `lolsim/data/champions.json` | 英雄可打位置（第一個為主位置）、強度、傷害類型、標籤、克制 |
| `lolsim/data/results.json` | 已賽結果與抽籤 |

賽果填寫範例：

```json
{
  "swiss": {
    "pairings": {
      "1": [["GEN", "IG"], ["HLE", "TES"], ...],
      "2": [["GEN", "HLE"], ...]
    },
    "results": [
      {"round": 1, "teams": ["GEN", "IG"], "winner": "GEN"},
      {"round": 1, "teams": ["T1", "LOS"], "winner": "T1"}
    ]
  },
  "knockout_pairings": [["GEN", "G2"], ["T1", "AL"], ["HLE", "DK"], ["BLG", "TES"]],
  "knockout": [{"teams": ["GEN", "G2"], "winner": "GEN"}]
}
```
`knockout_pairings` 依籤表順序：第 1、2 組勝者在四強相遇，第 3、4 組勝者相遇。

## 注意事項
- 隊伍 rating、招牌英雄與英雄強度是 **賽前人工估計**（參考各賽區戰績、賽前預測與 26.20 版本職業賽趨勢），
  不是官方數據；初始校準後的晉級率與公開的賽前模型大致一致（例如 GEN ≈ 89%、LOS ≈ 5%）。
- 選手層級（個別選手英雄池）尚未建模，以隊伍各路的招牌英雄代替。
- 官方賽制細節（如八強種子規則）如與實際不同，可直接用 `knockout_pairings` 指定。

## 測試
```bash
python -m unittest
```

## 英雄池儀表板

```bash
python -m lolsim --seed 2026 dashboard --sims 3000      # 輸出 reports/dashboard.html
```
仿直播數據牆：每隊一張卡片，列出晉級/四強/決賽/冠軍機率與每屆預測戰績；每個位置列出選手、評分、勝率與英雄池
（右下角 = 該隊此位置的選用比例、下方 = 預測勝率），滑過或點擊英雄可看版本強度、熟練加成、被禁率、克制關係。
選手名稱在 `lolsim/data/players.json`（目前只填了 MKOI、KC、TLAW、LYON 部分，其餘顯示位置）。
有網路時英雄頭像會從 Riot Data Dragon 載入，否則顯示英雄縮寫。
