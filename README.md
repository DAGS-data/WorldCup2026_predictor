# 🌍 World Cup 2026 Predictor

A statistical prediction engine and interactive dashboard for the **2026 FIFA World Cup** (Canada, Mexico, United States). Models all 48 teams, 16 groups, and the full knockout bracket using a hybrid **ELO + Poisson** framework.

---

## 🏗 Architecture

```
wc2026-predictor/
├── backend/
│   ├── main.py            # FastAPI server (6 endpoints)
│   ├── elo.py             # ELO rating system
│   ├── predictor.py       # Poisson match predictor + bracket simulator
│   ├── performance.py     # Team performance rating (1–100 scale)
│   └── data/
│       ├── teams.json     # 48 national teams with stats
│       └── matches.json   # 48 group-stage matches
└── frontend/
    └── index.html         # Single-page dashboard (vanilla JS, dark theme)
```

- **Backend:** Python 3.12, FastAPI, NumPy, SciPy, Pydantic
- **Frontend:** Single HTML file, vanilla JavaScript, Chart.js CDN, no build step
- **Data:** Static JSON files — no database required

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Start API server
python backend/main.py          # → http://localhost:8000

# Open dashboard
open frontend/index.html        # or just double-click it
```

API docs at http://localhost:8000/docs (auto-generated Swagger UI).

---

## 🧠 Mathematical Models

The predictor combines two complementary approaches: **ELO ratings** for relative team strength and a **Poisson model** for goal-scoring probabilities.

---

### 1. ELO Rating System (`elo.py`)

Adapted from the chess rating system, ELO assigns each team a single scalar score that represents their strength. Higher-rated teams are expected to beat lower-rated ones.

#### Expected Score

For a match between team A (rating \( R_A \)) and team B (rating \( R_B \)), the expected score for team A is:

\[
E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}
\]

- \( E_A \in [0, 1] \) → win probability before accounting for draws
- **400-point rule:** A team rated 400 points higher is expected to win ~91% of the time
- \( E_B = 1 - E_A \)

#### Home Advantage

Host nations receive a flat **+100 ELO bonus**. If team A is a host:

\[
E_A = \frac{1}{1 + 10^{(R_B - (R_A + 100)) / 400}}
\]

#### Rating Update (K-Factor)

After a match, both teams' ratings are updated using the **World Cup K-factor** (\( K = 30 \)):

\[
R'_A = R_A + K \cdot (S_A - E_A)
\]

Where \( S_A \) is the actual result:
- \( S_A = 1.0 \) for a win
- \( S_A = 0.5 \) for a draw
- \( S_A = 0.0 \) for a loss

A win against a stronger opponent earns more points than a win against a weaker one. The total ELO in the system is conserved — what one team gains, the other loses.

#### Win Probability (quick estimate)

\[
P(\text{A wins}) = E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}
\]

This is a fast, draw-free estimate used for quick comparisons. The full match prediction uses the Poisson model below.

---

### 2. Poisson Match Predictor (`predictor.py`)

Football goals are rare, discrete events — exactly what the **Poisson distribution** models. Instead of predicting a binary win/loss, we model the full probability distribution over possible scores.

#### Expected Goals (xG)

Each team's expected goals in a match depends on the ELO difference:

\[
\lambda_A = \max\left(0.25,\; \lambda_{\text{base}} + \frac{R_A - R_B}{400}\right)
\]

Where:
- \( \lambda_{\text{base}} = 1.35 \) — average goals per team per World Cup group match (empirical)
- **Every 100 ELO points** of advantage shifts expected goals by **+0.25**
- Floor of 0.25 prevents degenerate zero-goal predictions
- Host teams add +100 to their ELO before computing \( \lambda \)

#### Score Probability

The probability of a specific score \( g_A : g_B \) is:

\[
P(g_A, g_B) = \underbrace{\frac{\lambda_A^{g_A} \cdot e^{-\lambda_A}}{g_A!}}_{\text{Poisson}(g_A \mid \lambda_A)} \times \underbrace{\frac{\lambda_B^{g_B} \cdot e^{-\lambda_B}}{g_B!}}_{\text{Poisson}(g_B \mid \lambda_B)}
\]

Goals are treated as **independent Poisson processes** — a standard assumption in football analytics (Dixon-Coles, Maher models).

#### Win/Draw/Loss Probabilities

Summing over all possible scores \( g_A, g_B \in [0, 8] \):

\[
\begin{aligned}
P(\text{A wins}) &= \sum_{g_A=0}^{8} \sum_{g_B=0}^{g_A-1} P(g_A, g_B) \\[4pt]
P(\text{Draw})   &= \sum_{g=0}^{8} P(g, g) \\[4pt]
P(\text{B wins}) &= \sum_{g_B=0}^{8} \sum_{g_A=0}^{g_B-1} P(g_A, g_B)
\end{aligned}
\]

Scores above 8 goals are truncated (combined probability < 0.001%).

#### Predicted Score

The most likely single scoreline (maximum joint probability):

\[
\text{predicted score} = \arg\max_{g_A, g_B}\; P(g_A, g_B)
\]

#### Example

| | ELO | xG |
|---|-----|----|
| Argentina | 1902 | 1.78 |
| Nigeria | 1523 | 0.92 |

- **Argentina wins:** 61.2%
- **Draw:** 20.8%
- **Nigeria wins:** 18.0%
- **Predicted score:** 2–1

---

### 3. Performance Rating (`performance.py`)

Each team gets a composite **1–100 rating** that combines three factors:

\[
\text{rating} = 1 + 99 \times \big(0.65 \cdot \text{ELO}_{\text{norm}} + 0.25 \cdot \text{Form} + 0.10 \cdot \text{GD}_{\text{factor}}\big)
\]

| Component | Weight | Description |
|-----------|--------|-------------|
| **ELO normalized** | 65% | ELO mapped to [0, 1] using competitive range 1380–1960 |
| **Form score** | 25% | Weighted average of last 10 match results (recent matches weighted higher — decay factor 0.93 per match) |
| **Goal difference factor** | 10% | Average goal difference per match, clamped to [-3, +3] and normalized to [0, 1] |

A 1–10 shorthand version is also provided for compact display (team cards in the UI).

#### Form Weighting

Recent form matters more. A win 1 match ago counts ~2× more than a win 10 matches ago:

| Matches ago | Weight |
|-------------|--------|
| 1 | 1.00 |
| 2 | 0.93 |
| 3 | 0.86 |
| 4 | 0.79 |
| … | … |
| 10 | 0.46 |

---

### 4. Tournament Bracket Simulator

The `/api/bracket` endpoint simulates the **entire knockout stage** from the group stage results:

1. **Group stage:** For each group, all 3 matches are predicted using the Poisson model. Teams are ranked by points → goal difference → goals for.
2. **R32 fixtures:** Group winners face runners-up from adjacent groups (A1 vs B2, B1 vs A2, etc.) — **16 matches**.
3. **Knockout progression:** Each round is simulated sequentially — R32 → R16 → Quarter-finals → Semi-finals → Final. For each match, the team with the higher win probability advances.
4. **Champion:** The bracket converges to a single predicted winner.

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/teams` | GET | All 48 teams with performance ratings |
| `/api/teams/{id}` | GET | Single team detail |
| `/api/matches` | GET | All 48 group matches with predictions |
| `/api/matches/{id}` | GET | Single match prediction |
| `/api/bracket` | GET | Full tournament bracket simulation |
| `/api/groups` | GET | All 16 groups with teams and matches |
| `/api/predict?team1_id=X&team2_id=Y` | GET | Head-to-head prediction for any two teams |
| `/api/retrain` | GET | Invalidate caches, reload data |

All responses include CORS headers (`Access-Control-Allow-Origin: *`).

### Example Response: `/api/matches/0`

```json
{
  "id": 0,
  "group": "A",
  "team1_id": 0,
  "team2_id": 1,
  "team1_name": "Mexico",
  "team1_flag": "🇲🇽",
  "team2_name": "Senegal",
  "team2_flag": "🇸🇳",
  "date": "2026-06-11",
  "venue": "Estadio Azteca, Mexico City",
  "prediction": {
    "team1_win_prob": 52.3,
    "draw_prob": 24.1,
    "team2_win_prob": 23.6,
    "predicted_score": "2-1",
    "team1_xg": 1.47,
    "team2_xg": 1.12,
    "key_factors": [
      "ELO advantage: Mexico (+45 pts)",
      "Home advantage: Mexico"
    ]
  }
}
```

---

## 🎨 Dashboard Features

The frontend (`frontend/index.html`) is a dark-themed single-page application:

- **Team Cards** — Horizontal scroll of 48 teams with flag, name, and color-coded rating badge
- **Group View** — 16-card grid showing 3 teams each with match schedules
- **Bracket Tree** — Full R32→Final knockout tree with probability bars
- **Team Detail Panel** — Slide-in overlay with stats table, form history, Chart.js trend chart
- **Match Panel** — VS layout with 3-segment probability bar, predicted score, xG display
- **Data Fetching** — All endpoints fetched in parallel with `Promise.allSettled` for resilience

---

## 📊 Data Format

### Team Object

```json
{
  "id": 0,
  "name": "Mexico",
  "flag_emoji": "🇲🇽",
  "group": "A",
  "confederation": "CONCACAF",
  "fifa_rank": 14,
  "elo_rating": 1845,
  "is_host": true,
  "recent_form": [
    {"result": "W", "goals_for": 3, "goals_against": 1},
    {"result": "D", "goals_for": 1, "goals_against": 1}
  ]
}
```

### Match Object

```json
{
  "id": 0,
  "group": "A",
  "team1_id": 0,
  "team2_id": 1,
  "date": "2026-06-11",
  "venue": "Estadio Azteca, Mexico City",
  "matchday": 1
}
```

---

## 🔮 Limitations & Assumptions

- **Static data:** Team ratings and form are snapshots — the `/api/retrain` endpoint reloads JSON but doesn't pull live data.
- **Poisson independence:** Goals are treated as independent events. The model doesn't account for in-game correlation (e.g., a red card affecting both teams' scoring rates).
- **No extra time in group stage:** Group matches are modeled as 90-minute results (no ET/penalties). Knockout rounds are decided by win probability in regulation.
- **Venue effects beyond home advantage:** Only host nations get a bonus. Altitude, travel distance, and climate are not modeled.
- **No player-level data:** Predictions are team-level only. Injuries, lineups, and tactical matchups are not considered.

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| API Framework | FastAPI |
| Math | NumPy, SciPy (Poisson PMF) |
| Validation | Pydantic |
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Charts | Chart.js 4.x (CDN) |
| Data Format | JSON |

---

## 📝 License

MIT

---

*Data updated daily. Models: ELO + Poisson. WC 2026 Predictor v1.*
