# ⚽ World Cup 2026 Predictor — XGBoost Edition

A machine-learning powered prediction engine and interactive dashboard for the **2026 FIFA World Cup** (Canada, Mexico, United States). Models all 48 teams, 16 groups, and the full knockout bracket using a hybrid **XGBoost + Poisson** framework with 38 engineered features.

---

## 🏗 Architecture

```
wc2026-predictor/
├── backend/
│   ├── main.py                  # FastAPI server (9 endpoints)
│   ├── elo.py                   # ELO rating system
│   ├── predictor.py             # Poisson match predictor + bracket simulator
│   ├── performance.py           # Team performance rating (1–100 scale)
│   ├── feature_engineering.py   # 38 engineered features for XGBoost
│   └── models/
│       ├── xgboost_predictor.py # XGBoost classifier + Optuna tuning + SHAP
│       ├── xgboost_v1.json      # Trained model (91.7% accuracy)
│       └── feature_names.json   # Feature name index
│   └── data/
│       ├── teams.json           # 48 teams with ELO, FIFA rank, squad values
│       └── matches.json         # All tournament matches (ESPN API)
└── frontend/
    └── index.html               # Single-page dashboard (vanilla JS, white theme)
```

- **Backend:** Python 3.13, FastAPI, XGBoost, NumPy, SciPy, SHAP, Pydantic
- **Frontend:** Single HTML file, vanilla JavaScript, Chart.js CDN, no build step
- **Data:** JSON files + ESPN API — no database required

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Start API server
cd backend && python3 main.py          # → http://localhost:8000

# Open dashboard
open frontend/index.html               # or just double-click it
```

API docs at http://localhost:8000/docs (auto-generated Swagger UI).

---

## 🧠 Prediction Models

### Overview

The predictor combines two complementary approaches:

| Model | Use Case | Output |
|-------|----------|--------|
| **XGBoost** | Knockout matches | P(team advances) — binary classification |
| **ELO + Poisson** | Group stage matches | Win / Draw / Loss probabilities |

---

### 1. XGBoost Classifier — Knockout Prediction

The primary model for knockout-stage predictions. Trained on all completed World Cup 2026 matches using time-based cross-validation.

#### Model Architecture

| Property | Value |
|----------|-------|
| Model | XGBoost (gradient boosted trees) |
| Task | Binary classification: P(team advances) |
| Features | 38 engineered |
| Samples | 176 (from completed matches) |
| Hyperparameters | Optuna-optimized (200 rounds) |

#### Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 91.7% |
| **ROC AUC** | 0.977 |
| **Brier Score** | 0.068 |

#### Feature Engineering (38 features)

The model ingests engineered features across 8 categories:

**1. Basic Differentials (5 features)**
- `elo_diff` — ELO rating difference (team − opponent)
- `fifa_rank_diff` — FIFA rank difference (normalized, negative = better)
- `squad_value_ratio` — log ratio of squad market values (Transfermarkt data)
- `host_advantage` — boolean: is this team a host nation?
- `host_vs_away` — host vs non-host matchup

**2. Tournament Momentum (3 features)**
- `momentum` — exponentially weighted recent results with match importance scaling
- `opponent_momentum` — same for opponent
- `momentum_diff` — net momentum advantage

Momentum formula (exponential decay λ = 0.85):

$$M = \frac{\sum_{i=0}^{n} r_i \cdot w_i \cdot (0.85)^i}{\sum_{i=0}^{n} w_i \cdot (0.85)^i}$$

Where $r_i$ is the match result (1.0 = win, 0.5 = draw, 0.35 = close loss to stronger opponent, 0.2 = close loss, 0.0 = loss) and $w_i$ is the match importance weight (group = 1.0, R32 = 2.0, ... Final = 6.0).

**3. Overperformance (3 features)**
- `overperformance` — goals scored above expectation, adjusted for opponent quality
- `opponent_overperformance` — same for opponent
- `overperformance_diff` — net overperformance advantage

Captures the "Cape Verde effect" — teams that systematically outperform their pre-tournament ratings.

$$\text{over} = \frac{1}{n}\sum_i \big(g_i - E[g_i]\big) \cdot (1 - 0.3 \cdot \frac{\text{elo}_\text{opp} - 1500}{400})$$

**4. Physical Factors (3 features)**
- `rest_days_diff` — rest day advantage
- `extra_time_diff` — extra time played (negative values = fresher)
- `cumulative_mins_ratio` — accumulated playing time ratio

**5. Performance Metrics (5 features)**
- `goals_scored_per_match`, `goals_conceded_per_match`, `goal_diff_per_match`
- `clean_sheet_rate` — proportion of matches with zero goals conceded
- `comeback_rate` — proportion of wins after trailing

**6. Goal Consistency (1 feature)**
- `goal_consistency` — standard deviation of goals scored (lower = more consistent)

**7. Stage & Group Context (7 features)**
- `stage_importance`, `stage_is_knockout`, `stage_is_r16`, `stage_is_qf`, `stage_is_sf`
- `group_pts`, `group_pts_per_match`, `group_position`

**8. Confederation & Meta (11 features)**
- `same_confederation`, `team_is_uefa`, `team_is_conmebol`, `opp_is_uefa`, `opp_is_conmebol`
- `elo_diff_abs`, `elo_ratio`, `is_top5`, `opp_is_top5`, `coming_off_close_loss`

#### SHAP Explainability

Every prediction includes SHAP (SHapley Additive exPlanations) values showing which features drove the prediction:

```
Top factors (sample):
  elo_diff              +0.152  → favors team advancing
  overperformance       +0.089  → team outperforming expectations
  fifa_rank_diff        -0.034  → opponent has better ranking
```

#### Training

```python
from models.xgboost_predictor import KnockoutPredictor
from feature_engineering import build_dataset

X, y, feature_names = build_dataset()
predictor = KnockoutPredictor()
metrics = predictor.train(X, y, feature_names)
# Accuracy: 0.917, AUC: 0.977
```

---

### 2. ELO Rating System (`elo.py`)

Adapted from the chess rating system. Each team has a single scalar score representing their strength.

#### Expected Score

For team A (rating $R_A$) vs team B (rating $R_B$):

$$E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$

- $E_A \in [0, 1]$ — win probability before accounting for draws
- **400-point rule:** A team rated 400 points higher is expected to win ~91% of the time
- $E_B = 1 - E_A$

#### Home Advantage

Host nations receive a flat **+100 ELO bonus**:

$$E_A = \frac{1}{1 + 10^{(R_B - (R_A + 100)) / 400}}$$

#### Rating Update (K-Factor)

After a match, ratings update with World Cup K-factor ($K = 30$):

$$R'_A = R_A + K \cdot (S_A - E_A)$$

Where $S_A$ is the actual result: 1.0 (win), 0.5 (draw), 0.0 (loss).

---

### 3. Poisson Match Predictor (`predictor.py`)

Football goals are rare, discrete events — modeled by the Poisson distribution.

#### Expected Goals (xG)

$$\lambda_A = \max\left(0.25,\; \lambda_{\text{base}} + \frac{R_A - R_B}{400}\right)$$

- $\lambda_{\text{base}} = 1.35$ — average goals per team per World Cup group match (empirical)
- Every 100 ELO points → ±0.25 expected goals
- Floor of 0.25 prevents degenerate predictions

#### Score Probability

$$P(g_A, g_B) = \frac{\lambda_A^{g_A} \cdot e^{-\lambda_A}}{g_A!} \times \frac{\lambda_B^{g_B} \cdot e^{-\lambda_B}}{g_B!}$$

Goals are treated as independent Poisson processes (standard Dixon-Coles/Maher assumption).

#### Win/Draw/Loss

$$P(\text{A wins}) = \sum_{g_A=0}^{8} \sum_{g_B=0}^{g_A-1} P(g_A, g_B)$$

$$P(\text{Draw})   = \sum_{g=0}^{8} P(g, g)$$

$$P(\text{B wins}) = \sum_{g_B=0}^{8} \sum_{g_A=0}^{g_B-1} P(g_A, g_B)$$

Scores above 8 goals truncated (combined probability < 0.001%).

#### Example

| | ELO | xG |
|---|-----|----|
| Argentina | 2127 | 1.78 |
| Egypt | 1659 | 0.92 |

- **Argentina wins:** 61.2%
- **Draw:** 20.8%
- **Egypt wins:** 18.0%
- **Predicted score:** 2–1

---

### 4. Performance Rating (`performance.py`)

Composite 1–100 rating combining three weighted factors:

$$\text{rating} = 1 + 99 \times \big(0.65 \cdot \text{ELO}_{\text{norm}} + 0.25 \cdot \text{Form} + 0.10 \cdot \text{GD}_{\text{factor}}\big)$$

| Component | Weight | Description |
|-----------|--------|-------------|
| **ELO normalized** | 65% | ELO mapped to [0, 1] (range: 1380–1960) |
| **Form score** | 25% | Weighted recent results (decay factor 0.93) |
| **Goal difference factor** | 10% | Avg goal diff, clamped to [−3, +3] |

---

### 5. Tournament Bracket Simulator

The `/api/bracket` endpoint simulates the full knockout stage:

1. **Group stage:** All matches simulated via Poisson. Teams ranked by points → GD → GF.
2. **R32 fixtures:** Group winners vs runners-up (cross-group pattern).
3. **Knockout progression:** R32 → R16 → QF → SF → Final. Winner = higher win probability.
4. **Champion:** The bracket converges to a single predicted winner.

---

## 📊 Squad Values (Transfermarkt)

All 48 teams include squad market values sourced from [Transfermarkt](https://transfermarkt.com). These values (in millions EUR) represent the total market value of each national team's squad and feed into the XGBoost model as the `squad_value_ratio` feature.

**Top 10 most valuable squads:**

| Team | Squad Value |
|------|------------|
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | €1,470M |
| 🇫🇷 France | €1,230M |
| 🇧🇷 Brazil | €1,010M |
| 🇪🇸 Spain | €965M |
| 🇦🇷 Argentina | €950M |
| 🇩🇪 Germany | €840M |
| 🇳🇱 Netherlands | €785M |
| 🇵🇹 Portugal | €760M |
| 🇧🇪 Belgium | €595M |
| 🇳🇴 Norway | €480M |

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/teams` | GET | All 48 teams with performance ratings + squad values |
| `/api/teams/{id}` | GET | Single team detail |
| `/api/matches` | GET | All matches with predictions |
| `/api/matches/{id}` | GET | Single match prediction |
| `/api/bracket` | GET | Full tournament bracket simulation |
| `/api/groups` | GET | All 16 groups with teams and matches |
| `/api/predict?team1_id=X&team2_id=Y` | GET | Poisson prediction (head-to-head) |
| `/api/predict/v2?team1_id=X&team2_id=Y` | GET | **XGBoost prediction** with SHAP explanation |
| `/api/model-info` | GET | Model metadata and performance metrics |
| `/api/retrain` | GET | Invalidate caches, reload data |

All responses include CORS headers (`Access-Control-Allow-Origin: *`).

### Example: `/api/predict/v2?team1_id=36&team2_id=34`

```json
{
  "team1": {"name": "Argentina", "flag": "🇦🇷", "elo": 2127.3, "fifa_rank": 1},
  "team2": {"name": "England", "flag": "🏴", "elo": 2038.8, "fifa_rank": 4},
  "v2_prediction": {
    "team1_advance_prob": 0.81,
    "model": "XGBoost (50+ features, Optuna-tuned)",
    "features_used": 38
  },
  "v1_baseline": {
    "team1_win_prob": 61.2,
    "draw_prob": 20.8,
    "team2_win_prob": 18.0,
    "model": "ELO + Poisson"
  },
  "explanation": {
    "base_value": 0.369,
    "prediction": 0.81,
    "top_factors": [
      {"feature": "elo_diff", "value": 88.5, "shap": 0.152},
      {"feature": "overperformance", "value": 0.85, "shap": 0.089}
    ]
  }
}
```

---

## 🎨 Dashboard Features

The frontend (`frontend/index.html`) is a **white-themed** single-page application with 5 views:

| View | Description |
|------|-------------|
| **Teams** | 48 cards with flag, FIFA rank, squad value, and performance rating |
| **Matches** | All matches with predictions — Poisson for group, XGBoost for knockout |
| **Bracket** | Full R32 → Final knockout tree with predicted champion |
| **H2H** | ⚡ **Head-to-Head predictor** — pick any two teams, compare stats, get XGBoost win probability with SHAP explanation |
| **Team Detail** | Full stats, map, recent form, and quick predict vs any opponent |

### Key Features
- **XGBoost predictions** shown for all knockout matches with advancement probability
- **SHAP explanations** — understand why the model predicts what it does
- **Transfermarkt squad values** — displayed on every team card and detail page
- **White theme** — clean, modern light UI replacing the original dark theme
- **Lazy bracket loading** — bracket simulates on-demand to avoid slow initial load
- **Responsive** — works on desktop and mobile

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| ML Model | XGBoost (Optuna-tuned) |
| API Framework | FastAPI |
| Math | NumPy, SciPy (Poisson PMF) |
| Explainability | SHAP |
| Validation | Pydantic |
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Maps | Leaflet.js (CDN) |
| Flags | FlagCDN |
| Data Format | JSON |
| Data Source | ESPN API + Transfermarkt |

---

## 📝 License

MIT

---

*Data sourced from ESPN API and Transfermarkt. Models: XGBoost + ELO + Poisson. WC 2026 Predictor v7.*
