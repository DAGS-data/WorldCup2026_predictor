# ⚽ World Cup 2026 Predictor — XGBoost Edition

> **⚠️ LEGAL DISCLAIMER — READ BEFORE USE**
>
> This system is an **academic and experimental** tool. Predictions are statistical estimates based on mathematical models and real historical data. **They do NOT constitute betting advice, financial recommendations, or guarantees of results.** Football is inherently unpredictable and no model can account for all factors (injuries, referee decisions, weather, team morale, etc.).
>
> **We are not responsible for any misuse of these predictions.** If you choose to use this information for betting, you do so at your own risk. Gambling can cause addiction and significant financial losses. Play responsibly.

---

## 🎥 Demo

![World Cup 2026 Predictor Demo](docs/demo.gif)

## 📡 Data Sources (100% real)

Every number in this project has a verifiable source. **No synthetic or made-up data.**

| Source | What we get | Verification |
|--------|--------------|-------------|
| [ESPN API](https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world) | 48 teams, 100 matches, dates, scores, stages | Real-time |
| [FIFA/Coca-Cola World Ranking](https://www.fifa.com/fifa-world-ranking) | Official ranking of all 48 teams | June 2026 |
| [Transfermarkt](https://www.transfermarkt.com/vereins-statistik/wertvollstenationalmannschaften/marktwertetop) | Squad market value (26 players per team) | July 4, 2026 ✓ |
| [FlagCDN](https://flagcdn.com) | National flags (PNG 160px) | CDN |
| [CartoDB](https://carto.com) | Leaflet maps (OpenStreetMap) | CDN |

**Manually verified squad values:**

| # | Team | Value |
|---|--------|-------|
| 1 | 🇫🇷 France | €1,520M |
| 2 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | €1,360M |
| 3 | 🇪🇸 Spain | €1,220M |
| 4 | 🇵🇹 Portugal | €1,010M |
| 5 | 🇩🇪 Germany | €947M |
| 6 | 🇧🇷 Brazil | €928M |
| 7 | 🇦🇷 Argentina | €808M |
| 8 | 🇳🇱 Netherlands | €754M |
| 9 | 🇳🇴 Norway | €590M |
| 10 | 🇧🇪 Belgium | €548M |

---

## 🏗 Architecture

```
wc2026-predictor/
├── backend/
│   ├── main.py                  # FastAPI server (11 endpoints)
│   ├── elo.py                   # ELO system adapted for football
│   ├── predictor.py             # Poisson model + tournament simulator
│   ├── performance.py           # Composite performance rating (1–100)
│   ├── feature_engineering.py   # 38-feature pipeline
│   ├── requirements.txt         # Dependencies
│   ├── frontend/                # SPA served statically
│   │   └── index.html           # Dashboard (1055 lines, vanilla JS)
│   ├── models/
│   │   ├── xgboost_predictor.py # XGBoost classifier + Optuna + SHAP
│   │   ├── xgboost_v1.json      # Trained model (91.7% accuracy)
│   │   └── feature_names.json   # Feature index
│   └── data/
│       ├── teams.json           # 48 teams (ELO, FIFA, squad value)
│       ├── teams_enriched.json  # Precomputed ratings (response <5ms)
│       ├── matches.json         # 100 matches (ESPN API)
│       └── matches_enriched.json# Precomputed with predictions
├── frontend/
│   └── index.html               # Copy for local development
├── docs/
│   └── demo.mp4                 # Demo video
├── requirements.txt
└── README.md
```

- **Backend:** Python 3.13, FastAPI, XGBoost, NumPy, SciPy, SHAP, Pydantic
- **Frontend:** HTML5 + CSS3 + Vanilla JS, no build step
- **Data:** Precomputed JSON with `@lru_cache` (response <5ms)
- **88 completed matches**, 12 scheduled (R16 → Final)

---

## 🧠 Predictive Models

The system uses **three complementary models** with clearly defined roles:

| Model | What it predicts | When it's used |
|--------|---------------|-----------------|
| **XGBoost** | Knockout advancement probability | Knockout stages (R32 → Final) |
| **ELO + Poisson** | Win / Draw / Loss + most likely scoreline | All matches |
| **Performance Rating** | Composite rating 1–100 | Team view, comparisons |

---

## Model 1: XGBoost — Advancement Probability

### What it predicts

$$P(\text{team advances}) \in [0, 1]$$

For each knockout matchup, the model estimates each team's probability of advancing to the next round (in regular time, extra time, or penalties).

### How it works

XGBoost (eXtreme Gradient Boosting) is an ensemble of decision trees that builds models sequentially, where each new tree corrects the errors of the previous one. Instead of "comparing" two teams directly, the model receives **38 numerical features** describing the difference between both teams and outputs a probability.

```
features(elo_diff, momentum, squad_value, ...) → XGBoost → P(advance)
```

The model **doesn't know who the teams are**. It sees numerical differences:

| Example feature | Argentina vs England |
|-------------------|------------------------|
| `elo_diff` | +88.5 (Argentina better) |
| `squad_value_ratio` | −0.52 (England pricier squad) |
| `momentum_diff` | +0.13 (Argentina better form) |
| `rest_days_diff` | 0 (same rest) |

With these 38 numbers, XGBoost predicts team A's advancement probability. To eliminate order bias (the result changing if you swap teams), we use **dual symmetrization**:

$$P(\text{A advances}) = \frac{p_{\text{A vs B}} + (1 - p_{\text{B vs A}})}{2}$$

This guarantees $P(\text{A}) + P(\text{B}) = 1.0$ always.

### Architecture

| Property | Value |
|-----------|-------|
| Algorithm | XGBoost (Gradient Boosted Trees) |
| Type | Binary classifier |
| Features | 38 engineered variables |
| Samples | 176 (88 matches × 2 perspectives) |
| Optimization | Optuna (200 rounds, TPE sampler) |
| Validation | Time-based split (80% train, 20% validation) |

### Metrics

| Metric | Value |
|---------|-------|
| Accuracy | 91.7% |
| ROC AUC | 0.987 |
| Brier Score | 0.066 |

### Why XGBoost?

1. **Non-linear:** Captures complex feature interactions that linear models can't (e.g., "high ELO ONLY matters if the team also has positive momentum")
2. **Built-in regularization:** L1 + L2 on trees prevents automatic overfitting
3. **Handles correlated features:** If `elo_diff` and `elo_ratio` measure similar things, XGBoost uses the most informative one without confusion
4. **SHAP:** Every prediction comes with an explanation of which features contributed the most

### Explainability with SHAP

Every prediction includes SHAP values (SHapley Additive exPlanations), based on cooperative game theory:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F| - |S| - 1)!}{|F|!} \Bigl[f(S \cup \{j\}) - f(S)\Bigr]$$

Where $\phi_j$ is the marginal contribution of feature $j$ to the prediction. In the frontend, features are shown as bars: **green** if they help the team, **red** if they hurt it.

---

## Model 2: ELO + Poisson — Goals and Scoreline

### The ELO System — Adapted for Football

#### Methodology

The ELO system was created by Arpad Elo in 1960 for chess and is documented in his book *The Rating of Chessplayers, Past and Present* (1978). The idea is simple: each player (or team) has a numerical rating $R$ representing their relative strength. After each match, ratings are updated based on expected vs actual result.

The **update mechanism is 100% standard** (Elo, 1978). The **rating initialization** is feature engineering based on FIFA ranking.

#### ELO Initialization — Feature Engineering

There is no pre-existing ELO system covering all 48 World Cup teams. Initial ELO ratings were derived from the **FIFA pre-tournament ranking (June 2026)** via linear interpolation between two endpoints:

$$\text{ELO}_{\text{initial}} \approx 2100 - \frac{700}{89} \times (\text{FIFA Ranking} - 1)$$

**Why 2100 for #1?** In chess, a world champion is around 2850. A lower ceiling was chosen for football for three reasons: (1) there are no decades of ELO history in football — starting lower leaves room for ratings to grow with match results; (2) with a 700-point gap between #1 and #90, Argentina beats New Zealand with ~98% probability, which is realistic; (3) 2100 starts just below the actual post-tournament maximum (2127), leaving room for winners to climb.

**Why 1400 for #90?** This is the ELO chess floor for a serious amateur (~1200–1400). 1000 (absolute beginner) was not used because all World Cup teams are professional level.

**Why `(rank − 1)`?** So #1 doesn't lose any points. If the formula used just `rank`, Argentina (#1) would unnecessarily lose ~7.87 points. The `−1` anchors the first position at exactly 2100.

**Why 700/89?** That's 700 points of difference between the endpoints (2100 − 1400) spread across 89 ranking positions (#1 to #90). Each FIFA ranking position is "worth" ~7.87 ELO points. It's the simplest straight line connecting the two chosen endpoints, using FIFA ranking as the X axis.

With concrete examples:

| Team | FIFA Ranking | Estimated initial ELO |
|--------|:-----------:|:--------------------:|
| 🇦🇷 Argentina | #1 | ~2100 |
| 🇲🇽 Mexico | #15 | ~1990 |
| 🇺🇾 Uruguay | #16 | ~1982 |
| 🇨🇻 Cape Verde | #54 | ~1683 |
| 🇳🇿 New Zealand | #88 | ~1416 |

This is feature engineering: FIFA ranking is the real data source; the linear interpolation to an ELO scale (~1400–2100) is the transformation that feeds the model.

Then, **the 88 real tournament matches** updated these values using the standard Elo formula (K=30). Teams that won rose; teams that lost fell. The final range is **1382–2127** (Argentina rose from ~2100 to 2127; New Zealand fell from ~1416 to 1382).

#### Standard Update Formula (Elo, 1978)

**Expected score:**
$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

**Post-match update:**
$$R'_A = R_A + K \cdot (S_A - E_A)$$

Where:
- $S_A = 1.0$ (win), $0.5$ (draw), $0.0$ (loss)
- $K = 30$ (sensitivity factor for a short tournament)
- A team with 100 more ELO than its opponent has ~64% win probability

#### Home Advantage (+100 ELO)

The three host nations (Mexico, Canada, USA) receive +100 ELO points when playing at home:

$$E_A^{\text{home}} = \frac{1}{1 + 10^{(R_B - (R_A + 100))/400}}$$

This adjustment is common practice in football ELO models (Hvattum & Arntzen, 2010; *Forecasting Association Football Outcomes*).

#### Usage in XGBoost

The XGBoost model uses **3 of its 38 features** derived from ELO:

| Feature | Formula | Role in the model |
|---------|---------|-----------------|
| `elo_diff` | $R_A - R_B$ | Feature #0 — direct differential |
| `elo_diff_abs` | $|R_A - R_B|$ | Feature #33 — magnitude of the gap |
| `elo_ratio` | $\frac{R_A}{R_B} - 1$ | Feature #34 — relative advantage |

### Poisson Goal Model

Goals in football are rare, discrete events. The Poisson distribution is the standard model for this type of event (Maher, 1982; Dixon & Coles, 1997).

#### Expected Goals (xG)

$$\lambda_A = \max\left(0.25,\; 1.35 + \frac{R_A - R_B}{400}\right)$$

- $\lambda_{\text{base}} = 1.35$ — empirical average goals per team in the group stage
- Every 100 ELO points of difference = ±0.25 expected goals
- Floor of 0.25 to avoid predicting zero goals

With home advantage: $R_A \leftarrow R_A + 100$

#### Exact Scoreline Probability

$$P(g_A, g_B) = \frac{\lambda_A^{g_A} e^{-\lambda_A}}{g_A!} \times \frac{\lambda_B^{g_B} e^{-\lambda_B}}{g_B!}$$

Both teams' goals are assumed to be **independent Poisson processes** — the standard assumption in football analytics.

#### Aggregate Probabilities

$$P(\text{A wins}) = \sum_{g_A=0}^{8} \sum_{g_B=0}^{g_A-1} P(g_A, g_B)$$
$$P(\text{Draw}) = \sum_{g=0}^{8} P(g, g)$$
$$P(\text{B wins}) = \sum_{g_B=0}^{8} \sum_{g_A=0}^{g_B-1} P(g_A, g_B)$$

Scorelines above 8 goals are truncated (combined probability $< 0.001\%$).

#### Most Likely Scoreline

$$s_{\text{pred}} = \underset{g_A, g_B}{\arg\max}\; P(g_A, g_B)$$

### Example: Argentina vs England

| | ELO | xG |
|---|-----|----|
| 🇦🇷 Argentina | 2127 | 1.78 |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 2039 | 1.42 |

- Argentina wins: 52.3%
- Draw / Extra Time: 24.1%
- England wins: 23.6%
- Most likely scoreline: 2–1

---

## Model 3: Composite Performance Rating

Rating from **1–100** combining three factors:

$$\text{rating} = 1 + 99 \times (0.65 \cdot \text{ELO}_{\text{norm}} + 0.25 \cdot \text{Form} + 0.10 \cdot \text{GD}_{\text{factor}})$$

| Component | Weight | Formula |
|-----------|:----:|---------|
| Normalized ELO | 65% | $\frac{\text{ELO} - 1380}{2130 - 1380}$ |
| Recent form | 25% | Weighted average of last 10 matches ($\alpha = 0.93$) |
| Goal difference | 10% | Average GD, capped at $[-3, +3]$, normalized to $[0,1]$ |

Form uses exponential decay: a match from 1 round ago weighs ~2× more than one from 10 rounds ago.

---

## 📊 The 38 Features in Detail

### 1. Basic Differentials (5)

| Feature | Formula | Description |
|---------|---------|-------------|
| `elo_diff` | $R_A - R_B$ | ELO difference |
| `fifa_rank_diff` | $\frac{R_{\text{fifa},B} - R_{\text{fifa},A}}{50}$ | Normalized FIFA ranking difference |
| `squad_value_ratio` | $\ln\left(\frac{V_A}{\max(1, V_B)}\right)$ | Log-ratio of squad value |
| `host_advantage` | $\mathbf{1}[A \text{ is host}]$ | Home indicator |
| `host_vs_away` | $\mathbf{1}[A \text{ home} \land \neg B \text{ home}]$ | Pure home vs away |

### 2. Tournament Momentum (3)

$$M_A = \frac{\sum_{i=0}^{n-1} r_i \cdot w_i \cdot 0.85^{i}}{\sum_{i=0}^{n-1} w_i \cdot 0.85^{i}}$$

Where:
- $r_i$: 1.0 (win), 0.5 (draw), 0.35 (1-goal loss vs strong opponent), 0.20 (narrow loss), 0.0 (clear loss)
- $w_i$: match importance (group=1.0, R32=2.0, ..., Final=6.0)

### 3. Overperformance (3)

Captures the "Cape Verde effect" — teams that systematically exceed expectations:

$$O_A = \frac{1}{n}\sum (g_i - \mathbb{E}[g_i]) \cdot \left(1 - 0.3 \cdot \frac{R_{\text{opp}} - 1500}{400}\right)$$

### 4. Physical Factors (3)

`rest_days_diff`, `extra_time_diff`, `cumulative_mins_ratio`

### 5. Performance Metrics (5)

`goals_scored_per_match`, `goals_conceded_per_match`, `goal_diff_per_match`, `clean_sheet_rate`, `comeback_rate`

### 6. Goal Consistency (1)

$$C_A = \sqrt{\frac{1}{n-1}\sum(g_i - \bar{g})^2}$$

### 7. Stage and Group (7)

`stage_importance`, `stage_is_knockout`, `stage_is_r16`, `stage_is_qf`, `stage_is_sf`, `group_pts`, `group_pts_per_match`, `group_position`

### 8. Confederation and Meta-features (11)

`same_confederation`, `team_is_uefa`, `team_is_conmebol`, `opp_is_uefa`, `opp_is_conmebol`, `elo_diff_abs`, `elo_ratio`, `is_top5`, `opp_is_top5`, `coming_off_close_loss`

---

## 🏆 Tournament Bracket

The `/api/bracket/v2` endpoint simulates the entire knockout bracket with XGBoost:

1. **R16:** 8 matches with precomputed predictions
2. **QF:** R16 winners paired by bracket order → XGBoost predicts each matchup
3. **SF:** QF winners → XGBoost
4. **Final:** SF winners → determines the predicted champion

The result is a JSON tree with winners, probabilities, and team info at each round. The frontend renders it as an interactive bracket.

---

## 🎨 Dashboard (Frontend)

Single-page application. Premium minimalist design with glass-morphism.

### 4 Views

| View | Hash | Content |
|-------|------|-----------|
| **🏠 Teams** | `#teams` | 48 cards with flag, FIFA ranking, squad value, recent form (W/D/L), and rating. Group filters. |
| **📅 Matches** | `#matches` | 100 matches with real scores, probability bars (XGBoost + Poisson), and predicted scoreline. Stage filters. |
| **⚔️ H2H** | `#h2h` | Comparator: pick 2 teams → XGBoost + SHAP + Poisson. |
| **🏆 Bracket** | `#bracket` | Full R16 → Final simulation with XGBoost. |

### UX

- Frosted glass nav (backdrop-filter blur)
- Animated skeleton loaders
- Responsive (3 → 2 → 1 column)
- No reloads (hash-based routing)
- Green = win, Gray = draw, Red = loss

---

## 📡 API Reference

| Endpoint | Description |
|----------|-------------|
| `GET /api/teams` | 48 teams with ratings and squad value |
| `GET /api/teams/{id}` | Team detail |
| `GET /api/matches` | 100 matches with precomputed predictions |
| `GET /api/matches/{id}` | Match detail |
| `GET /api/groups` | 16 groups (A–P) with teams and matches |
| `GET /api/bracket/v2` | Bracket simulation with **XGBoost** |
| `GET /api/predict/v2` | **XGBoost:** P(advance) + SHAP + Poisson baseline |
| `GET /api/predict` | Poisson prediction (legacy) |
| `GET /api/model-info` | Model metrics |
| `GET /api/retrain` | Invalidate caches |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
cd backend && python3 main.py    # → http://localhost:8000
```

### Deployment (Seenode)

| Field | Value |
|-------|-------|
| Provider | Seenode Basic ($3/mo) |
| Root directory | *(empty)* |
| Build command | `pip install -r requirements.txt` |
| Start command | `cd backend && python main.py` |
| Port | 8000 |

---

## 🛠 Tech Stack

| Layer | Technology |
|------|-----------|
| Language | Python 3.13 |
| ML | XGBoost (Optuna-tuned, 38 features) |
| API | FastAPI + Uvicorn |
| Math | NumPy, SciPy |
| Explainability | SHAP (TreeExplainer) |
| Data | Static JSON + `@lru_cache` |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Maps | Leaflet.js + CartoDB |
| Flags | FlagCDN |
| Deployment | Seenode |

---

## 📚 References

| # | Reference | Link |
|---|-----------|------|
| 1 | Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. | [DOI](https://doi.org/10.1145/2939672.2939785) |
| 2 | Lundberg, S. M. & Lee, S.-I. (2017). *SHAP: A Unified Approach to Interpreting Model Predictions*. | [Paper](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) |
| 3 | Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. | [DOI](https://doi.org/10.1145/3292500.3330701) |
| 4 | Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. | ISBN 0-668-04721-6 |
| 5 | Maher, M. J. (1982). *Modelling association football scores*. | [DOI](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x) |
| 6 | Dixon, M. J. & Coles, S. G. (1997). *Modelling association football scores and inefficiencies*. | [DOI](https://doi.org/10.1111/1467-9876.00065) |
| 7 | Bergstra, J. et al. (2011). *Algorithms for Hyper-Parameter Optimization*. | [Paper](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html) |

---

## 📝 License

MIT

---

*Data: ESPN API, Transfermarkt (July 2026), FIFA. Models: XGBoost + ELO + Poisson. 88 matches completed, 12 remaining.*
