# ⚽ World Cup 2026 Predictor — Logistic Regression Edition

> **⚠️ DISCLAIMER — READ BEFORE USE**
>
> This system is an **academic and experimental tool**. The predictions shown are statistical estimates based on mathematical models and historical data. **They DO NOT constitute betting advice, financial recommendations, or guarantees of results.** Football is inherently unpredictable and no model can account for all factors that influence a match (injuries, referee decisions, weather conditions, player morale, etc.).
>
> **We are not responsible for misuse of these predictions.** If you choose to use this information for sports betting, you do so at your own risk. Gambling can cause addiction and significant financial losses. Please gamble responsibly.

---

## 🏗 Architecture

```
wc2026-predictor/
├── backend/
│   ├── main.py                  # FastAPI server (9 endpoints)
│   ├── elo.py                   # ELO rating system (K=30, home +100)
│   ├── predictor.py             # Poisson model + tournament simulator
│   ├── performance.py           # Composite performance rating (1–100)
│   ├── feature_engineering.py   # 38-feature pipeline for XGBoost
│   └── models/
│       ├── logistic_predictor.py # Logistic Regression predictor (L2)
│       ├── logistic_v1.pkl       # Trained model (86% accuracy)
│       ├── xgboost_predictor.py  # Old XGBoost classifier (legacy)
│       ├── xgboost_v1.json       # Old XGBoost model (legacy)
│       └── feature_names.json    # Feature index (shared)
│   └── data/
│       ├── teams.json           # 48 teams with ELO, FIFA rank, squad value
│       ├── teams_enriched.json  # Pre-computed with ratings (instant response)
│       ├── matches.json         # 100 tournament matches (source: ESPN API)
│       └── matches_enriched.json# Pre-computed with XGBoost + Poisson predictions
├── frontend/
│   └── index.html               # SPA dashboard (1055 lines, vanilla JS)
├── requirements.txt             # Python deps (fastapi, xgboost, shap, …)
├── .gitignore
└── README.md
```

- **Backend:** Python 3.13, FastAPI, scikit-learn, NumPy, SciPy, Pydantic
- **Frontend:** Single HTML file, vanilla JavaScript, no build step, no bundler
- **Data:** Pre-computed JSON files for instant response (<5ms per request)
- **88 completed matches**, 12 scheduled (R16 onward)

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
cd backend && python3 main.py     # → http://localhost:8000

# 3. Open in browser
# The frontend SPA is served from the same server (no separate dev server needed)
open http://localhost:8000
```

### Production Deployment (Seenode)

The app is designed for **Seenode Basic** ($3/month, no cold starts):

| Setting | Value |
|---------|-------|
| Provider | Seenode |
| Plan | Basic ($3/mo → 512MB RAM, 1 vCPU) |
| Root directory | `backend` |
| Build command | `pip install -r ../requirements.txt` |
| Start command | `python main.py` |
| Domain | Custom domain supported (e.g. `wc2026.example.com`) |

**Why Seenode?**
- No cold starts — always-on container
- GitHub integration — auto-deploy on push
- Native FastAPI + Uvicorn support
- $3/month fits the lightweight stack (no DB, pre-computed JSON)

---

## 📡 Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| [ESPN API](https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world) | Teams, matches, results, stages | Real-time |
| [FIFA/Coca-Cola World Ranking](https://www.fifa.com/fifa-world-ranking) | Official FIFA ranking | June 2026 |
| [Transfermarkt](https://www.transfermarkt.com/vereins-statistik/wertvollstenationalmannschaften/marktwertetop) | Squad market values (€) — 48 teams verified | July 2026 ✓ |
| [FlagCDN](https://flagcdn.com) | National flags (160px PNG) | Static CDN |
| [CartoDB](https://carto.com) | Maps (Leaflet tiles) | CDN |
| [Google Fonts](https://fonts.google.com) | Inter typeface | CDN |

Squad values were verified against [Transfermarkt's National Team Rankings](https://www.transfermarkt.com/vereins-statistik/wertvollstenationalmannschaften/marktwertetop) on July 4, 2026. Each team's total squad market value is sourced directly from Transfermarkt's 26-player squad valuation.

---

## 🌐 Website — Dashboard Overview

The frontend is a **Single Page Application** served by FastAPI's `StaticFiles`. No build step, no framework — just vanilla JS, CSS3, and HTML5. The design follows a premium minimal aesthetic (Apple/SaaS-inspired) with glass-morphism navigation, subtle grid backgrounds, and micro-interactions.

### 4 Views

| View | URL hash | Content |
|------|----------|---------|
| **🏠 Teams** | `#teams` (default) | 48 team cards in a responsive 3-column grid. Each card shows: flag (circular, via FlagCDN), name, FIFA abbreviation, last 5 results as colored dots (W/D/L), and a color-coded performance rating badge (Elite/Strong/Average/Weak). Click a card → Detail view. Filter by confederation or group via pill buttons. |
| **📅 Matches** | `#matches` | 2-column grid of all 100 tournament matches. Each card shows: stage badge (completed/live/scheduled), team flags + names, score (if played), probability bar with dual-color breakdown (team1 win % / draw % / team2 win %), predicted score from Poisson model, and XGBoost advancement probability for knockout ties. Filter by stage (Group → Final). |
| **⚔️ H2H** | `#h2h` | Head-to-head comparator. Two dropdowns → pick any 2 teams from the 48. Results show: **V2 (XGBoost)** with P(advance) + SHAP explanation ("Why?" — top contributing features), **V1 baseline (ELO+Poisson)** with Win/Draw/Loss percentages and predicted score. Uses dual-prediction symmetrization for order-invariance. |
| **🔍 Detail** | `#detail/:id` | Full team profile. Hero section: large circular flag + team name + massive performance rating (1-100). Stats grid: FIFA rank, ELO rating, squad value, group position. "Home" badge for host nations. Form timeline: last 10 results as colored boxes (W/D/L) with opponent name and score. Embedded Leaflet map centered on the country. Built-in predictor: pick any opponent → instant XGBoost + Poisson prediction. |

### UX Features
- **Sticky nav** with glass-morphism blur backdrop
- **Skeleton loaders** (animated shimmer) while data fetches
- **Responsive** — 3-col → 2-col → 1-col at breakpoints
- **No page reloads** — hash-based routing, all data fetched once and cached
- **Color system:** Green (win), Gray (draw), Red (loss), Blue (accent)

---

## 🧠 Predictive Models

The system uses **three complementary models** with clearly defined roles:

---

### Model 1: Logistic Regression — Knockout Advancement Prediction

**Primary model** for knockout stages. Replaced XGBoost (v2) because:
- **No overfitting:** L2 regularization handles 38 features on 136 samples
- **Naturally calibrated:** logistic outputs are true probabilities by design
- **Interpretable:** each coefficient shows exactly what matters
- **Honest:** probabilities stay in the 50-65% range instead of extreme 80%+ (R32, R16, Quarter-finals, Semi-finals, Final).

#### What it predicts

The probability that a given team advances from a knockout tie (in regulation time, extra time, or penalties):

$$P(\text{advance}) \in [0, 1]$$

#### Model Architecture

| Property | Value |
|-----------|-------|
| Algorithm | Logistic Regression (L2-regularized) |
| Type | Binary classifier |
| Features | 38 engineered variables |
| Samples | 176 (88 completed matches × 2 perspectives) |
| Regularization | L2, C=0.1 (prevents overfitting) |
| Validation | Time-series CV, 5-fold (77.2% ± 7.4%) |
| Symmetrization | Dual-prediction averaging for order-invariance |

#### Dual-Prediction Symmetrization

XGBoost features are directional (`elo_diff`, `momentum_diff`, etc.) and tree models don't guarantee complementary probabilities from swapped inputs. To eliminate order bias:

$$P(\text{team1 advances}) = \frac{p_{\text{fwd}} + (1 - p_{\text{rev}})}{2}$$

Where:
- $p_{\text{fwd}}$ = XGBoost prediction with team1 as the subject (team1 vs team2 features)
- $p_{\text{rev}}$ = XGBoost prediction with team2 as the subject (team2 vs team1 features)

This guarantees $P(\text{A}) + P(\text{B}) = 1.0$ exactly.

#### Performance Metrics

| Metric | Value | Interpretation |
|---------|-------|----------------|
| **CV Accuracy** | 77.2% ± 7.4% | Time-series cross-validation (honest!) |
| **Test Accuracy** | 91.7% | Held-out 20% (chronological) |
| **Brier Score** | 0.096 | Well-calibrated (0 = perfect, 0.25 = random) |

#### Objective Function

XGBoost minimizes binary logistic loss (*logloss*):

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \Bigl[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\Bigr]$$

Where:
- $N = 176$ — training samples
- $y_i \in \{0, 1\}$ — actual outcome (1 = advanced, 0 = eliminated)
- $\hat{p}_i = \sigma(\hat{y}_i)$ — predicted probability from the tree ensemble

The model predicts $P(\text{team advances})$ from **both perspectives** (team A vs B and B vs A) and averages the results to guarantee order-invariance (see [Chen & Guestrin, 2016]).

#### Hyperparameter Optimization (Optuna)

The model was optimized using **Optuna**, a Bayesian optimization framework that explores the hyperparameter space via Tree-structured Parzen Estimators (TPE):

$$p(x \mid y) = \begin{cases} \ell(x) & \text{if } y < y^* \\ g(x) & \text{if } y \geq y^* \end{cases}$$

The TPE algorithm models $p(x \mid y < y^*)$ using one density $\ell(x)$ and $p(x \mid y \geq y^*)$ using another density $g(x)$, then samples configurations that maximize the ratio $\ell(x)/g(x)$ — focusing on promising regions of the search space (see [Akiba et al., 2019]).

Parameters optimized: `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight`.

#### Engineered Features (38 variables)

Features are grouped into 8 categories:

**1. Basic Differentials (5 features)**

| Feature | Formula | Description |
|---------|---------|-------------|
| `elo_diff` | $R_A - R_B$ | ELO rating difference |
| `fifa_rank_diff` | $\frac{R_{\text{fifa},B} - R_{\text{fifa},A}}{50}$ | Normalized FIFA rank difference (negative = better) |
| `squad_value_ratio` | $\ln\!\left(\frac{V_A}{\max(1,\, V_B)}\right)$ | Log-ratio of squad market values |
| `host_advantage` | $\mathbf{1}[A \text{ is host}]$ | Home team indicator |
| `host_vs_away` | $\mathbf{1}[A \text{ host} \land \neg B \text{ host}]$ | Pure home vs away |

**2. Tournament Momentum (3 features)**

Momentum measures recent performance with exponential weighting:

$$M_A = \frac{\sum_{i=0}^{n-1} r_i \cdot w_i \cdot \lambda^{\,i}}{\sum_{i=0}^{n-1} w_i \cdot \lambda^{\,i}}$$

Where:
- $r_i \in \{1.0,\; 0.5,\; 0.35,\; 0.20,\; 0.0\}$ — match outcome score
  - Win: 1.0
  - Draw: 0.5
  - Loss by 1 goal vs opponent rated $\geq 50$ ELO higher: 0.35 *(close loss bonus)*
  - Loss by 1 goal vs similarly-rated opponent: 0.20
  - Clear loss: 0.0
- $w_i \in \{1.0,\; 2.0,\; 3.0,\; 4.0,\; 5.0,\; 6.0\}$ — match importance (group=1.0, R32=2.0, …, Final=6.0)
- $\lambda = 0.85$ — exponential decay factor (recent matches weighted higher)

Features: `momentum`, `opponent_momentum`, `momentum_diff`

**3. Overperformance (3 features)**

Captures the "Cape Verde effect" — teams that systematically outperform expectations:

$$O_A = \frac{1}{n}\sum_{i=1}^{n} (g_i - \mathbb{E}[g_i]) \cdot \left(1 - 0.3 \cdot \frac{R_{\text{opp},i} - 1500}{400}\right)$$

Where $g_i$ are actual goals scored and $\mathbb{E}[g_i]$ are expected goals from ELO difference. The adjustment factor penalizes overperformance against strong opponents (harder to "overperform" against Brazil than against Haiti).

Features: `overperformance`, `opponent_overperformance`, `overperformance_diff`

**4. Physical Factors (3 features)**

| Feature | Description |
|---------|-------------|
| `rest_days_diff` | Difference in rest days since last match |
| `extra_time_diff` | Extra time minutes played (negative = fresher) |
| `cumulative_mins_ratio` | Ratio of accumulated tournament minutes |

**5. Performance Metrics (5 features)**

| Feature | Formula |
|---------|---------|
| `goals_scored_per_match` | $\frac{1}{n}\sum g_{F}$ |
| `goals_conceded_per_match` | $\frac{1}{n}\sum g_{A}$ |
| `goal_diff_per_match` | $\mu_F - \mu_A$ |
| `clean_sheet_rate` | $\frac{\text{matches with no goals conceded}}{n}$ |
| `comeback_rate` | $\frac{\text{comeback wins}}{n}$ |

**6. Goal Consistency (1 feature)**

$$C_A = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(g_i - \bar{g})^2}$$

Standard deviation of goals per match. Lower values indicate consistent scoring — a team that always scores 2 goals is more predictable than one alternating between 0 and 4.

**7. Stage & Group (7 features)**

`stage_importance`, `stage_is_knockout`, `stage_is_r16`, `stage_is_qf`, `stage_is_sf`, `group_pts`, `group_pts_per_match`, `group_position`

**8. Confederation & Meta-features (11 features)**

`same_confederation`, `team_is_uefa`, `team_is_conmebol`, `opp_is_uefa`, `opp_is_conmebol`, `elo_diff_abs`, `elo_ratio`, `is_top5`, `opp_is_top5`, `coming_off_close_loss`

#### Explainability with SHAP

Each prediction includes **SHAP** (*SHapley Additive exPlanations*) values, based on Shapley values from cooperative game theory:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F| - |S| - 1)!}{|F|!} \Bigl[f(S \cup \{j\}) - f(S)\Bigr]$$

Where:
- $F$ is the set of all features
- $S$ is a subset of features
- $f(S)$ is the model prediction using only the features in $S$
- $\phi_j$ is the marginal contribution of feature $j$

This explains **why** the model predicts what it predicts — e.g., "+15.2% due to ELO difference, −3.4% due to lower FIFA ranking" (see [Lundberg & Lee, 2017]).

On the H2H page, the top 8 SHAP contributions are displayed as horizontal bars: green for positive contributions (helping the team), red for negative (hurting the team).

#### Limitations

- Trained on data from a **single tournament** (WC 2026). Does not generalize to other contexts.
- Does not model injuries, tactical changes, or specific lineups.
- Assumes group stage performance patterns hold in knockout rounds.
- 176 training samples — limited statistical power for rare match outcomes.

---

### Model 2: ELO + Poisson — Score and Goal Prediction

**Complementary model** for Win/Draw/Loss breakdown and exact score probabilities.

#### 2.1 ELO Rating System

Adapted from chess (Elo, 1978). Each team has a rating $R$ representing its relative strength.

##### Expected Score

For a match between team A (rating $R_A$) and team B (rating $R_B$):

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)\,/\,400}}$$

Properties:
- $E_A \in [0, 1]$
- **400-point rule:** a team rated 400 points higher wins ~91% of the time
- $E_B = 1 - E_A$

##### Home Advantage

Host teams receive a fixed bonus of +100 ELO points:

$$E_A^{\text{home}} = \frac{1}{1 + 10^{(R_B - (R_A + 100))\,/\,400}}$$

##### Rating Update (K-Factor)

After each match, ratings are updated using the World Cup $K$-factor ($K = 30$):

$$R'_A = R_A + K \cdot (S_A - E_A)$$

Where $S_A$ is the actual result:
- Win: $S_A = 1.0$
- Draw: $S_A = 0.5$
- Loss: $S_A = 0.0$

Total ELO in the system is conserved — what one team gains, the other loses.

#### 2.2 Poisson Goal Model

Goals in football are rare, discrete events — naturally modeled by the Poisson distribution (see [Maher, 1982]; [Dixon & Coles, 1997]).

##### Expected Goals (xG)

$$\lambda_A = \max\!\left(0.25,\; \lambda_{\text{base}} + \frac{R_A - R_B}{400}\right)$$

Where:
- $\lambda_{\text{base}} = 1.35$ — empirical average goals per team in World Cup group stages
- Each 100 ELO points shifts expected goals by $\pm 0.25$
- Floor of $0.25$ prevents degenerate zero-goal predictions

With home advantage: $R_A \leftarrow R_A + 100$ for the host nation.

##### Exact Score Probability

$$P(g_A, g_B) = \underbrace{\frac{\lambda_A^{g_A}\, e^{-\lambda_A}}{g_A!}}_{\text{Poisson}(g_A \mid \lambda_A)} \;\times\; \underbrace{\frac{\lambda_B^{g_B}\, e^{-\lambda_B}}{g_B!}}_{\text{Poisson}(g_B \mid \lambda_B)}$$

Goals from both teams are assumed **independent Poisson processes** — the standard assumption in football analytics (Maher, 1982; Dixon & Coles, 1997).

##### Aggregate Probabilities

Summing over all possible scorelines ($g_i \in [0, 8]$):

$$P(\text{A wins}) = \sum_{g_A=0}^{8} \sum_{g_B=0}^{g_A-1} P(g_A, g_B)$$

$$P(\text{Draw}) = \sum_{g=0}^{8} P(g, g)$$

$$P(\text{B wins}) = \sum_{g_B=0}^{8} \sum_{g_A=0}^{g_B-1} P(g_A, g_B)$$

Scores above 8 goals are truncated (combined probability $< 0.001\%$).

##### Most Likely Score

$$s_{\text{pred}} = \underset{g_A,\, g_B}{\arg\max}\; P(g_A, g_B)$$

#### Example: Argentina vs England

| | ELO | xG |
|---|-----|----|
| 🇦🇷 Argentina | 2127 | 1.78 |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 2039 | 1.42 |

- **Argentina wins:** 52.3%
- **Draw / Extra Time:** 24.1%
- **England wins:** 23.6%
- **Most likely score:** 2–1

#### Limitations

- Poisson independence: does not model intra-match correlation (e.g., a red card affecting both teams)
- Does not consider lineups, injuries, or tactical factors
- Uses only ELO as a strength predictor
- No extra time in group stage (90-minute result assumed)

---

### Model 3: Composite Performance Rating

Each team receives a rating from **1–100** combining three weighted factors:

$$\text{rating} = 1 + 99 \times \Bigl(0.65 \cdot \text{ELO}_\text{norm} + 0.25 \cdot \text{Form} + 0.10 \cdot \text{GD}_\text{factor}\Bigr)$$

| Component | Weight | Description |
|-----------|--------|-------------|
| **Normalized ELO** | 65% | ELO mapped to $[0, 1]$ (competitive range: 1380–2127) |
| **Form** | 25% | Weighted average of last 10 matches (exponential decay $\alpha = 0.93$) |
| **Goal difference** | 10% | Average goal difference, clamped to $[-3, +3]$ |

Form is calculated using geometrically decreasing weights:

$$F = \frac{\sum_{i=0}^{9} r_i \cdot 0.93^{\,i}}{\sum_{i=0}^{9} 0.93^{\,i}}$$

Where $r_i$ is the result of the $i$-th most recent match. A match from 1 round ago weighs ~2× more than one from 10 rounds ago.

---

## 🏆 Tournament Bracket

The `/api/bracket/v2` endpoint simulates the entire knockout bracket using XGBoost dual-prediction:

1. **R16:** 8 matches with pre-computed XGBoost probabilities (from `matches_enriched.json`)
2. **QF:** 4 matches — winners of R16 paired by bracket order, XGBoost predicts each
3. **SF:** 2 matches — QF winners paired
4. **Final:** 1 match — SF winners, determines the predicted champion

The result is a JSON tree with winners, advancement probabilities, and team info (name, flag, ELO, FIFA rank) at every round. The frontend renders this as an interactive knockout bracket with connected lines, probability badges, and team flags.

---

## 📊 Squad Values (Transfermarkt)

The 48 qualifying teams with verified squad market values:

| # | Team | Value |
|---|------|-------|
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

## 📡 API Reference

| Endpoint | Description |
|----------|-------------|
| `GET /api/teams` | 48 teams with ratings, squad values, recent form, group position |
| `GET /api/teams/{id}` | Single team detail (0–47) |
| `GET /api/matches` | All 100 matches with XGBoost + Poisson predictions pre-computed |
| `GET /api/matches/{id}` | Single match detail |
| `GET /api/groups` | 16 groups (A–P) with teams and matches |
| `GET /api/bracket` | Full bracket simulation using **Poisson/ELO only** (legacy) |
| `GET /api/bracket/v2` | Full bracket simulation using **XGBoost dual-prediction** (recommended) |
| `GET /api/predict?team1_id=X&team2_id=Y` | Poisson prediction only (legacy) |
| `GET /api/predict/v2?team1_id=X&team2_id=Y` | **XGBoost:** P(advance) + SHAP explanation + Poisson baseline |
| `GET /api/model-info` | Model metrics (accuracy 91.7%, ROC AUC 0.987, Brier 0.066) |
| `GET /api/retrain` | Invalidate caches and reload data from disk |

### Response shapes

**`/api/teams`** — each team includes:
```json
{
  "id": 0, "name": "Mexico", "abbr": "MEX", "flag_emoji": "🇲🇽",
  "flag_code": "mx", "confederation": "CONCACAF", "is_host": true,
  "fifa_rank": 15, "elo_rating": 1872.5, "squad_value_millions": 191.85,
  "group": "A", "group_pos": 1,
  "recent_form": [{"opponent": "Ecuador", "result": "W", "goals_for": "2", "goals_against": "0", …}],
  "performance": {"rating_100": 89, "rating_10": 9, "form_score": 100.0, …}
}
```

**`/api/predict/v2`** — dual-model comparison:
```json
{
  "team1": {"name": "Argentina", "flag": "🇦🇷", "elo": 2127, "fifa_rank": 1},
  "team2": {"name": "England", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "elo": 2039, "fifa_rank": 3},
  "v2_prediction": {
    "team1_advance_prob": 0.6234,
    "model": "XGBoost (38 features, Optuna-tuned)",
    "features_used": 38
  },
  "v1_baseline": {
    "team1_win_prob": 52.3, "draw_prob": 24.1, "team2_win_prob": 23.6,
    "model": "ELO + Poisson"
  },
  "explanation": {
    "prediction": 0.6234, "base_value": 0.5,
    "top_factors": [
      {"feature": "elo_diff", "value": 88.0, "shap": 0.152},
      …
    ]
  }
}
```

**`/api/bracket/v2`** — full knockout simulation:
```json
{
  "model": "XGBoost (dual-prediction symmetrized)",
  "champion": {"id": 6, "name": "Argentina", "flag": "🇦🇷", "elo": 2127, …},
  "rounds": {
    "round_of_16": [{"match_id": 88, "team1": {…}, "team2": {…}, "team1_prob": 81.2, "team2_prob": 18.8, "winner_id": 6}, …],
    "quarter_finals": […],
    "semi_finals": […],
    "final": […]
  }
}
```

---

## 🛠 Tech Stack

| Layer | Technology |
|------|-----------|
| Language | Python 3.13 |
| ML | Logistic Regression (L2, C=0.1, 38 features) + XGBoost (legacy) |
| API | FastAPI + Uvicorn |
| Mathematics | NumPy, SciPy (Poisson PMF) |
| Explainability | Logistic coefficients (feature contributions) |
| Data | Static pre-computed JSON (`@lru_cache`) |
| Frontend | HTML5 + CSS3 + Vanilla JS (SPA, 1055 LOC) |
| Maps | Leaflet.js (OpenStreetMap tiles via CartoDB) |
| Flags | FlagCDN (160px PNG) |
| Typography | Inter (Google Fonts) |
| Deployment | Seenode Basic ($3/mo) |

---

## 📚 References

| # | Reference | DOI / Link |
|---|-----------|------------|
| 1 | **XGBoost:** Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016. | [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785) |
| 2 | **SHAP:** Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017. | [10.5555/3295222.3295230](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) |
| 3 | **Optuna:** Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD 2019. | [10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701) |
| 4 | **ELO System:** Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing. | ISBN 0-668-04721-6 |
| 5 | **Poisson Football:** Maher, M. J. (1982). *Modelling association football scores*. Statistica Neerlandica, 36(3), 109–118. | [10.1111/j.1467-9574.1982.tb00782.x](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x) |
| 6 | **Dixon-Coles:** Dixon, M. J. & Coles, S. G. (1997). *Modelling association football scores and inefficiencies in the football betting market*. JRSS C, 46(2), 265–280. | [10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065) |
| 7 | **Bayesian Opt / TPE:** Bergstra, J., Bardenet, R., Bengio, Y. & Kégl, B. (2011). *Algorithms for Hyper-Parameter Optimization*. NeurIPS 2011. | [10.5555/2986459.2986743](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html) |
| 8 | **FIFA Ranking:** FIFA/Coca-Cola Men's World Ranking. | [fifa.com](https://www.fifa.com/fifa-world-ranking) |
| 9 | **Squad Values:** Transfermarkt National Team Rankings. | [transfermarkt.com](https://www.transfermarkt.com/vereins-statistik/wertvollstenationalmannschaften/marktwertetop) |
| 10 | **ESPN API:** FIFA World Cup 2026 data. | [espn.com](https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world) |

---

## 📝 License

MIT

---

> **⚠️ REMINDER:** This project is an educational and experimental tool. It should not be used as a basis for betting decisions. Statistical models have inherent limitations and real football frequently defies predictions. Use at your own discretion and responsibility.

---

*Data: ESPN API, Transfermarkt (verified July 2026), FIFA. Models: XGBoost + ELO + Poisson. World Cup 2026 Predictor.*
