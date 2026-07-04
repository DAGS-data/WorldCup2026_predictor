# ⚽ World Cup 2026 Predictor — XGBoost Edition

> **⚠️ DISCLAIMER — LEER ANTES DE USAR**
>
> Este sistema es un proyecto **académico y experimental**. Las predicciones aquí mostradas son estimaciones estadísticas basadas en modelos matemáticos y datos históricos. **NO constituyen asesoría de apuestas, recomendación financiera ni garantía de resultados.** El fútbol es inherentemente impredecible y ningún modelo puede anticipar todos los factores que influyen en un partido (lesiones, decisiones arbitrales, condiciones climáticas, estado anímico de los jugadores, etc.).
>
> **No nos hacemos responsables por el uso indebido de estas predicciones.** Si usted decide utilizar esta información para apuestas deportivas, lo hace bajo su propio riesgo. Las apuestas pueden causar adicción y pérdidas financieras significativas. Juegue responsablemente.
>
> ---

---

## 🏗 Arquitectura

```
wc2026-predictor/
├── backend/
│   ├── main.py                  # FastAPI server (8 endpoints)
│   ├── elo.py                   # Sistema de rating ELO
│   ├── predictor.py             # Modelo Poisson + simulador de torneo
│   ├── performance.py           # Rating compuesto de rendimiento (1–100)
│   ├── feature_engineering.py   # Pipeline de 38 features para XGBoost
│   └── models/
│       ├── xgboost_predictor.py # Clasificador XGBoost + Optuna + SHAP
│       ├── xgboost_v1.json      # Modelo entrenado (91.7% accuracy)
│       └── feature_names.json   # Índice de features
│   └── data/
│       ├── teams.json           # 48 selecciones con ELO, FIFA rank, valor plantilla
│       ├── teams_enriched.json  # Pre-computado con ratings (respuesta instantánea)
│       ├── matches.json         # Todos los partidos del torneo (fuente: ESPN API)
│       └── matches_enriched.json# Pre-computado con predicciones XGBoost + Poisson
└── frontend/
    └── index.html               # Dashboard SPA (vanilla JS, tema claro)
```

- **Backend:** Python 3.13, FastAPI, XGBoost, NumPy, SciPy, SHAP, Pydantic
- **Frontend:** Single HTML file, vanilla JavaScript, sin build step
- **Datos:** Archivos JSON pre-computados para respuesta instantánea

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
cd backend && python3 main.py     # → http://localhost:8000
open frontend/index.html           # o doble clic
```

---

## 📡 Fuentes de Datos

| Fuente | Dato | Frecuencia |
|--------|------|------------|
| [ESPN API](https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world) | Equipos, partidos, resultados, stages | Tiempo real |
| [FIFA/Coca-Cola World Ranking](https://www.fifa.com/fifa-world-ranking) | Ranking FIFA oficial | Junio 2026 |
| [Transfermarkt](https://www.transfermarkt.com) | Valor de mercado de plantillas (€) | 2025/2026 |
| [FlagCDN](https://flagcdn.com) | Banderas de selecciones | CDN estático |
| [CartoDB](https://carto.com) | Mapas (Leaflet tiles) | CDN |

Los datos de partidos se obtienen del endpoint `fifa.world` de ESPN que cubre la Copa del Mundo 2026. Los valores de plantilla (`squad_value_millions`) provienen de Transfermarkt y representan el valor total de mercado de cada selección nacional en millones de euros.

---

## 🧠 Modelos Predictivos

El sistema utiliza **dos modelos complementarios** con roles claramente definidos:

### Modelo 1: XGBoost — Predicción de Avance en Eliminatoria

**Modelo principal** para fases de eliminación directa (R32, R16, Cuartos, Semis, Final).

#### ¿Qué predice?

$$P(\text{equipo avanza}) \in [0, 1]$$

Probabilidad de que un equipo gane la eliminatoria (en tiempo reglamentario, tiempo extra, o penales).

#### Arquitectura del Modelo

| Propiedad | Valor |
|-----------|-------|
| Algoritmo | XGBoost (Gradient Boosted Trees) |
| Tipo | Clasificador binario |
| Features | 38 variables de ingeniería |
| Muestras | 176 (partidos completados del torneo) |
| Hiperparámetros | Optimizados con Optuna (200 rondas) |
| Validación | Time-based split (80% train, 20% validation) |

#### Métricas de Rendimiento

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 91.7% | 91.7% de predicciones correctas en validación |
| **ROC AUC** | 0.977 | Excelente discriminación entre avance/eliminación |
| **Brier Score** | 0.068 | Calibración casi perfecta (0 = perfecto, 0.25 = aleatorio) |

#### Función Objetivo

XGBoost minimiza la pérdida logística binaria (*binary logloss*):

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]$$

Donde:
- $N = 176$ muestras de entrenamiento
- $y_i \in \{0, 1\}$ — resultado real (1 = avanzó, 0 = eliminado)
- $\hat{p}_i = \sigma(\hat{y}_i)$ — probabilidad predicha por el ensemble de árboles

#### Optimización de Hiperparámetros (Optuna)

El modelo fue optimizado usando **Optuna**, un framework de optimización bayesiana que explora el espacio de hiperparámetros mediante Tree-structured Parzen Estimators (TPE):

$$\text{TPE}(x) = \frac{p(x|y < y^*)}{p(x|y \geq y^*)} \cdot p(y < y^*)$$

Donde $y^*$ es un umbral de rendimiento (ej. accuracy > 0.85). Optuna muestrea configuraciones que maximizan esta razón, enfocándose en regiones prometedoras del espacio de búsqueda.

Parámetros optimizados: `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight`.

#### Features de Ingeniería (38 variables)

Las features se agrupan en 8 categorías:

**1. Diferenciales Básicos (5 features)**

| Feature | Fórmula | Descripción |
|---------|---------|-------------|
| `elo_diff` | $R_A - R_B$ | Diferencia de rating ELO |
| `fifa_rank_diff` | $\frac{R_{\text{fifa},B} - R_{\text{fifa},A}}{50}$ | Diferencia normalizada FIFA (negativo = mejor) |
| `squad_value_ratio` | $\ln\left(\frac{V_A}{\max(1, V_B)}\right)$ | Log-ratio de valor de plantilla |
| `host_advantage` | $\mathbb{1}[A \text{ es anfitrión}]$ | ¿Juega de local? |
| `host_vs_away` | $\mathbb{1}[A \text{ anfitrión} \land \neg B \text{ anfitrión}]$ | Local vs visitante puro |

**2. Momentum de Torneo (3 features)**

El momentum mide el rendimiento reciente ponderado exponencialmente:

$$M_A = \frac{\sum_{i=0}^{n-1} r_i \cdot w_i \cdot \lambda^i}{\sum_{i=0}^{n-1} w_i \cdot \lambda^i}$$

Donde:
- $r_i \in \{1.0, 0.5, 0.35, 0.2, 0.0\}$ — resultado del partido $i$
  - Victoria: 1.0
  - Empate: 0.5
  - Derrota por 1 gol vs rival $\geq 50$ ELO superior: 0.35
  - Derrota por 1 gol vs rival similar: 0.20
  - Derrota clara: 0.0
- $w_i \in \{1.0, 2.0, 3.0, 4.0, 5.0, 6.0\}$ — importancia del partido (grupo=1.0, R32=2.0, ..., Final=6.0)
- $\lambda = 0.85$ — factor de decaimiento exponencial (partidos más recientes pesan más)

Features: `momentum`, `opponent_momentum`, `momentum_diff`

**3. Sobre-rendimiento (3 features)**

Captura el "efecto Cabo Verde" — equipos que superan sistemáticamente las expectativas:

$$O_A = \frac{1}{n}\sum_{i=1}^{n} (g_i - \mathbb{E}[g_i]) \cdot \left(1 - 0.3 \cdot \frac{R_{\text{opp},i} - 1500}{400}\right)$$

Donde $g_i$ son los goles reales y $\mathbb{E}[g_i]$ los goles esperados según ELO. El factor de ajuste reduce el sobre-rendimiento contra rivales fuertes (es más difícil "sobre-rendir" contra Brasil que contra Haití).

Features: `overperformance`, `opponent_overperformance`, `overperformance_diff`

**4. Factores Físicos (3 features)**

| Feature | Descripción |
|---------|-------------|
| `rest_days_diff` | Diferencia de días de descanso |
| `extra_time_diff` | Minutos extra jugados (negativo = más fresco) |
| `cumulative_mins_ratio` | Ratio de minutos acumulados en el torneo |

**5. Métricas de Rendimiento (5 features)**

| Feature | Fórmula |
|---------|---------|
| `goals_scored_per_match` | $\frac{1}{n}\sum g_{\text{a favor}}$ |
| `goals_conceded_per_match` | $\frac{1}{n}\sum g_{\text{en contra}}$ |
| `goal_diff_per_match` | $\overline{g_{\text{favor}}} - \overline{g_{\text{contra}}}$ |
| `clean_sheet_rate` | $\frac{\text{partidos sin recibir gol}}{n}$ |
| `comeback_rate` | $\frac{\text{remontadas}}{n}$ |

**6. Consistencia Goleadora (1 feature)**

$$C_A = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(g_i - \bar{g})^2}$$

Desviación estándar de goles por partido. Valores bajos indican consistencia.

**7. Fase y Grupo (7 features)**

`stage_importance`, `stage_is_knockout`, `stage_is_r16`, `stage_is_qf`, `stage_is_sf`, `group_pts`, `group_pts_per_match`, `group_position`

**8. Confederación y Meta-features (11 features)**

`same_confederation`, `team_is_uefa`, `team_is_conmebol`, `opp_is_uefa`, `opp_is_conmebol`, `elo_diff_abs`, `elo_ratio`, `is_top5`, `opp_is_top5`, `coming_off_close_loss`

#### Explicabilidad con SHAP

Cada predicción incluye valores **SHAP** (*SHapley Additive exPlanations*), basados en la teoría de juegos cooperativos de Shapley:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[f(S \cup \{j\}) - f(S)\right]$$

Donde:
- $F$ es el conjunto de features
- $S$ es un subconjunto de features
- $f(S)$ es la predicción usando solo las features en $S$
- $\phi_j$ es la contribución marginal de la feature $j$

Esto permite explicar **por qué** el modelo predice lo que predice (ej: "+15.2% por diferencia de ELO, −3.4% por ranking FIFA inferior").

#### Limitaciones

- Entrenado con datos de UN solo torneo (WC 2026). No generaliza a otros contextos.
- No modela lesiones, cambios tácticos, ni alineaciones.
- Asume que los patrones de la fase de grupos se mantienen en eliminatorias.

---

### Modelo 2: ELO + Poisson — Predicción de Marcador y Goles

**Modelo complementario** para desglose Gana/Empate/Pierde y probabilidad de marcador exacto.

#### 2.1 Sistema de Rating ELO

Adaptado del ajedrez (Arpad Elo, 1960). Cada selección tiene un rating $R$ que representa su fuerza relativa.

##### Probabilidad Esperada

Para un partido entre A (rating $R_A$) y B (rating $R_B$):

$$E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$

Propiedades:
- $E_A \in [0, 1]$
- **Regla de los 400 puntos:** Un equipo con 400 puntos más gana ~91% de las veces
- $E_B = 1 - E_A$

##### Ventaja de Localía

Las selecciones anfitrionas reciben un bono fijo de +100 ELO:

$$E_A^{\text{local}} = \frac{1}{1 + 10^{(R_B - (R_A + 100)) / 400}}$$

##### Actualización de Rating (Factor K)

Tras cada partido, los ratings se actualizan con el factor K de Copa del Mundo ($K = 30$):

$$R'_A = R_A + K \cdot (S_A - E_A)$$

Donde $S_A$ es el resultado real:
- Victoria: $S_A = 1.0$
- Empate: $S_A = 0.5$
- Derrota: $S_A = 0.0$

El ELO total del sistema se conserva — lo que un equipo gana, el otro lo pierde.

#### 2.2 Modelo de Goles de Poisson

Los goles en fútbol son eventos raros y discretos — modelados naturalmente por la distribución de Poisson.

##### Goles Esperados (xG)

$$\lambda_A = \max\left(0.25,\; \lambda_{\text{base}} + \frac{R_A - R_B}{400}\right)$$

Donde:
- $\lambda_{\text{base}} = 1.35$ — promedio empírico de goles por equipo en fase de grupos de Copas del Mundo
- Cada 100 puntos ELO desplazan los goles esperados en ±0.25
- Piso de 0.25 para evitar predicciones degeneradas (cero goles)

Con ventaja de localía: $R_A \leftarrow R_A + 100$ para el anfitrión.

##### Probabilidad de un Marcador Exacto

$$P(g_A, g_B) = \underbrace{\frac{\lambda_A^{g_A} \cdot e^{-\lambda_A}}{g_A!}}_{\text{Poisson}(g_A \mid \lambda_A)} \times \underbrace{\frac{\lambda_B^{g_B} \cdot e^{-\lambda_B}}{g_B!}}_{\text{Poisson}(g_B \mid \lambda_B)}$$

Los goles de ambos equipos se asumen **procesos de Poisson independientes** — supuesto estándar en analítica de fútbol (modelos Dixon-Coles, Maher).

##### Probabilidades Agregadas

Sumando sobre todos los marcadores posibles ($g_i \in [0, 8]$):

$$P(\text{A gana}) = \sum_{g_A=0}^{8} \sum_{g_B=0}^{g_A-1} P(g_A, g_B)$$

$$P(\text{Empate}) = \sum_{g=0}^{8} P(g, g)$$

$$P(\text{B gana}) = \sum_{g_B=0}^{8} \sum_{g_A=0}^{g_B-1} P(g_A, g_B)$$

Marcadores con más de 8 goles se truncan (probabilidad combinada < 0.001%).

##### Marcador Más Probable

$$\text{marcador}_{\text{pred}} = \arg\max_{g_A, g_B}\; P(g_A, g_B)$$

#### Ejemplo: Argentina vs Inglaterra

| | ELO | xG |
|---|-----|----|
| 🇦🇷 Argentina | 2127 | 1.78 |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra | 2039 | 1.42 |

- **Argentina gana:** 52.3%
- **Empate / Tiempo Extra:** 24.1%
- **Inglaterra gana:** 23.6%
- **Marcador más probable:** 2–1

#### Limitaciones

- Independencia de Poisson: no modela correlación intra-partido (ej: expulsión que afecta a ambos)
- No considera alineaciones, lesiones ni factores tácticos
- Solo usa ELO como predictor de fuerza
- Sin tiempo extra en fase de grupos (se asume resultado en 90 minutos)

---

### Modelo 3: Rating de Rendimiento Compuesto

Cada selección recibe un rating **1–100** que combina tres factores ponderados:

$$\text{rating} = 1 + 99 \times \big(0.65 \cdot \text{ELO}_{\text{norm}} + 0.25 \cdot \text{Forma} + 0.10 \cdot \text{GD}_{\text{factor}}\big)$$

| Componente | Peso | Descripción |
|-----------|------|-------------|
| **ELO normalizado** | 65% | ELO mapeado a $[0, 1]$ (rango competitivo: 1380–2127) |
| **Forma** | 25% | Media ponderada de últimos 10 partidos (decaimiento exponencial $\alpha = 0.93$) |
| **Diferencia de goles** | 10% | Promedio de diferencia de goles, acotada a $[-3, +3]$ |

La forma se calcula con pesos geométricamente decrecientes:

$$F = \frac{\sum_{i=0}^{9} r_i \cdot 0.93^i}{\sum_{i=0}^{9} 0.93^i}$$

Donde $r_i$ es el resultado del $i$-ésimo partido más reciente. Un partido hace 1 jornada pesa ~2× más que uno hace 10 jornadas.

---

## 🎨 Dashboard

El frontend es una SPA (*Single Page Application*) con 4 vistas:

| Vista | Contenido |
|-------|-----------|
| **Teams** | 48 cards con bandera, FIFA rank, valor plantilla y rating |
| **Matches** | Todos los partidos: XGBoost (avance) + Poisson (G/ET/P) |
| **H2H** | Comparador cara a cara: pick 2 equipos, XGBoost + SHAP |
| **Detail** | Ficha completa de cada selección con mapa, forma y predictor |

---

## 📊 Squad Values (Transfermarkt)

Las 48 selecciones incluyen el valor de mercado de sus plantillas según Transfermarkt:

| # | Selección | Valor |
|---|-----------|-------|
| 1 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra | €1,470M |
| 2 | 🇫🇷 Francia | €1,230M |
| 3 | 🇧🇷 Brasil | €1,010M |
| 4 | 🇪🇸 España | €965M |
| 5 | 🇦🇷 Argentina | €950M |
| 6 | 🇩🇪 Alemania | €840M |
| 7 | 🇳🇱 Países Bajos | €785M |
| 8 | 🇵🇹 Portugal | €760M |
| 9 | 🇧🇪 Bélgica | €595M |
| 10 | 🇳🇴 Noruega | €480M |

---

## 📡 API Reference

| Endpoint | Descripción |
|----------|-------------|
| `GET /api/teams` | 48 equipos con ratings y squad values |
| `GET /api/teams/{id}` | Detalle de una selección |
| `GET /api/matches` | Partidos con predicciones XGBoost + Poisson |
| `GET /api/matches/{id}` | Detalle de un partido |
| `GET /api/groups` | 16 grupos con equipos y partidos |
| `GET /api/predict/v2?team1_id=X&team2_id=Y` | **XGBoost:** P(avanzar) + SHAP + Poisson baseline |
| `GET /api/model-info` | Métricas y metadata del modelo |
| `GET /api/retrain` | Invalidar cachés y recargar datos |

---

## 🛠 Tech Stack

| Capa | Tecnología |
|------|-----------|
| Lenguaje | Python 3.13 |
| ML | XGBoost (Optuna-tuned) |
| API | FastAPI + Uvicorn |
| Matemáticas | NumPy, SciPy (Poisson PMF) |
| Explicabilidad | SHAP |
| Datos | JSON estático pre-computado |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Mapas | Leaflet.js |
| Banderas | FlagCDN |

---

## 📝 Licencia

MIT

---

> **⚠️ RECORDATORIO:** Este proyecto es una herramienta educativa y experimental. No debe utilizarse como base para decisiones de apuestas. Los modelos estadísticos tienen limitaciones inherentes y el fútbol real frecuentemente desafía las predicciones. Utilice bajo su propio criterio y responsabilidad.

---

*Datos: ESPN API, Transfermarkt, FIFA. Modelos: XGBoost + ELO + Poisson. World Cup 2026 Predictor v9.*
