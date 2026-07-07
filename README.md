# ⚽ World Cup 2026 Predictor — XGBoost Edition

> **⚠️ AVISO LEGAL — LEER ANTES DE USAR**
>
> Este sistema es una herramienta **académica y experimental**. Las predicciones son estimaciones estadísticas basadas en modelos matemáticos y datos históricos reales. **NO constituyen consejo de apuestas, recomendaciones financieras ni garantía de resultados.** El fútbol es inherentemente impredecible y ningún modelo puede considerar todos los factores (lesiones, decisiones arbitrales, clima, moral del equipo, etc.).
>
> **No nos hacemos responsables del mal uso de estas predicciones.** Si decides usar esta información para apuestas, lo haces bajo tu propio riesgo. Las apuestas pueden causar adicción y pérdidas financieras significativas. Juega con responsabilidad.

---

## 🎥 Demo

![World Cup 2026 Predictor Demo](docs/demo.gif)

## 📡 Fuentes de datos (100% reales)

Cada número en este proyecto tiene una fuente verificable. **No hay datos sintéticos ni inventados.**

| Fuente | Qué obtenemos | Verificación |
|--------|--------------|-------------|
| [ESPN API](https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world) | 48 equipos, 100 partidos, fechas, scores, etapas | Tiempo real |
| [FIFA/Coca-Cola World Ranking](https://www.fifa.com/fifa-world-ranking) | Ranking oficial de las 48 selecciones | Junio 2026 |
| [Transfermarkt](https://www.transfermarkt.com/vereins-statistik/wertvollstenationalmannschaften/marktwertetop) | Valor de mercado del plantel (26 jugadores por selección) | Julio 4, 2026 ✓ |
| [FlagCDN](https://flagcdn.com) | Banderas nacionales (PNG 160px) | CDN |
| [CartoDB](https://carto.com) | Mapas Leaflet (OpenStreetMap) | CDN |

**Valores de plantel verificados manualmente:**

| # | Equipo | Valor |
|---|--------|-------|
| 1 | 🇫🇷 Francia | €1,520M |
| 2 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra | €1,360M |
| 3 | 🇪🇸 España | €1,220M |
| 4 | 🇵🇹 Portugal | €1,010M |
| 5 | 🇩🇪 Alemania | €947M |
| 6 | 🇧🇷 Brasil | €928M |
| 7 | 🇦🇷 Argentina | €808M |
| 8 | 🇳🇱 Países Bajos | €754M |
| 9 | 🇳🇴 Noruega | €590M |
| 10 | 🇧🇪 Bélgica | €548M |

---

## 🏗 Arquitectura

```
wc2026-predictor/
├── backend/
│   ├── main.py                  # FastAPI server (11 endpoints)
│   ├── elo.py                   # Sistema ELO adaptado a fútbol
│   ├── predictor.py             # Modelo Poisson + simulador de torneo
│   ├── performance.py           # Rating compuesto de rendimiento (1–100)
│   ├── feature_engineering.py   # Pipeline de 38 features
│   ├── requirements.txt         # Dependencias
│   ├── frontend/                # SPA servida estáticamente
│   │   └── index.html           # Dashboard (1055 líneas, vanilla JS)
│   ├── models/
│   │   ├── xgboost_predictor.py # Clasificador XGBoost + Optuna + SHAP
│   │   ├── xgboost_v1.json      # Modelo entrenado (91.7% accuracy)
│   │   └── feature_names.json   # Índice de features
│   └── data/
│       ├── teams.json           # 48 equipos (ELO, FIFA, valor plantel)
│       ├── teams_enriched.json  # Precomputado con ratings (respuesta <5ms)
│       ├── matches.json         # 100 partidos (ESPN API)
│       └── matches_enriched.json# Precomputado con predicciones
├── frontend/
│   └── index.html               # Copia para desarrollo local
├── docs/
│   └── demo.mp4                 # Video demo
├── requirements.txt
└── README.md
```

- **Backend:** Python 3.13, FastAPI, XGBoost, NumPy, SciPy, SHAP, Pydantic
- **Frontend:** HTML5 + CSS3 + Vanilla JS, sin build step
- **Datos:** JSON precomputados con `@lru_cache` (respuesta <5ms)
- **88 partidos completados**, 12 programados (R16 → Final)

---

## 🧠 Modelos Predictivos

El sistema usa **tres modelos complementarios** con roles claramente definidos:

| Modelo | ¿Qué predice? | ¿Cuándo se usa? |
|--------|---------------|-----------------|
| **XGBoost** | Probabilidad de avanzar en eliminatoria | Fases knockout (R32 → Final) |
| **ELO + Poisson** | Win / Draw / Loss + marcador más probable | Todos los partidos |
| **Performance Rating** | Rating compuesto 1–100 | Vista de equipos, comparaciones |

---

## Modelo 1: XGBoost — Probabilidad de Avance

### Qué predice

$$P(\text{equipo avanza}) \in [0, 1]$$

Para cada cruce eliminatorio, el modelo estima qué probabilidad tiene cada equipo de pasar a la siguiente ronda (en tiempo regular, tiempo extra o penales).

### Cómo funciona

XGBoost (eXtreme Gradient Boosting) es un ensemble de árboles de decisión que construye modelos secuencialmente, donde cada nuevo árbol corrige los errores del anterior. En lugar de "comparar" dos equipos directamente, el modelo recibe **38 características numéricas** que describen la diferencia entre ambos equipos y devuelve una probabilidad.

```
features(elo_diff, momentum, squad_value, ...) → XGBoost → P(avance)
```

El modelo **no sabe quiénes son los equipos**. Ve diferencias numéricas:

| Feature de ejemplo | Argentina vs Inglaterra |
|-------------------|------------------------|
| `elo_diff` | +88.5 (Argentina mejor) |
| `squad_value_ratio` | −0.52 (Inglaterra plantel más caro) |
| `momentum_diff` | +0.13 (Argentina mejor racha) |
| `rest_days_diff` | 0 (mismo descanso) |

Con estos 38 números, XGBoost predice la probabilidad de que el equipo A avance. Para eliminar el sesgo de orden (que el resultado cambie si intercambias los equipos), usamos **simetrización dual**:

$$P(\text{A avanza}) = \frac{p_{\text{A vs B}} + (1 - p_{\text{B vs A}})}{2}$$

Esto garantiza $P(\text{A}) + P(\text{B}) = 1.0$ siempre.

### Arquitectura

| Propiedad | Valor |
|-----------|-------|
| Algoritmo | XGBoost (Gradient Boosted Trees) |
| Tipo | Clasificador binario |
| Features | 38 variables engineered |
| Muestras | 176 (88 partidos × 2 perspectivas) |
| Optimización | Optuna (200 rondas, TPE sampler) |
| Validación | Time-based split (80% train, 20% validación) |

### Métricas

| Métrica | Valor |
|---------|-------|
| Accuracy | 91.7% |
| ROC AUC | 0.987 |
| Brier Score | 0.066 |

### ¿Por qué XGBoost?

1. **No lineal:** Captura interacciones complejas entre features que modelos lineales no pueden (ej: "ELO alto SOLO importa si el equipo también tiene momentum positivo")
2. **Regularización incorporada:** L1 + L2 en los árboles evita overfitting automático
3. **Maneja features correlacionados:** Si `elo_diff` y `elo_ratio` miden cosas similares, XGBoost usa el más informativo sin confundirse
4. **SHAP:** Cada predicción viene con una explicación de qué features contribuyeron más

### Explicabilidad con SHAP

Cada predicción incluye valores SHAP (SHapley Additive exPlanations), basados en teoría de juegos cooperativos:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F| - |S| - 1)!}{|F|!} \Bigl[f(S \cup \{j\}) - f(S)\Bigr]$$

Donde $\phi_j$ es la contribución marginal del feature $j$ a la predicción. En el frontend, los features se muestran como barras: **verde** si ayudan al equipo, **rojo** si lo perjudican.

---

## Modelo 2: ELO + Poisson — Goles y Marcador

### El Sistema ELO — Adaptado al Fútbol

#### Metodología

El sistema ELO fue creado por Arpad Elo en 1960 para el ajedrez y está documentado en su libro *The Rating of Chessplayers, Past and Present* (1978). La idea es simple: cada jugador (o equipo) tiene un rating numérico $R$ que representa su fuerza relativa. Después de cada partido, los ratings se actualizan según el resultado esperado vs el real.

La **mecánica de actualización es 100% estándar** (Elo, 1978). La **inicialización de ratings** es feature engineering basado en el ranking FIFA.

#### Inicialización de ELO — Feature Engineering

No existe un sistema ELO preexistente para las 48 selecciones del Mundial. Los ELO iniciales se derivaron del **ranking FIFA pre-torneo (junio 2026)** mediante un mapeo lineal aproximado:

$$\text{ELO}_{\text{inicial}} \approx 2100 - \frac{700}{89} \times (\text{Ranking FIFA} - 1)$$

Es decir, el equipo #1 del ranking FIFA arranca en ~2100, el #90 en ~1400, y los intermedios se distribuyen proporcionalmente. Con ejemplos concretos:

| Equipo | Ranking FIFA | ELO inicial estimado |
|--------|:-----------:|:--------------------:|
| 🇦🇷 Argentina | #1 | ~2100 |
| 🇲🇽 México | #15 | ~1990 |
| 🇺🇾 Uruguay | #16 | ~1982 |
| 🇨🇻 Cabo Verde | #54 | ~1683 |
| 🇳🇿 Nueva Zelanda | #88 | ~1416 |

Esto es feature engineering: el ranking FIFA es la fuente de datos real; el mapeo lineal a una escala ELO (~1400–2100) es la transformación que alimenta al modelo. No es una metodología publicada — es una decisión de diseño para ubicar a los equipos en un rango numérico comparable.

Luego, **los 88 partidos reales del torneo** actualizaron estos valores con la fórmula estándar de Elo (K=30). Los equipos que ganaron subieron; los que perdieron bajaron. El rango final es **1382–2127** (Argentina subió de ~2100 a 2127; Nueva Zelanda bajó de ~1416 a 1382).

#### Fórmula estándar de actualización (Elo, 1978)

**Resultado esperado:**
$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

**Actualización post-partido:**
$$R'_A = R_A + K \cdot (S_A - E_A)$$

Donde:
- $S_A = 1.0$ (victoria), $0.5$ (empate), $0.0$ (derrota)
- $K = 30$ (factor de sensibilidad para torneo corto)
- Un equipo con 100 puntos ELO más que su rival tiene ~64% de probabilidad de ganar

#### Ventaja de localía (+100 ELO)

Las tres naciones anfitrionas (México, Canadá, EE.UU.) reciben +100 puntos ELO cuando juegan en casa:

$$E_A^{\text{local}} = \frac{1}{1 + 10^{(R_B - (R_A + 100))/400}}$$

Este ajuste es una práctica común en modelos ELO de fútbol (Hvattum & Arntzen, 2010; *Forecasting Association Football Outcomes*).

#### Uso en XGBoost

El modelo XGBoost usa **3 de sus 38 features** derivados del ELO:

| Feature | Cálculo | Rol en el modelo |
|---------|---------|-----------------|
| `elo_diff` | $R_A - R_B$ | Feature #0 — diferencial directo |
| `elo_diff_abs` | $|R_A - R_B|$ | Feature #33 — magnitud de la diferencia |
| `elo_ratio` | $\frac{R_A}{R_B} - 1$ | Feature #34 — ventaja relativa |

### Modelo Poisson de Goles

Los goles en fútbol son eventos raros y discretos. La distribución de Poisson es el modelo estándar para este tipo de eventos (Maher, 1982; Dixon & Coles, 1997).

#### Goles esperados (xG)

$$\lambda_A = \max\left(0.25,\; 1.35 + \frac{R_A - R_B}{400}\right)$$

- $\lambda_{\text{base}} = 1.35$ — promedio empírico de goles por equipo en fase de grupos
- Cada 100 puntos ELO de diferencia = ±0.25 goles esperados
- Piso de 0.25 para evitar predicciones de cero goles

Con localía: $R_A \leftarrow R_A + 100$

#### Probabilidad de marcador exacto

$$P(g_A, g_B) = \frac{\lambda_A^{g_A} e^{-\lambda_A}}{g_A!} \times \frac{\lambda_B^{g_B} e^{-\lambda_B}}{g_B!}$$

Los goles de ambos equipos se asumen **procesos de Poisson independientes** — el supuesto estándar en analítica de fútbol.

#### Probabilidades agregadas

$$P(\text{A gana}) = \sum_{g_A=0}^{8} \sum_{g_B=0}^{g_A-1} P(g_A, g_B)$$
$$P(\text{Empate}) = \sum_{g=0}^{8} P(g, g)$$
$$P(\text{B gana}) = \sum_{g_B=0}^{8} \sum_{g_A=0}^{g_B-1} P(g_A, g_B)$$

Se truncan marcadores arriba de 8 goles (probabilidad combinada $< 0.001\%$).

#### Marcador más probable

$$s_{\text{pred}} = \underset{g_A, g_B}{\arg\max}\; P(g_A, g_B)$$

### Ejemplo: Argentina vs Inglaterra

| | ELO | xG |
|---|-----|----|
| 🇦🇷 Argentina | 2127 | 1.78 |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra | 2039 | 1.42 |

- Argentina gana: 52.3%
- Empate / Tiempo Extra: 24.1%
- Inglaterra gana: 23.6%
- Marcador más probable: 2–1

---

## Modelo 3: Rating Compuesto de Rendimiento

Rating de **1–100** que combina tres factores:

$$\text{rating} = 1 + 99 \times (0.65 \cdot \text{ELO}_{\text{norm}} + 0.25 \cdot \text{Forma} + 0.10 \cdot \text{GD}_{\text{factor}})$$

| Componente | Peso | Cálculo |
|-----------|:----:|---------|
| ELO normalizado | 65% | $\frac{\text{ELO} - 1380}{2130 - 1380}$ |
| Forma reciente | 25% | Promedio ponderado de últimos 10 partidos ($\alpha = 0.93$) |
| Diferencia de gol | 10% | GD promedio, acotado a $[-3, +3]$, normalizado a $[0,1]$ |

La forma usa decaimiento exponencial: un partido de hace 1 fecha pesa ~2× más que uno de hace 10 fechas.

---

## 📊 Los 38 Features en Detalle

### 1. Diferenciales Básicos (5)

| Feature | Fórmula | Descripción |
|---------|---------|-------------|
| `elo_diff` | $R_A - R_B$ | Diferencia de ELO |
| `fifa_rank_diff` | $\frac{R_{\text{fifa},B} - R_{\text{fifa},A}}{50}$ | Diferencia normalizada de ranking FIFA |
| `squad_value_ratio` | $\ln\left(\frac{V_A}{\max(1, V_B)}\right)$ | Log-ratio de valor de plantel |
| `host_advantage` | $\mathbf{1}[A \text{ es anfitrión}]$ | Indicador de localía |
| `host_vs_away` | $\mathbf{1}[A \text{ local} \land \neg B \text{ local}]$ | Local puro vs visitante |

### 2. Momentum del Torneo (3)

$$M_A = \frac{\sum_{i=0}^{n-1} r_i \cdot w_i \cdot 0.85^{i}}{\sum_{i=0}^{n-1} w_i \cdot 0.85^{i}}$$

Donde:
- $r_i$: 1.0 (victoria), 0.5 (empate), 0.35 (derrota por 1 gol vs rival fuerte), 0.20 (derrota ajustada), 0.0 (derrota clara)
- $w_i$: importancia del partido (grupo=1.0, R32=2.0, ..., Final=6.0)

### 3. Sobre-rendimiento (3)

Captura el "efecto Cabo Verde" — equipos que sistemáticamente superan expectativas:

$$O_A = \frac{1}{n}\sum (g_i - \mathbb{E}[g_i]) \cdot \left(1 - 0.3 \cdot \frac{R_{\text{opp}} - 1500}{400}\right)$$

### 4. Factores Físicos (3)

`rest_days_diff`, `extra_time_diff`, `cumulative_mins_ratio`

### 5. Métricas de Rendimiento (5)

`goals_scored_per_match`, `goals_conceded_per_match`, `goal_diff_per_match`, `clean_sheet_rate`, `comeback_rate`

### 6. Consistencia de Gol (1)

$$C_A = \sqrt{\frac{1}{n-1}\sum(g_i - \bar{g})^2}$$

### 7. Etapa y Grupo (7)

`stage_importance`, `stage_is_knockout`, `stage_is_r16`, `stage_is_qf`, `stage_is_sf`, `group_pts`, `group_pts_per_match`, `group_position`

### 8. Confederación y Meta-features (11)

`same_confederation`, `team_is_uefa`, `team_is_conmebol`, `opp_is_uefa`, `opp_is_conmebol`, `elo_diff_abs`, `elo_ratio`, `is_top5`, `opp_is_top5`, `coming_off_close_loss`

---

## 🏆 Bracket del Torneo

El endpoint `/api/bracket/v2` simula todo el cuadro eliminatorio con XGBoost:

1. **R16:** 8 partidos con predicciones precomputadas
2. **QF:** Ganadores de R16 emparejados por orden de bracket → XGBoost predice cada cruce
3. **SF:** Ganadores de QF → XGBoost
4. **Final:** Ganadores de SF → determina el campeón predicho

El resultado es un árbol JSON con ganadores, probabilidades e información de cada equipo en cada ronda. El frontend lo renderiza como un bracket interactivo.

---

## 🎨 Dashboard (Frontend)

SPA de una sola página. Diseño premium minimalista con glass-morphism.

### 4 Vistas

| Vista | Hash | Contenido |
|-------|------|-----------|
| **🏠 Teams** | `#teams` | 48 tarjetas con bandera, ranking FIFA, valor plantel, forma reciente (W/D/L) y rating. Filtros por grupo. |
| **📅 Matches** | `#matches` | 100 partidos con scores reales, barras de probabilidad (XGBoost + Poisson) y marcador predicho. Filtros por etapa. |
| **⚔️ H2H** | `#h2h` | Comparador: elegí 2 equipos → XGBoost + SHAP + Poisson. |
| **🏆 Bracket** | `#bracket` | Simulación completa R16 → Final con XGBoost. |

### UX

- Nav con vidrio esmerilado (backdrop-filter blur)
- Skeleton loaders animados
- Responsive (3 → 2 → 1 columna)
- Sin recargas (routing por hash)
- Verde = victoria, Gris = empate, Rojo = derrota

---

## 📡 API Reference

| Endpoint | Descripción |
|----------|-------------|
| `GET /api/teams` | 48 equipos con ratings y valor de plantel |
| `GET /api/teams/{id}` | Detalle de un equipo |
| `GET /api/matches` | 100 partidos con predicciones precomputadas |
| `GET /api/matches/{id}` | Detalle de un partido |
| `GET /api/groups` | 16 grupos (A–P) con equipos y partidos |
| `GET /api/bracket/v2` | Simulación del bracket con **XGBoost** |
| `GET /api/predict/v2` | **XGBoost:** P(avance) + SHAP + baseline Poisson |
| `GET /api/predict` | Predicción Poisson (legacy) |
| `GET /api/model-info` | Métricas del modelo |
| `GET /api/retrain` | Invalidar cachés |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
cd backend && python3 main.py    # → http://localhost:8000
```

### Deployment (Seenode)

| Campo | Valor |
|-------|-------|
| Provider | Seenode Basic ($3/mo) |
| Root directory | *(vacío)* |
| Build command | `pip install -r requirements.txt` |
| Start command | `cd backend && python main.py` |
| Port | 8000 |

---

## 🛠 Tech Stack

| Capa | Tecnología |
|------|-----------|
| Lenguaje | Python 3.13 |
| ML | XGBoost (Optuna-tuned, 38 features) |
| API | FastAPI + Uvicorn |
| Matemáticas | NumPy, SciPy |
| Explainability | SHAP (TreeExplainer) |
| Datos | JSON estático + `@lru_cache` |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Mapas | Leaflet.js + CartoDB |
| Banderas | FlagCDN |
| Deployment | Seenode |

---

## 📚 Referencias

| # | Referencia | Link |
|---|-----------|------|
| 1 | Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. | [DOI](https://doi.org/10.1145/2939672.2939785) |
| 2 | Lundberg, S. M. & Lee, S.-I. (2017). *SHAP: A Unified Approach to Interpreting Model Predictions*. | [Paper](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) |
| 3 | Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. | [DOI](https://doi.org/10.1145/3292500.3330701) |
| 4 | Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. | ISBN 0-668-04721-6 |
| 5 | Maher, M. J. (1982). *Modelling association football scores*. | [DOI](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x) |
| 6 | Dixon, M. J. & Coles, S. G. (1997). *Modelling association football scores and inefficiencies*. | [DOI](https://doi.org/10.1111/1467-9876.00065) |
| 7 | Bergstra, J. et al. (2011). *Algorithms for Hyper-Parameter Optimization*. | [Paper](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html) |

---

## 📝 Licencia

MIT

---

*Datos: ESPN API, Transfermarkt (julio 2026), FIFA. Modelos: XGBoost + ELO + Poisson. 88 partidos completados, 12 por jugarse.*
