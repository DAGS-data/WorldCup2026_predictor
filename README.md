# ⚽ World Cup 2026 Predictor — Logistic Regression Edition

> **⚠️ AVISO LEGAL — LEER ANTES DE USAR**
>
> Este sistema es una herramienta **académica y experimental**. Las predicciones mostradas son estimaciones estadísticas basadas en modelos matemáticos y datos históricos. **NO constituyen consejo de apuestas, recomendaciones financieras ni garantía de resultados.** El fútbol es inherentemente impredecible y ningún modelo puede considerar todos los factores que influyen en un partido (lesiones, decisiones arbitrales, clima, moral del equipo, etc.).
>
> **No nos hacemos responsables del mal uso de estas predicciones.** Si decides usar esta información para apuestas deportivas, lo haces bajo tu propio riesgo. Las apuestas pueden causar adicción y pérdidas financieras significativas. Juega con responsabilidad.

---

## 🎥 Demo

https://github.com/user-attachments/assets/COLOCAR_URL_DEL_VIDEO_AQUI

*(El video muestra la app completa: Teams, Matches, H2H, Bracket y la simulación del torneo.)*

---

## 📋 Índice

1. [¿De dónde salen los datos? (100% reales)](#-de-dónde-salen-los-datos-100-reales)
2. [Arquitectura del proyecto](#-arquitectura)
3. [La historia de los modelos](#-la-historia-de-los-modelos)
4. [Modelo 1: Regresión Logística (Principal)](#modelo-1-regresión-logística-l2---predicción-de-avance)
5. [Modelo 2: ELO + Poisson (Goles)](#modelo-2-elo--poisson---predicción-de-goles)
6. [Modelo 3: Rating de Rendimiento](#modelo-3-rating-compuesto-de-rendimiento)
7. [Pruebas Bayesianas](#-pruebas-bayesianas)
8. [Dashboard (Frontend)](#-dashboard-frontend)
9. [API Reference](#-api-reference)
10. [Deployment (Seenode)](#-deployment-seenode)
11. [Referencias Académicas](#-referencias-académicas)

---

## 📡 ¿De dónde salen los datos? (100% reales)

**Todos los datos de este proyecto son reales.** No hay datos sintéticos, simulados ni inventados. Cada número tiene una fuente verificable:

### Equipos y partidos: ESPN API

| Fuente | Endpoint | Qué obtuvimos |
|--------|----------|---------------|
| [ESPN API](https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world) | `/teams`, `/events` | 48 equipos clasificados, fechas reales, scores reales, etapas |

Los 88 partidos completados (fase de grupos + R32) tienen scores **reales** extraídos de ESPN. Los 12 partidos restantes (R16 en adelante) están programados con fechas oficiales de FIFA.

### Ranking FIFA

| Fuente | Dato | Fecha |
|--------|------|-------|
| [FIFA/Coca-Cola World Ranking](https://www.fifa.com/fifa-world-ranking) | Ranking oficial de las 48 selecciones | Junio 2026 |

Cada equipo tiene su posición FIFA real al momento del torneo.

### Valor de plantilla: Transfermarkt

| Fuente | Dato | Verificación |
|--------|------|-------------|
| [Transfermarkt](https://www.transfermarkt.com/vereins-statistik/wertvollstenationalmannschaften/marktwertetop) | Valor de mercado del plantel (€) — 26 jugadores por selección | Julio 4, 2026 ✓ |

Cada equipo tiene su valor de plantel **verificado manualmente** contra Transfermarkt. Ejemplos reales:

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
| ... | ... | ... |
| 48 | 🇨🇼 Curazao | €18M |

### Banderas y mapas

| Fuente | Uso |
|--------|-----|
| [FlagCDN](https://flagcdn.com) | Banderas nacionales (PNG 160px) |
| [CartoDB](https://carto.com) | Mapas Leaflet (OpenStreetMap) |
| [Google Fonts](https://fonts.google.com) | Tipografía Inter |

### Sistema ELO

Las puntuaciones ELO fueron calculadas usando el sistema estándar de ajedrez adaptado a fútbol (Elo, 1978), con:
- **K = 30** (factor de actualización para Mundial)
- **Ventaja local = +100 puntos ELO** para las naciones anfitrionas (México, Canadá, EE.UU.)

---

## 🏗 Arquitectura

```
wc2026-predictor/
├── backend/
│   ├── main.py                  # FastAPI server (11 endpoints)
│   ├── elo.py                   # Sistema ELO (K=30, localía +100)
│   ├── predictor.py             # Modelo Poisson + simulador de torneo
│   ├── performance.py           # Rating compuesto (1–100)
│   ├── feature_engineering.py   # Pipeline de 38 features
│   ├── requirements.txt         # Dependencias Python
│   ├── frontend/                # SPA servida estáticamente
│   │   └── index.html           # Dashboard (1055 líneas, vanilla JS)
│   ├── models/
│   │   ├── logistic_predictor.py # Predictor de regresión logística (L2)
│   │   ├── logistic_v1.pkl       # Modelo entrenado (86% test accuracy)
│   │   ├── feature_names_logistic.json # Índice de features
│   │   ├── xgboost_predictor.py  # XGBoost legacy (v2)
│   │   ├── xgboost_v1.json       # XGBoost legacy
│   │   └── feature_names.json    # Índice compartido
│   └── data/
│       ├── teams.json           # 48 equipos (ELO, FIFA rank, valor plantel)
│       ├── teams_enriched.json  # Precomputado con ratings
│       ├── matches.json         # 100 partidos (ESPN API)
│       └── matches_enriched.json# Precomputado con predicciones
├── frontend/
│   └── index.html               # Copia local para desarrollo
├── docs/
│   └── demo.mp4                 # Video demo de la app
├── requirements.txt             # Dependencias (raíz del repo)
├── .gitignore
└── README.md
```

- **Backend:** Python 3.13, FastAPI, scikit-learn, NumPy, SciPy, Pydantic
- **Frontend:** HTML5 + CSS3 + Vanilla JS, sin build step
- **Datos:** JSON precomputados con `@lru_cache` (respuesta <5ms)
- **88 partidos completados**, 12 programados (R16 → Final)

---

## 🧪 La historia de los modelos

Este proyecto pasó por **4 iteraciones de modelos** antes de llegar a la versión final. Aquí la historia completa:

### v0 — Solo Poisson/ELO

El punto de partida. Modelo puramente matemático: ELO → expected goals → distribución Poisson → probabilidades Win/Draw/Loss. Sin machine learning, sin overfitting. Confiable pero limitado: no aprende de los datos del torneo.

### v1 — XGBoost (árboles de decisión)

Agregamos XGBoost con 38 features engineered y optimización Optuna. Reportaba 91.7% de accuracy. Pero...

**Problema detectado:** Con solo 176 muestras (88 partidos × 2 perspectivas) y 38 features, XGBoost estaba **sobreajustando**. Las predicciones eran extremas: 81% Argentina vs 19% Inglaterra. Los árboles de decisión pueden memorizar ruido con pocos datos.

### v2 — Prueba Bayesiana

Para verificar nuestras sospechas, construimos un modelo Bayesiano (Bradley-Terry jerárquico con PyMC). Resultados:

| Métrica | Bayesiano | XGBoost |
|---------|:---------:|:-------:|
| Argentina vs Inglaterra | 65.6% | ~62% |
| 95% intervalo de credibilidad | [33.7%, 89.2%] | *(no reporta)* |
| Incertidumbre | ✅ Cuantificada | ❌ Oculta |

**Conclusión:** El intervalo de credibilidad bayesiano es **enorme** (55 puntos porcentuales). Con 136 muestras para 48 parámetros, NADIE puede tener certeza. XGBoost escondía esta incertidumbre detrás de un solo número.

### v3 — Regresión Logística (VERSIÓN FINAL)

Probamos 4 modelos simples con validación cruzada de series temporales:

| Modelo | CV Accuracy | ¿Overfitting? |
|--------|:-----------:|:-------------:|
| Naive Bayes | 60.7% | ❌ Malo |
| Ridge Classifier | 75.2% | ✅ No |
| Decision Tree (depth=3) | 77.9% | 🟡 Algo |
| **Logistic Regression (L2)** | **77.2% ± 7.4%** | ✅ **No** |

**¿Por qué Regresión Logística?**

1. **No overfittea:** Regularización L2 (C=0.1) maneja 38 features en 136 muestras
2. **Calibración natural:** La salida logística ES una probabilidad por diseño
3. **Interpretable:** Cada coeficiente te dice exactamente qué importa
4. **Honesta:** Argentina vs Inglaterra = 56.7% (no 81%, no 62% — una ligera ventaja)
5. **Sin hiperparámetros:** No hay tuning dramático, no hay Optuna, no hay magia

---

## Modelo 1: Regresión Logística (L2) — Predicción de Avance

**Modelo principal** para fases eliminatorias (R32, R16, Cuartos, Semis, Final).

### Qué predice

La probabilidad de que un equipo avance de una eliminatoria:

$$P(\text{avance}) \in [0, 1]$$

### Ecuación del modelo

$$P(\text{equipo A avanza}) = \sigma\left(\beta_0 + \sum_{j=1}^{38} \beta_j \cdot x_j\right)$$

Donde:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ — función sigmoide
- $\beta_j$ — coeficiente aprendido para el feature $j$ (regularizado con L2)
- $x_j$ — valor del feature estandarizado
- $\beta_0$ — intercepto (log-odds base)
- $C = 0.1$ — fuerza de regularización (más bajo = más regularización)

### Simetrización dual

Los features son direccionales (`elo_diff`, `momentum_diff`). Para eliminar el sesgo de orden:

$$P(\text{A avanza}) = \frac{p_{\text{fwd}} + (1 - p_{\text{rev}})}{2}$$

Donde:
- $p_{\text{fwd}}$ = predicción con equipo A como sujeto
- $p_{\text{rev}}$ = predicción con equipo B como sujeto

Esto garantiza $P(\text{A}) + P(\text{B}) = 1.0$ exactamente.

### Arquitectura del modelo

| Propiedad | Valor |
|-----------|-------|
| Algoritmo | Regresión Logística (L2-regularizada) |
| Tipo | Clasificador binario |
| Features | 38 variables engineered |
| Muestras | 176 (88 partidos × 2 perspectivas) |
| Regularización | L2, C=0.1 |
| Validación | Time-series CV, 5-fold |
| Preprocesamiento | StandardScaler (z-score) |

### Métricas

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **CV Accuracy** | 77.2% ± 7.4% | Time-series cross-validation (honesta) |
| **Test Accuracy** | 86.1% | Held-out 20% (cronológico) |
| **Brier Score** | 0.096 | Bien calibrado (0 = perfecto, 0.25 = aleatorio) |

### Features más importantes (top 10 coeficientes)

| # | Feature | Coeficiente | ¿Qué significa? |
|---|---------|:-----------:|-----------------|
| 1 | `elo_diff` | +0.52 | Diferencia de ELO — lo más importante |
| 2 | `elo_ratio` | +0.51 | Ratio de ELO (respalda a `elo_diff`) |
| 3 | `squad_value_ratio` | +0.36 | Plantel más caro = ventaja real |
| 4 | `opp_is_top5` | −0.31 | Enfrentar a un top 5 te baja la probabilidad |
| 5 | `goal_consistency` | −0.26 | Ser errático en goles es malo |
| 6 | `cumulative_mins_ratio` | +0.21 | Minutos acumulados importan (fatiga) |
| 7 | `fifa_rank_diff` | +0.17 | Ranking FIFA también pesa |
| 8 | `same_confederation` | +0.15 | Misma confederación → más parejo |
| 9 | `host_advantage` | +0.10 | Ser local ayuda |
| 10 | `host_vs_away` | +0.10 | Local vs visitante puro |

### Explicabilidad (sin SHAP)

Cada predicción incluye la **contribución de cada feature**:

$$\text{contribución}_j = \beta_j \cdot x_j^{\text{scaled}}$$

En el frontend H2H, las contribuciones se muestran como barras: **verde** (ayuda al equipo) y **rojo** (lo perjudica).

Ejemplo real: Argentina vs Inglaterra

```
📊 Logistic: Argentina avanza (56.7%)

🔍 ¿Por qué? (contribuciones)
  opp_is_top5        coef: -0.305   contrib: -0.748  ← jugar contra top5 perjudica
  group_pts          coef: -0.098   contrib: -0.445  
  cumulative_mins    coef: +0.212   contrib: +0.351  ← mejor descanso ayuda
  elo_diff           coef: +0.515   contrib: +0.150  ← ELO superior
  goals_scored       coef: +0.100   contrib: +0.138  ← más goles anotados
```

---

## Los 38 Features en detalle

### 1. Diferenciales Básicos (5 features)

| Feature | Fórmula | Descripción |
|---------|---------|-------------|
| `elo_diff` | $R_A - R_B$ | Diferencia de rating ELO |
| `fifa_rank_diff` | $\frac{R_{\text{fifa},B} - R_{\text{fifa},A}}{50}$ | Diferencia normalizada de ranking FIFA |
| `squad_value_ratio` | $\ln\left(\frac{V_A}{\max(1, V_B)}\right)$ | Log-ratio del valor de plantel |
| `host_advantage` | $\mathbf{1}[A \text{ es anfitrión}]$ | Indicador de localía |
| `host_vs_away` | $\mathbf{1}[A \text{ local} \land \neg B \text{ local}]$ | Local puro vs visitante puro |

### 2. Momentum del Torneo (3 features)

$$M_A = \frac{\sum_{i=0}^{n-1} r_i \cdot w_i \cdot \lambda^{i}}{\sum_{i=0}^{n-1} w_i \cdot \lambda^{i}}$$

Donde:
- $r_i \in \{1.0, 0.5, 0.35, 0.20, 0.0\}$ — puntuación del resultado
- $w_i \in \{1.0, ..., 6.0\}$ — importancia del partido (grupo=1.0, Final=6.0)
- $\lambda = 0.85$ — decaimiento exponencial

Features: `momentum`, `opponent_momentum`, `momentum_diff`

### 3. Sobre-rendimiento (3 features)

El "efecto Cabo Verde" — equipos que superan expectativas:

$$O_A = \frac{1}{n}\sum_{i=1}^{n} (g_i - \mathbb{E}[g_i]) \cdot \left(1 - 0.3 \cdot \frac{R_{\text{opp},i} - 1500}{400}\right)$$

Features: `overperformance`, `opponent_overperformance`, `overperformance_diff`

### 4. Factores Físicos (3 features)

| Feature | Descripción |
|---------|-------------|
| `rest_days_diff` | Diferencia de días de descanso |
| `extra_time_diff` | Minutos extra jugados (negativo = más fresco) |
| `cumulative_mins_ratio` | Ratio de minutos acumulados |

### 5. Métricas de Rendimiento (5 features)

| Feature | Fórmula |
|---------|---------|
| `goals_scored_per_match` | $\frac{1}{n}\sum g_F$ |
| `goals_conceded_per_match` | $\frac{1}{n}\sum g_A$ |
| `goal_diff_per_match` | $\mu_F - \mu_A$ |
| `clean_sheet_rate` | $\frac{\text{partidos sin recibir gol}}{n}$ |
| `comeback_rate` | $\frac{\text{remontadas}}{n}$ |

### 6. Consistencia de Gol (1 feature)

$$C_A = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(g_i - \bar{g})^2}$$

Desviación estándar de goles por partido. Menor = más consistente.

### 7. Etapa y Grupo (7 features)

`stage_importance`, `stage_is_knockout`, `stage_is_r16`, `stage_is_qf`, `stage_is_sf`, `group_pts`, `group_pts_per_match`, `group_position`

### 8. Confederación y Meta-features (11 features)

`same_confederation`, `team_is_uefa`, `team_is_conmebol`, `opp_is_uefa`, `opp_is_conmebol`, `elo_diff_abs`, `elo_ratio`, `is_top5`, `opp_is_top5`, `coming_off_close_loss`

---

## Modelo 2: ELO + Poisson — Predicción de Goles

**Modelo complementario** para desglose Win/Draw/Loss y probabilidades de marcador exacto.

### Sistema ELO

Adaptado de ajedrez (Elo, 1978). Cada equipo tiene un rating $R$:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

**Ventaja local:** +100 puntos ELO para naciones anfitrionas.

**Actualización post-partido (K=30):**

$$R'_A = R_A + 30 \cdot (S_A - E_A)$$

### Modelo Poisson de Goles

Los goles en fútbol son eventos raros y discretos — modelados naturalmente con Poisson (Maher, 1982; Dixon & Coles, 1997).

**Goles esperados (xG):**

$$\lambda_A = \max\left(0.25,\; 1.35 + \frac{R_A - R_B}{400}\right)$$

**Probabilidad de marcador exacto:**

$$P(g_A, g_B) = \frac{\lambda_A^{g_A} e^{-\lambda_A}}{g_A!} \times \frac{\lambda_B^{g_B} e^{-\lambda_B}}{g_B!}$$

**Probabilidades agregadas:**

$$P(\text{A gana}) = \sum_{g_A=0}^{8} \sum_{g_B=0}^{g_A-1} P(g_A, g_B)$$
$$P(\text{Empate}) = \sum_{g=0}^{8} P(g, g)$$

---

## Modelo 3: Rating Compuesto de Rendimiento

Rating de **1–100** combinando tres factores:

$$\text{rating} = 1 + 99 \times (0.65 \cdot \text{ELO}_{\text{norm}} + 0.25 \cdot \text{Forma} + 0.10 \cdot \text{GD}_{\text{factor}})$$

| Componente | Peso | Descripción |
|-----------|:----:|-------------|
| ELO normalizado | 65% | ELO mapeado a [0,1] (rango competitivo: 1380–2127) |
| Forma | 25% | Promedio ponderado de últimos 10 partidos ($\alpha=0.93$) |
| Diferencia de gol | 10% | GD promedio, acotado a [-3, +3] |

---

## 🔬 Pruebas Bayesianas

Para validar el modelo, construimos un **Modelo Jerárquico Bayesiano Bradley-Terry** con PyMC:

$$P(\text{A vence a B}) = \sigma(\alpha_A - \alpha_B + \beta \cdot \text{localía})$$
$$\alpha_i \sim \mathcal{N}(\mu_i^{\text{ELO}}, 0.5)$$

### Resultados clave

| Hallazgo | Dato |
|----------|------|
| Accuracy (ELO prior) | 89.7% sobre 68 partidos con ganador claro |
| Intervalos de credibilidad | **~1.8 unidades de ancho** en habilidad $\alpha$ |
| Equivalente en ELO | **~360 puntos ELO de incertidumbre** |
| Argentina vs Inglaterra | 65.6% [33.7%, 89.2%] |

### Conclusión bayesiana

> Con 136 muestras para 48 parámetros, ningún modelo puede ser preciso. La diferencia entre XGBoost y Logistic Regression no es de accuracy — es de **honestidad epistémica**. El bayesiano cuantifica la incertidumbre; la regresión logística es conservadora; XGBoost esconde la ignorancia detrás de números precisos pero incorrectos.

---

## 🎨 Dashboard (Frontend)

SPA de una sola página (1055 líneas, vanilla JS). Diseño premium minimalista con glass-morphism, sombras suaves y micro-interacciones.

### 4 Vistas

| Vista | Hash | Contenido |
|-------|------|-----------|
| **🏠 Teams** | `#teams` | 48 tarjetas de equipo con bandera circular, ranking FIFA, valor de plantel, forma reciente (W/D/L) y rating de rendimiento. Filtros por grupo. |
| **📅 Matches** | `#matches` | 100 partidos con scores reales, barras de probabilidad (Win/ET/Lose de Poisson + avance de Logistic), y marcador predicho. Filtros por etapa y grupo. |
| **⚔️ H2H** | `#h2h` | Comparador cabeza a cabeza. Dos dropdowns → predicción de regresión logística con contribuciones de features explicadas. |
| **🏆 Bracket** | `#bracket` | Simulación completa del torneo (R16 → Final) usando regresión logística. Campeón predicho con probabilidades por ronda. |

### Características UX

- **Nav con glass-morphism** (backdrop-filter blur)
- **Skeleton loaders** animados mientras carga
- **Responsive** (3-columnas → 2 → 1 en móvil)
- **Sin recargas** — routing por hash, datos cacheados
- **Sistema de color:** Verde (victoria), Gris (empate), Rojo (derrota), Azul (acento)

---

## 📡 API Reference

| Endpoint | Descripción |
|----------|-------------|
| `GET /api/teams` | 48 equipos con ratings, valor de plantel, forma reciente |
| `GET /api/teams/{id}` | Detalle de un equipo |
| `GET /api/matches` | 100 partidos con predicciones precomputadas |
| `GET /api/matches/{id}` | Detalle de un partido |
| `GET /api/groups` | 16 grupos (A–P) con equipos y partidos |
| `GET /api/bracket` | Simulación del torneo con Poisson/ELO (legacy) |
| `GET /api/bracket/v3` | **Simulación con Regresión Logística (recomendado)** |
| `GET /api/predict/v3` | **Predicción Logística: P(avance) + contribuciones + baseline Poisson** |
| `GET /api/predict/v2` | Predicción XGBoost + SHAP (legacy) |
| `GET /api/predict` | Predicción Poisson solamente (legacy) |
| `GET /api/model-info` | Métricas y metadata del modelo |
| `GET /api/retrain` | Invalidar cachés y recargar datos |

---

## 🚀 Deployment (Seenode)

La app está desplegada en [Seenode](https://seenode.com) (plan Basic, $3/mes, sin cold starts).

| Campo | Valor |
|-------|-------|
| Provider | Seenode |
| Plan | Basic ($3/mo — 512MB RAM, 1 vCPU) |
| Repo | `DAGS-data/WorldCup2026_predictor` |
| Root directory | *(vacío — raíz del repo)* |
| Build command | `pip install -r requirements.txt` |
| Start command | `cd backend && python main.py` |
| Port | 8000 |

### ¿Por qué Seenode?

- Sin cold starts — siempre encendido
- Integración con GitHub — deploy automático al hacer push
- Soporte nativo de FastAPI + Uvicorn
- $3/mes para el stack ligero (sin DB, JSON precomputado)

---

## 🛠 Tech Stack

| Capa | Tecnología |
|------|-----------|
| Lenguaje | Python 3.13 |
| ML | scikit-learn (LogisticRegression, StandardScaler) |
| API | FastAPI + Uvicorn |
| Matemáticas | NumPy, SciPy (Poisson PMF) |
| Datos | JSON estático precomputado con `@lru_cache` |
| Frontend | HTML5 + CSS3 + Vanilla JS (SPA) |
| Mapas | Leaflet.js (OpenStreetMap vía CartoDB) |
| Banderas | FlagCDN (PNG 160px) |
| Tipografía | Inter (Google Fonts) |
| Deployment | Seenode Basic ($3/mo) |

---

## 📚 Referencias Académicas

| # | Referencia | Link |
|---|-----------|------|
| 1 | Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016. | [DOI](https://doi.org/10.1145/2939672.2939785) |
| 2 | Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017. | [Paper](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) |
| 3 | Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD 2019. | [DOI](https://doi.org/10.1145/3292500.3330701) |
| 4 | Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing. | ISBN 0-668-04721-6 |
| 5 | Maher, M. J. (1982). *Modelling association football scores*. Statistica Neerlandica, 36(3). | [DOI](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x) |
| 6 | Dixon, M. J. & Coles, S. G. (1997). *Modelling association football scores*. JRSS C, 46(2). | [DOI](https://doi.org/10.1111/1467-9876.00065) |
| 7 | Bergstra, J. et al. (2011). *Algorithms for Hyper-Parameter Optimization*. NeurIPS 2011. | [Paper](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html) |

---

## 📝 Licencia

MIT

---

> **⚠️ RECORDATORIO:** Este proyecto es una herramienta educativa y experimental. No debe usarse como base para decisiones de apuestas. Los modelos estadísticos tienen limitaciones inherentes y el fútbol real frecuentemente desafía las predicciones. Úsalo bajo tu propia discreción y responsabilidad.

---

*Datos: ESPN API, Transfermarkt (verificado julio 2026), FIFA. Modelo principal: Regresión Logística (L2, C=0.1). 38 features engineered. 88 partidos completados. World Cup 2026 Predictor.*
