#!/usr/bin/env python3
"""
CatBoost + Poisson Loss — V2 improved
More features, hyperparameter tuning, calibration.
"""
import json
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from catboost import CatBoostRegressor, Pool

# ── Load data ──
with open("backend/data/teams.json") as f:
    teams = json.load(f)
with open("backend/data/matches.json") as f:
    matches_raw = json.load(f)

team_by_id = {t["id"]: t for t in teams}

# ── Sort completed matches chronologically ──
completed = []
for m in matches_raw:
    if (m.get("status") or "").startswith("completed") and m.get("score"):
        t1g = int(m["score"]["team1"])
        t2g = int(m["score"]["team2"])
        completed.append({
            "date": m["date"],
            "stage": m.get("stage", "group"),
            "team1_id": m["team1_id"],
            "team2_id": m["team2_id"],
            "team1_goals": t1g,
            "team2_goals": t2g,
            "is_home_team1": team_by_id[m["team1_id"]].get("is_host", False),
        })
completed.sort(key=lambda m: m["date"])

# ── State trackers ──
team_goals_for = defaultdict(list)
team_goals_against = defaultdict(list)
team_match_dates = defaultdict(list)
team_h2h = defaultdict(lambda: defaultdict(list))  # team_h2h[tid1][tid2] = [(gf, ga, date), ...]

K = 32
for t in teams:
    t["_elo"] = 2100 - (700.0 / 89.0) * (t["fifa_rank"] - 1)

# ── Feature helpers ──
def safe_mean(arr, default=0):
    return float(np.mean(arr)) if len(arr) > 0 else default

def ewma(arr, alpha=0.3, default=0):
    """Exponential weighted moving average (recent = higher weight)."""
    if not arr:
        return default
    weights = np.exp([alpha * i for i in range(len(arr))])
    weights = weights / weights.sum()
    return float(np.sum(np.array(arr) * weights))

def rest_days(dates, match_date_str, default=5):
    """Days since last match."""
    if not dates:
        return default
    last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    match_date = datetime.strptime(match_date_str, "%Y-%m-%d")
    return (match_date - last_date).days


def get_team_features(team_id, opponent_id, stage, is_home, match_date):
    tid, oid = team_id, opponent_id
    t = team_by_id[tid]
    opp = team_by_id[oid]
    
    gf = team_goals_for[tid]
    ga = team_goals_against[tid]
    dates = team_match_dates[tid]
    n = len(gf)
    
    feats = {}
    
    # ── Core differentials ──
    feats["elo"] = t["_elo"]
    feats["elo_diff"] = t["_elo"] - opp["_elo"]
    feats["fifa_rank"] = t.get("fifa_rank", 90)
    feats["fifa_rank_diff"] = t.get("fifa_rank", 90) - opp.get("fifa_rank", 90)
    feats["squad_value"] = np.log1p(t.get("squad_value_millions", 0))
    feats["squad_value_ratio"] = np.log1p(t.get("squad_value_millions", 1)) / max(np.log1p(opp.get("squad_value_millions", 1)), 0.01)
    feats["is_host"] = 1 if is_home else 0
    feats["is_host_team"] = 1 if t.get("is_host", False) else 0
    
    # ── Raw recent form (last 5) ──
    for i in range(5):
        idx = n - 1 - i
        feats[f"gf_l{i+1}"] = gf[idx] if idx >= 0 else 0
        feats[f"ga_l{i+1}"] = ga[idx] if idx >= 0 else 0
        # Goal difference
        feats[f"gd_l{i+1}"] = (gf[idx] - ga[idx]) if idx >= 0 else 0
    
    # ── Aggregated form (EWMA-weighted) ──
    if n > 0:
        feats["avg_gf"] = safe_mean(gf[-5:])
        feats["avg_ga"] = safe_mean(ga[-5:])
        feats["ewma_gf"] = ewma(gf[-8:], alpha=0.4)
        feats["ewma_ga"] = ewma(ga[-8:], alpha=0.4)
        feats["goal_diff_avg"] = safe_mean([gf[i] - ga[i] for i in range(max(0, n-5), n)])
        
        # Trends: acceleration (how fast is form changing?)
        if n >= 3:
            recent_gf = gf[-3:]
            older_gf = gf[max(0, n-6):n-3]
            feats["gf_trend"] = safe_mean(recent_gf) - safe_mean(older_gf)
            recent_ga = ga[-3:]
            older_ga = ga[max(0, n-6):n-3]
            feats["ga_trend"] = safe_mean(recent_ga) - safe_mean(older_ga)
        else:
            feats["gf_trend"] = 0
            feats["ga_trend"] = 0
        
        # Clean sheet rate
        clean_sheets = sum(1 for g in ga[-5:] if g == 0)
        feats["clean_sheet_rate"] = clean_sheets / min(n, 5)
        
        # BTTS rate (both teams to score)
        btts = sum(1 for i in range(max(0, n-5), n) if gf[i] > 0 and ga[i] > 0)
        feats["btts_rate"] = btts / min(n, 5)
        
        # Over 2.5 rate
        over25 = sum(1 for i in range(max(0, n-5), n) if gf[i] + ga[i] > 2)
        feats["over25_rate"] = over25 / min(n, 5)
        
        # Scoring consistency (std of goals scored)
        feats["gf_std"] = float(np.std(gf[-5:])) if n >= 2 else 0
        
        # Win rate
        wins = sum(1 for i in range(max(0, n-5), n) if gf[i] > ga[i])
        feats["win_rate"] = wins / min(n, 5)
        
        feats["matches_played"] = n
    else:
        feats["avg_gf"] = feats["avg_ga"] = feats["ewma_gf"] = feats["ewma_ga"] = 0
        feats["goal_diff_avg"] = 0
        feats["gf_trend"] = feats["ga_trend"] = 0
        feats["clean_sheet_rate"] = feats["btts_rate"] = feats["over25_rate"] = 0
        feats["gf_std"] = feats["win_rate"] = 0
        feats["matches_played"] = 0
    
    # ── Rest / fatigue ──
    feats["rest_days"] = rest_days(dates, match_date)
    feats["rest_days_log"] = np.log1p(feats["rest_days"])
    
    # ── H2H history ──
    h2h = team_h2h[tid][oid]
    h2h_n = len(h2h)
    if h2h_n > 0:
        feats["h2h_avg_gf"] = safe_mean([h[0] for h in h2h[-3:]])
        feats["h2h_avg_ga"] = safe_mean([h[1] for h in h2h[-3:]])
        feats["h2h_matches"] = h2h_n
        feats["h2h_goal_diff_avg"] = safe_mean([h[0] - h[1] for h in h2h[-3:]])
        feats["h2h_days_since"] = rest_days([h[2] for h in h2h], match_date, default=365)
    else:
        feats["h2h_avg_gf"] = feats["h2h_avg_ga"] = feats["h2h_goal_diff_avg"] = 0
        feats["h2h_matches"] = 0
        feats["h2h_days_since"] = 365
    
    # ── Tournament stage ──
    stage_order = {"group": 0, "round_of_32": 1, "round_of_16": 2, "quarter_finals": 3}
    feats["stage_num"] = stage_order.get(stage, 0)
    
    # ── Confederation ──
    feats["confederation"] = t.get("confederation", "Unknown")
    
    # ── Performance metrics ──
    perf = t.get("performance", {})
    feats["perf_rating"] = perf.get("rating_100", 50)
    feats["perf_form"] = perf.get("form_score", 50)
    
    return feats


# ── Build training data ──
X_rows = []
y_home = []
y_away = []

for match in completed:
    t1_id, t2_id = match["team1_id"], match["team2_id"]
    stage = match["stage"]
    
    f1 = get_team_features(t1_id, t2_id, stage, is_home=match["is_home_team1"], match_date=match["date"])
    f2 = get_team_features(t2_id, t1_id, stage, is_home=False, match_date=match["date"])
    
    row = {}
    for k, v in f1.items():
        row[f"t1_{k}"] = v
    for k, v in f2.items():
        row[f"t2_{k}"] = v
    
    X_rows.append(row)
    y_home.append(match["team1_goals"])
    y_away.append(match["team2_goals"])
    
    # Update ELO
    elo1, elo2 = team_by_id[t1_id]["_elo"], team_by_id[t2_id]["_elo"]
    gd = match["team1_goals"] - match["team2_goals"]
    if gd > 0: r1, r2 = 1, 0
    elif gd < 0: r1, r2 = 0, 1
    else: r1, r2 = 0.5, 0.5
    e1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    team_by_id[t1_id]["_elo"] = elo1 + K * (r1 - e1)
    team_by_id[t2_id]["_elo"] = elo2 + K * (r2 - (1 - e1))
    
    # Track goals & H2H
    team_goals_for[t1_id].append(match["team1_goals"])
    team_goals_against[t1_id].append(match["team2_goals"])
    team_goals_for[t2_id].append(match["team2_goals"])
    team_goals_against[t2_id].append(match["team1_goals"])
    team_match_dates[t1_id].append(match["date"])
    team_match_dates[t2_id].append(match["date"])
    team_h2h[t1_id][t2_id].append((match["team1_goals"], match["team2_goals"], match["date"]))
    team_h2h[t2_id][t1_id].append((match["team2_goals"], match["team1_goals"], match["date"]))

print(f"Training samples: {len(X_rows)}")

# ── Feature matrix ──
categorical_cols = sorted([k for k in X_rows[0] if k.endswith("_confederation")])
numeric_cols = sorted([k for k in X_rows[0] if k not in categorical_cols])
all_cols = numeric_cols + categorical_cols

X_num = np.array([[row[k] for k in numeric_cols] for row in X_rows], dtype=np.float32)
X_cat = np.array([[row[k] for k in categorical_cols] for row in X_rows], dtype=str)
X_all = np.hstack([X_num, X_cat.astype(object)])
cat_indices = list(range(len(numeric_cols), len(all_cols)))

y_home_arr = np.array(y_home, dtype=np.float32)
y_away_arr = np.array(y_away, dtype=np.float32)

print(f"Features: {len(all_cols)} ({len(numeric_cols)} numeric + {len(categorical_cols)} categorical)")

# ── Train/validation split (chronological: last 20%) ──
split_idx = int(len(X_all) * 0.8)
X_train, X_val = X_all[:split_idx], X_all[split_idx:]
y_h_tr, y_h_val = y_home_arr[:split_idx], y_home_arr[split_idx:]
y_a_tr, y_a_val = y_away_arr[:split_idx], y_away_arr[split_idx:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# ── Hyperparameter sweep ──
def train_and_eval(params, X_tr, y_tr, X_v, y_v, name):
    tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
    v_pool = Pool(X_v, y_v, cat_features=cat_indices)
    
    model = CatBoostRegressor(
        loss_function="Poisson",
        random_seed=42,
        verbose=False,
        early_stopping_rounds=30,
        **params
    )
    model.fit(tr_pool, eval_set=v_pool, verbose=False)
    
    preds = model.predict(v_pool)
    mae = float(np.mean(np.abs(preds - y_v)))
    poisson_dev = float(2 * np.mean(y_v * np.log((y_v + 1e-8) / (preds + 1e-8)) - (y_v - preds)))
    return model, mae, poisson_dev, model.tree_count_

param_grid = [
    {"iterations": 400, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 3},
    {"iterations": 400, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 5},
    {"iterations": 400, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 7},
    {"iterations": 500, "learning_rate": 0.02, "depth": 4, "l2_leaf_reg": 3},
]

print("\n--- Hyperparameter sweep ---")
best_home, best_away = None, None
best_h_score, best_a_score = 1e9, 1e9

for i, params in enumerate(param_grid):
    m_h, mae_h, dev_h, iters_h = train_and_eval(params, X_train, y_h_tr, X_val, y_h_val, f"Home [{i}]")
    m_a, mae_a, dev_a, iters_a = train_and_eval(params, X_train, y_a_tr, X_val, y_a_val, f"Away [{i}]")
    
    score_h = mae_h + dev_h * 0.5
    score_a = mae_a + dev_a * 0.5
    
    mark_h = "★" if score_h < best_h_score else ""
    mark_a = "★" if score_a < best_a_score else ""
    
    if score_h < best_h_score:
        best_h_score = score_h
        best_home = (m_h, mae_h, dev_h, params)
    if score_a < best_a_score:
        best_a_score = score_a
        best_away = (m_a, mae_a, dev_a, params)
    
    print(f"  [{i}] lr={params['learning_rate']} d={params['depth']} l2={params['l2_leaf_reg']}: "
          f"Home MAE={mae_h:.3f} Dev={dev_h:.3f} ({iters_h} iters) {mark_h} | "
          f"Away MAE={mae_a:.3f} Dev={dev_a:.3f} ({iters_a} iters) {mark_a}")

print(f"\nBest home: MAE={best_home[1]:.3f} Dev={best_home[2]:.3f} params={best_home[3]}")
print(f"Best away: MAE={best_away[1]:.3f} Dev={best_away[2]:.3f} params={best_away[3]}")

# ── Retrain best models on all training data ──
def train_final(params, X_tr, y_tr, name):
    pool = Pool(X_tr, y_tr, cat_features=cat_indices)
    model = CatBoostRegressor(
        loss_function="Poisson",
        random_seed=42,
        verbose=False,
        **params
    )
    model.fit(pool, verbose=False)
    return model

model_home = train_final(best_home[3], X_train, y_h_tr, "home_final")
model_away = train_final(best_away[3], X_train, y_a_tr, "away_final")

# ── Evaluate & calibrate ──
v_pool = Pool(X_val, cat_features=cat_indices)
pred_h = model_home.predict(v_pool)
pred_a = model_away.predict(v_pool)
pred_total = pred_h + pred_a
actual_total = y_h_val + y_a_val

print(f"\n--- Validation results ---")
print(f"Home MAE: {np.mean(np.abs(pred_h - y_h_val)):.3f}")
print(f"Away MAE: {np.mean(np.abs(pred_a - y_a_val)):.3f}")
print(f"Total MAE: {np.mean(np.abs(pred_total - actual_total)):.3f}")
print(f"Mean actual: home={np.mean(y_h_val):.2f} away={np.mean(y_a_val):.2f} total={np.mean(actual_total):.2f}")
print(f"Mean predicted: home={np.mean(pred_h):.2f} away={np.mean(pred_a):.2f} total={np.mean(pred_total):.2f}")

# Calibration: adjust λ so mean predicted ≈ mean actual
cal_home = np.mean(y_h_val) / max(np.mean(pred_h), 0.01)
cal_away = np.mean(y_a_val) / max(np.mean(pred_a), 0.01)
print(f"\nCalibration factors: home={cal_home:.3f}, away={cal_away:.3f}")

pred_h_cal = pred_h * cal_home
pred_a_cal = pred_a * cal_away
pred_total_cal = pred_h_cal + pred_a_cal
print(f"After calibration:")
print(f"  Home MAE: {np.mean(np.abs(pred_h_cal - y_h_val)):.3f}")
print(f"  Away MAE: {np.mean(np.abs(pred_a_cal - y_a_val)):.3f}")
print(f"  Total MAE: {np.mean(np.abs(pred_total_cal - actual_total)):.3f}")
print(f"  Mean predicted total: {np.mean(pred_total_cal):.2f}")

# ── Sample predictions ──
print(f"\n--- Sample predictions (validation) ---")
for i in range(min(12, len(X_val))):
    mi = split_idx + i
    m = completed[mi]
    t1 = team_by_id[m["team1_id"]]["name"]
    t2 = team_by_id[m["team2_id"]]["name"]
    real = f"{m['team1_goals']}-{m['team2_goals']}"
    raw = f"({pred_h[i]:.1f}, {pred_a[i]:.1f})"
    cal = f"({pred_h_cal[i]:.1f}, {pred_a_cal[i]:.1f})"
    print(f"  {t1} vs {t2}: real {real} | raw {raw} | cal {cal}")

# ── Feature importance ──
print(f"\n--- Top 20 features (home goals) ---")
imps = model_home.get_feature_importance()
feat_imp = sorted(zip(all_cols, imps), key=lambda x: x[1], reverse=True)
for name, imp in feat_imp[:20]:
    print(f"  {name}: {imp:.4f}")

# ── Save models ──
model_home.save_model("backend/models/goals_home_v2.cbm")
model_away.save_model("backend/models/goals_away_v2.cbm")

metadata = {
    "feature_names": all_cols,
    "categorical_features": categorical_cols,
    "numeric_features": numeric_cols,
    "cat_indices": cat_indices,
    "mae_home": float(np.mean(np.abs(pred_h_cal - y_h_val))),
    "mae_away": float(np.mean(np.abs(pred_a_cal - y_a_val))),
    "total_mae": float(np.mean(np.abs(pred_total_cal - actual_total))),
    "calibration_home": float(cal_home),
    "calibration_away": float(cal_away),
    "train_samples": len(X_train),
    "val_samples": len(X_val),
    "best_params_home": best_home[3],
    "best_params_away": best_away[3],
}
with open("backend/models/goals_metadata_v2.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Models saved: goals_home_v2.cbm, goals_away_v2.cbm")
print(f"✅ Metadata saved: goals_metadata_v2.json")
