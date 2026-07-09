#!/usr/bin/env python3
"""
CatBoost + Poisson Loss — V3 (lean & calibrated)
Keeps best features from V1, adds calibration, hyperparameter tuning.
"""
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
from catboost import CatBoostRegressor, Pool

# ── Load ──
with open("backend/data/teams.json") as f:
    teams = json.load(f)
with open("backend/data/matches.json") as f:
    matches_raw = json.load(f)

team_by_id = {t["id"]: t for t in teams}

# ── Sort ──
completed = []
for m in matches_raw:
    if (m.get("status") or "").startswith("completed") and m.get("score"):
        completed.append({
            "date": m["date"], "stage": m.get("stage", "group"),
            "team1_id": m["team1_id"], "team2_id": m["team2_id"],
            "team1_goals": int(m["score"]["team1"]),
            "team2_goals": int(m["score"]["team2"]),
            "is_home_team1": team_by_id[m["team1_id"]].get("is_host", False),
        })
completed.sort(key=lambda m: m["date"])

# ── State ──
team_gf = defaultdict(list)
team_ga = defaultdict(list)
team_dates = defaultdict(list)
K = 32
for t in teams:
    t["_elo"] = 2100 - (700.0 / 89.0) * (t["fifa_rank"] - 1)

def safe_mean(arr): return float(np.mean(arr)) if arr else 0.0
def rest_days(dates, match_date): 
    if not dates: return 5
    return (datetime.strptime(match_date, "%Y-%m-%d") - datetime.strptime(dates[-1], "%Y-%m-%d")).days

def get_team_features(tid, oid, stage, is_home, match_date):
    t = team_by_id[tid]
    opp = team_by_id[oid]
    gf = team_gf[tid]
    ga = team_ga[tid]
    n = len(gf)
    
    f = {}
    # Core differentials
    f["elo"] = t["_elo"]
    f["elo_diff"] = t["_elo"] - opp["_elo"]
    f["fifa_rank_diff"] = t.get("fifa_rank", 90) - opp.get("fifa_rank", 90)
    f["squad_value_log"] = np.log1p(t.get("squad_value_millions", 0))
    f["squad_value_ratio"] = np.log1p(t.get("squad_value_millions", 1)) / max(np.log1p(opp.get("squad_value_millions", 1)), 0.01)
    f["is_host"] = 1 if is_home else 0
    f["is_host_nation"] = 1 if t.get("is_host", False) else 0
    f["confederation"] = t.get("confederation", "Unknown")
    
    # Raw last 5 games
    for i in range(5):
        idx = n - 1 - i
        f[f"gf_l{i+1}"] = gf[idx] if idx >= 0 else 0
        f[f"ga_l{i+1}"] = ga[idx] if idx >= 0 else 0
    
    # Aggregated form
    if n > 0:
        last5_gf = gf[-5:] if len(gf) >= 5 else gf
        last5_ga = ga[-5:] if len(ga) >= 5 else ga
        f["avg_gf"] = safe_mean(last5_gf)
        f["avg_ga"] = safe_mean(last5_ga)
        f["goal_diff_avg"] = safe_mean([gf[i] - ga[i] for i in range(max(0, n-5), n)])
        # trend: last 3 vs previous 3
        if n >= 4:
            f["gf_trend"] = safe_mean(gf[-3:]) - safe_mean(gf[max(0, n-6):n-3])
        else:
            f["gf_trend"] = 0
        # over 2.5 rate
        f["over25_rate"] = sum(1 for i in range(max(0, n-5), n) if gf[i] + ga[i] > 2) / min(n, 5)
        # clean sheet rate
        f["clean_sheet_rate"] = sum(1 for g in ga[-5:] if g == 0) / min(n, 5)
        f["matches_played"] = min(n, 10)
    else:
        f["avg_gf"] = f["avg_ga"] = f["goal_diff_avg"] = f["gf_trend"] = 0.0
        f["over25_rate"] = f["clean_sheet_rate"] = 0.0
        f["matches_played"] = 0
    
    # Rest days
    f["rest_days"] = min(rest_days(team_dates[tid], match_date), 14)
    
    # Stage
    stage_map = {"group": 0, "round_of_32": 1, "round_of_16": 2, "quarter_finals": 3}
    f["stage_num"] = stage_map.get(stage, 0)
    
    # Performance
    perf = t.get("performance", {})
    f["perf_rating"] = perf.get("rating_100", 50)
    f["perf_form"] = perf.get("form_score", 50)
    
    return f

# ── Build ──
X_rows, y_home, y_away = [], [], []

for m in completed:
    t1_id, t2_id = m["team1_id"], m["team2_id"]
    f1 = get_team_features(t1_id, t2_id, m["stage"], m["is_home_team1"], m["date"])
    f2 = get_team_features(t2_id, t1_id, m["stage"], False, m["date"])
    
    row = {}
    for k, v in f1.items(): row[f"t1_{k}"] = v
    for k, v in f2.items(): row[f"t2_{k}"] = v
    X_rows.append(row)
    y_home.append(m["team1_goals"])
    y_away.append(m["team2_goals"])
    
    # Update ELO
    elo1, elo2 = team_by_id[t1_id]["_elo"], team_by_id[t2_id]["_elo"]
    gd = m["team1_goals"] - m["team2_goals"]
    r1 = 1 if gd > 0 else (0 if gd < 0 else 0.5)
    e1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    team_by_id[t1_id]["_elo"] = elo1 + K * (r1 - e1)
    team_by_id[t2_id]["_elo"] = elo2 + K * ((1 - r1) - (1 - e1))
    
    team_gf[t1_id].append(m["team1_goals"])
    team_ga[t1_id].append(m["team2_goals"])
    team_gf[t2_id].append(m["team2_goals"])
    team_ga[t2_id].append(m["team1_goals"])
    team_dates[t1_id].append(m["date"])
    team_dates[t2_id].append(m["date"])

# ── Feature matrix ──
cat_cols = sorted([k for k in X_rows[0] if k.endswith("_confederation")])
num_cols = sorted([k for k in X_rows[0] if k not in cat_cols])
all_cols = num_cols + cat_cols
cat_idx = list(range(len(num_cols), len(all_cols)))

X_num = np.array([[r[k] for k in num_cols] for r in X_rows], dtype=np.float32)
X_cat = np.array([[r[k] for k in cat_cols] for r in X_rows], dtype=str)
X_all = np.hstack([X_num, X_cat.astype(object)])
y_h = np.array(y_home, dtype=np.float32)
y_a = np.array(y_away, dtype=np.float32)

print(f"Samples: {len(X_rows)}, Features: {len(all_cols)}")

# ── Split ──
split = int(len(X_all) * 0.8)
X_tr, X_va = X_all[:split], X_all[split:]
yh_tr, yh_va = y_h[:split], y_h[split:]
ya_tr, ya_va = y_a[:split], y_a[split:]
print(f"Train: {len(X_tr)}, Val: {len(X_va)}")

# ── Hyperparameter sweep ──
def run(params, Xtr, ytr, Xva, yva):
    trp = Pool(Xtr, ytr, cat_features=cat_idx)
    vap = Pool(Xva, yva, cat_features=cat_idx)
    m = CatBoostRegressor(loss_function="Poisson", random_seed=42, verbose=False, early_stopping_rounds=30, **params)
    m.fit(trp, eval_set=vap, verbose=False)
    p = m.predict(vap)
    mae = float(np.mean(np.abs(p - yva)))
    dev = float(2 * np.mean(yva * np.log((yva + 1e-8) / (p + 1e-8)) - (yva - p)))
    return m, mae, dev

params_grid = [
    {"iterations": 300, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 3},
    {"iterations": 300, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 5},
    {"iterations": 400, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 5},
    {"iterations": 400, "learning_rate": 0.03, "depth": 5, "l2_leaf_reg": 3},
]

best_h, best_a = None, None
best_h_score, best_a_score = 1e9, 1e9

print("\n--- Hyperparameter sweep ---")
for i, p in enumerate(params_grid):
    mh, mae_h, dev_h = run(p, X_tr, yh_tr, X_va, yh_va)
    ma, mae_a, dev_a = run(p, X_tr, ya_tr, X_va, ya_va)
    sh, sa = mae_h + dev_h*0.3, mae_a + dev_a*0.3
    print(f"  [{i}] lr={p['learning_rate']} d={p['depth']} l2={p['l2_leaf_reg']}: "
          f"H MAE={mae_h:.3f} Dev={dev_h:.3f} {'★' if sh<best_h_score else ''} | "
          f"A MAE={mae_a:.3f} Dev={dev_a:.3f} {'★' if sa<best_a_score else ''}")
    if sh < best_h_score:
        best_h_score = sh
        best_h = (p, mh, mae_h, dev_h)
    if sa < best_a_score:
        best_a_score = sa
        best_a = (p, ma, mae_a, dev_a)

print(f"\nBest home: MAE={best_h[2]:.3f} Dev={best_h[3]:.3f} p={best_h[0]}")
print(f"Best away: MAE={best_a[2]:.3f} Dev={best_a[3]:.3f} p={best_a[0]}")

# ── Retrain on all training data ──
model_home = CatBoostRegressor(loss_function="Poisson", random_seed=42, verbose=False, **best_h[0])
model_home.fit(X_tr, yh_tr, cat_features=cat_idx, verbose=False)

model_away = CatBoostRegressor(loss_function="Poisson", random_seed=42, verbose=False, **best_a[0])
model_away.fit(X_tr, ya_tr, cat_features=cat_idx, verbose=False)

# ── Calibrate ──
vap = Pool(X_va, cat_features=cat_idx)
ph = model_home.predict(vap)
pa = model_away.predict(vap)

cal_h = float(np.mean(yh_va) / max(np.mean(ph), 0.01))
cal_a = float(np.mean(ya_va) / max(np.mean(pa), 0.01))

ph_cal = ph * cal_h
pa_cal = pa * cal_a
total_pred_cal = ph_cal + pa_cal
total_actual = yh_va + ya_va

print(f"\n--- Validation (after calibration) ---")
print(f"Home MAE: {np.mean(np.abs(ph_cal - yh_va)):.3f}  (raw: {np.mean(np.abs(ph - yh_va)):.3f})")
print(f"Away MAE: {np.mean(np.abs(pa_cal - ya_va)):.3f}  (raw: {np.mean(np.abs(pa - ya_va)):.3f})")
print(f"Total MAE: {np.mean(np.abs(total_pred_cal - total_actual)):.3f}")
print(f"Mean actual total: {np.mean(total_actual):.2f}, pred: {np.mean(total_pred_cal):.2f}")
print(f"Calibration: home={cal_h:.3f}, away={cal_a:.3f}")

# ── Samples ──
print("\n--- Sample predictions ---")
for i in range(min(10, len(X_va))):
    mi = split + i
    m = completed[mi]
    t1 = team_by_id[m["team1_id"]]["name"]
    t2 = team_by_id[m["team2_id"]]["name"]
    print(f"  {t1} vs {t2}: {m['team1_goals']}-{m['team2_goals']}  →  pred ({ph_cal[i]:.1f}, {pa_cal[i]:.1f}) total={total_pred_cal[i]:.1f}")

# ── Feature importance ──
print("\n--- Top 15 features (home goals) ---")
imps = model_home.get_feature_importance()
for name, imp in sorted(zip(all_cols, imps), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {name}: {imp:.4f}")

# ── Save ──
model_home.save_model("backend/models/goals_home_v3.cbm")
model_away.save_model("backend/models/goals_away_v3.cbm")
meta = {
    "feature_names": all_cols, "categorical_features": cat_cols, "cat_indices": cat_idx,
    "mae_home_cal": float(np.mean(np.abs(ph_cal - yh_va))),
    "mae_away_cal": float(np.mean(np.abs(pa_cal - ya_va))),
    "total_mae_cal": float(np.mean(np.abs(total_pred_cal - total_actual))),
    "calibration_home": cal_h, "calibration_away": cal_a,
    "best_params_home": best_h[0], "best_params_away": best_a[0],
    "train_samples": len(X_tr), "val_samples": len(X_va),
}
with open("backend/models/goals_metadata_v3.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"\n✅ goals_home_v3.cbm + goals_away_v3.cbm saved")
