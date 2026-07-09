#!/usr/bin/env python3
"""
CatBoost + Poisson Loss model for goal prediction
Predicts goals scored by each team in a match.
"""
import json
import numpy as np
from collections import defaultdict
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit

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

# ── State trackers (updated match-by-match) ──
team_goals_for = defaultdict(list)     # goals scored each match
team_goals_against = defaultdict(list) # goals conceded each match
team_match_count = defaultdict(int)     # matches played
team_elo_history = {}                   # current ELO for each team

# Initialize ELO from current teams.json (these are post-tournament, but close enough)
# Better approach: recompute ELO from scratch match-by-match
K = 32

# We actually need pre-tournament ELOs for proper time-series. 
# Let's use the formula: ELO_init = 2100 - (700/89) * (fifa_rank - 1)
for t in teams:
    elo_init = 2100 - (700.0 / 89.0) * (t["fifa_rank"] - 1)
    team_elo_history[t["id"]] = elo_init

# ── Feature extraction for a match ──
def get_team_features(team_id, opponent_id, team_elo, opp_elo, stage, is_home):
    """Extract features for ONE team going into a match."""
    tid = team_id
    t = team_by_id.get(tid, {})
    opp = team_by_id.get(opponent_id, {})
    
    gf = team_goals_for.get(tid, [])
    ga = team_goals_against.get(tid, [])
    n = len(gf)  # matches played
    
    feats = {}
    
    # Core differentials
    feats["elo"] = team_elo
    feats["elo_diff"] = team_elo - opp_elo
    feats["fifa_rank"] = t.get("fifa_rank", 90)
    feats["fifa_rank_diff"] = t.get("fifa_rank", 90) - opp.get("fifa_rank", 90)
    feats["squad_value"] = t.get("squad_value_millions", 0)
    feats["squad_value_ratio"] = feats["squad_value"] / max(opp.get("squad_value_millions", 1), 1)
    feats["is_host"] = 1 if is_home else 0
    feats["confederation"] = t.get("confederation", "Unknown")
    
    # Recent form (last 5 matches, padded with 0 if fewer)
    for i in range(5):
        idx = n - 1 - i
        feats[f"gf_last_{i+1}"] = gf[idx] if idx >= 0 else 0
        feats[f"ga_last_{i+1}"] = ga[idx] if idx >= 0 else 0
    
    # Aggregated form
    if n > 0:
        feats["avg_gf"] = np.mean(gf[-5:]) if len(gf) >= 1 else gf[-1]
        feats["avg_ga"] = np.mean(ga[-5:]) if len(ga) >= 1 else ga[-1]
        feats["gf_trend"] = np.mean(gf[-3:]) - np.mean(gf[-5:]) if len(gf) >= 5 else 0
        feats["ga_trend"] = np.mean(ga[-3:]) - np.mean(ga[-5:]) if len(ga) >= 5 else 0
        feats["goal_diff_avg"] = feats["avg_gf"] - feats["avg_ga"]
        feats["matches_played"] = n
    else:
        feats["avg_gf"] = 0
        feats["avg_ga"] = 0
        feats["gf_trend"] = 0
        feats["ga_trend"] = 0
        feats["goal_diff_avg"] = 0
        feats["matches_played"] = 0
    
    # Tournament stage
    stage_order = {"group": 0, "round_of_32": 1, "round_of_16": 2, "quarter_finals": 3}
    feats["stage_num"] = stage_order.get(stage, 0)
    
    # Performance metrics from teams.json
    perf = t.get("performance", {})
    feats["perf_rating"] = perf.get("rating_100", 50)
    feats["perf_form"] = perf.get("form_score", 50)
    
    return feats


# ── Build training data ──
X_rows = []
y_home = []
y_away = []

for match in completed:
    t1_id = match["team1_id"]
    t2_id = match["team2_id"]
    stage = match["stage"]
    
    elo1 = team_elo_history[t1_id]
    elo2 = team_elo_history[t2_id]
    
    # Features for team1 (home perspective)
    f1 = get_team_features(t1_id, t2_id, elo1, elo2, stage, is_home=match["is_home_team1"])
    f2 = get_team_features(t2_id, t1_id, elo2, elo1, stage, is_home=False)  # team2 is visitor
    
    # Combine features: prefix with t1_ and t2_
    row = {}
    for k, v in f1.items():
        row[f"t1_{k}"] = v
    for k, v in f2.items():
        row[f"t2_{k}"] = v
    
    X_rows.append(row)
    y_home.append(match["team1_goals"])
    y_away.append(match["team2_goals"])
    
    # Update ELO after match
    actual_diff = match["team1_goals"] - match["team2_goals"]
    if actual_diff > 0:
        result_t1, result_t2 = 1, 0
    elif actual_diff < 0:
        result_t1, result_t2 = 0, 1
    else:
        result_t1, result_t2 = 0.5, 0.5
    
    expected_t1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    expected_t2 = 1 - expected_t1
    team_elo_history[t1_id] = elo1 + K * (result_t1 - expected_t1)
    team_elo_history[t2_id] = elo2 + K * (result_t2 - expected_t2)
    
    # Track goals
    team_goals_for[t1_id].append(match["team1_goals"])
    team_goals_against[t1_id].append(match["team2_goals"])
    team_goals_for[t2_id].append(match["team2_goals"])
    team_goals_against[t2_id].append(match["team1_goals"])
    team_match_count[t1_id] += 1
    team_match_count[t2_id] += 1

print(f"Training samples: {len(X_rows)}")

# ── Convert to arrays ──
# Separate numeric and categorical columns
categorical_cols = ["t1_confederation", "t2_confederation"]

# Build feature matrix
all_keys = sorted(X_rows[0].keys())
numeric_keys = [k for k in all_keys if k not in categorical_cols]

X_numeric = np.array([[row[k] for k in numeric_keys] for row in X_rows], dtype=np.float32)
X_categorical = np.array([[row[k] for k in categorical_cols] for row in X_rows], dtype=str)

X_all = np.hstack([X_numeric, X_categorical.astype(object)])  # numpy object array for CatBoost
feature_names = numeric_keys + categorical_cols

y_home_arr = np.array(y_home, dtype=np.float32)
y_away_arr = np.array(y_away, dtype=np.float32)

# ── Train CatBoost (Poisson) with TimeSeriesSplit ──
def train_poisson_model(X_train, y_train, X_val, y_val, name):
    train_pool = Pool(X_train, y_train, cat_features=list(range(len(numeric_keys), len(feature_names))))
    val_pool = Pool(X_val, y_val, cat_features=list(range(len(numeric_keys), len(feature_names))))
    
    model = CatBoostRegressor(
        loss_function="Poisson",
        iterations=500,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=30,
    )
    model.fit(train_pool, eval_set=val_pool, verbose=False)
    
    preds = model.predict(val_pool)
    mae = np.mean(np.abs(preds - y_val))
    poisson_dev = 2 * np.mean(y_val * np.log((y_val + 1e-8) / (preds + 1e-8)) - (y_val - preds))
    
    print(f"{name}: MAE={mae:.3f}  Poisson deviance={poisson_dev:.3f}  Iters={model.tree_count_}")
    return model, mae

# Use last 20% for validation
split_idx = int(len(X_all) * 0.8)
X_train, X_val = X_all[:split_idx], X_all[split_idx:]
y_home_train, y_home_val = y_home_arr[:split_idx], y_home_arr[split_idx:]
y_away_train, y_away_val = y_away_arr[:split_idx], y_away_arr[split_idx:]

print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}")
print(f"Feature count: {len(feature_names)}")
print()

model_home, mae_home = train_poisson_model(X_train, y_home_train, X_val, y_home_val, "Home goals")
model_away, mae_away = train_poisson_model(X_train, y_away_train, X_val, y_away_val, "Away goals")

# ── Combined evaluation ──
pred_home = model_home.predict(Pool(X_val, cat_features=list(range(len(numeric_keys), len(feature_names)))))
pred_away = model_away.predict(Pool(X_val, cat_features=list(range(len(numeric_keys), len(feature_names)))))

pred_total = pred_home + pred_away
actual_total = y_home_val + y_away_val
total_mae = np.mean(np.abs(pred_total - actual_total))
print(f"\nCombined total goals MAE: {total_mae:.3f}")
print(f"Mean actual goals: {np.mean(actual_total):.2f}, Mean predicted: {np.mean(pred_total):.2f}")

# Show some predictions
print("\n--- Sample predictions (validation set) ---")
for i in range(min(10, len(X_val))):
    # Find original match
    match_idx = split_idx + i
    m = completed[match_idx]
    t1 = team_by_id[m["team1_id"]]["name"]
    t2 = team_by_id[m["team2_id"]]["name"]
    print(f"  {t1} vs {t2}: actual {m['team1_goals']}-{m['team2_goals']}, "
          f"pred λ=({pred_home[i]:.2f}, {pred_away[i]:.2f}), "
          f"total actual={m['team1_goals']+m['team2_goals']} pred={pred_total[i]:.2f}")

# Feature importance
print("\n--- Top features (home goals model) ---")
importances = model_home.get_feature_importance()
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
for name, imp in feat_imp[:15]:
    print(f"  {name}: {imp:.4f}")

# ── Save models ──
model_home.save_model("backend/models/goals_home_v1.cbm")
model_away.save_model("backend/models/goals_away_v1.cbm")

# Save metadata
metadata = {
    "feature_names": feature_names,
    "categorical_features": categorical_cols,
    "numeric_features": numeric_keys,
    "mae_home": float(mae_home),
    "mae_away": float(mae_away),
    "total_mae": float(total_mae),
    "train_samples": len(X_train),
    "val_samples": len(X_val),
}
with open("backend/models/goals_metadata_v1.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Models saved: backend/models/goals_home_v1.cbm, goals_away_v1.cbm")
print("✅ Metadata saved: backend/models/goals_metadata_v1.json")
