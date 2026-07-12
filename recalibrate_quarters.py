#!/usr/bin/env python3
"""Recalibrate model with real Round of 16 results for the quarters branch."""

import json
import sys
from pathlib import Path

# Add backend to path
BACKEND = Path(__file__).parent / "backend"
sys.path.insert(0, str(BACKEND))

from elo import calculate_new_ratings
from feature_engineering import build_dataset, build_team_history, compute_match_features
from models.xgboost_predictor import KnockoutPredictor
from models.logistic_predictor import LogisticPredictor
from predictor import calculate_match_probabilities, get_prediction_factors
from performance import calculate_performance_rating

DATA_DIR = BACKEND / "data"

# ──────────────────────────────────────────────
# 1. Real Round of 16 results
# ──────────────────────────────────────────────
# Format: (match_id, team1_score, team2_score, status, penalties?)
# penalties: optional "p1-p2" string
REAL_R16 = {
    88: (0, 3, "completed"),        # Canada vs Morocco
    89: (0, 1, "completed"),        # Paraguay vs France
    90: (1, 2, "completed"),        # Brazil vs Norway
    91: (2, 3, "completed"),        # Mexico vs England
    92: (0, 1, "completed"),        # Portugal vs Spain
    93: (1, 4, "completed"),        # USA vs Belgium
    94: (3, 2, "completed"),        # Argentina vs Egypt
    95: (0, 0, "completed_penalties", "4-3"),  # Switzerland vs Colombia (Swiss won pens)
}

# ──────────────────────────────────────────────
# 2. Load current data
# ──────────────────────────────────────────────
with open(DATA_DIR / "matches.json") as f:
    matches = json.load(f)

with open(DATA_DIR / "teams.json") as f:
    teams_raw = json.load(f)
teams = {t["id"]: t for t in teams_raw}

print("=== UPDATING ROUND OF 16 RESULTS ===\n")

# ──────────────────────────────────────────────
# 3. Update matches.json with real results
# ──────────────────────────────────────────────
for match in matches:
    mid = match["id"]
    if mid in REAL_R16:
        entry = REAL_R16[mid]
        s1, s2, status = entry[0], entry[1], entry[2]
        match["score"] = {"team1": str(s1), "team2": str(s2)}
        match["status"] = status
        if len(entry) > 3:
            match["score"]["penalties"] = entry[3]
        
        t1_name = teams.get(match["team1_id"], {}).get("name", "?")
        t2_name = teams.get(match["team2_id"], {}).get("name", "?")
        print(f"  #{mid} {t1_name} {s1}-{s2} {t2_name} [{status}]")

# Save updated matches.json
with open(DATA_DIR / "matches.json", "w") as f:
    json.dump(matches, f, indent=2, ensure_ascii=False)
print("\n✓ matches.json updated\n")

# ──────────────────────────────────────────────
# 4. Recalculate ELO for all completed matches
# ──────────────────────────────────────────────
print("=== RECALCULATING ELO RATINGS ===\n")

# Reset ALL ELO to initial values (from teams.json base)
# Actually, we need to track what the initial ELO was BEFORE recalibration.
# The original model was already trained on matches with their ELO values at the time.
# For proper recalibration, we need to re-run ELO updates through ALL completed matches
# using the initial ELO from teams.json as starting point.

# Get initial ELO from teams.json
initial_elo = {}
for t in teams_raw:
    initial_elo[t["id"]] = t["elo_rating"]
    
# Get ALL completed matches in chronological order
all_completed = sorted(
    [m for m in matches if m.get("score") and m.get("status", "").startswith("completed")],
    key=lambda m: m["date"]
)

print(f"Processing {len(all_completed)} completed matches for ELO update...")

# Current ELO tracker (starts at initial, gets updated match by match)
current_elo = dict(initial_elo)

elo_updates = 0
for m in all_completed:
    t1_id = m["team1_id"]
    t2_id = m["team2_id"]
    if t1_id is None or t2_id is None:
        continue
    
    t1 = teams.get(t1_id, {})
    t2 = teams.get(t2_id, {})
    
    elo1 = current_elo.get(t1_id, t1.get("elo_rating", 1500))
    elo2 = current_elo.get(t2_id, t2.get("elo_rating", 1500))
    
    score = m["score"]
    s1 = int(score["team1"])
    s2 = int(score["team2"])
    
    # Determine winner (handle penalties)
    t1_won = s1 > s2
    if m.get("status", "").endswith("_penalties"):
        pen = score.get("penalties", "")
        if pen:
            try:
                p1, p2 = map(int, pen.split("-"))
                t1_won = p1 > p2
            except:
                pass
    
    # ELO result from team1's perspective
    if s1 > s2 or (s1 == s2 and t1_won):
        result = "W"
    elif s1 == s2:
        result = "D"
    else:
        result = "L"
    
    t1_is_home = t1.get("is_host", False)
    new_elo1, new_elo2 = calculate_new_ratings(elo1, elo2, result, t1_is_home)
    
    current_elo[t1_id] = new_elo1
    current_elo[t2_id] = new_elo2
    elo_updates += 1

print(f"  Updated {elo_updates} match ELO ratings")

# Write updated ELO to teams.json
for t in teams_raw:
    new_elo = current_elo.get(t["id"], t["elo_rating"])
    old_elo = t["elo_rating"]
    t["elo_rating"] = round(new_elo)
    if abs(old_elo - t["elo_rating"]) >= 5:
        diff = t['elo_rating'] - old_elo
        print(f"  {t['name']}: {old_elo} → {t['elo_rating']} ({diff:+.0f})")

with open(DATA_DIR / "teams.json", "w") as f:
    json.dump(teams_raw, f, indent=2, ensure_ascii=False)
print("\n✓ teams.json updated with recalculated ELO\n")

# Refresh teams dict
teams = {t["id"]: t for t in teams_raw}

# ──────────────────────────────────────────────
# 5. Build training dataset with updated data
# ──────────────────────────────────────────────
print("=== BUILDING TRAINING DATASET ===\n")
X, y, feature_names = build_dataset()

# ──────────────────────────────────────────────
# 6. Retrain models
# ──────────────────────────────────────────────
print("\n=== TRAINING XGBOOST ===\n")
xgb = KnockoutPredictor()
xgb.train(X, y, feature_names)

print("\n=== TRAINING LOGISTIC REGRESSION ===\n")
logreg = LogisticPredictor()
logreg.train(X, y, feature_names)

# ──────────────────────────────────────────────
# 7. Regenerate teams_enriched.json
# ──────────────────────────────────────────────
print("\n=== REGENERATING TEAMS ENRICHED ===\n")
teams_enriched = []
for t in teams_raw:
    enriched = dict(t)
    enriched["performance"] = calculate_performance_rating(
        t["elo_rating"], t.get("recent_form", [])
    )
    teams_enriched.append(enriched)

with open(DATA_DIR / "teams_enriched.json", "w") as f:
    json.dump(teams_enriched, f, indent=2, ensure_ascii=False)
print("✓ teams_enriched.json regenerated\n")

# ──────────────────────────────────────────────
# 8. Regenerate matches_enriched.json
# ──────────────────────────────────────────────
print("=== REGENERATING MATCHES ENRICHED ===\n")

# Rebuild team history for match enrichment
team_history = build_team_history(matches, teams)

# Enrich all matches
xgb_count = 0
for m in matches:
    t1_id = m.get("team1_id")
    t2_id = m.get("team2_id")
    if t1_id is None or t2_id is None:
        continue
    t1 = teams.get(t1_id, {})
    t2 = teams.get(t2_id, {})
    if not t1 or not t2:
        continue

    m["team1_name"] = t1["name"]
    m["team1_flag"] = t1["flag_emoji"]
    m["team2_name"] = t2["name"]
    m["team2_flag"] = t2["flag_emoji"]

    # Poisson baseline prediction
    probs = calculate_match_probabilities(
        t1["elo_rating"], t2["elo_rating"],
        t1.get("is_host", False), t2.get("is_host", False))
    factors = get_prediction_factors(t1, t2)
    m["prediction"] = {**probs, "key_factors": factors}

    # XGBoost dual-prediction for knockout matches
    stage = m.get("stage", "group")
    if stage != "group":
        t1_hist = team_history.get(t1_id, [])
        t2_hist = team_history.get(t2_id, [])
        try:
            fwd = compute_match_features(
                t1_id, t2_id, t1, t2, t1_hist, t2_hist,
                stage, m.get("date", "")
            )
            rev = compute_match_features(
                t2_id, t1_id, t2, t1, t2_hist, t1_hist,
                stage, m.get("date", "")
            )
            prob_fwd = xgb.predict(fwd)
            prob_rev = xgb.predict(rev)
            prob_sym = round((prob_fwd + (1.0 - prob_rev)) / 2.0, 4)
            m["xgb_prediction"] = {
                "team1_advance_prob": round(prob_sym * 100, 1),
                "team2_advance_prob": round((1 - prob_sym) * 100, 1),
                "model": "XGBoost (symmetric dual-prediction)",
            }
            xgb_count += 1
        except Exception as e:
            print(f"  WARN: XGB for {t1['name']} vs {t2['name']}: {e}")

with open(DATA_DIR / "matches_enriched.json", "w") as f:
    json.dump(matches, f, indent=2, ensure_ascii=False)
print(f"✓ matches_enriched.json regenerated ({xgb_count} knockout matches with XGBoost)\n")

# ──────────────────────────────────────────────
# 9. Print quarter-final predictions
# ──────────────────────────────────────────────
print("=" * 60)
print("QUARTER-FINAL PREDICTIONS")
print("=" * 60)

# QF matchups based on real R16 results + bracket structure
# From Wikipedia: the bracket shows:
# Match 97 (QF1): France vs Morocco  (winner #89 vs winner #88)
# Match 98 (QF2): Spain vs Belgium   (winner #92 vs winner #93)
# Match 99 (QF3): Norway vs England  (winner #90 vs winner #91)
# Match 100 (QF4): Argentina vs Switzerland (winner #94 vs winner #95)

qf_matchups = [
    (32, 9, "France", "Morocco"),       # #89 winner vs #88 winner
    (28, 24, "Spain", "Belgium"),        # #92 winner vs #93 winner
    (35, 44, "Norway", "England"),       # #90 winner vs #91 winner
    (36, 7, "Argentina", "Switzerland"), # #94 winner vs #95 winner
]

for t1_id, t2_id, t1_name, t2_name in qf_matchups:
    t1 = teams.get(t1_id, {})
    t2 = teams.get(t2_id, {})
    
    # Poisson
    probs = calculate_match_probabilities(
        t1["elo_rating"], t2["elo_rating"],
        t1.get("is_host", False), t2.get("is_host", False))
    
    # XGBoost
    t1_hist = team_history.get(t1_id, [])
    t2_hist = team_history.get(t2_id, [])
    fwd = compute_match_features(t1_id, t2_id, t1, t2, t1_hist, t2_hist, "quarter_finals", "2026-07-09")
    rev = compute_match_features(t2_id, t1_id, t2, t1, t2_hist, t1_hist, "quarter_finals", "2026-07-09")
    prob_fwd = xgb.predict(fwd)
    prob_rev = xgb.predict(rev)
    prob_sym = round((prob_fwd + (1.0 - prob_rev)) / 2.0, 4)
    
    print(f"\n{t1_name} ({t1['elo_rating']}) vs {t2_name} ({t2['elo_rating']})")
    print(f"  Poisson:     {t1_name} {probs['team1_win_prob']:.1f}% | Draw {probs['draw_prob']:.1f}% | {t2_name} {probs['team2_win_prob']:.1f}%")
    print(f"  Predicted:   {probs['predicted_score']}")
    print(f"  XGBoost:     {t1_name} {prob_sym*100:.1f}% | {t2_name} {(1-prob_sym)*100:.1f}%")
    print(f"  Key factors: {', '.join(factors)}")

print("\n✓ Done! Models retrained with real R16 results.")
print("  Updated files: matches.json, teams.json, teams_enriched.json, matches_enriched.json")
print("  Retrained: xgboost_v1.json, logistic_v1.pkl")
