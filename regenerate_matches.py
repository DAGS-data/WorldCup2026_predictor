#!/usr/bin/env python3
"""Regenerate matches_enriched.json with symmetric XGBoost predictions.

Dual-prediction symmetrization: predict from BOTH perspectives and average,
eliminating the order bias where swapping team1/team2 gave different results.
"""
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from models.xgboost_predictor import KnockoutPredictor
from feature_engineering import build_team_history, compute_match_features
from predictor import calculate_match_probabilities, get_prediction_factors

DATA_DIR = Path(__file__).parent / "backend" / "data"

# Load data
with open(DATA_DIR / "teams.json") as f:
    teams_raw = json.load(f)
teams = {t["id"]: t for t in teams_raw}

with open(DATA_DIR / "matches.json") as f:
    matches = json.load(f)

# Build per-team history chronologically
team_history = build_team_history(matches, teams)

# Init XGBoost predictor
xgb = KnockoutPredictor()
print(f"Loaded: {xgb.get_metrics()}")

count = 0
for m in matches:
    t1_id = m.get("team1_id")
    t2_id = m.get("team2_id")
    if t1_id is None or t2_id is None:
        continue
    t1 = teams.get(t1_id)
    t2 = teams.get(t2_id)
    if not t1 or not t2:
        continue

    m["team1_name"] = t1["name"]
    m["team1_flag"] = t1["flag_emoji"]
    m["team2_name"] = t2["name"]
    m["team2_flag"] = t2["flag_emoji"]

    # Poisson baseline
    probs = calculate_match_probabilities(
        t1["elo_rating"], t2["elo_rating"],
        t1.get("is_host", False), t2.get("is_host", False))
    factors = get_prediction_factors(t1, t2)
    m["prediction"] = {**probs, "key_factors": factors}

    # XGBoost dual-prediction symmetrization (knockout only)
    stage = m.get("stage", "group")
    if stage != "group":
        t1_hist = team_history.get(t1_id, [])
        t2_hist = team_history.get(t2_id, [])

        try:
            fwd = compute_match_features(
                t1_id, t2_id, t1, t2, t1_hist, t2_hist,
                stage, m.get("date", "2026-07-04")
            )
            rev = compute_match_features(
                t2_id, t1_id, t2, t1, t2_hist, t1_hist,
                stage, m.get("date", "2026-07-04")
            )
            prob_fwd = xgb.predict(fwd)
            prob_rev = xgb.predict(rev)
            prob_sym = round((prob_fwd + (1.0 - prob_rev)) / 2.0, 4)

            m["xgb_prediction"] = {
                "team1_advance_prob": round(prob_sym * 100, 1),
                "team2_advance_prob": round((1 - prob_sym) * 100, 1),
                "model": "XGBoost (symmetric dual-prediction)",
            }
            count += 1
        except Exception as e:
            print(f"  WARN: XGBoost failed for {t1['name']} vs {t2['name']}: {e}")

# Save
out_path = DATA_DIR / "matches_enriched.json"
with open(out_path, "w") as f:
    json.dump(matches, f, indent=2, ensure_ascii=False)

print(f"\nDone. {count} matches with symmetric XGBoost predictions.")
print(f"Saved to {out_path}")
