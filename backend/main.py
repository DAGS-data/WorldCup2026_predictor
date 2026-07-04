"""World Cup 2026 Predictor — FastAPI backend."""

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from performance import calculate_performance_rating
from predictor import calculate_match_probabilities, get_prediction_factors, simulate_tournament
from models.xgboost_predictor import KnockoutPredictor

# Init XGBoost predictor (lazy load)
_xgb_predictor = None

def get_xgb_predictor():
    global _xgb_predictor
    if _xgb_predictor is None:
        _xgb_predictor = KnockoutPredictor()
    return _xgb_predictor

DATA_DIR = Path(__file__).parent / "data"

app = FastAPI(title="WC 2026 Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_teams() -> list[dict]:
    with open(DATA_DIR / "teams.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_matches() -> list[dict]:
    with open(DATA_DIR / "matches.json") as f:
        return json.load(f)


def enrich_team(team: dict) -> dict:
    """Add computed performance rating to a team dict."""
    perf = calculate_performance_rating(team["elo_rating"], team.get("recent_form", []))
    return {**team, "performance": perf}


def enrich_match(match: dict, teams: list[dict]) -> dict:
    """Add prediction data to a match dict."""
    t1 = next((t for t in teams if t["id"] == match["team1_id"]), None)
    t2 = next((t for t in teams if t["id"] == match["team2_id"]), None)
    if not t1 or not t2:
        return match

    probs = calculate_match_probabilities(
        t1["elo_rating"],
        t2["elo_rating"],
        t1.get("is_host", False),
        t2.get("is_host", False),
    )
    factors = get_prediction_factors(t1, t2)

    return {
        **match,
        "team1_name": t1["name"],
        "team1_flag": t1["flag_emoji"],
        "team2_name": t2["name"],
        "team2_flag": t2["flag_emoji"],
        "prediction": {**probs, "key_factors": factors},
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/teams")
def get_teams():
    teams = load_teams()
    return [enrich_team(t) for t in teams]


@app.get("/api/teams/{team_id}")
def get_team(team_id: int):
    teams = load_teams()
    team = next((t for t in teams if t["id"] == team_id), None)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return enrich_team(team)


@app.get("/api/matches")
def get_matches():
    teams = load_teams()
    matches = load_matches()
    return [enrich_match(m, teams) for m in matches]


@app.get("/api/matches/{match_id}")
def get_match(match_id: int):
    teams = load_teams()
    matches = load_matches()
    match = next((m for m in matches if m["id"] == match_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return enrich_match(match, teams)


@app.get("/api/bracket")
def get_bracket():
    teams = load_teams()
    matches = load_matches()

    teams_by_group: dict[str, list[dict]] = {}
    for team in teams:
        g = team["group"]
        teams_by_group.setdefault(g, []).append(team)

    return simulate_tournament(teams_by_group, matches)


@app.get("/api/groups")
def get_groups():
    teams = load_teams()
    matches = load_matches()
    groups: dict[str, dict] = {}

    for team in teams:
        g = team["group"]
        if g not in groups:
            groups[g] = {"group": g, "teams": [], "matches": []}
        groups[g]["teams"].append(enrich_team(team))

    for match in matches:
        g = match.get("group", "")
        if g in groups:
            groups[g]["matches"].append(enrich_match(match, teams))

    return sorted(groups.values(), key=lambda x: x["group"])


@app.get("/api/predict")
def predict_match(team1_id: int, team2_id: int):
    teams = load_teams()
    t1 = next((t for t in teams if t["id"] == team1_id), None)
    t2 = next((t for t in teams if t["id"] == team2_id), None)
    if not t1 or not t2:
        raise HTTPException(status_code=404, detail="Team not found")

    probs = calculate_match_probabilities(
        t1["elo_rating"],
        t2["elo_rating"],
        t1.get("is_host", False),
        t2.get("is_host", False),
    )
    factors = get_prediction_factors(t1, t2)
    perf1 = calculate_performance_rating(t1["elo_rating"], t1.get("recent_form", []))
    perf2 = calculate_performance_rating(t2["elo_rating"], t2.get("recent_form", []))

    return {
        "team1": {**t1, "performance": perf1},
        "team2": {**t2, "performance": perf2},
        "prediction": {**probs, "key_factors": factors},
    }


@app.get("/api/retrain")
def retrain():
    """Invalidate caches and reload data (simulates model retraining)."""
    load_teams.cache_clear()
    load_matches.cache_clear()
    return {"status": "ok", "message": "Model data reloaded successfully"}


@app.get("/api/predict/v2")
def predict_match_v2(team1_id: int, team2_id: int):
    """
    XGBoost-powered prediction: P(team advances) with 50+ features.
    Returns advancement probability + SHAP explanations.
    """
    teams = load_teams()
    t1 = next((t for t in teams if t["id"] == team1_id), None)
    t2 = next((t for t in teams if t["id"] == team2_id), None)
    if not t1 or not t2:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # Get historical data for features
    matches = load_matches()
    
    # Build team histories from completed matches
    from collections import defaultdict
    from feature_engineering import compute_match_features
    
    t1_hist = []
    t2_hist = []
    
    for m in sorted(
        [m for m in matches if m.get("score") and m.get("status", "").startswith("completed")],
        key=lambda x: x["date"]
    ):
        if m.get("team1_id") == team1_id or m.get("team2_id") == team1_id:
            role = "home" if m["team1_id"] == team1_id else "away"
            t1_hist.append({
                "score": m["score"], "stage": m.get("stage","group"),
                "date": m["date"], "team_role": role,
                "team_elo": t1["elo_rating"], "opponent_elo": t2["elo_rating"],
                "status": m["status"], "clean_sheet": int(m["score"]["team2" if role=="home" else "team1"]) == 0,
                "comeback": False, "advanced": False,
            })
        if m.get("team1_id") == team2_id or m.get("team2_id") == team2_id:
            role = "home" if m["team1_id"] == team2_id else "away"
            t2_hist.append({
                "score": m["score"], "stage": m.get("stage","group"),
                "date": m["date"], "team_role": role,
                "team_elo": t2["elo_rating"], "opponent_elo": t1["elo_rating"],
                "status": m["status"], "clean_sheet": int(m["score"]["team2" if role=="home" else "team1"]) == 0,
                "comeback": False, "advanced": False,
            })
    
    # Compute features
    features = compute_match_features(
        team1_id, team2_id, t1, t2,
        t1_hist, t2_hist,
        "round_of_16", "2026-07-04"  # stage, date
    )
    
    # Predict
    try:
        xgb = get_xgb_predictor()
        prob = xgb.predict(features)
        explanation = xgb.explain(features)
    except Exception as e:
        prob = None
        explanation = {"error": str(e)}
    
    # Also get Poisson baseline for comparison
    poisson = calculate_match_probabilities(
        t1["elo_rating"], t2["elo_rating"],
        t1.get("is_host", False), t2.get("is_host", False)
    )
    
    return {
        "team1": {"name": t1["name"], "flag": t1["flag_emoji"], "elo": t1["elo_rating"], "fifa_rank": t1.get("fifa_rank")},
        "team2": {"name": t2["name"], "flag": t2["flag_emoji"], "elo": t2["elo_rating"], "fifa_rank": t2.get("fifa_rank")},
        "v2_prediction": {
            "team1_advance_prob": prob,
            "model": "XGBoost (50+ features, Optuna-tuned)",
            "features_used": len(features),
        },
        "v1_baseline": {
            "team1_win_prob": poisson["team1_win_prob"],
            "draw_prob": poisson["draw_prob"],
            "team2_win_prob": poisson["team2_win_prob"],
            "model": "ELO + Poisson",
        },
        "explanation": explanation,
    }


@app.get("/")
def root():
    return {"message": "WC 2026 Predictor API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
