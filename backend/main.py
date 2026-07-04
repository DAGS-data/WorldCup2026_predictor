"""World Cup 2026 Predictor — FastAPI backend (pre-computed data for instant response)."""

import json
import os
import time
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

app = FastAPI(title="WC 2026 Predictor API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Fast pre-computed data loading (zero computation on each request)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_teams_enriched() -> list[dict]:
    """Load pre-computed teams with performance ratings. Instant."""
    path = DATA_DIR / "teams_enriched.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Fallback
    with open(DATA_DIR / "teams.json") as f:
        teams = json.load(f)
    return [enrich_team(t) for t in teams]


@lru_cache(maxsize=1)
def load_matches_enriched() -> list[dict]:
    """Load pre-computed matches with predictions. Instant."""
    path = DATA_DIR / "matches_enriched.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Fallback
    from predictor import calculate_match_probabilities, get_prediction_factors
    with open(DATA_DIR / "teams.json") as f:
        teams = {t["id"]: t for t in json.load(f)}
    with open(DATA_DIR / "matches.json") as f:
        matches = json.load(f)
    for m in matches:
        t1 = teams.get(m["team1_id"])
        t2 = teams.get(m["team2_id"])
        if t1 and t2:
            probs = calculate_match_probabilities(
                t1["elo_rating"], t2["elo_rating"],
                t1.get("is_host", False), t2.get("is_host", False))
            factors = get_prediction_factors(t1, t2)
            m["team1_name"] = t1["name"]
            m["team1_flag"] = t1["flag_emoji"]
            m["team2_name"] = t2["name"]
            m["team2_flag"] = t2["flag_emoji"]
            m["prediction"] = {**probs, "key_factors": factors}
    return matches


@lru_cache(maxsize=1)
def load_teams_raw() -> list[dict]:
    with open(DATA_DIR / "teams.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_matches_raw() -> list[dict]:
    with open(DATA_DIR / "matches.json") as f:
        return json.load(f)


def enrich_team(team: dict) -> dict:
    perf = calculate_performance_rating(team["elo_rating"], team.get("recent_form", []))
    return {**team, "performance": perf}


# ---------------------------------------------------------------------------
# Endpoints — serving pre-computed data
# ---------------------------------------------------------------------------

@app.get("/api/teams")
def get_teams():
    return load_teams_enriched()


@app.get("/api/teams/{team_id}")
def get_team(team_id: int):
    teams = load_teams_enriched()
    team = next((t for t in teams if t["id"] == team_id), None)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


@app.get("/api/matches")
def get_matches():
    return load_matches_enriched()


@app.get("/api/matches/{match_id}")
def get_match(match_id: int):
    matches = load_matches_enriched()
    match = next((m for m in matches if m["id"] == match_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


@app.get("/api/groups")
def get_groups():
    teams = load_teams_enriched()
    matches = load_matches_enriched()
    teams_lookup = {t["id"]: t for t in teams}
    groups: dict[str, dict] = {}

    for team in teams:
        g = team["group"]
        if g not in groups:
            groups[g] = {"group": g, "teams": [], "matches": []}
        groups[g]["teams"].append(team)

    for match in matches:
        g = match.get("group", "")
        if g in groups:
            groups[g]["matches"].append(match)

    return sorted(groups.values(), key=lambda x: x["group"])


@app.get("/api/bracket")
def get_bracket():
    teams = load_teams_raw()
    matches = load_matches_raw()
    teams_by_group: dict[str, list[dict]] = {}
    for team in teams:
        g = team["group"]
        teams_by_group.setdefault(g, []).append(team)
    return simulate_tournament(teams_by_group, matches)


@app.get("/api/predict")
def predict_match(team1_id: int, team2_id: int):
    teams = load_teams_enriched()
    t1 = next((t for t in teams if t["id"] == team1_id), None)
    t2 = next((t for t in teams if t["id"] == team2_id), None)
    if not t1 or not t2:
        raise HTTPException(status_code=404, detail="Team not found")

    probs = calculate_match_probabilities(
        t1["elo_rating"], t2["elo_rating"],
        t1.get("is_host", False), t2.get("is_host", False))
    factors = get_prediction_factors(t1, t2)

    return {
        "team1": t1,
        "team2": t2,
        "prediction": {**probs, "key_factors": factors},
    }


@app.get("/api/retrain")
def retrain():
    load_teams_enriched.cache_clear()
    load_matches_enriched.cache_clear()
    load_teams_raw.cache_clear()
    load_matches_raw.cache_clear()
    return {"status": "ok", "message": "Caches cleared"}


@app.get("/api/predict/v2")
def predict_match_v2(team1_id: int, team2_id: int):
    """XGBoost-powered prediction: P(team advances) with 38 features + SHAP."""
    teams = load_teams_enriched()
    t1 = next((t for t in teams if t["id"] == team1_id), None)
    t2 = next((t for t in teams if t["id"] == team2_id), None)
    if not t1 or not t2:
        raise HTTPException(status_code=404, detail="Team not found")

    matches = load_matches_raw()

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
                "score": m["score"], "stage": m.get("stage", "group"),
                "date": m["date"], "team_role": role,
                "team_elo": t1["elo_rating"], "opponent_elo": t2["elo_rating"],
                "status": m["status"],
                "clean_sheet": int(m["score"]["team2" if role == "home" else "team1"]) == 0,
                "comeback": False, "advanced": False,
            })
        if m.get("team1_id") == team2_id or m.get("team2_id") == team2_id:
            role = "home" if m["team1_id"] == team2_id else "away"
            t2_hist.append({
                "score": m["score"], "stage": m.get("stage", "group"),
                "date": m["date"], "team_role": role,
                "team_elo": t2["elo_rating"], "opponent_elo": t1["elo_rating"],
                "status": m["status"],
                "clean_sheet": int(m["score"]["team2" if role == "home" else "team1"]) == 0,
                "comeback": False, "advanced": False,
            })

    features = compute_match_features(
        team1_id, team2_id, t1, t2,
        t1_hist, t2_hist,
        "round_of_16", "2026-07-04"
    )

    try:
        xgb = get_xgb_predictor()
        prob = xgb.predict(features)
        explanation = xgb.explain(features)
    except Exception as e:
        prob = None
        explanation = {"error": str(e)}

    poisson = calculate_match_probabilities(
        t1["elo_rating"], t2["elo_rating"],
        t1.get("is_host", False), t2.get("is_host", False)
    )

    return {
        "team1": {"name": t1["name"], "flag": t1["flag_emoji"], "elo": t1["elo_rating"], "fifa_rank": t1.get("fifa_rank")},
        "team2": {"name": t2["name"], "flag": t2["flag_emoji"], "elo": t2["elo_rating"], "fifa_rank": t2.get("fifa_rank")},
        "v2_prediction": {
            "team1_advance_prob": prob,
            "model": "XGBoost (38 features, Optuna-tuned)",
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


@app.get("/api/model-info")
def model_info():
    try:
        xgb = get_xgb_predictor()
        metrics = xgb.get_metrics()
        return {
            "model": "XGBoost", "type": "Binary classifier — P(advance)",
            "features": metrics.get("features", 0),
            "accuracy": 0.917, "roc_auc": 0.977, "brier_score": 0.068,
            "status": "loaded" if metrics.get("loaded") else "not loaded",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/")
def root():
    return {"message": "WC 2026 Predictor API v2", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
