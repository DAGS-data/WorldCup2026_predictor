"""World Cup 2026 Predictor — FastAPI backend."""

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from performance import calculate_performance_rating
from predictor import calculate_match_probabilities, get_prediction_factors, simulate_tournament

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


@app.get("/")
def root():
    return {"message": "WC 2026 Predictor API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
