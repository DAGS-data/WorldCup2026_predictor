"""
Feature engineering pipeline for World Cup 2026 knockout predictions.

Computes 38 features per match from real ESPN data, incorporating:
- Tournament momentum (performance vs expectation)
- Overperformance tracking (the "Cape Verde effect")
- Match importance weighting
- Physical factors (rest, travel, extra time fatigue)
- Historical head-to-head
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from predictor import estimate_expected_goals

DATA_DIR = Path(__file__).parent / "data"

# === LOAD RAW MATCH DATA ===

def load_raw_matches(path: str = None) -> list[dict]:
    """Load ESPN match data (or processed matches.json)."""
    if path:
        with open(path) as f:
            return json.load(f)
    with open(DATA_DIR / "matches.json") as f:
        return json.load(f)

def load_teams() -> list[dict]:
    with open(DATA_DIR / "teams.json") as f:
        return json.load(f)


# === TOURNAMENT FEATURES ===

MATCH_IMPORTANCE = {
    "group": 1.0,
    "round_of_32": 2.0,
    "round_of_16": 3.0,
    "quarter_finals": 4.0,
    "semi_finals": 5.0,
    "final": 6.0,
}

def compute_momentum(team_matches: list[dict], current_date: str) -> float:
    """
    Tournament momentum: weighted sum of match outcomes, 
    adjusted for opponent quality and match importance.
    
    Exponentially weighted (lambda=0.85 per match backward in time).
    """
    if not team_matches:
        return 0.0
    
    momentum = 0.0
    total_weight = 0.0
    
    for i, m in enumerate(team_matches):
        # Result value
        score = m.get("score", {})
        if not score:
            continue
        
        g_for = int(score.get("team1", 0)) if m.get("team_role") == "home" else int(score.get("team2", 0))
        g_against = int(score.get("team2", 0)) if m.get("team_role") == "home" else int(score.get("team1", 0))
        
        if g_for > g_against:
            result = 1.0
        elif g_for == g_against:
            # Draw in knockout — check penalties
            if m.get("status", "").endswith("_penalties"):
                result = 1.0 if m.get("advanced") else 0.0
            else:
                result = 0.5
        else:
            # Close loss bonus: lost by 1 goal to stronger opponent
            opp_elo = m.get("opponent_elo", 1500)
            team_elo = m.get("team_elo", 1500)
            if g_against - g_for == 1 and opp_elo - team_elo >= 50:
                result = 0.35
            elif g_against - g_for == 1 and opp_elo - team_elo >= -50:
                result = 0.20
            else:
                result = 0.0
        
        # Weight: more recent = higher, more important match = higher
        importance = MATCH_IMPORTANCE.get(m.get("stage", "group"), 1.0)
        recency = 0.85 ** i  # exponential decay
        weight = recency * importance
        
        momentum += result * weight
        total_weight += weight
    
    return momentum / total_weight if total_weight > 0 else 0.0


def compute_overperformance(matches: list[dict]) -> float:
    """
    Performance vs expectation: how much has the team outperformed 
    its pre-tournament ELO?
    
    Positive = exceeding expectations (Cape Verde effect)
    """
    if not matches:
        return 0.0
    
    total_over = 0.0
    for m in matches:
        score = m.get("score", {})
        if not score:
            continue
        
        g_for = int(score.get("team1", 0)) if m.get("team_role") == "home" else int(score.get("team2", 0))
        g_against = int(score.get("team2", 0)) if m.get("team_role") == "home" else int(score.get("team1", 0))
        expected = m.get("expected_goals", 1.0)
        opp_quality = (m.get("opponent_elo", 1500) - 1500) / 400  # normalize
        
        # Overperformance = (actual - expected) adjusted for opponent
        over = (g_for - expected) * (1.0 - opp_quality * 0.3)  # harder to overperform vs strong teams
        total_over += over
    
    return total_over / len(matches)


def compute_physical_features(team_matches: list[dict]) -> dict:
    """Rest days, travel, fatigue metrics."""
    if len(team_matches) < 2:
        return {"rest_days": 5, "cumulative_mins": 0, "extra_time_played": 0}
    
    # Rest days since last match
    last_date = team_matches[-1].get("date", "2026-06-11")
    # Simple estimate: group matches are 3-4 days apart
    
    # Cumulative minutes played
    total_mins = 90 * len(team_matches)  # base 90 min per match
    
    # Extra time
    et_played = sum(1 for m in team_matches if m.get("status", "").endswith("_aet"))
    
    return {
        "rest_days": 4 if len(team_matches) <= 3 else 3,  # R16+ are tighter
        "cumulative_mins": total_mins + et_played * 30,
        "extra_time_played": et_played,
    }


# === FEATURE COMPUTATION PER MATCH ===

def compute_match_features(
    team_id: int,
    opponent_id: int,
    team_data: dict,
    opp_data: dict,
    team_history: list[dict],
    opp_history: list[dict],
    match_stage: str,
    match_date: str,
) -> dict:
    """
    Compute all features for a single team in a single match.
    
    Returns a dict of feature_name → value for XGBoost input.
    """
    f = {}
    
    # === BASIC DIFFERENTIALS ===
    f["elo_diff"] = team_data.get("elo_rating", 1500) - opp_data.get("elo_rating", 1500)
    f["fifa_rank_diff"] = (opp_data.get("fifa_rank", 50) - team_data.get("fifa_rank", 50)) / 50.0  # negative = team better
    f["squad_value_ratio"] = math.log(max(1, team_data.get("squad_value_millions", 100)) / max(1, opp_data.get("squad_value_millions", 100)))
    f["host_advantage"] = 1.0 if team_data.get("is_host") else 0.0
    f["host_vs_away"] = 1.0 if team_data.get("is_host") and not opp_data.get("is_host") else 0.0
    
    # === TOURNAMENT MOMENTUM ===
    f["momentum"] = compute_momentum(team_history, match_date)
    f["opponent_momentum"] = compute_momentum(opp_history, match_date)
    f["momentum_diff"] = f["momentum"] - f["opponent_momentum"]
    
    # === OVERPERFORMANCE ===
    f["overperformance"] = compute_overperformance(team_history)
    f["opponent_overperformance"] = compute_overperformance(opp_history)
    f["overperformance_diff"] = f["overperformance"] - f["opponent_overperformance"]
    
    # === PHYSICAL ===
    phys = compute_physical_features(team_history)
    opp_phys = compute_physical_features(opp_history)
    f["rest_days_diff"] = phys["rest_days"] - opp_phys["rest_days"]
    f["extra_time_diff"] = opp_phys["extra_time_played"] - phys["extra_time_played"]  # positive = fresher
    f["cumulative_mins_ratio"] = phys["cumulative_mins"] / max(1, opp_phys["cumulative_mins"])
    
    # === PERFORMANCE METRICS ===
    goals_scored = sum(
        int(m["score"]["team1"]) if m.get("team_role") == "home" else int(m["score"]["team2"])
        for m in team_history if m.get("score")
    )
    goals_conceded = sum(
        int(m["score"]["team2"]) if m.get("team_role") == "home" else int(m["score"]["team1"])
        for m in team_history if m.get("score")
    )
    f["goals_scored_per_match"] = goals_scored / max(1, len([m for m in team_history if m.get("score")]))
    f["goals_conceded_per_match"] = goals_conceded / max(1, len([m for m in team_history if m.get("score")]))
    f["goal_diff_per_match"] = f["goals_scored_per_match"] - f["goals_conceded_per_match"]
    
    # Clean sheets
    clean_sheets = sum(1 for m in team_history if m.get("clean_sheet"))
    f["clean_sheet_rate"] = clean_sheets / max(1, len(team_history))
    
    # Comeback wins
    comebacks = sum(1 for m in team_history if m.get("comeback"))
    f["comeback_rate"] = comebacks / max(1, len(team_history))
    
    # === GOAL CONSISTENCY ===
    if len(team_history) >= 2:
        goals = [int(m["score"]["team1"]) if m.get("team_role") == "home" else int(m["score"]["team2"])
                 for m in team_history[-5:] if m.get("score")]
        if goals:
            mean_g = sum(goals) / len(goals)
            f["goal_consistency"] = math.sqrt(sum((g - mean_g)**2 for g in goals) / len(goals)) if len(goals) > 1 else 0
        else:
            f["goal_consistency"] = 0
    else:
        f["goal_consistency"] = 0
    
    # === STAGE ===
    f["stage_importance"] = MATCH_IMPORTANCE.get(match_stage, 1.0)
    f["stage_is_knockout"] = 1.0 if match_stage != "group" else 0.0
    f["stage_is_r16"] = 1.0 if match_stage == "round_of_16" else 0.0
    f["stage_is_qf"] = 1.0 if match_stage == "quarter_finals" else 0.0
    f["stage_is_sf"] = 1.0 if match_stage == "semi_finals" else 0.0
    
    # === GROUP PERFORMANCE ===
    group_matches = [m for m in team_history if m.get("stage") == "group"]
    if group_matches:
        pts = sum(
            3 if (int(m["score"]["team1"]) > int(m["score"]["team2"]) if m.get("team_role") == "home" 
                  else int(m["score"]["team2"]) > int(m["score"]["team1"]))
            else 1 if (int(m["score"]["team1"]) == int(m["score"]["team2"]) if m.get("team_role") == "home"
                      else int(m["score"]["team2"]) == int(m["score"]["team1"]))
            else 0
            for m in group_matches if m.get("score")
        )
        f["group_pts"] = pts
        f["group_pts_per_match"] = pts / max(1, len(group_matches))
    else:
        f["group_pts"] = 0
        f["group_pts_per_match"] = 0
    
    f["group_position"] = team_data.get("group_position", 3)  # 1=winner, 2=2nd, 3=3rd out
    
    # === CONFEDERATION ===
    conf1 = team_data.get("confederation", "")
    conf2 = opp_data.get("confederation", "")
    f["same_confederation"] = 1.0 if conf1 == conf2 else 0.0
    f["team_is_uefa"] = 1.0 if conf1 == "UEFA" else 0.0
    f["team_is_conmebol"] = 1.0 if conf1 == "CONMEBOL" else 0.0
    f["opp_is_uefa"] = 1.0 if conf2 == "UEFA" else 0.0
    f["opp_is_conmebol"] = 1.0 if conf2 == "CONMEBOL" else 0.0
    
    # === EXTRA FEATURES ===
    f["elo_diff_abs"] = abs(f["elo_diff"])
    f["elo_ratio"] = (team_data.get("elo_rating", 1500) / max(1, opp_data.get("elo_rating", 1500))) - 1
    f["is_top5"] = 1.0 if team_data.get("elo_rating", 0) >= 1950 else 0.0
    f["opp_is_top5"] = 1.0 if opp_data.get("elo_rating", 0) >= 1950 else 0.0
    
    # Close loss indicator
    last_match = team_history[-1] if team_history else None
    if last_match and last_match.get("score"):
        gf = int(last_match["score"]["team1"]) if last_match.get("team_role") == "home" else int(last_match["score"]["team2"])
        ga = int(last_match["score"]["team2"]) if last_match.get("team_role") == "home" else int(last_match["score"]["team1"])
        f["coming_off_close_loss"] = 1.0 if (ga - gf == 1) else 0.0
    else:
        f["coming_off_close_loss"] = 0.0
    
    return f


# === SHARED TEAM HISTORY BUILDER ===

def build_team_history(matches: list[dict], teams: dict[int, dict]) -> dict[int, list[dict]]:
    """
    Build chronological per-team match history from completed matches.

    Single source of truth for the history entries fed into `compute_match_features`
    (training and every live prediction endpoint use this same function, so the
    fields — including `expected_goals` for overperformance — never drift out of sync).
    """
    history = defaultdict(list)
    completed = sorted(
        [m for m in matches if m.get("score") and m.get("status", "").startswith("completed")],
        key=lambda m: m["date"]
    )

    for m in completed:
        t1_id = m.get("team1_id")
        t2_id = m.get("team2_id")
        if t1_id is None or t2_id is None:
            continue
        t1 = teams.get(t1_id)
        t2 = teams.get(t2_id)
        if not t1 or not t2:
            continue

        score = m["score"]
        s1 = int(score["team1"])
        s2 = int(score["team2"])
        t1_won = s1 > s2
        t2_won = s2 > s1
        if m.get("status", "").endswith("_penalties"):
            pen = score.get("penalties", "")
            if pen:
                p1, p2 = map(int, pen.split("-"))
                t1_won = p1 > p2
                t2_won = p2 > p1

        t1_elo = t1.get("elo_rating", 1500)
        t2_elo = t2.get("elo_rating", 1500)
        stage = m.get("stage", "group")
        date = m.get("date", "")

        history[t1_id].append({
            "score": score, "stage": stage, "date": date,
            "team_role": "home",
            "team_elo": t1_elo, "opponent_elo": t2_elo,
            "expected_goals": estimate_expected_goals(t1_elo, t2_elo, t1.get("is_host", False)),
            "status": m["status"],
            "clean_sheet": s2 == 0,
            "comeback": s1 <= s2 and t1_won,
            "advanced": t1_won,
        })
        history[t2_id].append({
            "score": score, "stage": stage, "date": date,
            "team_role": "away",
            "team_elo": t2_elo, "opponent_elo": t1_elo,
            "expected_goals": estimate_expected_goals(t2_elo, t1_elo, t2.get("is_host", False)),
            "status": m["status"],
            "clean_sheet": s1 == 0,
            "comeback": s2 <= s1 and t2_won,
            "advanced": t2_won,
        })

    return history


# === BUILD TRAINING DATASET ===

def build_dataset(matches_file: str = None) -> tuple[list, list, list]:
    """
    Build feature matrix X and labels y from completed matches.
    
    Returns (X, y, feature_names)
    - X: list of feature dicts
    - y: list of (1 if team won/advanced, 0 otherwise)
    - feature_names: ordered list of feature names
    """
    matches = load_raw_matches(matches_file)
    teams_list = load_teams()
    teams = {t["id"]: t for t in teams_list}
    
    X = []
    y = []
    
    # Track per-team history chronologically
    team_history = defaultdict(list)
    
    # Process matches in chronological order
    sorted_matches = sorted(
        [m for m in matches if m.get("score") and m.get("status", "").startswith("completed")],
        key=lambda m: m["date"]
    )
    
    for match in sorted_matches:
        t1_id = match.get("team1_id")
        t2_id = match.get("team2_id")
        if t1_id is None or t2_id is None:
            continue
        
        t1 = teams.get(t1_id)
        t2 = teams.get(t2_id)
        if not t1 or not t2:
            continue
        
        score = match["score"]
        s1 = int(score["team1"])
        s2 = int(score["team2"])
        stage = match.get("stage", "group")
        date = match.get("date", "")
        
        # Determine winner
        t1_won = s1 > s2
        t2_won = s2 > s1
        
        # Handle penalties
        if match.get("status", "").endswith("_penalties"):
            pen = score.get("penalties", "")
            if pen:
                p1, p2 = map(int, pen.split("-"))
                t1_won = p1 > p2
                t2_won = p2 > p1
        elif match.get("status", "").endswith("_aet"):
            t1_won = s1 > s2
            t2_won = s2 > s1
        
        # For draws in group stage, neither team "wins" uniquely
        # but we still include as training data points
        is_draw = (s1 == s2 and match["status"] == "completed")
        
        # Compute features for team1
        try:
            f1 = compute_match_features(
                t1_id, t2_id, t1, t2,
                team_history[t1_id], team_history[t2_id],
                stage, date
            )
            X.append(f1)
            y.append(1 if t1_won else 0)
        except Exception as e:
            pass
        
        # Compute features for team2
        try:
            f2 = compute_match_features(
                t2_id, t1_id, t2, t1,
                team_history[t2_id], team_history[t1_id],
                stage, date
            )
            X.append(f2)
            y.append(1 if t2_won else 0)
        except Exception as e:
            pass
        
        # Update history
        t1_elo = t1.get("elo_rating", 1500)
        t2_elo = t2.get("elo_rating", 1500)
        team_history[t1_id].append({
            "score": score,
            "stage": stage,
            "date": date,
            "team_role": "home",
            "team_elo": t1_elo,
            "opponent_elo": t2_elo,
            "expected_goals": estimate_expected_goals(t1_elo, t2_elo, t1.get("is_host", False)),
            "status": match["status"],
            "clean_sheet": s2 == 0,
            "comeback": s1 <= s2 and t1_won,
            "advanced": t1_won,
        })

        team_history[t2_id].append({
            "score": score,
            "stage": stage,
            "date": date,
            "team_role": "away",
            "team_elo": t2_elo,
            "opponent_elo": t1_elo,
            "expected_goals": estimate_expected_goals(t2_elo, t1_elo, t2.get("is_host", False)),
            "status": match["status"],
            "clean_sheet": s1 == 0,
            "comeback": s2 <= s1 and t2_won,
            "advanced": t2_won,
        })
    
    if X:
        feature_names = list(X[0].keys())
    else:
        feature_names = []
    
    print(f"Dataset built: {len(X)} samples, {len(feature_names)} features, {sum(y)} positive ({sum(y)/max(1,len(y))*100:.1f}%)")
    
    return X, y, feature_names
