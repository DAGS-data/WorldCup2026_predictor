"""
Feature extraction for CatBoost Poisson goal model.
Must exactly match the features used during training (train_goals.py V1).
"""
from collections import defaultdict

# Stage order
STAGE_ORDER = {"group": 0, "round_of_32": 1, "round_of_16": 2, "quarter_finals": 3}


def get_goal_features(team, opponent, match_history, stage="round_of_16", is_home=False):
    """
    team: dict from teams.json (with elo_rating, fifa_rank, squad_value_millions, etc.)
    opponent: same
    match_history: list of {goals_for, goals_against} for this team, chronologically
    stage: stage name
    is_home: bool
    
    Returns dict of features matching train_goals.py V1.
    """
    gf = [m["goals_for"] for m in match_history]
    ga = [m["goals_against"] for m in match_history]
    n = len(gf)

    f = {}

    # Core differentials
    f["elo"] = team.get("elo_rating", 1500)
    f["elo_diff"] = team.get("elo_rating", 1500) - opponent.get("elo_rating", 1500)
    f["fifa_rank"] = team.get("fifa_rank", 90)
    f["fifa_rank_diff"] = team.get("fifa_rank", 90) - opponent.get("fifa_rank", 90)
    f["squad_value"] = team.get("squad_value_millions", 0)
    f["squad_value_ratio"] = (
        team.get("squad_value_millions", 1) / max(opponent.get("squad_value_millions", 1), 1)
    )
    f["is_host"] = 1 if is_home else 0
    f["confederation"] = team.get("confederation", "Unknown")

    # Recent form (last 5 matches)
    for i in range(5):
        idx = n - 1 - i
        f[f"gf_last_{i+1}"] = gf[idx] if idx >= 0 else 0
        f[f"ga_last_{i+1}"] = ga[idx] if idx >= 0 else 0

    # Aggregated form
    if n > 0:
        def _mean(arr):
            return sum(arr) / len(arr) if arr else 0.0

        last5_gf = gf[-5:] if len(gf) >= 5 else gf
        last5_ga = ga[-5:] if len(ga) >= 5 else ga
        f["avg_gf"] = _mean(last5_gf)
        f["avg_ga"] = _mean(last5_ga)
        f["gf_trend"] = _mean(gf[-3:]) - _mean(gf[-5:]) if len(gf) >= 5 else 0.0
        f["ga_trend"] = _mean(ga[-3:]) - _mean(ga[-5:]) if len(ga) >= 5 else 0.0
        f["goal_diff_avg"] = _mean([gf[i] - ga[i] for i in range(max(0, n - 5), n)])
        f["matches_played"] = n
    else:
        f["avg_gf"] = 0.0
        f["avg_ga"] = 0.0
        f["gf_trend"] = 0.0
        f["ga_trend"] = 0.0
        f["goal_diff_avg"] = 0.0
        f["matches_played"] = 0

    # Tournament stage
    f["stage_num"] = STAGE_ORDER.get(stage, 0)

    # Performance
    perf = team.get("performance", {})
    f["perf_rating"] = perf.get("rating_100", 50)
    f["perf_form"] = perf.get("form_score", 50)

    return f


def build_match_history(team_id, matches_raw, teams):
    """Build chronological match history for a team from completed matches."""
    completed = [
        m for m in matches_raw
        if m.get("status", "").startswith("completed") and m.get("score")
    ]
    completed.sort(key=lambda m: m["date"])

    history = []
    for m in completed:
        if m["team1_id"] == team_id:
            history.append({
                "goals_for": int(m["score"]["team1"]),
                "goals_against": int(m["score"]["team2"]),
            })
        elif m["team2_id"] == team_id:
            history.append({
                "goals_for": int(m["score"]["team2"]),
                "goals_against": int(m["score"]["team1"]),
            })
    return history
