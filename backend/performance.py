"""Team performance rating calculator for World Cup 2026."""

from typing import Any


def calculate_form_score(recent_form: list[dict[str, Any]]) -> float:
    """
    Weighted form score from last 10 matches (0.0-1.0).
    Most recent matches weighted higher.
    """
    if not recent_form:
        return 0.5

    weights = [1.0, 0.93, 0.86, 0.79, 0.73, 0.67, 0.61, 0.56, 0.51, 0.46]
    result_values = {"W": 1.0, "D": 0.5, "L": 0.0}

    total_weight = 0.0
    weighted_score = 0.0

    for i, match in enumerate(recent_form[:10]):
        w = weights[i] if i < len(weights) else 0.4
        score = result_values.get(match.get("result", "D"), 0.5)
        weighted_score += score * w
        total_weight += w

    return weighted_score / total_weight if total_weight > 0 else 0.5


def calculate_goal_difference_factor(recent_form: list[dict[str, Any]]) -> float:
    """Average goal difference per match, normalised to 0-1."""
    if not recent_form:
        return 0.5
    total_gd = sum(
        int(m.get("goals_for", 0)) - int(m.get("goals_against", 0))
        for m in recent_form[:10]
    )
    avg_gd = total_gd / len(recent_form[:10])
    # Clamp to [-3, 3] then map to [0, 1]
    clamped = max(-3.0, min(3.0, avg_gd))
    return (clamped + 3.0) / 6.0


def calculate_performance_rating(elo_rating: float, recent_form: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Full performance rating.
    Returns rating_100 (1-100), rating_10 (1-10), plus component breakdown.
    """
    # ELO component: typical competitive range 1400-2000
    min_elo, max_elo = 1400, 2130
    elo_norm = max(0.0, min(1.0, (elo_rating - min_elo) / (max_elo - min_elo)))

    form = calculate_form_score(recent_form)
    gd_factor = calculate_goal_difference_factor(recent_form)

    # Weighted blend: 65% ELO, 25% form, 10% goal difference
    combined = 0.65 * elo_norm + 0.25 * form + 0.10 * gd_factor

    rating_100 = max(1, min(100, round(1 + combined * 99)))
    rating_10 = max(1, min(10, round(1 + combined * 9)))

    return {
        "rating_100": rating_100,
        "rating_10": rating_10,
        "form_score": round(form * 100, 1),
        "elo_contribution": round(elo_norm * 100, 1),
        "gd_factor": round(gd_factor * 100, 1),
    }
