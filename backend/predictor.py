"""Match outcome predictor using ELO + Poisson model."""

import math
from typing import Any

from elo import get_expected_score

BASE_GOALS = 1.35  # average goals scored per team per WC group match


def poisson_pmf(lam: float, k: int) -> float:
    """P(X=k) for Poisson(lam)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def estimate_expected_goals(team_elo: float, opponent_elo: float, team_is_host: bool = False) -> float:
    """
    Expected goals for a team in a single match.
    ELO difference drives the adjustment: each 100 ELO points ≈ +0.25 goals.
    """
    home_bonus = 100 if team_is_host else 0
    elo_diff = (team_elo + home_bonus) - opponent_elo
    adjustment = elo_diff / 400.0
    return max(0.25, BASE_GOALS + adjustment)


def calculate_match_probabilities(
    team1_elo: float,
    team2_elo: float,
    team1_is_host: bool = False,
    team2_is_host: bool = False,
) -> dict[str, Any]:
    """
    Full Poisson-based match probability model.
    Returns win/draw/loss probabilities, predicted score, and xG.
    """
    lam1 = estimate_expected_goals(team1_elo, team2_elo, team1_is_host)
    lam2 = estimate_expected_goals(team2_elo, team1_elo, team2_is_host)

    max_goals = 8
    win1 = draw = win2 = 0.0
    best_score = (0, 0)
    best_prob = 0.0

    for g1 in range(max_goals + 1):
        p1 = poisson_pmf(lam1, g1)
        if p1 < 1e-9:
            continue
        for g2 in range(max_goals + 1):
            prob = p1 * poisson_pmf(lam2, g2)
            if g1 > g2:
                win1 += prob
            elif g1 == g2:
                draw += prob
            else:
                win2 += prob
            if prob > best_prob:
                best_prob = prob
                best_score = (g1, g2)

    total = win1 + draw + win2 or 1.0

    return {
        "team1_win_prob": round(win1 / total * 100, 1),
        "draw_prob": round(draw / total * 100, 1),
        "team2_win_prob": round(win2 / total * 100, 1),
        "predicted_score": f"{best_score[0]}-{best_score[1]}",
        "team1_xg": round(lam1, 2),
        "team2_xg": round(lam2, 2),
    }


def get_prediction_factors(team1: dict[str, Any], team2: dict[str, Any]) -> list[str]:
    """Return up to 3 human-readable prediction factors."""
    factors: list[str] = []

    elo_diff = team1["elo_rating"] - team2["elo_rating"]
    if abs(elo_diff) >= 30:
        stronger = team1["name"] if elo_diff > 0 else team2["name"]
        factors.append(f"ELO advantage: {stronger} (+{abs(elo_diff):.0f} pts)")

    val1 = team1.get("squad_value_millions", 0)
    val2 = team2.get("squad_value_millions", 0)
    if abs(val1 - val2) >= 80:
        richer = team1["name"] if val1 > val2 else team2["name"]
        factors.append(f"Squad value edge: {richer} (${abs(val1-val2):.0f}M)")

    if team1.get("is_host"):
        factors.append(f"Home advantage: {team1['name']}")
    if team2.get("is_host"):
        factors.append(f"Home advantage: {team2['name']}")

    return factors[:3]


def simulate_tournament(teams_by_group: dict[str, list[dict]], matches: list[dict]) -> dict[str, Any]:
    """
    Simulate full tournament bracket from group stage through Final.
    Returns predicted knockout bracket structure.
    """
    # Determine group winners and runners-up from predicted group standings
    group_results: dict[str, list[dict]] = {}
    for group, group_teams in teams_by_group.items():
        standings = {t["id"]: {"team": t, "pts": 0, "gd": 0, "gf": 0} for t in group_teams}
        group_matches = [m for m in matches if m.get("group") == group]

        for match in group_matches:
            t1 = match["team1_id"]
            t2 = match["team2_id"]
            t1_elo = next((t["elo_rating"] for t in group_teams if t["id"] == t1), 1500)
            t2_elo = next((t["elo_rating"] for t in group_teams if t["id"] == t2), 1500)
            probs = calculate_match_probabilities(t1_elo, t2_elo)
            score = probs["predicted_score"].split("-")
            g1, g2 = int(score[0]), int(score[1])

            if t1 in standings and t2 in standings:
                standings[t1]["gf"] += g1
                standings[t1]["gd"] += g1 - g2
                standings[t2]["gf"] += g2
                standings[t2]["gd"] += g2 - g1
                if g1 > g2:
                    standings[t1]["pts"] += 3
                elif g1 == g2:
                    standings[t1]["pts"] += 1
                    standings[t2]["pts"] += 1
                else:
                    standings[t2]["pts"] += 3

        ranked = sorted(
            standings.values(),
            key=lambda x: (x["pts"], x["gd"], x["gf"]),
            reverse=True,
        )
        group_results[group] = [r["team"] for r in ranked]

    # Build R32 fixtures (16 groups → 32 teams, winner vs runner-up cross-pattern)
    groups = sorted(group_results.keys())
    r32_fixtures = []
    for i in range(0, len(groups), 2):
        g1, g2 = groups[i], groups[i + 1]
        r32_fixtures.append({
            "team1": group_results[g1][0],
            "team2": group_results[g2][1],
        })
        r32_fixtures.append({
            "team1": group_results[g2][0],
            "team2": group_results[g1][1],
        })

    def simulate_ko_round(fixtures: list[dict]) -> list[dict]:
        winners = []
        for f in fixtures:
            t1, t2 = f["team1"], f["team2"]
            probs = calculate_match_probabilities(t1["elo_rating"], t2["elo_rating"])
            winner = t1 if probs["team1_win_prob"] >= probs["team2_win_prob"] else t2
            winners.append({"team1": t1, "team2": t2, "predicted_winner": winner, "probs": probs})
        return winners

    r32 = simulate_ko_round(r32_fixtures)
    r16_fixtures = [{"team1": m["predicted_winner"], "team2": r32[i+1]["predicted_winner"]}
                    for i, m in enumerate(r32) if i % 2 == 0]
    r16 = simulate_ko_round(r16_fixtures)
    qf_fixtures = [{"team1": r16[i]["predicted_winner"], "team2": r16[i+1]["predicted_winner"]}
                   for i in range(0, len(r16), 2)]
    qf = simulate_ko_round(qf_fixtures)
    sf_fixtures = [{"team1": qf[0]["predicted_winner"], "team2": qf[1]["predicted_winner"]},
                   {"team1": qf[2]["predicted_winner"], "team2": qf[3]["predicted_winner"]}]
    sf = simulate_ko_round(sf_fixtures)
    final_fixture = [{"team1": sf[0]["predicted_winner"], "team2": sf[1]["predicted_winner"]}]
    final = simulate_ko_round(final_fixture)

    def simplify(rounds):
        return [
            {
                "team1": m["team1"]["name"],
                "team1_flag": m["team1"]["flag_emoji"],
                "team2": m["team2"]["name"],
                "team2_flag": m["team2"]["flag_emoji"],
                "predicted_winner": m["predicted_winner"]["name"],
                "team1_win_prob": m["probs"]["team1_win_prob"],
                "draw_prob": m["probs"]["draw_prob"],
                "team2_win_prob": m["probs"]["team2_win_prob"],
            }
            for m in rounds
        ]

    return {
        "round_of_32": simplify(r32),
        "round_of_16": simplify(r16),
        "quarter_finals": simplify(qf),
        "semi_finals": simplify(sf),
        "final": simplify(final),
        "predicted_champion": final[0]["predicted_winner"]["name"],
        "predicted_champion_flag": final[0]["predicted_winner"]["flag_emoji"],
    }
