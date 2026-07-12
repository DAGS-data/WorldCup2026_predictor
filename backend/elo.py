"""ELO rating system for World Cup 2026 predictions."""

K_FACTOR = 30  # World Cup K-factor
HOME_ADVANTAGE = 100  # ELO points for host nations

INITIAL_ELO_CEILING = 2100  # ELO assigned to FIFA rank #1
INITIAL_ELO_FLOOR = 1400  # ELO assigned to FIFA rank #90
INITIAL_ELO_SPAN_RANKS = 89  # ranks #1..#90


def initial_elo_from_fifa_rank(fifa_rank: float) -> float:
    """
    Seed ELO from the pre-tournament FIFA ranking via linear interpolation.

    No historical ELO exists for all 48 World Cup teams, so this is the
    starting point before any match updates: rank #1 -> 2100, rank #90 -> 1400.
    """
    return INITIAL_ELO_CEILING - (INITIAL_ELO_CEILING - INITIAL_ELO_FLOOR) / INITIAL_ELO_SPAN_RANKS * (fifa_rank - 1)


def get_expected_score(team_elo: float, opponent_elo: float, is_home: bool = False) -> float:
    """Calculate expected score (0-1) using ELO formula."""
    adjusted_team_elo = team_elo + (HOME_ADVANTAGE if is_home else 0)
    return 1 / (1 + 10 ** ((opponent_elo - adjusted_team_elo) / 400))


def update_elo(team_elo: float, opponent_elo: float, result: float, is_home: bool = False) -> float:
    """Update ELO rating after a match. result: 1=win, 0.5=draw, 0=loss"""
    expected = get_expected_score(team_elo, opponent_elo, is_home)
    return team_elo + K_FACTOR * (result - expected)


def calculate_new_ratings(
    team_elo: float,
    opponent_elo: float,
    result: str,
    team_is_home: bool = False,
) -> tuple[float, float]:
    """
    Calculate new ELO ratings for both teams after a match.
    result: 'W' | 'D' | 'L' from team's perspective.
    Returns (new_team_elo, new_opponent_elo).
    """
    scores = {"W": (1.0, 0.0), "D": (0.5, 0.5), "L": (0.0, 1.0)}
    team_score, opp_score = scores.get(result, (0.5, 0.5))

    new_team_elo = update_elo(team_elo, opponent_elo, team_score, team_is_home)
    new_opp_elo = update_elo(opponent_elo, team_elo, opp_score, not team_is_home)
    return new_team_elo, new_opp_elo


def elo_win_probability(team_elo: float, opponent_elo: float, team_is_home: bool = False) -> float:
    """Quick ELO-based win probability (not accounting for draws)."""
    return get_expected_score(team_elo, opponent_elo, team_is_home)
