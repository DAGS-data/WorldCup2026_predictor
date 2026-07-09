"""
Dixon-Coles Bayesian goal predictor (replaces CatBoost+Poisson).
Loads pre-trained parameters and predicts scorelines with DC adjustment.
"""
import json
import math
import pickle
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent
DC_PARAMS_PATH = MODEL_DIR / "dixon_coles_v1.pkl"


class DixonColesPredictor:
    """Predict goals using hierarchical Bayesian attack/defense strengths."""

    def __init__(self):
        self.att = None
        self.def_str = None
        self.home_adv = 0.3
        self.rho = 0.0
        self.team_names = []
        self.team_ids = []
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not DC_PARAMS_PATH.exists():
            raise FileNotFoundError(
                "Dixon-Coles model not found. Run train_dixon_coles.py first."
            )
        with open(DC_PARAMS_PATH, "rb") as f:
            params = pickle.load(f)
        self.att = np.array(params["att"])
        self.def_str = np.array(params["def"])
        self.home_adv = params["home_adv"]
        self.rho = params["rho"]
        self.team_names = params["team_names"]
        self.team_ids = params["team_ids"]
        self._loaded = True

    def predict(self, team1_id: int, team2_id: int) -> dict:
        """
        Predict λ_home, λ_away and top scorelines using Dixon-Coles.
        """
        self._ensure_loaded()

        idx1 = self.team_ids.index(team1_id)
        idx2 = self.team_ids.index(team2_id)

        att1, def1 = self.att[idx1], self.def_str[idx1]
        att2, def2 = self.att[idx2], self.def_str[idx2]

        log_lam_home = att1 + def2 + self.home_adv
        log_lam_away = att2 + def1

        lam_home = math.exp(log_lam_home)
        lam_away = math.exp(log_lam_away)
        rho = float(self.rho)

        # Compute Dixon-Coles adjusted scoreline probabilities
        top = self._dc_scorelines(lam_home, lam_away, rho, n=10)

        return {
            "home_lambda": round(lam_home, 3),
            "away_lambda": round(lam_away, 3),
            "top_scorelines": top,
        }

    @staticmethod
    def _poisson_prob(lam: float, k: int) -> float:
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return (lam ** k) * math.exp(-lam) / math.factorial(k)

    def _dc_adjustment(self, h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
        """Dixon-Coles τ adjustment factor for low-scoring games."""
        if rho <= 0:
            return 1.0
        if h == 0 and a == 0:
            return max(0.0, 1.0 - lam_h * lam_a * rho)
        elif h == 0 and a == 1:
            return 1.0 + lam_h * rho
        elif h == 1 and a == 0:
            return 1.0 + lam_a * rho
        elif h == 1 and a == 1:
            return max(0.0, 1.0 - rho)
        else:
            return 1.0

    def _dc_scorelines(self, lam_h: float, lam_a: float, rho: float, n: int = 10) -> dict:
        """Top N scorelines with Dixon-Coles correction."""
        scores = []
        for h in range(8):
            ph = self._poisson_prob(lam_h, h)
            for a in range(8):
                pa = self._poisson_prob(lam_a, a)
                raw_p = ph * pa
                tau = self._dc_adjustment(h, a, lam_h, lam_a, rho)
                adj_p = raw_p * tau
                scores.append((h, a, adj_p))

        # Normalize
        total = sum(p for _, _, p in scores)
        scores = [(h, a, p / max(total, 1e-8)) for h, a, p in scores]
        scores.sort(key=lambda x: x[2], reverse=True)

        return {
            "top": [
                {
                    "home_goals": h,
                    "away_goals": a,
                    "probability": round(p * 100, 1),
                }
                for h, a, p in scores[:n]
            ],
        }


# Singleton
_predictor = None


def get_goals_predictor() -> DixonColesPredictor:
    global _predictor
    if _predictor is None:
        _predictor = DixonColesPredictor()
    return _predictor
