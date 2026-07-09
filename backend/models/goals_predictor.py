"""
CatBoost Poisson goal predictor — predicts λ (expected goals) for each team,
then computes most likely scorelines.
"""
import json
import math
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent


class PoissonGoalPredictor:
    """Predict expected goals using CatBoost + Poisson loss, then sample scorelines."""

    def __init__(self):
        self.model_home = None
        self.model_away = None
        self.metadata = {}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from catboost import CatBoostRegressor

        home_path = MODEL_DIR / "goals_home_v1.cbm"
        away_path = MODEL_DIR / "goals_away_v1.cbm"
        meta_path = MODEL_DIR / "goals_metadata_v1.json"

        if not home_path.exists() or not away_path.exists():
            raise FileNotFoundError("Goal models not found. Run train_goals.py first.")

        self.model_home = CatBoostRegressor()
        self.model_home.load_model(str(home_path))
        self.model_away = CatBoostRegressor()
        self.model_away.load_model(str(away_path))

        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)

        self._loaded = True

    def predict(self, t1_features: dict, t2_features: dict) -> dict:
        """
        t1_features: dict of features for team1 (home)
        t2_features: dict of features for team2 (away)
        
        Returns {home_lambda, away_lambda, top_scorelines: [(score, prob), ...]}
        """
        self._ensure_loaded()

        # Build row with t1_ and t2_ prefixes
        row = {}
        for k, v in t1_features.items():
            row[f"t1_{k}"] = v
        for k, v in t2_features.items():
            row[f"t2_{k}"] = v

        # Get feature names and order from metadata
        feature_names = self.metadata.get("feature_names", sorted(row.keys()))

        # Build feature vector in the right order
        cat_features_set = set(self.metadata.get("categorical_features", []))
        X = np.array([[row.get(k, 0) for k in feature_names]], dtype=object)

        lam_home = float(self.model_home.predict(X)[0])
        lam_away = float(self.model_away.predict(X)[0])

        # Compute top scorelines
        top_scorelines = self._top_scorelines(lam_home, lam_away, n=5)

        return {
            "home_lambda": round(lam_home, 3),
            "away_lambda": round(lam_away, 3),
            "top_scorelines": top_scorelines,
        }

    @staticmethod
    def _poisson_prob(lam: float, k: int) -> float:
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return (lam ** k) * math.exp(-lam) / math.factorial(k)

    def _top_scorelines(self, lam_h: float, lam_a: float, n: int = 5) -> list:
        """Return top N most likely scorelines + full probability matrix."""
        scores = []
        matrix = {}
        for h in range(7):
            ph = self._poisson_prob(lam_h, h)
            for a in range(7):
                pa = self._poisson_prob(lam_a, a)
                p = round(ph * pa, 6)
                scores.append((h, a, p))
                matrix[f"{h}-{a}"] = round(p * 100, 2)
        scores.sort(key=lambda x: x[2], reverse=True)
        return {
            "top": [{"home_goals": h, "away_goals": a, "probability": round(p * 100, 1)}
                    for h, a, p in scores[:n]],
            "matrix_7x7": matrix,
        }


# Singleton
_predictor = None


def get_goals_predictor() -> PoissonGoalPredictor:
    global _predictor
    if _predictor is None:
        _predictor = PoissonGoalPredictor()
    return _predictor
