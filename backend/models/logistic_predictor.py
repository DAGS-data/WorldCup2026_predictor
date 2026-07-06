"""
Logistic Regression predictor for World Cup knockout predictions.
Replaces XGBoost — simpler, no overfitting, naturally calibrated.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "logistic_v1.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names_logistic.json"


class LogisticPredictor:
    """L2-regularized logistic regression for knockout advancement probability."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._loaded = False

    def train(self, X: list[dict], y: list[int], feature_names: list[str]):
        """Train logistic regression with time-based split validation."""
        self.feature_names = feature_names

        X_np = np.array([[d.get(f, 0.0) for f in feature_names] for d in X])
        y_np = np.array(y)

        # Time-based split
        split_idx = int(len(X_np) * 0.8)
        X_train, X_val = X_np[:split_idx], X_np[split_idx:]
        y_train, y_val = y_np[:split_idx], y_np[split_idx:]

        print(f"Training: {len(X_train)} samples | Validation: {len(X_val)} samples")
        print(f"  Train positive rate: {y_train.mean():.1%}")
        print(f"  Val positive rate: {y_val.mean():.1%}")

        # Standardize
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        # Train
        self.model = LogisticRegression(
            C=0.1,                  # L2 regularization strength
            max_iter=2000,
            random_state=42,
        )
        self.model.fit(X_train_s, y_train)

        # Evaluate
        acc = self.model.score(X_val_s, y_val)
        probs = self.model.predict_proba(X_val_s)[:, 1]
        from sklearn.metrics import brier_score_loss, roc_auc_score
        brier = brier_score_loss(y_val, probs)
        try:
            auc = roc_auc_score(y_val, probs)
        except:
            auc = 0.5

        print(f"\n=== Validation Metrics ===")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  ROC AUC:  {auc:.3f}")
        print(f"  Brier:    {brier:.4f}")

        # Feature importance (absolute coefficients)
        coef_abs = np.abs(self.model.coef_[0])
        top_idx = np.argsort(coef_abs)[-10:][::-1]
        print(f"\n=== Top 10 Features ===")
        for rank, i in enumerate(top_idx, 1):
            print(f"  {rank}. {feature_names[i]}: {self.model.coef_[0][i]:+.4f}")

        self._save()
        return {"accuracy": acc, "auc": auc, "brier": brier}

    def predict(self, features: dict) -> float:
        """Predict P(team advances). Returns probability in [0, 1]."""
        if not self._loaded:
            self._load()

        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        X_s = self.scaler.transform(X)
        prob = float(self.model.predict_proba(X_s)[0][1])
        return round(prob, 4)

    def explain(self, features: dict) -> dict:
        """Explain prediction using logistic coefficients (SHAP-free)."""
        if not self._loaded:
            self._load()

        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        X_s = self.scaler.transform(X)

        prob = float(self.model.predict_proba(X_s)[0][1])
        coefs = self.model.coef_[0]
        intercept = float(self.model.intercept_[0])

        # Per-feature contribution: coef × scaled_value
        contributions = []
        for i, feat in enumerate(self.feature_names):
            contrib = float(coefs[i] * X_s[0][i])
            contributions.append({
                "feature": feat,
                "value": round(float(features.get(feat, 0.0)), 4),
                "coefficient": round(float(coefs[i]), 4),
                "contribution": round(contrib, 4),
            })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "prediction": round(prob, 4),
            "intercept": round(intercept, 4),
            "model_type": "Logistic Regression (L2, C=0.1)",
            "top_factors": contributions[:8],
        }

    def _save(self):
        """Save model + scaler + feature names."""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(data, f)
        with open(FEATURE_NAMES_PATH, "w") as f:
            json.dump(self.feature_names, f)
        print(f"Model saved to {MODEL_PATH}")

    def _load(self):
        """Load model from disk."""
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_names = data["feature_names"]
            self._loaded = True
        else:
            self._loaded = True

    def get_metrics(self) -> dict:
        if not self._loaded:
            self._load()
        return {
            "model_path": str(MODEL_PATH),
            "features": len(self.feature_names) if self.feature_names else 0,
            "loaded": self.model is not None,
            "model_type": "Logistic Regression (L2, C=0.1)",
        }


# ============================================================
# TRAINING SCRIPT
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from feature_engineering import build_dataset

    print("=" * 60)
    print("BUILDING DATASET")
    print("=" * 60)
    X, y, feature_names = build_dataset()

    print(f"\n{'=' * 60}")
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 60)
    predictor = LogisticPredictor()
    metrics = predictor.train(X, y, feature_names)

    print(f"\n{'=' * 60}")
    print("SAMPLE PREDICTION")
    print("=" * 60)
    if X:
        sample = X[-1]
        prob = predictor.predict(sample)
        print(f"Sample prediction: {prob:.1%}")
        print(f"Features: {len(sample)}")
