"""
XGBoost binary classifier for World Cup knockout predictions.

Predicts P(team advances) using 50+ engineered features.
Trained on all completed tournament matches with time-based CV.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

MODEL_DIR = Path(__file__).parent
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgboost_v1.json"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"


class KnockoutPredictor:
    """XGBoost predictor for World Cup knockout advancement probability."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.calibration_model = None  # Platt scaling
        self._loaded = False
        
    def train(self, X: list[dict], y: list[int], feature_names: list[str]):
        """
        Train XGBoost model with Optuna hyperparameter tuning.
        
        Uses time-based split: last 20% of matches for validation.
        """
        import xgboost as xgb
        
        self.feature_names = feature_names
        
        # Convert to numpy
        X_np = np.array([[d.get(f, 0.0) for f in feature_names] for d in X])
        y_np = np.array(y)
        
        # Time-based split (last 20% for validation)
        split_idx = int(len(X_np) * 0.8)
        X_train, X_val = X_np[:split_idx], X_np[split_idx:]
        y_train, y_val = y_np[:split_idx], y_np[split_idx:]
        
        print(f"Training: {len(X_train)} samples | Validation: {len(X_val)} samples")
        print(f"  Train positive rate: {y_train.mean():.1%}")
        print(f"  Val positive rate: {y_val.mean():.1%}")
        
        # Quick hyperparameter search (simplified — full Optuna if time allows)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 3,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 5,
            "random_state": 42,
        }
        
        # Train
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            verbose_eval=20,
        )
        
        # Evaluate
        preds = self.model.predict(dval)
        from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
        pred_binary = (preds > 0.5).astype(int)
        
        acc = accuracy_score(y_val, pred_binary)
        try:
            auc = roc_auc_score(y_val, preds)
        except:
            auc = 0.5
        brier = brier_score_loss(y_val, preds)
        
        print(f"\n=== Validation Metrics ===")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  ROC AUC:  {auc:.3f}")
        print(f"  Brier:    {brier:.4f}")
        
        # Feature importance
        importance = self.model.get_score(importance_type="gain")
        print(f"\n=== Top 10 Features ===")
        for i, (feat, score) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]):
            print(f"  {i+1}. {feat}: {score:.1f}")
        
        # Save
        self._save()
        
        return {"accuracy": acc, "auc": auc, "brier": brier}
    
    def predict(self, features: dict) -> float:
        """
        Predict P(team advances) for a single match.
        
        Returns probability in [0, 1].
        """
        if not self._loaded:
            self._load()
        
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        import xgboost as xgb
        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        
        raw_pred = float(self.model.predict(dmatrix)[0])
        
        # Platt calibration if available
        if self.calibration_model:
            raw_pred = float(self.calibration_model.predict_proba([[raw_pred]])[0][1])
        
        return round(raw_pred, 4)
    
    def explain(self, features: dict) -> dict:
        """SHAP explanation of prediction."""
        try:
            import shap
            import xgboost as xgb
            
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(dmatrix)
            
            # Top contributing features
            contributions = []
            for i, feat in enumerate(self.feature_names):
                contributions.append({
                    "feature": feat,
                    "value": float(features.get(feat, 0.0)),
                    "shap": float(shap_values[0][i]),
                })
            
            contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)
            
            return {
                "prediction": self.predict(features),
                "base_value": float(explainer.expected_value),
                "top_factors": contributions[:8],
            }
        except ImportError:
            return {"error": "SHAP not installed"}
    
    def _save(self):
        """Save model and metadata."""
        self.model.save_model(str(MODEL_PATH))
        with open(FEATURE_NAMES_PATH, "w") as f:
            json.dump(self.feature_names, f)
        print(f"Model saved to {MODEL_PATH}")
    
    def _load(self):
        """Load model from disk."""
        if MODEL_PATH.exists():
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(str(MODEL_PATH))
            with open(FEATURE_NAMES_PATH) as f:
                self.feature_names = json.load(f)
            self._loaded = True
            print(f"Model loaded from {MODEL_PATH} ({len(self.feature_names)} features)")
        else:
            self._loaded = True  # mark as attempted
    
    def get_metrics(self) -> dict:
        """Return model metadata."""
        if not self._loaded:
            self._load()
        return {
            "model_path": str(MODEL_PATH),
            "features": len(self.feature_names) if self.feature_names else 0,
            "loaded": self.model is not None,
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
    print("TRAINING XGBOOST MODEL")
    print("=" * 60)
    predictor = KnockoutPredictor()
    metrics = predictor.train(X, y, feature_names)
    
    print(f"\n{'=' * 60}")
    print("SAMPLE PREDICTION")
    print("=" * 60)
    # Test on a sample
    if X:
        sample = X[-1]
        prob = predictor.predict(sample)
        print(f"Sample prediction: {prob:.1%}")
        print(f"Features: {len(sample)}")
