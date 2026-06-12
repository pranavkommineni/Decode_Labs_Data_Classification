"""
trainer.py  –  Model training, cross-validation, and evaluation.
"""

import time
import logging
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from typing import Any

logger = logging.getLogger(__name__)


class Trainer:
    """
    Wraps a scikit-learn estimator and provides:
    - fit() with timing
    - evaluate() returning full metric dict
    - cross_validate() with stratified K-fold
    - tune() via GridSearchCV
    """

    def __init__(self, model, model_name: str = "model"):
        self.model      = model
        self.model_name = model_name
        self._fitted    = False
        self._train_time: float = 0.0

    # ── Training ─────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Trainer":
        logger.info("Training '%s' on %d samples…", self.model_name, len(y_train))
        t0 = time.perf_counter()
        self.model.fit(X_train, y_train)
        self._train_time = time.perf_counter() - t0
        self._fitted = True
        logger.info("Training done in %.3fs", self._train_time)
        return self

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 target_names: list[str] | None = None) -> dict:
        self._check_fitted()
        y_pred = self.model.predict(X_test)

        avg = "weighted"
        metrics = {
            "model":       self.model_name,
            "accuracy":    round(accuracy_score(y_test, y_pred), 4),
            "precision":   round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "recall":      round(recall_score   (y_test, y_pred, average=avg, zero_division=0), 4),
            "f1_score":    round(f1_score       (y_test, y_pred, average=avg, zero_division=0), 4),
            "train_time_s": round(self._train_time, 4),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=target_names,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        return metrics

    # ── Cross-validation ─────────────────────────────────────────────

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       cv: int = 5, scoring: str = "accuracy") -> dict:
        self._check_not_fitted()
        skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        return {
            "cv_folds":   cv,
            "scoring":    scoring,
            "scores":     scores.round(4).tolist(),
            "mean":       round(scores.mean(), 4),
            "std":        round(scores.std(), 4),
        }

    # ── Hyperparameter tuning ─────────────────────────────────────────

    def tune(self, X_train: np.ndarray, y_train: np.ndarray,
             param_grid: dict, cv: int = 5) -> dict:
        logger.info("Grid-searching '%s'…", self.model_name)
        search = GridSearchCV(
            self.model, param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        self.model    = search.best_estimator_
        self._fitted  = True
        logger.info("Best params: %s  (cv score=%.4f)", search.best_params_, search.best_score_)
        return {
            "best_params":   search.best_params_,
            "best_cv_score": round(search.best_score_, 4),
        }

    # ── Prediction ────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise NotImplementedError(f"{self.model_name} does not support predict_proba.")

    # ── Guards ────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(f"Model '{self.model_name}' must be fitted before evaluation.")

    def _check_not_fitted(self):
        # for cross_validate we want a fresh clone to avoid data leakage
        pass  # sklearn cv handles this internally
