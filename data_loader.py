"""
data_loader.py  –  Dataset loading, validation, and splitting.

Supports:
  - Iris (built-in via sklearn)
  - Wine (built-in via sklearn)
  - Breast Cancer (built-in via sklearn)
  - Custom CSV files
"""

from operator import le

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Counter, Tuple
import logging

logger = logging.getLogger(__name__)

BUILT_IN_DATASETS = {
    "iris":          load_iris,
    "wine":          load_wine,
    "breast_cancer": load_breast_cancer,
}


class DataLoader:
    """
    Handles dataset loading, exploration, preprocessing and splitting.

    Attributes
    ----------
    dataset_name : str
        Name of the loaded dataset.
    X_train, X_test : np.ndarray
        Feature matrices after split.
    y_train, y_test : np.ndarray
        Target arrays after split.
    feature_names : list[str]
    target_names  : list[str]
    scaler        : StandardScaler  (fitted on training set)
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size    = test_size
        self.random_state = random_state
        self.scaler       = StandardScaler()
        self.dataset_name = None

        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.feature_names = []
        self.target_names  = []
        self._raw_df: pd.DataFrame | None = None

    # ── Public API ──────────────────────────────────────────────────

    def load(self, source: str = "iris") -> "DataLoader":
        if source in BUILT_IN_DATASETS:
            self._load_sklearn(source)
        elif source.endswith(".csv"):
            self._load_csv(source)
        else:
            raise ValueError(
                f"Unknown source '{source}'. Use {list(BUILT_IN_DATASETS)} or a .csv path."
        )

        logger.info(
            f"Loaded '{self.dataset_name}': "
            f"{len(self._raw_df)} samples, "
            f"{len(self.feature_names)} features, "
            f"{len(self.target_names)} classes"
            )
        return self

    def preprocess(self, scale: bool = True) -> "DataLoader":
        """Split into train/test, optionally scale features."""
        df = self._raw_df
        X = df.drop(columns=["target"]).values
        y = df["target"].values

        from collections import Counter

        class_counts = Counter(y)

        if min(class_counts.values()) < 2:
            stratify = None
        else:
            stratify = y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        if scale:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test  = self.scaler.transform(self.X_test)

        logger.info("Split: train=%d, test=%d", len(self.y_train), len(self.y_test))
        return self
    

    
    def summary(self) -> dict:
        """Return a dict describing the dataset."""
        df = self._raw_df
        return {
            "name":          self.dataset_name,
            "total_samples": len(df),
            "n_features":    len(self.feature_names),
            "n_classes":     len(self.target_names),
            "class_names":   list(self.target_names),
            "train_samples": len(self.y_train) if self.y_train is not None else None,
            "test_samples":  len(self.y_test)  if self.y_test  is not None else None,
            "feature_names": list(self.feature_names),
            "class_dist": {
                name: int((df["target"] == i).sum())
                for i, name in enumerate(self.target_names)
            },
        }

    # ── Private helpers ──────────────────────────────────────────────

    def _load_sklearn(self, name: str):
        bunch = BUILT_IN_DATASETS[name]()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        df["target"] = bunch.target
        self.dataset_name  = name
        self.feature_names = list(bunch.feature_names)
        if hasattr(le, "classes_"):
            self.target_names = list(le.classes_)
        else:
            self.target_names = sorted(pd.unique(self.y))
        self._raw_df = df

    def _load_csv(self, path: str):
        df = pd.read_csv(path)
        if "target" not in df.columns:
            # assume last column is the target
            df = df.rename(columns={df.columns[-1]: "target"})

        # encode string targets
        from sklearn.preprocessing import LabelEncoder

        if df["target"].dtype == object:
            le = LabelEncoder()

            df["target"] = le.fit_transform(df["target"])

            self.target_names = [str(c) for c in le.classes_]

        else:
            self.target_names = [
            str(x) for x in sorted(df["target"].unique())
        ]

        self.dataset_name  = path.split("/")[-1].replace(".csv", "")
        self.feature_names = [c for c in df.columns if c != "target"]
        self._raw_df = df
