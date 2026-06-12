"""
models.py  –  Classifier registry.

Each entry wraps a scikit-learn estimator with its default
hyper-parameter grid for optional tuning.
"""

from sklearn.linear_model      import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm                import SVC
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.naive_bayes        import GaussianNB


MODEL_REGISTRY: dict[str, dict] = {
    "logistic_regression": {
        "class":  LogisticRegression,
        "params": {"max_iter": 1000, "random_state": 42},
        "grid": {
            "C":       [0.01, 0.1, 1, 10, 100],
            "solver":  ["lbfgs", "liblinear"],
        },
        "description": "Fast linear baseline — great for high-dimensional data.",
    },
    "decision_tree": {
        "class":  DecisionTreeClassifier,
        "params": {"random_state": 42},
        "grid": {
            "max_depth":        [3, 5, 7, None],
            "min_samples_leaf": [1, 2, 5],
        },
        "description": "Interpretable rule-based splits; prone to overfit.",
    },
    "random_forest": {
        "class":  RandomForestClassifier,
        "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "grid": {
            "n_estimators": [50, 100, 200],
            "max_depth":    [None, 5, 10],
        },
        "description": "Ensemble of trees — robust and usually top performer.",
    },
    "gradient_boosting": {
        "class":  GradientBoostingClassifier,
        "params": {"n_estimators": 100, "random_state": 42},
        "grid": {
            "n_estimators":  [50, 100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth":     [3, 5],
        },
        "description": "Sequential boosting — often best accuracy on tabular data.",
    },
    "svm": {
        "class":  SVC,
        "params": {"probability": True, "random_state": 42},
        "grid": {
            "C":      [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
        },
        "description": "Support Vector Machine — strong for small/medium datasets.",
    },
    "knn": {
        "class":  KNeighborsClassifier,
        "params": {},
        "grid": {
            "n_neighbors": [3, 5, 7, 11],
            "weights":     ["uniform", "distance"],
        },
        "description": "K-Nearest Neighbours — simple, no training phase.",
    },
    "naive_bayes": {
        "class":  GaussianNB,
        "params": {},
        "grid": {},
        "description": "Probabilistic — extremely fast, works well with small data.",
    },
}


def build_model(name: str):
    """Instantiate a classifier by registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[name]
    return entry["class"](**entry["params"])
