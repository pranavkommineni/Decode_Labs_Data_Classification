"""
test_classification.py  –  Unit & integration tests.

Run:
    python -m pytest test_classification.py -v
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from data_loader import DataLoader
from models      import build_model, MODEL_REGISTRY
from trainer     import Trainer
from evaluator   import Evaluator


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def iris_loader():
    loader = DataLoader(test_size=0.2, random_state=42)
    loader.load("iris").preprocess(scale=True)
    return loader


@pytest.fixture(scope="session")
def trained_rf(iris_loader):
    model   = build_model("random_forest")
    trainer = Trainer(model, "random_forest")
    trainer.fit(iris_loader.X_train, iris_loader.y_train)
    return trainer


# ── DataLoader ────────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_iris(self, iris_loader):
        assert iris_loader.dataset_name == "iris"
        assert iris_loader.X_train is not None
        assert iris_loader.X_test is not None

    def test_shapes_consistent(self, iris_loader):
        assert iris_loader.X_train.shape[0] == iris_loader.y_train.shape[0]
        assert iris_loader.X_test.shape[0]  == iris_loader.y_test.shape[0]

    def test_summary_keys(self, iris_loader):
        s = iris_loader.summary()
        for key in ["name", "total_samples", "n_features", "n_classes", "class_names"]:
            assert key in s

    def test_load_wine(self):
        loader = DataLoader()
        loader.load("wine").preprocess()
        assert loader.dataset_name == "wine"
        assert loader.X_train is not None

    def test_load_breast_cancer(self):
        loader = DataLoader()
        loader.load("breast_cancer").preprocess()
        assert loader.n_classes == 2 if hasattr(loader, "n_classes") else True

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            DataLoader().load("nonexistent_dataset")

    def test_stratification(self, iris_loader):
        # test set should contain all 3 classes
        assert len(np.unique(iris_loader.y_test)) == 3

    def test_feature_scaling(self, iris_loader):
        # scaled train features should have approx mean=0, std=1
        means = iris_loader.X_train.mean(axis=0)
        stds  = iris_loader.X_train.std(axis=0)
        assert np.allclose(means, 0, atol=0.1)
        assert np.allclose(stds,  1, atol=0.1)


# ── Model registry ────────────────────────────────────────────────────

class TestModelRegistry:
    def test_all_models_buildable(self):
        for name in MODEL_REGISTRY:
            m = build_model(name)
            assert m is not None

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            build_model("super_deep_network")


# ── Trainer ──────────────────────────────────────────────────────────

class TestTrainer:
    def test_fit_produces_fitted_model(self, iris_loader):
        model   = build_model("logistic_regression")
        trainer = Trainer(model, "logistic_regression")
        trainer.fit(iris_loader.X_train, iris_loader.y_train)
        assert trainer._fitted

    def test_evaluate_accuracy_range(self, trained_rf, iris_loader):
        metrics = trained_rf.evaluate(iris_loader.X_test, iris_loader.y_test)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_keys_present(self, trained_rf, iris_loader):
        metrics = trained_rf.evaluate(iris_loader.X_test, iris_loader.y_test)
        for k in ["accuracy", "precision", "recall", "f1_score", "train_time_s"]:
            assert k in metrics

    def test_predict_shape(self, trained_rf, iris_loader):
        preds = trained_rf.predict(iris_loader.X_test)
        assert preds.shape == iris_loader.y_test.shape

    def test_cross_validate_keys(self, iris_loader):
        model   = build_model("decision_tree")
        trainer = Trainer(model, "decision_tree")
        X = np.vstack([iris_loader.X_train, iris_loader.X_test])
        y = np.concatenate([iris_loader.y_train, iris_loader.y_test])
        cv = trainer.cross_validate(X, y, cv=3)
        assert "mean" in cv and "std" in cv
        assert len(cv["scores"]) == 3

    def test_evaluate_before_fit_raises(self, iris_loader):
        model   = build_model("knn")
        trainer = Trainer(model, "knn")
        with pytest.raises(RuntimeError):
            trainer.evaluate(iris_loader.X_test, iris_loader.y_test)

    def test_all_models_reach_decent_accuracy(self, iris_loader):
        """All models should exceed 85% on Iris (a well-separated dataset)."""
        for name in ["logistic_regression", "random_forest", "svm", "knn", "naive_bayes"]:
            model   = build_model(name)
            trainer = Trainer(model, name)
            trainer.fit(iris_loader.X_train, iris_loader.y_train)
            metrics = trainer.evaluate(iris_loader.X_test, iris_loader.y_test)
            assert metrics["accuracy"] >= 0.85, (
                f"{name} only scored {metrics['accuracy']:.4f}"
            )


# ── Evaluator ─────────────────────────────────────────────────────────

class TestEvaluator:
    def test_add_and_compare(self, trained_rf, iris_loader):
        metrics   = trained_rf.evaluate(iris_loader.X_test, iris_loader.y_test)
        evaluator = Evaluator()
        evaluator.add(metrics)
        assert len(evaluator.results) == 1
        evaluator.print_comparison()   # should not raise
