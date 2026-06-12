"""
main.py  –  Entry point for the Data Classification pipeline.

Usage:
    python main.py                        # default: iris, all models
    python main.py --dataset wine         # use wine dataset
    python main.py --dataset breast_cancer --models random_forest svm
    python main.py --dataset data/my.csv  # custom CSV
    python main.py --tune                 # enable hyperparameter search
    python main.py --cv 10               # 10-fold cross-validation

Run:
    python main.py --help
"""

import argparse
import logging
import sys

from data_loader import DataLoader, BUILT_IN_DATASETS
from models      import MODEL_REGISTRY, build_model
from trainer     import Trainer
from evaluator   import Evaluator

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Data Classification Pipeline")
    parser.add_argument(
        "--dataset", default="iris",
        help=f"Built-in name {list(BUILT_IN_DATASETS)} or path to a CSV file. (default: iris)"
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
        help="Which models to train (default: all)"
    )
    parser.add_argument(
        "--cv", type=int, default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data held out for testing (default: 0.2)"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run GridSearchCV hyper-parameter tuning for each model"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip saving PNG charts"
    )
    return parser.parse_args()


# ── Main pipeline ────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load & preprocess data ────────────────────────────────────
    print("\n" + "═" * 60)
    print("  📊  Data Classification Pipeline")
    print("═" * 60)

    loader = DataLoader(test_size=args.test_size)
    loader.load(args.dataset).preprocess(scale=True)

    summary = loader.summary()
    print(f"\n  Dataset : {summary['name']}")
    print(f"  Samples : {summary['total_samples']}")
    print(f"  Features: {summary['n_features']}")
    print(f"  Classes : {summary['n_classes']}  → {summary['class_names']}")
    print(f"  Train/Test split: {summary['train_samples']} / {summary['test_samples']}")
    print()

    evaluator = Evaluator()

    # ── 2. Train each model ──────────────────────────────────────────
    for model_name in args.models:
        print(f"  ▶  {model_name}")

        sklearn_model = build_model(model_name)
        trainer       = Trainer(sklearn_model, model_name)

        # Cross-validation (on full scaled data)
        import numpy as np
        X_all = np.vstack([loader.X_train, loader.X_test])
        y_all = np.concatenate([loader.y_train, loader.y_test])
        cv_result = trainer.cross_validate(X_all, y_all, cv=args.cv)
        print(f"     CV ({args.cv}-fold): {cv_result['mean']:.4f} ± {cv_result['std']:.4f}")

        # Optional tuning
        if args.tune:
            param_grid = MODEL_REGISTRY[model_name]["grid"]
            if param_grid:
                tune_result = trainer.tune(loader.X_train, loader.y_train, param_grid, cv=args.cv)
                print(f"     Best params: {tune_result['best_params']}")
            else:
                trainer.fit(loader.X_train, loader.y_train)
        else:
            trainer.fit(loader.X_train, loader.y_train)

        # Evaluate on held-out test set
        metrics = trainer.evaluate(
            loader.X_test, loader.y_test,
            target_names=loader.target_names,
        )
        evaluator.add(metrics)
        print(f"     Test accuracy: {metrics['accuracy']:.4f}  |  F1: {metrics['f1_score']:.4f}")

    # ── 3. Report ────────────────────────────────────────────────────
    evaluator.print_comparison()
    evaluator.save_comparison_csv()
    evaluator.save_json()

    if not args.no_plots:
        evaluator.plot_metric_comparison()
        evaluator.plot_confusion_matrices(loader.target_names)
        print("  📈  Charts saved to results/")

    print("\n  Done. Results saved to results/\n")


if __name__ == "__main__":
    main()
