# 📊 Data Classification Using AI

A full end-to-end supervised classification pipeline using **scikit-learn**, supporting multiple datasets and 7 classifiers with cross-validation, optional hyperparameter tuning, and rich output artefacts.

## Architecture

```
project2_data_classification/
│
├── main.py               ← CLI entry point & pipeline orchestrator
├── data_loader.py        ← Dataset loading, validation, train/test split, scaling
├── models.py             ← Registry of 7 classifiers with hyper-parameter grids
├── trainer.py            ← Fit, cross-validate, tune (GridSearchCV), evaluate
├── evaluator.py          ← Multi-model comparison table + charts
├── test_classification.py← 20+ unit & integration tests (pytest)
├── requirements.txt
└── results/              ← Auto-created at runtime
    ├── comparison.csv
    ├── results.json
    ├── model_comparison.png
    └── cm_<model>.png     (one per model)
```

## Classifiers

| Name | Notes |
|---|---|
| `logistic_regression` | Fast linear baseline |
| `decision_tree` | Interpretable |
| `random_forest` | Robust ensemble ⭐ |
| `gradient_boosting` | Often best accuracy |
| `svm` | Strong on small data |
| `knn` | No training phase |
| `naive_bayes` | Extremely fast |

## Quick Start

```bash
pip install -r requirements.txt

# Run on Iris (default)
python main.py

# Run on Wine dataset with only two models
python main.py --dataset wine --models random_forest svm

# Run on Breast Cancer with hyperparameter tuning + 10-fold CV
python main.py --dataset breast_cancer --tune --cv 10

# Custom CSV (last column = target)
python main.py --dataset data/my_data.csv
```

## Sample Output

```
══════════════════════════════════════════════════════════
  📊  Data Classification Pipeline
══════════════════════════════════════════════════════════

  Dataset : iris
  Samples : 150
  Features: 4
  Classes : 3  → ['setosa', 'versicolor', 'virginica']
  Train/Test split: 120 / 30

  ▶  random_forest
     CV (5-fold): 0.9667 ± 0.0167
     Test accuracy: 1.0000  |  F1: 1.0000
  ...

────────────────────────────────────────────────────────────────────────────
  Model                     Accuracy  Precision     Recall         F1  Train(s)
────────────────────────────────────────────────────────────────────────────
🥇 random_forest              1.0000     1.0000     1.0000     1.0000    0.1523
   svm                        1.0000     1.0000     1.0000     1.0000    0.0031
   ...
```

## Running Tests

```bash
python -m pytest test_classification.py -v
```

All 20+ tests should pass. Key coverage:
- DataLoader: shapes, stratification, scaling, unknown-source errors
- Model registry: all 7 models instantiate correctly
- Trainer: fit, evaluate, cross-validate, predict, guard conditions
- All 5 core models reach ≥ 85% accuracy on Iris
