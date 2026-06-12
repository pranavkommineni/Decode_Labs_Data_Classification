"""
evaluator.py  –  Multi-model comparison and result reporting.

Saves:
  results/comparison_report.txt
  results/confusion_matrix_<model>.txt
"""

import os
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import ConfusionMatrixDisplay

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("results")


class Evaluator:
    """Aggregates per-model metrics, prints comparison tables, saves artefacts."""

    def __init__(self):
        RESULTS_DIR.mkdir(exist_ok=True)
        self.results: list[dict] = []

    def add(self, metrics: dict):
        self.results.append(metrics)

    # ── Console report ────────────────────────────────────────────────

    def print_comparison(self):
        if not self.results:
            print("No results yet.")
            return

        header = f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Train(s)':>10}"
        sep    = "─" * len(header)
        print(sep)
        print(header)
        print(sep)

        sorted_results = sorted(self.results, key=lambda r: r["accuracy"], reverse=True)
        for i, r in enumerate(sorted_results):
            prefix = "🥇 " if i == 0 else "   "
            print(
                f"{prefix}{r['model']:<23} "
                f"{r['accuracy']:>10.4f} "
                f"{r['precision']:>10.4f} "
                f"{r['recall']:>10.4f} "
                f"{r['f1_score']:>10.4f} "
                f"{r['train_time_s']:>10.4f}"
            )
        print(sep)
        best = sorted_results[0]
        print(f"\n✅  Best model: {best['model']}  (accuracy={best['accuracy']:.4f})\n")

    def print_report(self, model_name: str):
        for r in self.results:
            if r["model"] == model_name:
                print(f"\n── Classification Report: {model_name} ──")
                print(r["classification_report"])
                return
        print(f"No results found for '{model_name}'.")

    # ── File artefacts ────────────────────────────────────────────────

    def save_comparison_csv(self):
        path = RESULTS_DIR / "comparison.csv"
        keys = ["model", "accuracy", "precision", "recall", "f1_score", "train_time_s"]
        with open(path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in self.results:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        logger.info("Saved %s", path)

    def save_json(self):
        path = RESULTS_DIR / "results.json"
        exportable = [{k: v for k, v in r.items() if k != "classification_report"}
                      for r in self.results]
        path.write_text(json.dumps(exportable, indent=2))
        logger.info("Saved %s", path)

    def plot_confusion_matrices(self, target_names: list[str]):
        """Save one PNG per model showing its confusion matrix."""
        for r in self.results:
            cm = np.array(r["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(max(4, len(target_names)), max(4, len(target_names))))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
            disp.plot(ax=ax, colorbar=True, cmap="Blues")
            ax.set_title(f"Confusion Matrix – {r['model']}")
            fig.tight_layout()
            out = RESULTS_DIR / f"cm_{r['model'].replace(' ', '_')}.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            logger.info("Saved %s", out)

    def plot_metric_comparison(self):
        """Bar chart comparing all models across key metrics."""
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        models  = [r["model"] for r in self.results]
        x       = np.arange(len(models))
        width   = 0.2
        colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 6))
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            vals = [r[metric] for r in self.results]
            ax.bar(x + i * width, vals, width, label=metric.replace("_", " ").title(), color=color)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.legend(loc="lower right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        out = RESULTS_DIR / "model_comparison.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info("Saved %s", out)
