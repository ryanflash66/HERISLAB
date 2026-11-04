"""
Evaluate the trained autoencoder for anomaly detection (Track A).

Loads the v2 bundle (weights, norm stats), computes reconstruction error on
the v2 test_normal and test_fault arrays, calibrates a threshold, and reports
metrics.

Steps:
  1. Load the bundle (hlab.bundle, HER-126)
  2. Compute per-image MSE on test/normal and test/fault (hlab.datasets, HER-127)
  3. Find optimal threshold (maximizes F1 score; hlab.metrics, HER-122)
  4. Report AUROC, precision, recall, F1
  5. Save reconstruction error distributions for visualization
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np

from hlab.bundle import Bundle
from hlab.datasets import Dataset
from hlab.metrics import auroc, best_f1_threshold, confusion, from_groups
from hlab.paths import RESULTS

# Keys of the confusion dict that go into *_eval_metrics.npy (unchanged since v2).
SAVED_KEYS = ("precision", "recall", "f1", "tp", "fp", "fn", "tn")


def score_errors(normal_errors, fault_errors):
    """AUROC, best-F1 threshold and the confusion at it, for two error arrays."""
    scores, labels = from_groups(normal_errors, fault_errors)
    threshold = best_f1_threshold(scores, labels)
    metrics = {k: v for k, v in confusion(scores, labels, threshold).items() if k in SAVED_KEYS}
    return auroc(scores, labels), threshold, metrics


def print_metrics(auroc_value, threshold, metrics):
    print(f"\n  AUROC: {auroc_value:.4f}")
    print(f"\n  Best threshold: {threshold:.6f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"    TP: {metrics['tp']:4d}  (faults correctly detected)")
    print(f"    FP: {metrics['fp']:4d}  (normals wrongly flagged)")
    print(f"    FN: {metrics['fn']:4d}  (faults missed)")
    print(f"    TN: {metrics['tn']:4d}  (normals correctly passed)")


def main():
    bundle = Bundle.load("v2")
    print(f"Device: {bundle.device}")

    # Load model
    print("\nLoading model...")
    bundle.model
    print(f"  Loaded {bundle.weights.name}")

    # Load test data
    print("\nLoading test data...")
    dataset = Dataset("v2")
    test_normal = dataset.split("test_normal").x
    test_fault = dataset.split("test_fault").x
    print(f"  Test normal: {test_normal.shape[0]} images")
    print(f"  Test fault:  {test_fault.shape[0]} images")

    # Compute reconstruction errors
    print("\nComputing reconstruction errors...")
    normal_errors = bundle.score_normalized(test_normal)
    fault_errors = bundle.score_normalized(test_fault)

    print(f"\n  Normal errors -- mean: {normal_errors.mean():.6f}, std: {normal_errors.std():.6f}")
    print(f"    min: {normal_errors.min():.6f}, max: {normal_errors.max():.6f}")
    print(f"\n  Fault errors  -- mean: {fault_errors.mean():.6f}, std: {fault_errors.std():.6f}")
    print(f"    min: {fault_errors.min():.6f}, max: {fault_errors.max():.6f}")

    # Separation check
    separation = fault_errors.mean() - normal_errors.mean()
    print(f"\n  Mean separation (fault - normal): {separation:.6f}")

    # AUROC, threshold, confusion
    auroc_value, threshold, metrics = score_errors(normal_errors, fault_errors)
    print_metrics(auroc_value, threshold, metrics)

    # Save results
    os.makedirs(RESULTS, exist_ok=True)
    np.save(RESULTS / "normal_errors.npy", normal_errors)
    np.save(RESULTS / "fault_errors.npy", fault_errors)
    np.save(RESULTS / "eval_metrics.npy", {
        "auroc": auroc_value,
        "threshold": threshold,
        **metrics,
    })

    print(f"\nResults saved to {RESULTS}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
