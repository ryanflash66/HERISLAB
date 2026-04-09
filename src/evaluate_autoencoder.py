"""
Evaluate the trained autoencoder for anomaly detection (Track A).

Loads the best model, computes reconstruction error on test/normal
and test/fault images, calibrates a threshold, and reports metrics.

Steps:
  1. Load best model weights
  2. Compute per-image MSE on test/normal and test/fault
  3. Find optimal threshold (maximizes F1 score)
  4. Report AUROC, precision, recall, F1
  5. Save reconstruction error distributions for visualization
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import torch
from pathlib import Path
from train_autoencoder import ThermalAutoencoder

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CA_Preprocessed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_reconstruction_errors(model, data, batch_size=32):
    """Compute per-image mean squared error between input and reconstruction."""
    model.eval()
    errors = []

    # Add channel dim if needed: (N, 240, 320) -> (N, 1, 240, 320)
    if data.ndim == 3:
        data = data[:, np.newaxis, :, :]

    dataset = torch.from_numpy(data).float()

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size].to(DEVICE)
            output = model(batch)
            # Per-image MSE: average over (channel, height, width)
            mse = ((batch - output) ** 2).mean(dim=(1, 2, 3))
            errors.extend(mse.cpu().numpy())

    return np.array(errors)


def find_best_threshold(normal_errors, fault_errors):
    """Find threshold that maximizes F1 score.

    Labels: normal=0 (negative), fault=1 (positive).
    Prediction: error > threshold -> fault (positive).
    """
    all_errors = np.concatenate([normal_errors, fault_errors])
    labels = np.concatenate([
        np.zeros(len(normal_errors)),
        np.ones(len(fault_errors)),
    ])

    # Try many thresholds between min and max error
    thresholds = np.linspace(all_errors.min(), all_errors.max(), 1000)
    best_f1 = 0
    best_thresh = 0
    best_metrics = {}

    for thresh in thresholds:
        preds = (all_errors > thresh).astype(int)

        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                "threshold": float(thresh),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(tp), "fp": int(fp),
                "fn": int(fn), "tn": int(tn),
            }

    return best_thresh, best_metrics


def compute_auroc(normal_errors, fault_errors):
    """Compute AUROC by sweeping thresholds (no sklearn needed)."""
    all_errors = np.concatenate([normal_errors, fault_errors])
    labels = np.concatenate([
        np.zeros(len(normal_errors)),
        np.ones(len(fault_errors)),
    ])

    # Sort by error descending
    sorted_indices = np.argsort(-all_errors)
    sorted_labels = labels[sorted_indices]

    # Count positives and negatives
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    # Walk through sorted predictions, accumulating TPR and FPR
    tp = 0
    fp = 0
    tpr_points = [0.0]
    fpr_points = [0.0]

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_points.append(tp / n_pos)
        fpr_points.append(fp / n_neg)

    # Trapezoidal integration
    auroc = 0.0
    for i in range(1, len(fpr_points)):
        auroc += (fpr_points[i] - fpr_points[i-1]) * (tpr_points[i] + tpr_points[i-1]) / 2

    return auroc


def main():
    print(f"Device: {DEVICE}")

    # Load model
    print("\nLoading model...")
    model = ThermalAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "autoencoder_best.pth", weights_only=True))
    print("  Loaded autoencoder_best.pth")

    # Load test data
    print("\nLoading test data...")
    test_normal = np.load(DATA_DIR / "test_normal.npy")
    test_fault = np.load(DATA_DIR / "test_fault.npy")
    print(f"  Test normal: {test_normal.shape[0]} images")
    print(f"  Test fault:  {test_fault.shape[0]} images")

    # Compute reconstruction errors
    print("\nComputing reconstruction errors...")
    normal_errors = compute_reconstruction_errors(model, test_normal)
    fault_errors = compute_reconstruction_errors(model, test_fault)

    print(f"\n  Normal errors -- mean: {normal_errors.mean():.6f}, std: {normal_errors.std():.6f}")
    print(f"    min: {normal_errors.min():.6f}, max: {normal_errors.max():.6f}")
    print(f"\n  Fault errors  -- mean: {fault_errors.mean():.6f}, std: {fault_errors.std():.6f}")
    print(f"    min: {fault_errors.min():.6f}, max: {fault_errors.max():.6f}")

    # Separation check
    separation = fault_errors.mean() - normal_errors.mean()
    print(f"\n  Mean separation (fault - normal): {separation:.6f}")

    # AUROC
    auroc = compute_auroc(normal_errors, fault_errors)
    print(f"\n  AUROC: {auroc:.4f}")

    # Find best threshold
    threshold, metrics = find_best_threshold(normal_errors, fault_errors)
    print(f"\n  Best threshold: {threshold:.6f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"    TP: {metrics['tp']:4d}  (faults correctly detected)")
    print(f"    FP: {metrics['fp']:4d}  (normals wrongly flagged)")
    print(f"    FN: {metrics['fn']:4d}  (faults missed)")
    print(f"    TN: {metrics['tn']:4d}  (normals correctly passed)")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(RESULTS_DIR / "normal_errors.npy", normal_errors)
    np.save(RESULTS_DIR / "fault_errors.npy", fault_errors)
    np.save(RESULTS_DIR / "eval_metrics.npy", {
        "auroc": auroc,
        "threshold": threshold,
        **metrics,
    })

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
