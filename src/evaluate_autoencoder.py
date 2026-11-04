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

Metrics come from hlab.metrics (HER-122): Mann-Whitney AUROC and an exact
best-F1 search over the distinct error values.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import torch
from pathlib import Path
from train_autoencoder import ThermalAutoencoder
from hlab.metrics import auroc, best_f1_threshold, confusion, from_groups

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CA_Preprocessed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keys of the confusion dict that go into *_eval_metrics.npy (unchanged since v2).
SAVED_KEYS = ("precision", "recall", "f1", "tp", "fp", "fn", "tn")


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

    # AUROC, threshold, confusion
    auroc_value, threshold, metrics = score_errors(normal_errors, fault_errors)
    print_metrics(auroc_value, threshold, metrics)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(RESULTS_DIR / "normal_errors.npy", normal_errors)
    np.save(RESULTS_DIR / "fault_errors.npy", fault_errors)
    np.save(RESULTS_DIR / "eval_metrics.npy", {
        "auroc": auroc_value,
        "threshold": threshold,
        **metrics,
    })

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
