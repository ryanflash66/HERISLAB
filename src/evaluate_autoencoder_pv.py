"""
Evaluate the PV-specialist autoencoder (HER-90).

Loads the PV-specialist weights and scores them against the PV-only held-out
test set produced by src/preprocess_pv.py. Reuses the threshold calibration
+ AUROC helpers from evaluate_autoencoder.py so the numbers are directly
comparable to the v2 baseline.

Inputs:
  models/autoencoder_pv_best.pth
  data/CA_Preprocessed/pv/test_normal.npy
  data/CA_Preprocessed/pv/test_fault.npy   (if present; HER-87 deliverable)

Outputs:
  results/pv_normal_errors.npy
  results/pv_fault_errors.npy   (if test_fault.npy exists)
  results/pv_eval_metrics.npy   (only when we have faults to calibrate against)
  results/eval_log_pv.txt       (console tee)
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

from train_autoencoder import ThermalAutoencoder
from evaluate_autoencoder import (
    compute_reconstruction_errors,
    find_best_threshold,
    compute_auroc,
)

ROOT = Path(__file__).resolve().parent.parent
PV_DATA_DIR = ROOT / "data" / "CA_Preprocessed" / "pv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = MODEL_DIR / "autoencoder_pv_best.pth"
NORMAL_NPY = PV_DATA_DIR / "test_normal.npy"
FAULT_NPY = PV_DATA_DIR / "test_fault.npy"
LOG_PATH = RESULTS_DIR / "eval_log_pv.txt"


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_PATH.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_handle)

    try:
        print(f"Device: {DEVICE}")
        if DEVICE.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # --- Load PV specialist model ---
        if not MODEL_PATH.exists():
            print(f"ERROR: PV specialist weights not found at {MODEL_PATH}")
            print("Run src/train_autoencoder_pv.py first.")
            return
        print("\nLoading PV specialist model...")
        model = ThermalAutoencoder().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        print(f"  Loaded {MODEL_PATH.name}")

        # --- Test normals ---
        if not NORMAL_NPY.exists():
            print(f"ERROR: {NORMAL_NPY} not found. Run src/preprocess_pv.py first.")
            return
        print("\nLoading PV holdout normals...")
        test_normal = np.load(NORMAL_NPY)
        print(f"  Test normal: {test_normal.shape[0]} images")

        print("\nComputing reconstruction errors on holdout normals...")
        normal_errors = compute_reconstruction_errors(model, test_normal)
        print(f"  mean: {normal_errors.mean():.6f}  std: {normal_errors.std():.6f}")
        print(f"  min:  {normal_errors.min():.6f}  max: {normal_errors.max():.6f}")

        np.save(RESULTS_DIR / "pv_normal_errors.npy", normal_errors)
        print(f"  Saved: {RESULTS_DIR / 'pv_normal_errors.npy'}")

        # --- Test faults (if available) ---
        if not FAULT_NPY.exists():
            print(f"\nSKIP fault eval: {FAULT_NPY} not found.")
            print("Waiting on HER-87 fault dataset. Full metrics / threshold calibration")
            print("are deferred until test_fault.npy exists.")
            print("\n--- Partial eval complete (normals only). ---")
            return

        print("\nLoading PV fault test set...")
        test_fault = np.load(FAULT_NPY)
        print(f"  Test fault: {test_fault.shape[0]} images")

        print("\nComputing reconstruction errors on faults...")
        fault_errors = compute_reconstruction_errors(model, test_fault)
        print(f"  mean: {fault_errors.mean():.6f}  std: {fault_errors.std():.6f}")
        print(f"  min:  {fault_errors.min():.6f}  max: {fault_errors.max():.6f}")

        # --- Metrics ---
        separation = fault_errors.mean() - normal_errors.mean()
        print(f"\n  Mean separation (fault - normal): {separation:.6f}")

        auroc = compute_auroc(normal_errors, fault_errors)
        print(f"  AUROC: {auroc:.4f}")

        threshold, metrics = find_best_threshold(normal_errors, fault_errors)
        print(f"\n  Best F1 threshold: {threshold:.6f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"\n  Confusion matrix:")
        print(f"    TP: {metrics['tp']:4d}  (faults correctly detected)")
        print(f"    FP: {metrics['fp']:4d}  (normals wrongly flagged)")
        print(f"    FN: {metrics['fn']:4d}  (faults missed)")
        print(f"    TN: {metrics['tn']:4d}  (normals correctly passed)")

        # v2 comparison
        print("\n  v2 baseline (for comparison):")
        print("    AUROC: 0.9023   F1: 0.9807   Precision: 0.9621   Recall: 1.0000")

        # Save
        np.save(RESULTS_DIR / "pv_fault_errors.npy", fault_errors)
        np.save(RESULTS_DIR / "pv_eval_metrics.npy", {
            "auroc": auroc,
            "threshold": threshold,
            **metrics,
        })
        print(f"\nSaved: {RESULTS_DIR / 'pv_fault_errors.npy'}")
        print(f"Saved: {RESULTS_DIR / 'pv_eval_metrics.npy'}")

        print("\n--- PV specialist evaluation complete. ---")
    finally:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main()
