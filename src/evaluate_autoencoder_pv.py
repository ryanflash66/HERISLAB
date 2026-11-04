"""
Evaluate the PV-specialist autoencoder (HER-90).

Loads the "pv" bundle and scores it against the PV-only test split produced
by src/preprocess_pv.py. That split is random per frame, so frames from the
same video can sit in both train and test (not independent). Reuses
score_errors from evaluate_autoencoder.py (metrics from hlab.metrics), so the
metric code is the same as the v2 baseline; the test sets and the
preprocessing differ, so the numbers are not a like-for-like comparison.

Inputs (through hlab.bundle / hlab.datasets):
  models/autoencoder_pv_best.pth
  data/CA_Preprocessed/pv/test_normal.npy
  data/CA_Preprocessed/pv/test_fault.npy   (if present; HER-87 deliverable)

Outputs:
  results/pv_normal_errors.npy
  results/pv_fault_errors.npy   (if test_fault.npy exists)
  results/pv_eval_metrics.npy   (only when we have faults to calibrate against)
  results/eval_log_pv.txt       (console tee)

--bundle pv_matched (HER-107) scores the model trained on
data/CA_Preprocessed/pv_matched/ against that folder's arrays and writes the
same files with a pv_matched_ prefix (eval_log_pv_matched.txt).

Usage:
    venv/Scripts/python.exe src/evaluate_autoencoder_pv.py [--bundle pv|pv_matched]
"""

import argparse
import sys

import numpy as np

from evaluate_autoencoder import print_metrics, score_errors
from hlab.bundle import Bundle
from hlab.datasets import Dataset
from hlab.paths import RESULTS

BUNDLE_NAMES = ("pv", "pv_matched")   # bundle name = dataset name = output prefix


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a PV-specialist bundle.")
    parser.add_argument("--bundle", choices=BUNDLE_NAMES, default="pv")
    name = parser.parse_args(argv).bundle
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_handle = (RESULTS / f"eval_log_{name}.txt").open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_handle)

    try:
        # --- Load PV specialist bundle ---
        try:
            bundle = Bundle.load(name)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            print(f"Run the {name} preprocess script and src/train.py --config {name} first.")
            return
        print(f"Device: {bundle.device}")
        if bundle.device.type == "cuda":
            import torch
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("\nLoading PV specialist model...")
        bundle.model
        print(f"  Loaded {bundle.weights.name}")

        # --- Test normals ---
        dataset = Dataset(name)
        if not dataset.has_split("holdout"):
            print(f"ERROR: {dataset} has no holdout array. Run src/preprocess_pv.py first.")
            return
        print("\nLoading PV holdout normals...")
        test_normal = dataset.split("holdout").x
        print(f"  Test normal: {test_normal.shape[0]} images")

        print("\nComputing reconstruction errors on holdout normals...")
        normal_errors = bundle.score_normalized(test_normal)
        print(f"  mean: {normal_errors.mean():.6f}  std: {normal_errors.std():.6f}")
        print(f"  min:  {normal_errors.min():.6f}  max: {normal_errors.max():.6f}")

        np.save(RESULTS / f"{name}_normal_errors.npy", normal_errors)
        print(f"  Saved: {RESULTS / f'{name}_normal_errors.npy'}")

        # --- Test faults (if available) ---
        if not dataset.has_split("fault"):
            print(f"\nSKIP fault eval: {dataset} has no fault array.")
            print("Waiting on HER-87 fault dataset. Full metrics / threshold calibration")
            print("are deferred until test_fault.npy exists.")
            print("\n--- Partial eval complete (normals only). ---")
            return

        print("\nLoading PV fault test set...")
        test_fault = dataset.split("fault").x
        print(f"  Test fault: {test_fault.shape[0]} images")

        print("\nComputing reconstruction errors on faults...")
        fault_errors = bundle.score_normalized(test_fault)
        print(f"  mean: {fault_errors.mean():.6f}  std: {fault_errors.std():.6f}")
        print(f"  min:  {fault_errors.min():.6f}  max: {fault_errors.max():.6f}")

        # --- Metrics ---
        separation = fault_errors.mean() - normal_errors.mean()
        print(f"\n  Mean separation (fault - normal): {separation:.6f}")

        auroc, threshold, metrics = score_errors(normal_errors, fault_errors)
        print_metrics(auroc, threshold, metrics)

        # v2 comparison
        print("\n  v2 baseline (for comparison):")
        print("    AUROC: 0.9023   F1: 0.9807   Precision: 0.9621   Recall: 1.0000")

        # Save
        np.save(RESULTS / f"{name}_fault_errors.npy", fault_errors)
        np.save(RESULTS / f"{name}_eval_metrics.npy", {
            "auroc": auroc,
            "threshold": threshold,
            **metrics,
        })
        print(f"\nSaved: {RESULTS / f'{name}_fault_errors.npy'}")
        print(f"Saved: {RESULTS / f'{name}_eval_metrics.npy'}")

        print("\n--- PV specialist evaluation complete. ---")
    finally:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main()
