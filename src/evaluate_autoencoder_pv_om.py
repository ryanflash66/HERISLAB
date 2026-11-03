"""
Evaluate the O&M-only PV specialist (HER-114 retrain).

Scores models/autoencoder_pv_om_best.pth on

  1. the O&M-only holdout (data/CA_Preprocessed/pv_om/test_normal.npy),
  2. the PVMD fault set (data/CA_Preprocessed/pv_om/test_fault.npy, 1,000 files),
  3. the THM aerial-survey frames, which this model never saw: the 84 THM
     frames held out by the original pv split and the full 1,075-frame set,
     preprocessed exactly as in preprocess_pv.py and z-scored with the O&M
     norm stats.

Outputs (all new; results/pv_*.npy from the THM+O&M model are not touched):
  results/pv_om_normal_errors.npy
  results/pv_om_fault_errors.npy
  results/pv_om_eval_metrics.npy      auroc, re-tuned threshold, confusion
  results/pv_om_thm_errors.npy        1,075 errors, THM file order
  results/pv_om_thm_errors.csv        frame, pv_split (train/holdout), error, flagged
  results/eval_log_pv_om.txt

Usage:
    venv/Scripts/python.exe src/evaluate_autoencoder_pv_om.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch

from train_autoencoder import ThermalAutoencoder
from evaluate_autoencoder import compute_reconstruction_errors, find_best_threshold, compute_auroc
from preprocess_pv import load_image_raw

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CA_Preprocessed" / "pv_om"
ORIG_DIR = ROOT / "data" / "CA_Preprocessed" / "pv"
THM_DIR = ROOT / "data" / "CA_Training_Data" / "train" / "normal" / "pv_thermal_inspection"
MODEL_PATH = ROOT / "models" / "autoencoder_pv_om_best.pth"
RESULTS_DIR = ROOT / "results"
LOG_PATH = RESULTS_DIR / "eval_log_pv_om.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Threshold of the THM+O&M model (results/pv_eval_metrics.npy), for reference only.
OLD_PV_THR = 0.027865182608366013


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def confusion(normal, fault, thr):
    tp = int((fault > thr).sum()); fn = len(fault) - tp
    fp = int((normal > thr).sum()); tn = len(normal) - fp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / len(fault)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=p, recall=r, f1=f1)


def original_holdout_thm():
    """THM frame names in the HOLDOUT section of the original pv split manifest."""
    names, sec = set(), None
    for line in (ORIG_DIR / "split_manifest.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "[TRAIN]": sec = "tr"; continue
        if line == "[HOLDOUT]": sec = "ho"; continue
        if sec == "ho" and "pv_thermal_inspection/" in line:
            names.add(Path(line).name)
    return names


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_PATH.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_handle)
    try:
        print(f"Device: {DEVICE}")
        print(f"Model:  {MODEL_PATH}")
        model = ThermalAutoencoder().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

        stats = np.load(DATA_DIR / "norm_stats.npy", allow_pickle=True).item()
        print(f"O&M norm stats: mean={stats['mean']:.4f} std={stats['std']:.4f}")

        # --- O&M holdout + PVMD faults ---
        test_normal = np.load(DATA_DIR / "test_normal.npy")
        test_fault = np.load(DATA_DIR / "test_fault.npy")
        print(f"\nO&M holdout: {test_normal.shape[0]}   PVMD faults: {test_fault.shape[0]}")
        normal_errors = compute_reconstruction_errors(model, test_normal)
        fault_errors = compute_reconstruction_errors(model, test_fault)
        for lab, e in [("healthy O&M", normal_errors), ("fault", fault_errors)]:
            print(f"  {lab:12s} mean {e.mean():.6f}  median {np.median(e):.6f}  std {e.std():.6f}  "
                  f"min {e.min():.6f}  max {e.max():.6f}")
        np.save(RESULTS_DIR / "pv_om_normal_errors.npy", normal_errors)
        np.save(RESULTS_DIR / "pv_om_fault_errors.npy", fault_errors)

        auroc = compute_auroc(normal_errors, fault_errors)
        threshold, metrics = find_best_threshold(normal_errors, fault_errors)
        print(f"\n  AUROC (1,000 fault files): {auroc:.4f}")
        print(f"  Re-tuned best-F1 threshold: {threshold:.6f}")
        print(f"  Precision {metrics['precision']:.4f}  Recall {metrics['recall']:.4f}  F1 {metrics['f1']:.4f}")
        print(f"  TP {metrics['tp']}  FP {metrics['fp']}  FN {metrics['fn']}  TN {metrics['tn']}")
        c_old = confusion(normal_errors, fault_errors, OLD_PV_THR)
        print(f"  At the THM+O&M model's threshold {OLD_PV_THR:.6f} (reference only): "
              f"FP {c_old['fp']}  FN {c_old['fn']}  F1 {c_old['f1']:.4f}")
        print("  THM+O&M model, same fault set, 892 healthy (808 O&M + 84 THM): AUROC 0.9741  F1 0.9440  FP 87")

        # --- THM frames (never seen by this model) ---
        print(f"\nScoring THM frames from {THM_DIR} ...")
        thm_files = sorted(p for p in THM_DIR.iterdir() if p.is_file())
        arrs = []
        for i, p in enumerate(thm_files):
            arrs.append(load_image_raw(p))
            if (i + 1) % 250 == 0 or i + 1 == len(thm_files):
                print(f"    loaded {i + 1}/{len(thm_files)}")
        thm = (np.stack(arrs) - stats["mean"]) / stats["std"]
        thm_errors = compute_reconstruction_errors(model, thm.astype(np.float32))
        ho_names = original_holdout_thm()
        split = np.array(["holdout" if p.name in ho_names else "train" for p in thm_files])
        flagged = thm_errors > threshold
        print(f"  THM all {len(thm_errors)}: median err {np.median(thm_errors):.4f}  "
              f"flagged at re-tuned thr {flagged.sum()} ({flagged.mean():.1%})")
        for s in ("holdout", "train"):
            m = split == s
            print(f"  THM {s:7s} {m.sum():4d}: median err {np.median(thm_errors[m]):.4f}  "
                  f"flagged {flagged[m].sum()} ({flagged[m].mean():.1%})")
        print(f"  (THM+O&M model flagged 81 of 84 held-out THM frames at its threshold)")
        np.save(RESULTS_DIR / "pv_om_thm_errors.npy", thm_errors)
        with (RESULTS_DIR / "pv_om_thm_errors.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["frame", "pv_split", "error", "flagged"])
            for p, s, e, f in zip(thm_files, split, thm_errors, flagged):
                w.writerow([p.name, s, f"{e:.6f}", int(f)])

        np.save(RESULTS_DIR / "pv_om_eval_metrics.npy", {
            "auroc": auroc, "threshold": threshold, **metrics,
            "old_threshold": OLD_PV_THR, "old_threshold_confusion": c_old,
            "thm_flag_rate_all": float(flagged.mean()),
            "thm_flag_rate_holdout84": float(flagged[split == "holdout"].mean()),
            "n_normal": int(len(normal_errors)), "n_fault": int(len(fault_errors)),
        })
        print("\nSaved results/pv_om_normal_errors.npy, pv_om_fault_errors.npy, pv_om_eval_metrics.npy, "
              "pv_om_thm_errors.npy, pv_om_thm_errors.csv")
    finally:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main()
