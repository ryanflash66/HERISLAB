"""
Evaluate the trained autoencoder for anomaly detection.

Loads the v2 bundle (weights, norm stats), computes reconstruction error on
the v2 test_normal and test_fault arrays, chooses the alarm threshold on the
validation half and reports the test half at it.

Steps:
  1. Load the bundle (hlab.bundle, HER-126)
  2. Compute per-image MSE on test/normal and test/fault (hlab.datasets, HER-127)
  3. Choose the best-F1 threshold on the validation half only and freeze it
     (Dataset.calibration, docs/data_manifests/v2_calibration_split.csv; HER-97)
  4. Report AUROC, precision, recall, F1 of the test half at that threshold
  5. Save reconstruction error distributions for visualization
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np

from hlab.bundle import Bundle
from hlab.datasets import TEST, VAL, Dataset
from hlab.metrics import auroc, best_f1_threshold, confusion, from_groups
from hlab.paths import RESULTS

# Keys of the confusion dict that go into *_eval_metrics.npy (unchanged since v2).
SAVED_KEYS = ("precision", "recall", "f1", "tp", "fp", "fn", "tn")
THRESHOLD_FROM = "best F1 on the validation half, frozen, applied once to the test half (HER-97)"


def score_errors(normal_errors, fault_errors):
    """AUROC, best-F1 threshold and the confusion at it, for two error arrays.

    The threshold is tuned on the errors it is scored on; the PV and v2 evaluators use calibrate().
    """
    scores, labels = from_groups(normal_errors, fault_errors)
    threshold = best_f1_threshold(scores, labels)
    metrics = {k: v for k, v in confusion(scores, labels, threshold).items() if k in SAVED_KEYS}
    return auroc(scores, labels), threshold, metrics


def _half(normal_errors, fault_errors, normal_role, fault_role, role, fault_groups=None):
    """(healthy errors, fault errors) of one half; with fault_groups, one mean error per group."""
    neg = np.asarray(normal_errors)[np.asarray(normal_role) == role]
    keep = np.asarray(fault_role) == role
    pos = np.asarray(fault_errors)[keep]
    if fault_groups is not None:
        g = np.asarray(fault_groups)[keep]
        pos = np.array([pos[g == k].mean() for k in np.unique(g)])
    return neg, pos


def _summary(neg, pos, threshold):
    scores, labels = from_groups(neg, pos)
    c = confusion(scores, labels, threshold)
    return dict(n_normal=int(len(neg)), n_fault=int(len(pos)), auroc=auroc(scores, labels),
                **{k: c[k] for k in (*SAVED_KEYS, "specificity")})


def calibrate(normal_errors, fault_errors, normal_role, fault_role, fault_groups=None):
    """Alarm threshold chosen on the VAL half, frozen, and applied once to the TEST half (HER-97).

    normal_role / fault_role: VAL | TEST per frame (Dataset.calibration). fault_groups: one id per
    fault frame (pixel-unique image); when given, each unique fault counts once on both halves (its
    mean error over the copies), and the per-file test numbers go under "test_files".
    Returns dict(threshold, threshold_from, val, test[, test_files]); val and test hold
    n_normal, n_fault, auroc and the confusion at the threshold.
    """
    val_neg, val_pos = _half(normal_errors, fault_errors, normal_role, fault_role, VAL, fault_groups)
    threshold = best_f1_threshold(*from_groups(val_neg, val_pos))
    test_neg, test_pos = _half(normal_errors, fault_errors, normal_role, fault_role, TEST, fault_groups)
    out = dict(threshold=threshold, threshold_from=THRESHOLD_FROM,
               val=_summary(val_neg, val_pos, threshold), test=_summary(test_neg, test_pos, threshold))
    if fault_groups is not None:
        out["test_files"] = _summary(*_half(normal_errors, fault_errors, normal_role, fault_role, TEST), threshold)
    return out


def saved_metrics(cal):
    """The *_eval_metrics.npy dict: the frozen threshold, the test half's numbers at the top level
    (the keys bundle readers and older scripts know), and the val / test summaries in full."""
    return {"threshold": cal["threshold"], "threshold_from": cal["threshold_from"],
            "auroc": cal["test"]["auroc"], **{k: cal["test"][k] for k in SAVED_KEYS},
            **{k: cal[k] for k in ("val", "test", "test_files") if k in cal}}


def print_calibration(cal):
    print(f"\n  Threshold (best F1 on the validation half, frozen): {cal['threshold']:.6f}")
    for key, title in (("val", "Validation half (threshold chosen here)"),
                       ("test", "Test half (scored once at the frozen threshold)"),
                       ("test_files", "Test half, every fault file")):
        if key not in cal:
            continue
        m = cal[key]
        print(f"\n  {title}: {m['n_normal']} healthy, {m['n_fault']} fault")
        print(f"    AUROC {m['auroc']:.4f}   precision {m['precision']:.4f}   recall {m['recall']:.4f}"
              f"   F1 {m['f1']:.4f}   specificity {m['specificity']:.4f}")
        print(f"    TP {m['tp']}  FP {m['fp']}  FN {m['fn']}  TN {m['tn']}")


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

    # Threshold on the validation half, frozen; the test half scored once at it (HER-97)
    cal = calibrate(normal_errors, fault_errors,
                    dataset.calibration("test_normal"), dataset.calibration("test_fault"))
    print_calibration(cal)

    # Save results
    os.makedirs(RESULTS, exist_ok=True)
    np.save(RESULTS / "normal_errors.npy", normal_errors)
    np.save(RESULTS / "fault_errors.npy", fault_errors)
    np.save(RESULTS / "eval_metrics.npy", saved_metrics(cal))

    print(f"\nResults saved to {RESULTS}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
