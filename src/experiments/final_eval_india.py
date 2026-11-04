"""
Final one-shot evaluation on TEST (plot 18) and CROSS_SITE, Oct 26.
Frozen before running, from the plot-16 work on Oct 25 (nothing below was tuned on test):

  model score   ae_peak3  = autoencoder, max of 3x3-averaged error / median error
                            (best pre-declared model score on plot 16: 0.640)
  original      ae_mean   = mean error per frame (the original primary, test 0.373 on Oct 25)
  baselines     input_peak3 (no model), temp p99 (C), hotspot delta p99-median (C)
  threshold     best F1 on VAL for each score, then frozen (HER-97)
  also          flag-everything F1 (HER-98)

Output: results/india/final_eval_log.txt (stdout)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset  # noqa: E402
from hlab.metrics import auroc, auroc_ci, best_f1_threshold, confusion  # noqa: E402
from hlab.panels import peak_scores, resolve_device  # noqa: E402


def all_scores(bundle, ds, split):
    s = ds.split(split)
    pk = peak_scores(bundle.model, s.x, bundle.device)
    f = lambda k: np.array([float(m[k]) if m[k] not in ("", "nan") else np.nan for m in s.meta])
    return s.labels, {
        "ae_peak3 (MODEL)": pk["peak3"],
        "ae_mean (original)": bundle.score_normalized(s.x, batch_size=64),
        "input_peak3 (no model)": pk["input_peak3 (no model)"],
        "temp p99 C (no model)": f("t_p99"),
        "hotspot delta C (no model)": f("t_p99") - f("t_median"),
    }


def main(device=None):
    bundle = Bundle.load("india", resolve_device(device))
    ds = Dataset("india")
    yv, sv = all_scores(bundle, ds, "val")
    thr = {k: best_f1_threshold(np.nan_to_num(v, nan=-1e9), yv) for k, v in sv.items()}
    for split in ("test", "cross_site"):
        y, s = all_scores(bundle, ds, split)
        print(f"\n======== {split.upper()}  ({int(y.sum())} faulty / {int((y == 0).sum())} healthy)")
        fl = confusion(np.ones(len(y)), y, 0)
        print(f"  flag everything: F1 {fl['f1']:.3f}, precision {fl['precision']:.3f}")
        for k, v in s.items():
            ok = ~np.isnan(v)
            a = auroc(v[ok], y[ok]); lo, hi = auroc_ci(v[ok], y[ok], n_boot=1000, seed=0)
            r = confusion(v[ok], y[ok], thr[k])
            print(f"  {k:28s} AUROC {a:.3f} (95% CI {lo:.3f} to {hi:.3f}) | at val threshold: "
                  f"precision {r['precision']:.3f} recall {r['recall']:.3f} F1 {r['f1']:.3f} specificity {r['specificity']:.3f}"
                  f"{'' if ok.all() else f'  [{(~ok).sum()} frames without temperature skipped]'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default=None)
    main(ap.parse_args().device)
