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
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_india import WEIGHTS, at_threshold, auroc, best_f1_threshold, boot_ci, errors, load  # noqa: E402
from score_peak_india import scores  # noqa: E402
from train_autoencoder import ThermalAutoencoder  # noqa: E402


def all_scores(model, split, dev):
    x, meta, y = load(split)
    pk = scores(model, x, dev); er = errors(model, x, dev)
    f = lambda k: np.array([float(m[k]) if m[k] not in ("", "nan") else np.nan for m in meta])
    return y, {
        "ae_peak3 (MODEL)": pk["peak3"],
        "ae_mean (original)": er["mean"],
        "input_peak3 (no model)": pk["input_peak3 (no model)"],
        "temp p99 C (no model)": f("t_p99"),
        "hotspot delta C (no model)": f("t_p99") - f("t_median"),
    }


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalAutoencoder().to(dev)
    model.load_state_dict(torch.load(WEIGHTS, map_location=dev)); model.eval()
    yv, sv = all_scores(model, "val", dev)
    thr = {k: best_f1_threshold(np.nan_to_num(v, nan=-1e9), yv) for k, v in sv.items()}
    for split in ("test", "cross_site"):
        y, s = all_scores(model, split, dev)
        print(f"\n======== {split.upper()}  ({int(y.sum())} faulty / {int((y == 0).sum())} healthy)")
        fl = at_threshold(np.ones(len(y)), y, 0)
        print(f"  flag everything: F1 {fl['f1']:.3f}, precision {fl['precision']:.3f}")
        for k, v in s.items():
            ok = ~np.isnan(v)
            a = auroc(v[ok], y[ok]); lo, hi = boot_ci(v[ok], y[ok])
            r = at_threshold(v[ok], y[ok], thr[k])
            print(f"  {k:28s} AUROC {a:.3f} (95% CI {lo:.3f} to {hi:.3f}) | at val threshold: "
                  f"precision {r['precision']:.3f} recall {r['recall']:.3f} F1 {r['f1']:.3f} specificity {r['specificity']:.3f}"
                  f"{'' if ok.all() else f'  [{(~ok).sum()} frames without temperature skipped]'}")


if __name__ == "__main__":
    main()
