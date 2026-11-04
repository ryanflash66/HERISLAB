"""
Evaluate the India autoencoder (milestone 06: HER-97, HER-98).

  - anomaly score (primary, as planned) = mean squared reconstruction error per image
  - secondary scores (added Oct 25 before seeing model results): p99 and p99.9
    of the per-pixel error map, because faults are small (median 1 box per frame)
    and a whole-image mean can drown them; thresholds for every score come from val
  - threshold chosen on VAL (plot 16) at best F1, then frozen (HER-97)
  - reported once on TEST (plot 18) and on CROSS_SITE (certified, other sites)
  - next to every result (HER-98): flag-everything F1, specificity, and
    no-model baselines whose direction is also fixed on val:
      image stats on the model input (mean, std, p1, p99)
      temperature stats (p99, max, hotspot delta = p99 - median)  [no model at all]
  - test AUROC with a 95% bootstrap interval

Outputs: results/india/eval_log.txt (stdout), results/india/{split}_errors.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset  # noqa: E402
from hlab.metrics import auroc, auroc_ci, best_f1_threshold, confusion  # noqa: E402
from hlab.panels import resolve_device  # noqa: E402
from hlab.paths import RESULTS  # noqa: E402

RES = RESULTS / "india"


def load(ds, split):
    s = ds.split(split)
    return s.x, s.meta, s.labels


def errors(model, x, dev):
    """Return dict of per-image scores from the per-pixel squared error map."""
    out = {"mean": [], "p99": [], "p99.9": []}
    with torch.no_grad():
        for i in range(0, len(x), 64):
            b = torch.from_numpy(x[i:i + 64]).unsqueeze(1).to(dev)
            e = ((model(b) - b) ** 2).flatten(1)
            out["mean"].append(e.mean(1).cpu().numpy())
            q = torch.quantile(e.float(), torch.tensor([0.99, 0.999], device=dev), dim=1)
            out["p99"].append(q[0].cpu().numpy()); out["p99.9"].append(q[1].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def baselines(x, meta):
    flat = x.reshape(len(x), -1)
    f = lambda k: np.array([float(m[k]) for m in meta])
    return {
        "img mean": flat.mean(1), "img std": flat.std(1),
        "img p1": np.percentile(flat, 1, axis=1), "img p99": np.percentile(flat, 99, axis=1),
        "temp p99 (C)": f("t_p99"), "temp max (C)": f("t_max"),
        "hotspot delta p99-median (C)": f("t_p99") - f("t_median"),
    }


def main(device=None):
    RES.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)
    model = Bundle.load("india", dev).model
    ds = Dataset("india")

    data = {s: load(ds, s) for s in ("val", "test", "cross_site")}
    err = {}
    for s, (x, meta, y) in data.items():
        err[s] = errors(model, x, dev)
        np.save(RES / f"{s}_errors.npy", err[s]["mean"])
        np.save(RES / f"{s}_errors_p999.npy", err[s]["p99.9"])

    yv = data["val"][2]
    for score in ("mean", "p99", "p99.9"):
        tag = "PRIMARY" if score == "mean" else "secondary"
        thr = best_f1_threshold(err["val"][score], yv)
        print(f"######## score = {score} pixel error ({tag}); threshold from VAL: {thr:.6f}\n")
        for s in ("val", "test", "cross_site"):
            y = data[s][2]; e = err[s][score]
            m = confusion(e, y, thr)
            flag_all = confusion(np.ones(len(y)), y, 0)["f1"]
            ci = auroc_ci(e, y, n_boot=1000, seed=0) if s == "test" else None
            print(f"== {s.upper()}  ({len(y)-y.sum()} healthy / {y.sum()} faulty)")
            print(f"  AUROC {auroc(e, y):.3f}" + (f"  (95% CI {ci[0]:.3f} to {ci[1]:.3f})" if ci is not None else ""))
            print(f"  at frozen threshold: precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
                  f"F1 {m['f1']:.3f}  specificity {m['specificity']:.3f}")
            print(f"  TP {m['tp']}  FP {m['fp']}  FN {m['fn']}  TN {m['tn']}")
            print(f"  flag-everything F1: {flag_all:.3f}\n")

    yt = data["test"][2]
    print("== NO-MODEL BASELINES (direction fixed on val, AUROC on test)")
    bv = baselines(data["val"][0], data["val"][1]); bt = baselines(data["test"][0], data["test"][1])
    for k in bv:
        if np.isnan(bv[k]).any():
            continue
        sign = 1 if auroc(bv[k], yv) >= 0.5 else -1
        print(f"  {k:32s} val {auroc(sign*bv[k], yv):.3f}   test {auroc(sign*bt[k], yt):.3f}")
    for score in ("mean", "p99", "p99.9"):
        print(f"  {'AUTOENCODER ' + score + ' error':32s} val {auroc(err['val'][score], yv):.3f}   test {auroc(err['test'][score], yt):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda when available)")
    main(ap.parse_args().device)
