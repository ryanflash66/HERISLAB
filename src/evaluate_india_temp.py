"""
Evaluate the temperature-scale autoencoder (HER-82 retrain, Oct 26).

Default: VAL (plot 16) only. --test runs TEST (plot 18) and CROSS_SITE once.
Everything declared before running:
  model scores   ae_peak3 (primary: max 3x3-averaged error / median error)
                 ae_mean  (the original average-error score, for the record)
  no-model       input_peak3 = brightest 3x3 spot in degrees above the frame's
                 median, in units of the frame's spread (same rule as before, now on C)
  thresholds     best F1 on VAL, frozen, applied to TEST (HER-97)
  comparison     the old model's numbers (per-frame stretched input) printed beside
Outputs: results/india_temp/{val|test}_eval_log.txt (stdout), *_scores.npy
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_india import at_threshold, auroc, best_f1_threshold, boot_ci  # noqa: E402
from train_autoencoder import ThermalAutoencoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "CA_Preprocessed" / "india_temp"
RES = ROOT / "results" / "india_temp"
WEIGHTS = ROOT / "models" / "autoencoder_india_temp_best.pth"
OLD = {"val": {"ae_peak3": 0.640, "ae_mean": 0.376, "input_peak3": 0.600},
       "test": {"ae_peak3": 0.597, "ae_mean": 0.373, "input_peak3": 0.585},
       "cross_site": {"ae_peak3": 0.346, "ae_mean": 0.332, "input_peak3": 0.610}}


def load(split, ns):
    x = np.load(DATA / f"{split}.npy")
    meta = list(csv.DictReader(open(DATA / f"{split}_meta.csv", encoding="utf-8")))
    y = np.array([1 if m["label"] == "faulty" else 0 for m in meta])
    return x, ((x - ns["mean"]) / ns["std"]).astype(np.float32), meta, y


@torch.no_grad()
def scores(model, xc, xz, dev):
    out = {"ae_peak3": [], "ae_mean": [], "input_peak3": []}
    for i in range(0, len(xz), 64):
        b = torch.from_numpy(xz[i:i + 64]).unsqueeze(1).to(dev)
        e = (model(b) - b) ** 2; flat = e.flatten(1)
        out["ae_mean"].append(flat.mean(1).cpu().numpy())
        out["ae_peak3"].append((Fn.avg_pool2d(e, 3, 1, 1).flatten(1).max(1).values / flat.median(1).values.clamp_min(1e-12)).cpu().numpy())
        t = torch.from_numpy(xc[i:i + 64]).unsqueeze(1).to(dev); tf = t.flatten(1)
        out["input_peak3"].append(((Fn.avg_pool2d(t, 3, 1, 1).flatten(1).max(1).values - tf.median(1).values) / tf.std(1).clamp_min(1e-6)).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def report(split, s, y, thr=None):
    print(f"\n======== {split.upper()} ({int(y.sum())} faulty / {int((y == 0).sum())} healthy)   [old model, stretched input, in brackets]")
    if thr is not None:
        fl = at_threshold(np.ones(len(y)), y, 0); print(f"  flag everything: F1 {fl['f1']:.3f}")
    for k, v in s.items():
        a = auroc(v, y); lo, hi = boot_ci(v, y)
        line = f"  {k:12s} AUROC {a:.3f} (95% CI {lo:.3f} to {hi:.3f})  [old {OLD[split][k]:.3f}]"
        if thr is not None:
            r = at_threshold(v, y, thr[k]); line += f" | precision {r['precision']:.3f} recall {r['recall']:.3f} F1 {r['f1']:.3f}"
        print(line + ("  <- primary" if k == "ae_peak3" else ""))
    p, q = s["ae_peak3"], s["input_peak3"]; rng = np.random.default_rng(0)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; d = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]); d.append(auroc(p[idx], y[idx]) - auroc(q[idx], y[idx]))
    print(f"  model minus no-model rule: {auroc(p, y) - auroc(q, y):+.3f} (95% CI {np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ns = np.load(DATA / "norm_stats.npy", allow_pickle=True).item()
    model = ThermalAutoencoder().to(dev); model.load_state_dict(torch.load(WEIGHTS, map_location=dev)); model.eval()
    xc, xz, meta, y = load("val", ns); sv = scores(model, xc, xz, dev)
    report("val", sv, y); np.save(RES / "val_scores.npy", sv, allow_pickle=True)
    if not args.test:
        return
    thr = {k: best_f1_threshold(v, y) for k, v in sv.items()}
    for split in ("test", "cross_site"):
        xc, xz, meta, y = load(split, ns); s = scores(model, xc, xz, dev)
        report(split, s, y, thr); np.save(RES / f"{split}_scores.npy", s, allow_pickle=True)


if __name__ == "__main__":
    main()
